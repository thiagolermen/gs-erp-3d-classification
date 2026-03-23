"""
Radiance field ERP generation from 3D Gaussian Splats.

Instead of ray-mesh intersection (HSDC/SWHDC papers), we evaluate the
continuous 3DGS density field at N concentric sphere shells using
EgoNeRF-inspired exponential radial spacing.

Shell radii follow the EgoNeRF exponential spacing scheme
(Choi et al., CVPR 2023, §3.2):

    r_s = r_near * (r_far / r_near)^(s / (N-1)),  s = 0..N-1

where r_near and r_far are derived from percentiles of the Gaussian centre
distances from the object centroid (default: 5th and 95th percentiles).

For each shell at radius r_s, sample points are placed on the sphere surface:

    P(u, v) = centroid + r_s * ray_direction(u, v)

The density at a sample point p is the sum of all Gaussian contributions
evaluated via the 3DGS volumetric density formula
(Kerbl et al., SIGGRAPH 2023, eq. 3):

    density(p) = sum_i [ opacity_i * exp(-0.5 * mahalanobis²(p, Gaussian_i)) ]

where the squared Mahalanobis distance uses the Gaussian's rotation (from
its quaternion) and scale:

    mahalanobis²(p, i) = || R_i^T (p - mu_i) / s_i ||²  (element-wise division)

Spatial culling reduces the per-shell computation: only Gaussians whose
centre distance from the centroid is within cutoff_sigma * max_scale of the
current shell radius are evaluated.

References:
    Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field
    Rendering", SIGGRAPH 2023.
    Choi et al., "Balanced Spherical Grid for Egocentric View Synthesis",
    CVPR 2023.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.preprocessing.ply_loader import load_gaussian_ply


# ---------------------------------------------------------------------------
# Centroid
# ---------------------------------------------------------------------------

def compute_centroid(xyz: np.ndarray, opacity: np.ndarray) -> np.ndarray:
    """Compute the opacity-weighted centroid of a 3D Gaussian cloud.

    The centroid is the opacity-weighted mean of all Gaussian centre positions,
    analogous to the area-weighted centroid used for meshes in the HSDC paper
    §II-A but adapted for the 3DGS representation.

    Args:
        xyz:     (N, 3) float32/float64 — Gaussian centre positions.
        opacity: (N,)   float32/float64 — per-Gaussian opacity in [0, 1].

    Returns:
        centroid: (3,) float64 — opacity-weighted 3-D centroid.
    """
    xyz = xyz.astype(np.float64)
    w = opacity.astype(np.float64)
    total = w.sum()
    if total < 1e-12:
        # Degenerate case: all opacities near zero → use unweighted mean
        return xyz.mean(axis=0)
    # Opacity-weighted average: centroid = sum(w_i * xyz_i) / sum(w_i)
    return (w[:, np.newaxis] * xyz).sum(axis=0) / total  # (3,) float64


# ---------------------------------------------------------------------------
# Ray directions
# ---------------------------------------------------------------------------

def build_ray_directions(H: int, W: int) -> np.ndarray:
    """Build unit ray directions for a (H, W) ERP pixel grid.

    Maps each ERP pixel (u, v) to a unit 3-D direction using the standard
    equirectangular projection (ERP) camera model.

    Coordinate convention (matching HSDC paper §II-A, Eq. 1-2):
        theta = (u / W) * 2*pi - pi       — azimuth,   in [-pi, pi)
        phi   = pi/2 - (v / H) * pi       — elevation, in [-pi/2, pi/2]

    Direction vector:
        d_x = cos(phi) * cos(theta)
        d_y = cos(phi) * sin(theta)
        d_z = sin(phi)

    Args:
        H: ERP image height (number of rows, default 256).
        W: ERP image width  (number of columns, default 512).

    Returns:
        ray_dirs: (H*W, 3) float32 — unit direction per pixel, row-major
                  (row 0 is the top of the ERP image).
    """
    # Pixel-centre sampling
    u = np.arange(W, dtype=np.float64)  # (W,)
    v = np.arange(H, dtype=np.float64)  # (H,)
    u_grid, v_grid = np.meshgrid(u, v)  # (H, W) each

    # ERP → spherical angles
    # EgoNeRF / standard ERP: theta=0 is the image centre column
    theta = (u_grid / W) * 2.0 * np.pi - np.pi   # azimuth  in [-pi, pi)
    phi   = np.pi / 2.0 - (v_grid / H) * np.pi   # elevation in [pi/2, -pi/2]

    # Spherical → Cartesian unit direction
    cos_phi = np.cos(phi)
    d_x = cos_phi * np.cos(theta)
    d_y = cos_phi * np.sin(theta)
    d_z = np.sin(phi)

    # Stack and flatten: (H*W, 3) float32
    dirs = np.stack([d_x, d_y, d_z], axis=-1).reshape(-1, 3).astype(np.float32)
    return dirs


# ---------------------------------------------------------------------------
# Shell radii
# ---------------------------------------------------------------------------

def compute_shell_radii(
    r_dist: np.ndarray,
    n_shells: int,
    r_near_pct: float = 10.0,
    r_far_pct: float = 90.0,
) -> np.ndarray:
    """Compute EgoNeRF-style exponential radial shell spacing.

    Shell radii follow the exponential scheme from Choi et al., CVPR 2023, §3.2:

        r_s = r_near * (r_far / r_near)^(s / (N-1)),  s = 0, 1, ..., N-1

    r_near and r_far are derived from percentiles of the Gaussian centre
    distances from the object centroid, making the shell spacing adaptive to
    the actual object extent.

    Args:
        r_dist:    (N,) float array — distance of each Gaussian centre from
                   the object centroid.
        n_shells:  Number of concentric shells (= number of ERP channels).
        r_near_pct: Percentile of r_dist to use as r_near (default 10.0).
        r_far_pct:  Percentile of r_dist to use as r_far  (default 90.0).

    Returns:
        radii: (n_shells,) float32 — shell radii in ascending order.

    Note:
        If r_near == r_far (degenerate object with no spatial extent), falls
        back to linearly spaced radii over [r_near * 0.5, r_near * 1.5].
    """
    r_near = float(np.percentile(r_dist, r_near_pct))
    r_far  = float(np.percentile(r_dist, r_far_pct))

    # Guard against r_near = 0
    if r_near < 1e-12:
        r_near = max(float(r_dist.min()), 1e-6)

    if abs(r_far - r_near) < 1e-9 or r_far <= r_near:
        # Degenerate: fall back to linspace around r_near
        radii = np.linspace(r_near * 0.5, r_near * 1.5, n_shells, dtype=np.float32)
    else:
        # EgoNeRF exponential spacing — Choi et al. CVPR 2023, §3.2
        # r_s = r_near * (r_far / r_near)^(s / (N-1))
        s = np.arange(n_shells, dtype=np.float64)
        radii = (r_near * (r_far / r_near) ** (s / max(n_shells - 1, 1))).astype(
            np.float32
        )

    return radii


# ---------------------------------------------------------------------------
# Quaternion → rotation matrix
# ---------------------------------------------------------------------------

def quaternions_to_rotation_matrices(quats: np.ndarray) -> np.ndarray:
    """Convert an array of quaternions (w, x, y, z) to rotation matrices.

    Vectorised implementation using the standard formula
    (Kerbl et al., SIGGRAPH 2023, supplemental material).

    Args:
        quats: (N, 4) float array — quaternions in (w, x, y, z) order.
               Need not be pre-normalised; normalisation is applied internally.

    Returns:
        R: (N, 3, 3) float32 — corresponding rotation matrices.
    """
    q = quats.astype(np.float64)
    # Normalise each quaternion
    norms = np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-12)
    q = q / norms  # (N, 4)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Standard quaternion → rotation matrix (row-major)
    # Reference: Kerbl et al. SIGGRAPH 2023 supplemental
    R = np.empty((len(q), 3, 3), dtype=np.float32)

    R[:, 0, 0] = (1.0 - 2.0 * (y * y + z * z)).astype(np.float32)
    R[:, 0, 1] = (2.0 * (x * y - w * z)).astype(np.float32)
    R[:, 0, 2] = (2.0 * (x * z + w * y)).astype(np.float32)

    R[:, 1, 0] = (2.0 * (x * y + w * z)).astype(np.float32)
    R[:, 1, 1] = (1.0 - 2.0 * (x * x + z * z)).astype(np.float32)
    R[:, 1, 2] = (2.0 * (y * z - w * x)).astype(np.float32)

    R[:, 2, 0] = (2.0 * (x * z - w * y)).astype(np.float32)
    R[:, 2, 1] = (2.0 * (y * z + w * x)).astype(np.float32)
    R[:, 2, 2] = (1.0 - 2.0 * (x * x + y * y)).astype(np.float32)

    return R  # (N, 3, 3)


# ---------------------------------------------------------------------------
# Precompute Gaussian parameters
# ---------------------------------------------------------------------------

def precompute_gaussian_params(
    gs: dict,
    centroid: np.ndarray,
) -> dict:
    """Precompute per-Gaussian parameters for fast radiance field evaluation.

    Derives quantities needed by the inner loop of compute_radiance_field_erp:
      - R^T / scale: the pre-scaled inverse rotation used in the Mahalanobis
        distance computation (Kerbl et al. SIGGRAPH 2023, eq. 3).
      - r_dist: per-Gaussian distance from centroid (for spatial culling).
      - max_scale: maximum scale axis per Gaussian (for culling radius).

    The Mahalanobis distance is:
        mahal²(p, i) = || R_i^T (p - mu_i) / s_i ||²
    which we rewrite as:
        local = einsum('kj, pj -> pk', Rt_scaled[i], diff)  where diff = p - mu_i

    Rt_scaled[i, k, j] = R_i^T[k, j] / scale[i, k]

    Args:
        gs:       Dict returned by load_gaussian_ply (must contain 'xyz',
                  'opacity', 'rgb', 'rotation', 'scale').
        centroid: (3,) float64 — object centroid (from compute_centroid).

    Returns:
        A dict containing:
            'xyz'       : (N, 3) float32  — Gaussian centres.
            'opacity'   : (N,)   float32  — per-Gaussian opacity in [0, 1].
            'rgb'       : (N, 3) float32  — per-Gaussian colour.
            'Rt_scaled' : (N, 3, 3) float32 — R_i^T with each row divided by
                          the corresponding scale axis.
            'r_dist'    : (N,)   float32  — ||xyz[i] - centroid||.
            'max_scale' : (N,)   float32  — max(scale[i]) across axes.
    """
    xyz = gs["xyz"].astype(np.float32)
    opacity = gs["opacity"].astype(np.float32)
    rgb = gs["rgb"].astype(np.float32)
    scale = gs["scale"].astype(np.float32)          # (N, 3)
    rotation = gs["rotation"].astype(np.float32)    # (N, 4) — (w,x,y,z)

    centroid_f32 = centroid.astype(np.float32)

    # Distance of each Gaussian centre from the object centroid (for culling)
    diff_to_centroid = xyz - centroid_f32            # (N, 3)
    r_dist = np.linalg.norm(diff_to_centroid, axis=1).astype(np.float32)  # (N,)

    # Maximum scale per Gaussian (culling radius)
    max_scale = scale.max(axis=1).astype(np.float32)  # (N,)

    # Build rotation matrices R: (N, 3, 3)
    R = quaternions_to_rotation_matrices(rotation)  # (N, 3, 3)

    # R^T: (N, 3, 3) — transpose each matrix
    Rt = R.transpose(0, 2, 1)  # (N, 3, 3)

    # Pre-scale: Rt_scaled[i, k, j] = Rt[i, k, j] / scale[i, k]
    # scale has shape (N, 3); we need to divide each row k of Rt[i] by scale[i, k]
    # Rt[:, k, :] / scale[:, k, np.newaxis] for each k
    scale_safe = np.where(scale < 1e-12, 1e-12, scale)  # avoid division by zero
    Rt_scaled = Rt / scale_safe[:, :, np.newaxis]  # (N, 3, 3)

    return {
        "xyz": xyz,
        "opacity": opacity,
        "rgb": rgb,
        "Rt_scaled": Rt_scaled.astype(np.float32),
        "r_dist": r_dist,
        "max_scale": max_scale,
    }


# ---------------------------------------------------------------------------
# Radiance field ERP
# ---------------------------------------------------------------------------

def compute_radiance_field_erp(
    gs_precomp: dict,
    centroid: np.ndarray,
    ray_dirs: np.ndarray,       # (H*W, 3)
    shell_radii: np.ndarray,    # (N_shells,)
    H: int,
    W: int,
    cutoff_sigma: float = 3.0,
    batch_size: int = 4096,
    device: Optional[str] = None,
    add_color: bool = False,
) -> np.ndarray:
    """Evaluate the 3DGS radiance field at N concentric sphere shells.

    For each shell s at radius r_s, sample points are:

        P_s = centroid + r_s * ray_dirs           (H*W, 3)

    The accumulated density at each sample point is:

        density[s, p] = sum_i [ opacity_i * exp(-0.5 * mahal²(P_s[p], i)) ]

    where:
        mahal²(p, i) = || Rt_scaled[i] @ (p - xyz[i]) ||²

    Only Gaussians satisfying the spatial culling criterion are evaluated:

        |r_dist[i] - r_s| < cutoff_sigma * max_scale[i]

    Algorithm (pixel-chunk tiling):
        For each pixel chunk P, precompute A[i,p,k] = Rt_scaled[i,k,:] · dirs[p,:]
        once and reuse it for all N_shells iterations — only the scaling by r_s
        and the per-Gaussian offset b[i] = Rt_scaled[i] @ (xyz[i] - centroid)
        change between shells.  This reduces memory allocation by ~(N_shells × B)
        compared to the Gaussian-batch formulation and enables GPU acceleration.

    Args:
        gs_precomp:  Dict from precompute_gaussian_params.
        centroid:    (3,) float64 — object centroid.
        ray_dirs:    (H*W, 3) float32 — unit ray directions.
        shell_radii: (N_shells,) float32 — shell radii.
        H:           ERP height.
        W:           ERP width.
        cutoff_sigma: Truncation radius in units of max_scale (default 3.0).
        batch_size:  Pixels per processing chunk (default 4096).
        device:      Torch device string ('cuda', 'cpu') or None for auto-detect.
                     When None, CUDA is used if available, else CPU numpy.
        add_color:   If True, also accumulates opacity-weighted RGB in a single
                     pass and appends 3 colour channels after the density shells.
                     Returns (N_shells + 3, H, W) when True (default False).

    Returns:
        erp: (N_shells, H, W) float32 if add_color=False;
             (N_shells + 3, H, W) float32 if add_color=True — density first, RGB last.
    """
    # ── resolve device ───────────────────────────────────────────────────────
    use_torch = _TORCH_AVAILABLE
    if use_torch:
        if device is None:
            import torch as _torch
            dev = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        else:
            import torch as _torch
            dev = _torch.device(device)
        use_torch = True
    else:
        dev = None

    if use_torch and dev is not None and dev.type == "cuda":
        import torch as _torch
        # Try GPU with progressively smaller batch sizes, then fall back to CPU.
        bs = batch_size
        while bs >= 64:
            try:
                _torch.cuda.empty_cache()
                result = _compute_erp_torch(
                    gs_precomp, centroid, ray_dirs, shell_radii,
                    H, W, cutoff_sigma, bs, dev, add_color=add_color,
                )
                return result
            except _torch.cuda.OutOfMemoryError:
                _torch.cuda.empty_cache()
                bs //= 2
                import logging
                logging.getLogger(__name__).warning(
                    "CUDA OOM with batch_size=%d, retrying with %d", bs * 2, bs,
                )
        # All GPU attempts failed — fall back to CPU numpy
        import logging
        logging.getLogger(__name__).warning(
            "CUDA OOM persists at batch_size=%d; falling back to CPU numpy.", bs,
        )
        return _compute_erp_numpy(
            gs_precomp, centroid, ray_dirs, shell_radii,
            H, W, cutoff_sigma, batch_size, add_color=add_color,
        )
    elif use_torch:
        return _compute_erp_torch(
            gs_precomp, centroid, ray_dirs, shell_radii,
            H, W, cutoff_sigma, batch_size, dev, add_color=add_color,
        )
    return _compute_erp_numpy(
        gs_precomp, centroid, ray_dirs, shell_radii,
        H, W, cutoff_sigma, batch_size, add_color=add_color,
    )


def _compute_erp_numpy(
    gs_precomp: dict,
    centroid: np.ndarray,
    ray_dirs: np.ndarray,
    shell_radii: np.ndarray,
    H: int,
    W: int,
    cutoff_sigma: float,
    batch_size: int,
    add_color: bool = False,
) -> np.ndarray:
    """CPU numpy implementation — pixel-chunk tiling."""
    xyz       = gs_precomp["xyz"]        # (N, 3)
    opacity   = gs_precomp["opacity"]    # (N,)
    rgb       = gs_precomp["rgb"]        # (N, 3)  — only used when add_color=True
    Rt_scaled = gs_precomp["Rt_scaled"]  # (N, 3, 3)
    r_dist    = gs_precomp["r_dist"]     # (N,)
    max_scale = gs_precomp["max_scale"]  # (N,)

    n_pixels = H * W
    cutoff2  = float(cutoff_sigma ** 2)
    centroid_f32 = centroid.astype(np.float32)

    # Pre-compute b[i] = Rt_scaled[i] @ (xyz[i] - centroid) — once per sample.
    # b[i,k] is the Gaussian mean in its own scaled local frame.
    delta = xyz - centroid_f32                             # (N, 3)
    b     = np.einsum("ikj,ij->ik", Rt_scaled, delta)     # (N, 3)

    erp_acc = np.zeros((len(shell_radii), n_pixels), dtype=np.float32)
    color_acc = np.zeros((3, n_pixels), dtype=np.float32) if add_color else None

    for p0 in range(0, n_pixels, batch_size):
        p1          = min(p0 + batch_size, n_pixels)
        dirs_chunk  = ray_dirs[p0:p1]                      # (P, 3)

        # A[i,p,k] = Rt_scaled[i,k,:] · dirs_chunk[p,:]  — shared for all shells.
        # Kerbl et al. SIGGRAPH 2023, eq. 3 (rotation part only).
        A_all = np.einsum("ikj,pj->ipk", Rt_scaled, dirs_chunk)  # (N, P, 3)

        for s_idx, r_s in enumerate(shell_radii):
            r_s = float(r_s)

            # Spatial culling: Gaussians whose extent overlaps this shell radius
            cull  = np.abs(r_dist - r_s) < cutoff_sigma * max_scale   # (N,) bool
            if not cull.any():
                continue

            A_c  = A_all[cull]      # (M, P, 3)
            b_c  = b[cull]          # (M, 3)
            op_c = opacity[cull]    # (M,)

            # mahal[m,p,k] = r_s * A_c[m,p,k] - b_c[m,k]
            # Kerbl et al. SIGGRAPH 2023, eq. 3
            mahal  = r_s * A_c - b_c[:, np.newaxis, :]    # (M, P, 3)
            mahal2 = (mahal * mahal).sum(axis=-1)          # (M, P)

            contrib = np.where(
                mahal2 < cutoff2,
                op_c[:, np.newaxis] * np.exp(-0.5 * mahal2),
                0.0,
            )
            erp_acc[s_idx, p0:p1] += contrib.sum(axis=0)
            if add_color and color_acc is not None:
                rgb_c = rgb[cull]  # (M, 3)
                color_acc[:, p0:p1] += np.einsum("mp,mc->cp", contrib, rgb_c)

    density_erp = erp_acc.reshape(len(shell_radii), H, W)
    if not add_color or color_acc is None:
        return density_erp
    # Normalize color by total accumulated density
    total_density = erp_acc.sum(0, keepdims=True)  # (1, n_pixels)
    color_erp = (color_acc / (total_density + 1e-8)).reshape(3, H, W)
    return np.concatenate([density_erp, color_erp], axis=0)  # (N_shells+3, H, W)


def _compute_erp_torch(
    gs_precomp: dict,
    centroid: np.ndarray,
    ray_dirs: np.ndarray,
    shell_radii: np.ndarray,
    H: int,
    W: int,
    cutoff_sigma: float,
    batch_size: int,
    dev: "torch.device",  # noqa: F821
    add_color: bool = False,
) -> np.ndarray:
    """GPU/CPU torch implementation — pixel-chunk tiling."""
    import torch

    xyz       = torch.from_numpy(gs_precomp["xyz"]).to(dev)        # (N, 3)
    opacity   = torch.from_numpy(gs_precomp["opacity"]).to(dev)    # (N,)
    Rt_scaled = torch.from_numpy(gs_precomp["Rt_scaled"]).to(dev)  # (N, 3, 3)
    r_dist    = torch.from_numpy(gs_precomp["r_dist"]).to(dev)     # (N,)
    max_scale = torch.from_numpy(gs_precomp["max_scale"]).to(dev)  # (N,)
    dirs_t    = torch.from_numpy(ray_dirs).to(dev)                  # (H*W, 3)
    centroid_t = torch.from_numpy(centroid.astype(np.float32)).to(dev)
    rgb_t     = torch.from_numpy(gs_precomp["rgb"]).to(dev) if add_color else None  # (N, 3)

    n_pixels = H * W
    cutoff2  = float(cutoff_sigma ** 2)
    n_shells = len(shell_radii)

    # b[i] = Rt_scaled[i] @ (xyz[i] - centroid) — once per sample
    delta = xyz - centroid_t                                        # (N, 3)
    b     = torch.einsum("ikj,ij->ik", Rt_scaled, delta)           # (N, 3)

    erp_acc   = torch.zeros((n_shells, n_pixels), device=dev, dtype=torch.float32)
    color_acc = torch.zeros((3, n_pixels), device=dev, dtype=torch.float32) if add_color else None

    for p0 in range(0, n_pixels, batch_size):
        p1         = min(p0 + batch_size, n_pixels)
        dirs_chunk = dirs_t[p0:p1]                                  # (P, 3)

        # A[i,p,k] = Rt_scaled[i,k,:] · dirs_chunk[p,:] — shared for all shells
        A_all = torch.einsum("ikj,pj->ipk", Rt_scaled, dirs_chunk)  # (N, P, 3)

        for s_idx, r_s in enumerate(shell_radii):
            r_s = float(r_s)

            cull = (r_dist - r_s).abs() < cutoff_sigma * max_scale  # (N,) bool
            if not cull.any():
                continue

            A_c  = A_all[cull]    # (M, P, 3)
            b_c  = b[cull]        # (M, 3)
            op_c = opacity[cull]  # (M,)

            mahal  = r_s * A_c - b_c.unsqueeze(1)                   # (M, P, 3)
            mahal2 = (mahal * mahal).sum(dim=-1)                     # (M, P)
            del mahal

            contrib = torch.where(
                mahal2 < cutoff2,
                op_c.unsqueeze(1) * torch.exp(-0.5 * mahal2),
                torch.zeros_like(mahal2),
            )
            del mahal2
            erp_acc[s_idx, p0:p1] += contrib.sum(dim=0)
            if add_color and color_acc is not None and rgb_t is not None:
                rgb_c = rgb_t[cull]  # (M, 3)
                color_acc[:, p0:p1] += torch.einsum("mp,mc->cp", contrib, rgb_c)
            del contrib, A_c, b_c, op_c

        del A_all

    density_erp = erp_acc.reshape(n_shells, H, W).cpu().numpy()
    if not add_color or color_acc is None:
        del erp_acc, xyz, opacity, Rt_scaled, r_dist, max_scale, dirs_t, b
        return density_erp
    # Normalize color by total accumulated density
    total_density = erp_acc.sum(0, keepdim=True)  # (1, n_pixels)
    color_erp = (color_acc / (total_density + 1e-8)).reshape(3, H, W).cpu().numpy()
    del erp_acc, color_acc, xyz, opacity, Rt_scaled, r_dist, max_scale, dirs_t, b
    return np.concatenate([density_erp, color_erp], axis=0)  # (N_shells+3, H, W)


# ---------------------------------------------------------------------------
# Full pipeline convenience wrapper
# ---------------------------------------------------------------------------

def gaussian_ply_to_erp(
    ply_path: Path,
    n_shells: int = 8,
    H: int = 256,
    W: int = 512,
    cutoff_sigma: float = 3.0,
    batch_size: int = 4096,
    r_near_pct: float = 10.0,
    r_far_pct: float = 90.0,
    min_opacity: float = 0.05,
    add_color: bool = False,
    device: Optional[str] = None,
) -> np.ndarray:
    """Full pipeline: PLY file -> (N_shells, H, W) radiance field ERP.

    Convenience wrapper that calls load_gaussian_ply, compute_centroid,
    build_ray_directions, compute_shell_radii, precompute_gaussian_params,
    and compute_radiance_field_erp in sequence.

    The returned ERP contains raw accumulated densities (not normalised to
    [0, 1]); normalisation should be applied downstream (e.g. in the Dataset).

    Args:
        ply_path:    Path to a 3DGS point_cloud.ply file.
        n_shells:    Number of concentric sphere shells (ERP channels).
        H:           ERP height in pixels (default 256).
        W:           ERP width  in pixels (default 512).
        cutoff_sigma: Gaussian truncation radius in max_scale units (default 3.0).
        batch_size:  Pixels per processing chunk (default 4096).
        r_near_pct:  Percentile of Gaussian distances used as r_near (default 10).
        r_far_pct:   Percentile of Gaussian distances used as r_far  (default 90).
        min_opacity: Minimum opacity threshold for filtering floater Gaussians
                     before computing centroid and shell radii. Gaussians with
                     opacity <= min_opacity are discarded. Default: 0.05.
        add_color:   If True, appends 3 opacity-weighted RGB channels after the
                     density shells. Returns (N_shells + 3, H, W) instead of
                     (N_shells, H, W). Default: False.
        device:      Torch device ('cuda', 'cpu') or None (auto-detect).

    Returns:
        erp: (N_shells, H, W) float32 if add_color=False;
             (N_shells + 3, H, W) float32 if add_color=True — density first, RGB last.

    Raises:
        FileNotFoundError: If *ply_path* does not exist.
        ValueError:        If the PLY file is malformed.
    """
    import logging as _logging
    ply_path = Path(ply_path)

    gs = load_gaussian_ply(ply_path)

    # Filter low-opacity floater Gaussians before computing centroid and shells.
    # Floaters contaminate the opacity-weighted centroid and inflate r_far,
    # causing outer shells to land in empty space.
    if min_opacity > 0.0:
        mask = gs["opacity"] > min_opacity
        if mask.sum() < 10:
            # Too few Gaussians after filtering — use all as fallback
            _logging.getLogger(__name__).warning(
                "min_opacity=%.3f filtered %d/%d Gaussians in '%s'; "
                "using all Gaussians as fallback.",
                min_opacity, int((~mask).sum()), len(mask), ply_path,
            )
        else:
            gs = {k: (v[mask] if isinstance(v, np.ndarray) else int(mask.sum()))
                  for k, v in gs.items()}

    centroid   = compute_centroid(gs["xyz"], gs["opacity"])
    ray_dirs   = build_ray_directions(H, W)           # (H*W, 3)
    gs_precomp = precompute_gaussian_params(gs, centroid)
    shell_radii = compute_shell_radii(
        gs_precomp["r_dist"],
        n_shells=n_shells,
        r_near_pct=r_near_pct,
        r_far_pct=r_far_pct,
    )

    return compute_radiance_field_erp(
        gs_precomp=gs_precomp,
        centroid=centroid,
        ray_dirs=ray_dirs,
        shell_radii=shell_radii,
        H=H,
        W=W,
        cutoff_sigma=cutoff_sigma,
        batch_size=batch_size,
        device=device,
        add_color=add_color,
    )

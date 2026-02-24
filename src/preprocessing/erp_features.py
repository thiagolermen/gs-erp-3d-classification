"""
ERP feature extraction for the HSDC and SWHDC classification pipelines.

HSDC pipeline (12 channels):
    Each object is represented by first and last ray-mesh intersection points.
    For each intersection: depth (normalized), surface normals (Nx, Ny, Nz),
    ray-normal alignment (cosine), and gradient magnitude of the depth map
    (Gaussian derivatives, σ=2).  Zero-hit pixels are 0 in all channels.

    Channel layout (HSDC paper §II-B):
        0  d₁      — first-hit depth (normalized)
        1  Nx₁     — first-hit normal x
        2  Ny₁     — first-hit normal y
        3  Nz₁     — first-hit normal z
        4  align₁  — cos(ray, normal) at first hit
        5  grad₁   — gradient magnitude of d₁
        6  dₙ      — last-hit depth (normalized)
        7  Nxₙ     — last-hit normal x
        8  Nyₙ     — last-hit normal y
        9  Nzₙ     — last-hit normal z
        10 alignₙ  — cos(ray, normal) at last hit
        11 gradₙ   — gradient magnitude of dₙ

SWHDC pipeline (1 channel):
    External depth map: last-hit distance normalized by the maximum distance
    within the object.  Zero-hit pixels are 0.

References:
    HSDC paper §II-B — Stringhini et al., IEEE ICIP 2024
    SWHDC paper §IV-A — Stringhini et al., SIBGRAPI 2024
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from src.preprocessing.ray_casting import IntersectionData, process_mesh

# Number of channels for each pipeline
HSDC_CHANNELS: int = 12
SWHDC_CHANNELS: int = 1


# ---------------------------------------------------------------------------
# Gradient magnitude
# ---------------------------------------------------------------------------

def compute_gradient_magnitude(
    depth_map: np.ndarray,
    sigma: float = 2.0,
) -> np.ndarray:
    """Compute the gradient magnitude of a depth map via Gaussian derivatives.

    Applies first-order Gaussian derivatives along the x- and y-axes
    independently (scipy.ndimage.gaussian_filter with order parameter), then
    returns the L2 magnitude.  σ is set experimentally to 2 (HSDC paper §II-B).

    Args:
        depth_map: (H, W) float array — single-channel depth image.
        sigma:     Gaussian standard deviation (HSDC paper §II-B: σ=2).

    Returns:
        gradient: (H, W) float32 — gradient magnitude map.
    """
    d = depth_map.astype(np.float64)
    # order=[0,1] — derivative along axis-1 (x / columns) — HSDC paper §II-B
    grad_x = gaussian_filter(d, sigma=sigma, order=[0, 1])
    # order=[1,0] — derivative along axis-0 (y / rows)
    grad_y = gaussian_filter(d, sigma=sigma, order=[1, 0])
    return np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.float32)


# ---------------------------------------------------------------------------
# HSDC 12-channel ERP
# ---------------------------------------------------------------------------

def build_hsdc_erp(data: IntersectionData) -> np.ndarray:
    """Build a 12-channel ERP from ray-casting results (HSDC pipeline).

    Implements the feature extraction described in HSDC paper §II-B.
    All channels are 0 at pixels with no ray-mesh intersection.

    Args:
        data: IntersectionData produced by the ray-casting engine.

    Returns:
        erp: (12, H, W) float32 array.
    """
    H, W   = data.hit_mask.shape
    erp    = np.zeros((HSDC_CHANNELS, H, W), dtype=np.float32)
    mask   = data.hit_mask              # (H, W) bool
    max_d  = data.max_dist              # scalar

    # ------------------------------------------------------------------
    # Channels 0–5: first intersection
    # ------------------------------------------------------------------

    # Ch 0: normalized depth d₁ — HSDC paper §II-B
    erp[0] = np.where(mask, data.first_dist / max_d, 0.0)

    # Ch 1–3: surface normals (Nx, Ny, Nz) at first hit — HSDC paper §II-B
    for i in range(3):
        erp[1 + i] = np.where(mask, data.first_normal[:, :, i], 0.0)

    # Ch 4: ray-normal alignment at first hit
    # = cos(angle between ray direction and face normal) — HSDC paper §II-B
    # Both vectors are unit-length, so dot product equals the cosine.
    align_first = np.einsum("hwc,hwc->hw", data.ray_dirs, data.first_normal)
    erp[4] = np.where(mask, align_first, 0.0)

    # Ch 5: gradient magnitude of d₁ (Gaussian σ=2) — HSDC paper §II-B
    grad_first = compute_gradient_magnitude(erp[0], sigma=2.0)
    erp[5] = np.where(mask, grad_first, 0.0)

    # ------------------------------------------------------------------
    # Channels 6–11: last intersection
    # ------------------------------------------------------------------

    # Ch 6: normalized depth dₙ — HSDC paper §II-B
    erp[6] = np.where(mask, data.last_dist / max_d, 0.0)

    # Ch 7–9: surface normals (Nx, Ny, Nz) at last hit
    for i in range(3):
        erp[7 + i] = np.where(mask, data.last_normal[:, :, i], 0.0)

    # Ch 10: ray-normal alignment at last hit
    align_last = np.einsum("hwc,hwc->hw", data.ray_dirs, data.last_normal)
    erp[10] = np.where(mask, align_last, 0.0)

    # Ch 11: gradient magnitude of dₙ (Gaussian σ=2)
    grad_last = compute_gradient_magnitude(erp[6], sigma=2.0)
    erp[11] = np.where(mask, grad_last, 0.0)

    return erp  # (12, H, W) float32


# ---------------------------------------------------------------------------
# SWHDC 1-channel ERP
# ---------------------------------------------------------------------------

def build_swhdc_erp(data: IntersectionData) -> np.ndarray:
    """Build a 1-channel external depth ERP (SWHDC pipeline).

    Stores the last-hit distance normalized by the maximum object distance.
    This produces an external depth map of the object (SWHDC paper §IV-A).
    Zero-hit pixels are 0.

    Args:
        data: IntersectionData produced by the ray-casting engine.

    Returns:
        erp: (1, H, W) float32 array.
    """
    depth = np.where(data.hit_mask, data.last_dist / data.max_dist, 0.0)
    return depth[np.newaxis].astype(np.float32)  # (1, H, W)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def mesh_to_erp(
    path: Path,
    pipeline: str,
    width: int = 512,
    height: int = 256,
    batch_size: int = 32_768,
) -> np.ndarray:
    """Full pipeline: mesh file → ERP numpy array.

    Args:
        path:       Path to the mesh file (.off or .obj).
        pipeline:   'hsdc' for the 12-channel ERP, 'swhdc' for the 1-channel
                    depth ERP.
        width:      ERP width (both papers: 512).
        height:     ERP height (both papers: 256).
        batch_size: Ray-casting batch size.

    Returns:
        erp: (12, H, W) or (1, H, W) float32 array.

    Raises:
        ValueError: If *pipeline* is not 'hsdc' or 'swhdc'.
    """
    if pipeline not in ("hsdc", "swhdc"):
        raise ValueError(f"pipeline must be 'hsdc' or 'swhdc', got '{pipeline}'")

    data = process_mesh(path, width=width, height=height, batch_size=batch_size)

    if pipeline == "hsdc":
        return build_hsdc_erp(data)
    return build_swhdc_erp(data)

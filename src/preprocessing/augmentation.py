"""
Data augmentation for ERP images in the 3DGS radiance field pipeline.

All three augmentation primitives are applied independently to each training
sample with a default probability of 15%.  Augmentation is NEVER applied to
validation or test data.

The module operates on (C, H, W) float32 numpy arrays and is agnostic to the
number of channels C.  It is therefore compatible with both:
  - The original HSDC/SWHDC pipelines (C=12 or C=1 from mesh ray-casting).
  - The new radiance field pipeline (C=N_shells from 3D Gaussian Splats).

Augmentation primitives (HSDC paper §III-A / SWHDC paper §IV-A):
    1. 3D rotation: rotation angles uniformly sampled from
           x, y ∈ [0°, 15°],  z ∈ [0°, 45°]
       The ERP is treated as a spherical signal.  A proper spherical rotation
       is applied by remapping every output pixel's 3-D direction through the
       inverse rotation matrix, converting back to spherical coordinates, and
       resampling the input ERP via bilinear interpolation.

    2. Gaussian blur: σ ∈ [0.1, 2.0], applied channel-wise.

    3. Gaussian noise: mean ∈ [0, 0.001], σ ∈ [0, 0.03], added independently
       to each channel.

References:
    HSDC paper §III-A — Stringhini et al., IEEE ICIP 2024
    SWHDC paper §IV-A — Stringhini et al., SIBGRAPI 2024
    EgoNeRF pipeline — Choi et al., CVPR 2023 (new 3DGS data source)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# 3-D rotation augmentation
# ---------------------------------------------------------------------------

def rotate_erp_3d(
    erp: np.ndarray,
    angle_x_deg: float,
    angle_y_deg: float,
    angle_z_deg: float,
) -> np.ndarray:
    """Apply a 3-D rotation to an ERP image by spherical remapping.

    Works for any number of channels C in the input (C, H, W) array.

    For every output pixel (x_out, y_out):
      1. Compute the corresponding 3-D unit direction d_out using the
         spherical camera model (HSDC Eq. 1).
      2. Apply the inverse rotation: d_src = R^{-1} · d_out.
      3. Convert d_src back to (θ_src, φ_src) and then to source pixel
         coordinates (x_src, y_src).
      4. Bilinearly interpolate the input ERP at (x_src, y_src).

    The horizontal axis is treated as circular (wrap boundary); the vertical
    axis uses nearest-neighbour boundary clamping.

    Args:
        erp:           (C, H, W) float32 — input ERP image, any C ≥ 1.
        angle_x_deg:   Rotation around x-axis in degrees, ∈ [0, 15].
        angle_y_deg:   Rotation around y-axis in degrees, ∈ [0, 15].
        angle_z_deg:   Rotation around z-axis in degrees, ∈ [0, 45].

    Returns:
        rotated: (C, H, W) float32 — rotated ERP image.
    """
    C, H, W = erp.shape

    # Inverse rotation matrix — HSDC paper §III-A (3-D rotation augmentation)
    R_inv = (
        Rotation.from_euler("xyz", [angle_x_deg, angle_y_deg, angle_z_deg], degrees=True)
        .inv()
        .as_matrix()
    )  # (3, 3)

    # Build output pixel grid
    xs = np.arange(W, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(xs, ys)  # (H, W)

    # Output pixel → spherical angles — inverse of HSDC Eq. 2
    theta_out = 2.0 * np.pi * (x_grid + 0.5) / W   # (H, W)
    phi_out   = np.pi        * (y_grid + 0.5) / H   # (H, W)

    # Spherical → 3-D unit direction — HSDC Eq. 1
    dx = np.cos(theta_out) * np.sin(phi_out)
    dy = np.sin(theta_out) * np.sin(phi_out)
    dz = np.cos(phi_out)
    d_out = np.stack([dx, dy, dz], axis=-1)  # (H, W, 3)

    # Apply inverse rotation to obtain source directions
    # d_src[h,w] = R_inv @ d_out[h,w]  (broadcast matmul)
    d_src = d_out @ R_inv.T  # (H, W, 3)

    # Re-normalise (floating-point drift)
    norms = np.linalg.norm(d_src, axis=-1, keepdims=True).clip(min=1e-12)
    d_src = d_src / norms  # (H, W, 3)

    # 3-D direction → spherical angles
    # φ_src = arccos(z)  — HSDC Eq. 1 (inverse)
    phi_src   = np.arccos(np.clip(d_src[:, :, 2], -1.0, 1.0))  # (H, W) in [0, π]
    # θ_src = atan2(y, x), mapped to [0, 2π)
    theta_src = np.arctan2(d_src[:, :, 1], d_src[:, :, 0]) % (2.0 * np.pi)

    # Spherical angles → source pixel coordinates — HSDC Eq. 2
    x_src = theta_src / (2.0 * np.pi) * W - 0.5  # (H, W) — may be < 0 or ≥ W
    y_src = phi_src   / np.pi          * H - 0.5  # (H, W) — in [−0.5, H−0.5]

    # Circular wrap on horizontal axis; clamp vertical axis to valid range
    x_src = x_src % W
    y_src = np.clip(y_src, 0.0, H - 1.0)

    # Bilinear interpolation for each channel (scipy map_coordinates order=1)
    # Channel count C is not hardcoded — works for any C ≥ 1
    rotated = np.empty_like(erp)
    for c in range(C):
        rotated[c] = map_coordinates(
            erp[c],
            [y_src, x_src],
            order=1,
            mode="nearest",   # vertical poles: clamp
            prefilter=False,
        ).astype(np.float32)

    return rotated


# ---------------------------------------------------------------------------
# Gaussian blur
# ---------------------------------------------------------------------------

def gaussian_blur_erp(erp: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to each channel of an ERP image.

    Works for any number of channels C in the input (C, H, W) array.
    σ is sampled uniformly from [0.1, 2.0] during training
    (HSDC paper §III-A / SWHDC paper §IV-A).

    Args:
        erp:   (C, H, W) float32 — input ERP image, any C ≥ 1.
        sigma: Gaussian standard deviation, ∈ [0.1, 2.0].

    Returns:
        blurred: (C, H, W) float32.
    """
    # Channel count C is not hardcoded — iterate over erp.shape[0]
    blurred = np.empty_like(erp)
    for c in range(erp.shape[0]):
        blurred[c] = gaussian_filter(erp[c].astype(np.float64), sigma=sigma).astype(
            np.float32
        )
    return blurred


# ---------------------------------------------------------------------------
# Gaussian noise
# ---------------------------------------------------------------------------

def gaussian_noise_erp(
    erp: np.ndarray,
    mean: float,
    std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian noise to each channel of an ERP image.

    Works for any number of channels C in the input (C, H, W) array.
    mean ∈ [0, 0.001] and σ ∈ [0, 0.03] (HSDC paper §III-A / SWHDC §IV-A).

    Args:
        erp:  (C, H, W) float32 — input ERP image, any C ≥ 1.
        mean: Noise mean.
        std:  Noise standard deviation.
        rng:  Optional numpy random Generator for reproducibility.

    Returns:
        noisy: (C, H, W) float32.
    """
    if rng is None:
        rng = np.random.default_rng()
    # erp.shape is (C, H, W) — no channel count hardcoded
    noise = rng.normal(loc=mean, scale=std, size=erp.shape).astype(np.float32)
    return erp + noise


# ---------------------------------------------------------------------------
# Random erasing
# ---------------------------------------------------------------------------


def random_erasing_erp(
    erp: np.ndarray,
    area_range: tuple[float, float] = (0.02, 0.33),
    aspect_range: tuple[float, float] = (0.3, 3.3),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Zero out a random rectangular patch of an ERP image.

    The erased region covers between 2% and 33% of the total area.
    Aspect ratio is sampled log-uniformly from [0.3, 3.3].
    Forces the model to use global structure rather than memorise
    local density patterns (Zhong et al., 2020).

    Args:
        erp:          (C, H, W) float32 — input ERP image, any C >= 1.
        area_range:   Min/max fraction of total area to erase.
        aspect_range: Min/max aspect ratio of the erased patch.
        rng:          Optional numpy random Generator.

    Returns:
        erased: (C, H, W) float32 — ERP with one rectangular patch zeroed out.
    """
    if rng is None:
        rng = np.random.default_rng()

    C, H, W = erp.shape
    total_area = H * W

    # Sample target area and aspect ratio
    target_area = float(rng.uniform(area_range[0], area_range[1])) * total_area
    log_aspect = float(rng.uniform(np.log(aspect_range[0]), np.log(aspect_range[1])))
    aspect = np.exp(log_aspect)

    # Compute patch dimensions
    h = int(round(np.sqrt(target_area / aspect)))
    w = int(round(np.sqrt(target_area * aspect)))
    h = max(1, min(h, H))
    w = max(1, min(w, W))

    # Random top-left corner
    top  = int(rng.integers(0, max(H - h, 0) + 1))
    left = int(rng.integers(0, max(W - w, 0) + 1))

    result = erp.copy()
    result[:, top:top + h, left:left + w] = 0.0
    return result


# ---------------------------------------------------------------------------
# Horizontal flip (azimuthal rotation of 180°)
# ---------------------------------------------------------------------------

def random_horizontal_flip_erp(erp: np.ndarray) -> np.ndarray:
    """Flip the ERP image along the horizontal (width) axis.

    In equirectangular projection a horizontal flip corresponds to an
    azimuthal rotation of 180°, which is a valid rigid transformation for
    3D object classification — objects are symmetric under arbitrary
    azimuthal orientations.

    Args:
        erp: (C, H, W) float32 — input ERP image, any C ≥ 1.

    Returns:
        flipped: (C, H, W) float32 — horizontally flipped ERP.
    """
    return erp[:, :, ::-1].copy()


# ---------------------------------------------------------------------------
# Composite augmentation
# ---------------------------------------------------------------------------

def augment(
    erp: np.ndarray,
    prob: float = 0.3,
    flip_prob: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply all augmentation primitives to an ERP image.

    Augmentations applied:
      - Horizontal flip  with probability *flip_prob* (default 0.5).
        Equivalent to azimuthal rotation of 180° — free invariance for 3D objects.
      - 3-D rotation, Gaussian blur, Gaussian noise each with probability *prob*
        (default 0.3; original papers used 0.15, but 0.3 is better for training
        from scratch with a small dataset).

    The input may have any number of channels C ≥ 1.

    Augmentation parameter ranges:
        Rotation  x, y: [0°, 15°]   z: [0°, 45°]   (HSDC §III-A / SWHDC §IV-A)
        Blur      σ:    [0.1, 2.0]
        Noise     mean: [0, 0.001]  σ: [0, 0.03]

    Args:
        erp:       (C, H, W) float32 — ERP image, any C ≥ 1.
        prob:      Probability for each of the three paper augmentations (default 0.3).
        flip_prob: Probability of horizontal flip (default 0.5).
        rng:       Optional numpy random Generator; if None a fresh one is created.

    Returns:
        result: (C, H, W) float32 — augmented ERP image (may be unchanged).
    """
    if rng is None:
        rng = np.random.default_rng()

    result = erp.copy()

    # --- Horizontal flip (azimuthal rotation of 180°) ---
    if rng.random() < flip_prob:
        result = random_horizontal_flip_erp(result)

    # --- 3-D rotation ---
    if rng.random() < prob:
        ax = float(rng.uniform(0.0, 15.0))
        ay = float(rng.uniform(0.0, 15.0))
        az = float(rng.uniform(0.0, 45.0))
        result = rotate_erp_3d(result, ax, ay, az)

    # --- Gaussian blur ---
    if rng.random() < prob:
        sigma = float(rng.uniform(0.1, 2.0))
        result = gaussian_blur_erp(result, sigma)

    # --- Gaussian noise ---
    if rng.random() < prob:
        noise_mean = float(rng.uniform(0.0, 0.001))
        noise_std  = float(rng.uniform(0.0, 0.03))
        result = gaussian_noise_erp(result, noise_mean, noise_std, rng=rng)

    # --- Random erasing (Zhong et al., 2020) ---
    if rng.random() < prob:
        result = random_erasing_erp(result, rng=rng)

    return result

"""
src/preprocessing

Public API for the 3DGS radiance field ERP preprocessing pipeline.

This package converts 3D Gaussian Splat PLY files (ShapeSplats/ModelSplats
format) into radiance field ERP tensors ready for neural network training.

Modules:
    ply_loader      — binary PLY parser for 3DGS files
    radiance_field  — EgoNeRF-style exponential shell ERP generation
    augmentation    — channel-agnostic ERP augmentation (rotation, blur, noise)
    dataset         — PyTorch Dataset and DataLoader wrappers
"""

from src.preprocessing.ply_loader import load_gaussian_ply
from src.preprocessing.radiance_field import (
    compute_centroid,
    build_ray_directions,
    compute_shell_radii,
    quaternions_to_rotation_matrices,
    precompute_gaussian_params,
    compute_radiance_field_erp,
    gaussian_ply_to_erp,
)
from src.preprocessing.augmentation import (
    rotate_erp_3d,
    gaussian_blur_erp,
    gaussian_noise_erp,
    augment,
)

try:
    from src.preprocessing.dataset import (
        GaussianERPDataset,
        build_dataloaders,
        MODELNET10_CATEGORIES,
        MODELNET40_CATEGORIES,
    )
except ModuleNotFoundError:
    # torch not installed — Dataset/DataLoader unavailable (preprocessing still works)
    pass

__all__ = [
    # ply_loader
    "load_gaussian_ply",
    # radiance_field
    "compute_centroid",
    "build_ray_directions",
    "compute_shell_radii",
    "quaternions_to_rotation_matrices",
    "precompute_gaussian_params",
    "compute_radiance_field_erp",
    "gaussian_ply_to_erp",
    # augmentation
    "rotate_erp_3d",
    "gaussian_blur_erp",
    "gaussian_noise_erp",
    "augment",
    # dataset
    "GaussianERPDataset",
    "build_dataloaders",
    "MODELNET10_CATEGORIES",
    "MODELNET40_CATEGORIES",
]

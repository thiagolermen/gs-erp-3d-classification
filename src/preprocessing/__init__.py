"""
src/preprocessing

Public API for the ERP preprocessing pipeline.
"""

from src.preprocessing.ray_casting import (
    load_mesh,
    compute_centroid,
    generate_ray_directions,
    cast_rays,
    process_mesh,
    IntersectionData,
)
from src.preprocessing.erp_features import (
    build_hsdc_erp,
    build_swhdc_erp,
    compute_gradient_magnitude,
    mesh_to_erp,
)
from src.preprocessing.augmentation import (
    rotate_erp_3d,
    gaussian_blur_erp,
    gaussian_noise_erp,
    augment,
)
from src.preprocessing.dataset import (
    ERPDataset,
    build_dataloaders,
    precompute_dataset,
)

__all__ = [
    "load_mesh",
    "compute_centroid",
    "generate_ray_directions",
    "cast_rays",
    "process_mesh",
    "IntersectionData",
    "build_hsdc_erp",
    "build_swhdc_erp",
    "compute_gradient_magnitude",
    "mesh_to_erp",
    "rotate_erp_3d",
    "gaussian_blur_erp",
    "gaussian_noise_erp",
    "augment",
    "ERPDataset",
    "build_dataloaders",
    "precompute_dataset",
]

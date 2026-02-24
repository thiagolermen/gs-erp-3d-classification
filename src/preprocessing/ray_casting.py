"""
Spherical ray-casting engine for the ERP-ViT 3D Classification pipeline.

A virtual spherical camera is placed at the area-weighted centroid of the mesh.
Omnidirectional rays are cast outward for every pixel of a (height × width) ERP
grid.  For each ray, all intersections with the mesh surface are collected; the
first (closest) and last (farthest) intersection points are retained to capture
both inner and outer object geometry.

References:
    HSDC paper §II-A, §II-B — Stringhini et al., IEEE ICIP 2024
    SWHDC paper §III-A, §IV-A — Stringhini et al., SIBGRAPI 2024
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import trimesh.ray


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class IntersectionData:
    """Per-pixel ray-mesh intersection results for a single ERP image.

    All spatial arrays are shaped (H, W) or (H, W, 3) and stored as float32.
    Zero-hit pixels have all values set to 0.

    Attributes:
        hit_mask:      (H, W) bool — True where the ray intersected the mesh.
        first_dist:    (H, W) float32 — distance from centroid to first hit.
        first_normal:  (H, W, 3) float32 — face normal at the first hit.
        last_dist:     (H, W) float32 — distance from centroid to last hit.
        last_normal:   (H, W, 3) float32 — face normal at the last hit.
        ray_dirs:      (H, W, 3) float32 — unit ray direction per pixel.
        max_dist:      float — maximum last-hit distance (used for normalisation).
    """

    hit_mask: np.ndarray
    first_dist: np.ndarray
    first_normal: np.ndarray
    last_dist: np.ndarray
    last_normal: np.ndarray
    ray_dirs: np.ndarray
    max_dist: float


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a triangular mesh from a .off or .obj file.

    When the file contains a Scene (multi-mesh), the geometry with the most
    faces is extracted.

    Args:
        path: Absolute path to the mesh file.

    Returns:
        A single trimesh.Trimesh object.

    Raises:
        ValueError: If no valid geometry can be extracted from the file.
    """
    loaded = trimesh.load(str(path), force="mesh", process=False)

    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    # Scene: pick the largest mesh by face count
    if hasattr(loaded, "geometry") and loaded.geometry:
        meshes = list(loaded.geometry.values())
        mesh = max(meshes, key=lambda m: len(m.faces))
        if isinstance(mesh, trimesh.Trimesh):
            return mesh

    raise ValueError(f"Could not extract a Trimesh from '{path}'")


# ---------------------------------------------------------------------------
# Centroid
# ---------------------------------------------------------------------------

def compute_centroid(mesh: trimesh.Trimesh) -> np.ndarray:
    """Compute the area-weighted centroid of a triangle mesh.

    Each triangle contributes proportionally to its area.  This matches the
    centroid computation described in HSDC paper §II-A.

    Args:
        mesh: Input triangle mesh.

    Returns:
        centroid: (3,) float64 — 3-D centroid coordinates.
    """
    # mesh.triangles_center: (N_tri, 3) — geometric centre of each triangle
    # mesh.area_faces:        (N_tri,)  — area of each triangle
    # HSDC paper §II-A: centroid = area-weighted average of triangle centroids
    areas = mesh.area_faces  # (N_tri,)
    total_area = areas.sum()
    if total_area == 0.0:
        # Degenerate mesh: fall back to bounding-box centre
        return mesh.bounds.mean(axis=0)
    centroid = (mesh.triangles_center * areas[:, np.newaxis]).sum(axis=0) / total_area
    return centroid  # (3,) float64


# ---------------------------------------------------------------------------
# Ray directions
# ---------------------------------------------------------------------------

def generate_ray_directions(
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate unit ray directions for a (height × width) ERP pixel grid.

    Each pixel (x, y) maps to spherical angles (θ, φ) via pixel-centre
    sampling.  The 3-D unit direction is then derived from the spherical camera
    model (HSDC Eq. 1 / SWHDC Eq. 1):

        p = [cos(θ) sin(φ),  sin(θ) sin(φ),  cos(φ)]

    and the inverse ERP mapping (HSDC Eq. 2 / SWHDC Eq. 2):

        θ = 2π (x + 0.5) / w,    φ = π (y + 0.5) / h

    Args:
        width:  ERP image width  (default used in both papers: 512).
        height: ERP image height (default used in both papers: 256).

    Returns:
        theta:      (H, W) float64 — longitudinal angle in [0, 2π).
        phi:        (H, W) float64 — latitudinal angle in (0, π).
        directions: (H*W, 3) float64 — unit direction per pixel, row-major.
    """
    xs = np.arange(width)    # (W,)
    ys = np.arange(height)   # (H,)
    x_grid, y_grid = np.meshgrid(xs, ys)  # (H, W) each

    # Inverse ERP mapping — HSDC Eq. 2 / SWHDC Eq. 2
    theta = 2.0 * np.pi * (x_grid + 0.5) / width   # (H, W) in [0, 2π)
    phi   = np.pi        * (y_grid + 0.5) / height  # (H, W) in (0, π)

    # Spherical camera model — HSDC Eq. 1 / SWHDC Eq. 1
    dx = np.cos(theta) * np.sin(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(phi)

    directions = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)  # (H*W, 3)
    return theta, phi, directions


# ---------------------------------------------------------------------------
# Ray casting
# ---------------------------------------------------------------------------

def cast_rays(
    mesh: trimesh.Trimesh,
    centroid: np.ndarray,
    directions: np.ndarray,
    width: int,
    height: int,
    batch_size: int = 32_768,
) -> IntersectionData:
    """Cast omnidirectional rays from *centroid* and collect intersection data.

    Rays are cast in batches to bound peak memory usage.  For each ray, all
    intersection points are collected and sorted by distance; only the first
    and last are retained (HSDC paper §II-B).

    Single-hit rays are replicated as both first and last intersection
    (HSDC paper §II-B: "In cases of a single intersection, it is replicated as
    the first and last hit").

    Zero-hit pixels are left at 0 in all output arrays.

    Args:
        mesh:       Triangle mesh to intersect.
        centroid:   (3,) float64 — origin for all rays.
        directions: (H*W, 3) float64 — unit direction per ray (row-major).
        width:      ERP image width.
        height:     ERP image height.
        batch_size: Number of rays per intersection batch.

    Returns:
        IntersectionData with per-pixel first/last distances and normals.
    """
    n_rays = width * height

    # Output buffers
    first_dist   = np.zeros(n_rays, dtype=np.float32)
    first_normal = np.zeros((n_rays, 3), dtype=np.float32)
    last_dist    = np.zeros(n_rays, dtype=np.float32)
    last_normal  = np.zeros((n_rays, 3), dtype=np.float32)
    hit_mask     = np.zeros(n_rays, dtype=bool)

    # Process rays in batches
    for start in range(0, n_rays, batch_size):
        end  = min(start + batch_size, n_rays)
        dirs = directions[start:end]                    # (B, 3)
        origs = np.broadcast_to(centroid, (end - start, 3)).copy()

        # Trimesh returns all intersections with multiple_hits=True
        locs, idx_ray, idx_tri = mesh.ray.intersects_location(
            ray_origins=origs,
            ray_directions=dirs,
            multiple_hits=True,
        )

        if len(locs) == 0:
            continue

        # Distance from centroid to each hit — HSDC paper §II-B
        dists   = np.linalg.norm(locs - centroid, axis=1).astype(np.float32)  # (N_hits,)
        normals = mesh.face_normals[idx_tri].astype(np.float32)                # (N_hits, 3)

        # Global ray indices
        global_idx_ray = idx_ray + start  # (N_hits,)

        # Sort by (global_ray_index, distance) to identify first/last hits
        sort_order     = np.lexsort((dists, global_idx_ray))
        sorted_rays    = global_idx_ray[sort_order]
        sorted_dists   = dists[sort_order]
        sorted_normals = normals[sort_order]

        # First hit: first occurrence of each unique ray index
        _, first_idx = np.unique(sorted_rays, return_index=True)

        hit_mask[sorted_rays[first_idx]]     = True
        first_dist[sorted_rays[first_idx]]   = sorted_dists[first_idx]
        first_normal[sorted_rays[first_idx]] = sorted_normals[first_idx]

        # Last hit: last occurrence of each unique ray index
        # Trick: reverse, take first unique, then undo reversal
        _, last_idx_rev = np.unique(sorted_rays[::-1], return_index=True)
        last_idx = len(sorted_rays) - 1 - last_idx_rev

        last_dist[sorted_rays[last_idx]]   = sorted_dists[last_idx]
        last_normal[sorted_rays[last_idx]] = sorted_normals[last_idx]

    # Reshape to (H, W, ...)
    hit_mask     = hit_mask.reshape(height, width)
    first_dist   = first_dist.reshape(height, width)
    first_normal = first_normal.reshape(height, width, 3)
    last_dist    = last_dist.reshape(height, width)
    last_normal  = last_normal.reshape(height, width, 3)
    ray_dirs     = directions.reshape(height, width, 3).astype(np.float32)

    # Maximum last-hit distance for normalisation — HSDC paper §II-B
    max_dist = float(last_dist[hit_mask].max()) if hit_mask.any() else 1.0
    if max_dist == 0.0:
        max_dist = 1.0

    return IntersectionData(
        hit_mask=hit_mask,
        first_dist=first_dist,
        first_normal=first_normal,
        last_dist=last_dist,
        last_normal=last_normal,
        ray_dirs=ray_dirs,
        max_dist=max_dist,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def process_mesh(
    path: Path,
    width: int = 512,
    height: int = 256,
    batch_size: int = 32_768,
) -> IntersectionData:
    """Full pipeline: mesh file → IntersectionData.

    Loads the mesh, computes the area-weighted centroid, generates ERP ray
    directions, and runs ray-mesh intersection.

    Args:
        path:       Path to the mesh file (.off or .obj).
        width:      ERP width in pixels (both papers use 512).
        height:     ERP height in pixels (both papers use 256).
        batch_size: Rays per intersection batch (tune for available RAM/VRAM).

    Returns:
        IntersectionData for the mesh.
    """
    mesh     = load_mesh(path)
    centroid = compute_centroid(mesh)
    _, _, directions = generate_ray_directions(width, height)
    return cast_rays(mesh, centroid, directions, width, height, batch_size)

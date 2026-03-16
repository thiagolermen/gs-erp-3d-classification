"""
Unit tests for the 3DGS radiance field preprocessing pipeline.

Minimum required tests (per CLAUDE.md):
    - ERP generation pixel count (H*W rays / directions)
    - Channel count (N_shells output channels)
    - Depth (density) normalisation range — raw output is non-negative
    - Zero-density shells: shells with no relevant Gaussians are all-zero
    - PLY loader: correct keys and dtypes (using a synthetic in-memory PLY)
    - Augmentation: shape, dtype, in-place safety for arbitrary C channels

All tests that require 3DGS data use synthetic data arrays so that they run
without downloading the ShapeSplats dataset.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.radiance_field import (
    build_ray_directions,
    compute_centroid,
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


# ---------------------------------------------------------------------------
# Helpers for synthetic data
# ---------------------------------------------------------------------------

def _make_synthetic_gs(n: int = 64, seed: int = 0) -> dict:
    """Build a synthetic Gaussian cloud dict (same format as load_gaussian_ply)."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-0.5, 0.5, (n, 3)).astype(np.float32)
    opacity = rng.uniform(0.1, 0.9, (n,)).astype(np.float32)
    scale = rng.uniform(0.01, 0.1, (n, 3)).astype(np.float32)
    rgb = rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32)
    # Random unit quaternions
    q_raw = rng.standard_normal((n, 4)).astype(np.float32)
    rotation = q_raw / np.linalg.norm(q_raw, axis=1, keepdims=True)
    return {
        "xyz": xyz,
        "opacity": opacity,
        "scale": scale,
        "rgb": rgb,
        "rotation": rotation,
        "n_gaussians": n,
    }


def _write_synthetic_ply(path: Path, n: int = 8, seed: int = 1) -> None:
    """Write a minimal valid binary-little-endian PLY file for testing."""
    rng = np.random.default_rng(seed)
    props = ["x", "y", "z", "nx", "ny", "nz",
             "f_dc_0", "f_dc_1", "f_dc_2",
             "opacity",
             "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    n_props = len(props)

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
    ]
    for p in props:
        header_lines.append(f"property float {p}")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    data = rng.standard_normal((n, n_props)).astype(np.float32)
    # Make opacity logit-range sensible
    data[:, props.index("opacity")] = rng.uniform(-2.0, 2.0, n).astype(np.float32)
    # Make scale log-negative (so exp gives small positive values)
    for ax in ("scale_0", "scale_1", "scale_2"):
        data[:, props.index(ax)] = rng.uniform(-4.0, -2.0, n).astype(np.float32)

    with path.open("wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())


# ---------------------------------------------------------------------------
# Ray direction tests
# ---------------------------------------------------------------------------

class TestBuildRayDirections:
    def test_pixel_count(self) -> None:
        """build_ray_directions must produce exactly H*W direction vectors."""
        H, W = 256, 512
        dirs = build_ray_directions(H, W)
        assert dirs.shape == (H * W, 3), (
            f"Expected ({H * W}, 3), got {dirs.shape}"
        )

    def test_unit_length(self) -> None:
        """All direction vectors must be unit-length (within float32 tolerance)."""
        dirs = build_ray_directions(256, 512)
        norms = np.linalg.norm(dirs.astype(np.float64), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_dtype(self) -> None:
        """Output dtype must be float32."""
        dirs = build_ray_directions(64, 128)
        assert dirs.dtype == np.float32

    def test_small_resolution(self) -> None:
        """Works for small non-standard resolutions."""
        dirs = build_ray_directions(4, 8)
        assert dirs.shape == (32, 3)


# ---------------------------------------------------------------------------
# Centroid tests
# ---------------------------------------------------------------------------

class TestComputeCentroid:
    def test_uniform_opacity_is_mean(self) -> None:
        """With uniform opacity, centroid equals the coordinate mean."""
        rng = np.random.default_rng(42)
        xyz = rng.standard_normal((100, 3)).astype(np.float32)
        opacity = np.ones(100, dtype=np.float32)
        centroid = compute_centroid(xyz, opacity)
        np.testing.assert_allclose(centroid, xyz.mean(axis=0), atol=1e-5)

    def test_single_gaussian(self) -> None:
        """Single-Gaussian cloud: centroid equals that Gaussian's position."""
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        opacity = np.array([0.8], dtype=np.float32)
        centroid = compute_centroid(xyz, opacity)
        np.testing.assert_allclose(centroid, [1.0, 2.0, 3.0], atol=1e-6)

    def test_zero_opacity_fallback(self) -> None:
        """All-zero opacity falls back to unweighted mean (no crash)."""
        xyz = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=np.float32)
        opacity = np.zeros(2, dtype=np.float32)
        centroid = compute_centroid(xyz, opacity)
        np.testing.assert_allclose(centroid, [1.0, 1.0, 1.0], atol=1e-6)

    def test_output_shape(self) -> None:
        """Centroid output must be (3,)."""
        gs = _make_synthetic_gs(32)
        centroid = compute_centroid(gs["xyz"], gs["opacity"])
        assert centroid.shape == (3,)


# ---------------------------------------------------------------------------
# Shell radii tests
# ---------------------------------------------------------------------------

class TestComputeShellRadii:
    def test_output_shape(self) -> None:
        """compute_shell_radii returns (n_shells,) array."""
        r_dist = np.linspace(0.1, 1.0, 50).astype(np.float32)
        radii = compute_shell_radii(r_dist, n_shells=8)
        assert radii.shape == (8,)

    def test_ascending_order(self) -> None:
        """Shell radii must be non-decreasing."""
        r_dist = np.linspace(0.1, 1.0, 100).astype(np.float32)
        radii = compute_shell_radii(r_dist, n_shells=8)
        assert np.all(np.diff(radii) >= 0.0), "Shell radii are not ascending"

    def test_dtype(self) -> None:
        """Output dtype must be float32."""
        r_dist = np.ones(20, dtype=np.float32)
        radii = compute_shell_radii(r_dist, n_shells=4)
        assert radii.dtype == np.float32

    def test_degenerate_fallback(self) -> None:
        """Degenerate (all same distance) falls back to linspace without crash."""
        r_dist = np.full(30, 0.5, dtype=np.float32)
        radii = compute_shell_radii(r_dist, n_shells=4)
        assert radii.shape == (4,)
        assert np.all(np.isfinite(radii))

    def test_single_shell(self) -> None:
        """n_shells=1 returns a length-1 array without crash."""
        r_dist = np.linspace(0.1, 1.0, 50).astype(np.float32)
        radii = compute_shell_radii(r_dist, n_shells=1)
        assert radii.shape == (1,)


# ---------------------------------------------------------------------------
# Quaternion → rotation matrix tests
# ---------------------------------------------------------------------------

class TestQuaternionsToRotationMatrices:
    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        q = rng.standard_normal((10, 4)).astype(np.float32)
        R = quaternions_to_rotation_matrices(q)
        assert R.shape == (10, 3, 3)

    def test_orthogonality(self) -> None:
        """Rotation matrices must satisfy R @ R^T ≈ I."""
        rng = np.random.default_rng(1)
        q = rng.standard_normal((20, 4)).astype(np.float32)
        R = quaternions_to_rotation_matrices(q)  # (20, 3, 3)
        # R @ R^T should be identity for each matrix
        I_approx = np.einsum("nij,nkj->nik", R, R)  # (20, 3, 3)
        I_ref = np.eye(3)[np.newaxis].repeat(20, axis=0)
        np.testing.assert_allclose(I_approx, I_ref, atol=1e-4)

    def test_determinant_one(self) -> None:
        """Rotation matrices must have determinant ≈ +1."""
        rng = np.random.default_rng(2)
        q = rng.standard_normal((15, 4)).astype(np.float32)
        R = quaternions_to_rotation_matrices(q)
        dets = np.linalg.det(R.astype(np.float64))
        np.testing.assert_allclose(dets, 1.0, atol=1e-4)

    def test_dtype(self) -> None:
        rng = np.random.default_rng(3)
        q = rng.standard_normal((5, 4)).astype(np.float32)
        R = quaternions_to_rotation_matrices(q)
        assert R.dtype == np.float32


# ---------------------------------------------------------------------------
# Radiance field ERP tests
# ---------------------------------------------------------------------------

class TestComputeRadianceFieldERP:
    """Tests for the full compute_radiance_field_erp function."""

    _H = 16
    _W = 32
    _N_SHELLS = 4

    def _setup(self) -> tuple:
        gs = _make_synthetic_gs(n=32, seed=7)
        centroid = compute_centroid(gs["xyz"], gs["opacity"])
        ray_dirs = build_ray_directions(self._H, self._W)
        gs_precomp = precompute_gaussian_params(gs, centroid)
        shell_radii = compute_shell_radii(
            gs_precomp["r_dist"], n_shells=self._N_SHELLS
        )
        return gs_precomp, centroid, ray_dirs, shell_radii

    def test_output_shape(self) -> None:
        """ERP shape must be (N_shells, H, W)."""
        gs_precomp, centroid, ray_dirs, shell_radii = self._setup()
        erp = compute_radiance_field_erp(
            gs_precomp, centroid, ray_dirs, shell_radii,
            H=self._H, W=self._W
        )
        assert erp.shape == (self._N_SHELLS, self._H, self._W), (
            f"Expected ({self._N_SHELLS}, {self._H}, {self._W}), got {erp.shape}"
        )

    def test_output_dtype(self) -> None:
        """ERP output must be float32."""
        gs_precomp, centroid, ray_dirs, shell_radii = self._setup()
        erp = compute_radiance_field_erp(
            gs_precomp, centroid, ray_dirs, shell_radii,
            H=self._H, W=self._W
        )
        assert erp.dtype == np.float32

    def test_non_negative(self) -> None:
        """Density values must be non-negative (opacity * exp >= 0)."""
        gs_precomp, centroid, ray_dirs, shell_radii = self._setup()
        erp = compute_radiance_field_erp(
            gs_precomp, centroid, ray_dirs, shell_radii,
            H=self._H, W=self._W
        )
        assert erp.min() >= 0.0, f"Negative density found: {erp.min()}"

    def test_channel_count(self) -> None:
        """Number of channels equals n_shells."""
        gs_precomp, centroid, ray_dirs, shell_radii = self._setup()
        erp = compute_radiance_field_erp(
            gs_precomp, centroid, ray_dirs, shell_radii,
            H=self._H, W=self._W
        )
        assert erp.shape[0] == self._N_SHELLS

    def test_no_gaussians_shell_is_zero(self) -> None:
        """A shell with no relevant Gaussians must be all-zero."""
        gs_precomp, centroid, ray_dirs, shell_radii = self._setup()
        # Use an extreme cutoff_sigma = 0 so no Gaussian contributes to any shell
        erp_zero = compute_radiance_field_erp(
            gs_precomp, centroid, ray_dirs, shell_radii,
            H=self._H, W=self._W, cutoff_sigma=0.0
        )
        np.testing.assert_array_equal(erp_zero, 0.0)


# ---------------------------------------------------------------------------
# PLY loader tests
# ---------------------------------------------------------------------------

class TestLoadGaussianPLY:
    def test_valid_ply(self, tmp_path: Path) -> None:
        """load_gaussian_ply returns correct keys and dtypes."""
        from src.preprocessing.ply_loader import load_gaussian_ply
        ply_path = tmp_path / "point_cloud.ply"
        _write_synthetic_ply(ply_path, n=16)

        result = load_gaussian_ply(ply_path)

        assert set(result.keys()) >= {"xyz", "opacity", "scale", "rgb", "rotation", "n_gaussians"}
        assert result["xyz"].shape == (16, 3)
        assert result["opacity"].shape == (16,)
        assert result["scale"].shape == (16, 3)
        assert result["rgb"].shape == (16, 3)
        assert result["rotation"].shape == (16, 4)
        assert result["n_gaussians"] == 16

    def test_opacity_range(self, tmp_path: Path) -> None:
        """Sigmoid-transformed opacity must be in [0, 1]."""
        from src.preprocessing.ply_loader import load_gaussian_ply
        ply_path = tmp_path / "point_cloud.ply"
        _write_synthetic_ply(ply_path, n=32)
        result = load_gaussian_ply(ply_path)
        assert result["opacity"].min() >= 0.0
        assert result["opacity"].max() <= 1.0

    def test_scale_positive(self, tmp_path: Path) -> None:
        """exp-transformed scale must be positive."""
        from src.preprocessing.ply_loader import load_gaussian_ply
        ply_path = tmp_path / "point_cloud.ply"
        _write_synthetic_ply(ply_path, n=16)
        result = load_gaussian_ply(ply_path)
        assert np.all(result["scale"] > 0.0)

    def test_rgb_range(self, tmp_path: Path) -> None:
        """RGB (from SH DC) must be in [0, 1]."""
        from src.preprocessing.ply_loader import load_gaussian_ply
        ply_path = tmp_path / "point_cloud.ply"
        _write_synthetic_ply(ply_path, n=16)
        result = load_gaussian_ply(ply_path)
        assert result["rgb"].min() >= 0.0
        assert result["rgb"].max() <= 1.0

    def test_file_not_found(self, tmp_path: Path) -> None:
        """load_gaussian_ply raises FileNotFoundError for missing files."""
        from src.preprocessing.ply_loader import load_gaussian_ply
        with pytest.raises(FileNotFoundError):
            load_gaussian_ply(tmp_path / "nonexistent.ply")

    def test_dtypes(self, tmp_path: Path) -> None:
        """All array outputs must be float32."""
        from src.preprocessing.ply_loader import load_gaussian_ply
        ply_path = tmp_path / "point_cloud.ply"
        _write_synthetic_ply(ply_path, n=8)
        result = load_gaussian_ply(ply_path)
        for key in ("xyz", "opacity", "scale", "rgb", "rotation"):
            assert result[key].dtype == np.float32, (
                f"Key '{key}' dtype is {result[key].dtype}, expected float32"
            )


# ---------------------------------------------------------------------------
# Augmentation tests (channel-agnostic)
# ---------------------------------------------------------------------------

class TestAugmentation:
    """Tests for the three augmentation primitives and composite function.

    These tests are channel-agnostic: they run with both N_shells channels
    (new 3DGS pipeline) and the legacy 1-channel and 12-channel shapes.
    """

    @pytest.mark.parametrize("C", [1, 8, 12])
    def test_rotation_shape_preserved(self, C: int) -> None:
        rng = np.random.default_rng(0)
        erp = rng.random((C, 32, 64)).astype(np.float32)
        rotated = rotate_erp_3d(erp, 5.0, 10.0, 30.0)
        assert rotated.shape == erp.shape

    @pytest.mark.parametrize("C", [1, 8, 12])
    def test_rotation_dtype(self, C: int) -> None:
        rng = np.random.default_rng(0)
        erp = rng.random((C, 32, 64)).astype(np.float32)
        rotated = rotate_erp_3d(erp, 0.0, 0.0, 0.0)
        assert rotated.dtype == np.float32

    def test_zero_rotation_identity(self) -> None:
        """Zero rotation must leave the ERP essentially unchanged."""
        rng = np.random.default_rng(0)
        erp = rng.random((8, 32, 64)).astype(np.float32)
        rotated = rotate_erp_3d(erp, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(rotated, erp, atol=1e-4)

    @pytest.mark.parametrize("C", [1, 8, 12])
    def test_blur_shape_preserved(self, C: int) -> None:
        rng = np.random.default_rng(1)
        erp = rng.random((C, 32, 64)).astype(np.float32)
        blurred = gaussian_blur_erp(erp, sigma=1.0)
        assert blurred.shape == erp.shape
        assert blurred.dtype == np.float32

    @pytest.mark.parametrize("C", [1, 8, 12])
    def test_noise_shape_preserved(self, C: int) -> None:
        rng = np.random.default_rng(2)
        erp = rng.random((C, 32, 64)).astype(np.float32)
        noisy = gaussian_noise_erp(erp, mean=0.0, std=0.01)
        assert noisy.shape == erp.shape
        assert noisy.dtype == np.float32

    def test_noise_changes_values(self) -> None:
        rng = np.random.default_rng(3)
        erp = rng.random((8, 32, 64)).astype(np.float32)
        noisy = gaussian_noise_erp(erp, mean=0.0, std=0.01, rng=np.random.default_rng(5))
        assert not np.allclose(erp, noisy)

    @pytest.mark.parametrize("C", [1, 8, 12])
    def test_augment_shape_preserved(self, C: int) -> None:
        rng = np.random.default_rng(10)
        erp = rng.random((C, 32, 64)).astype(np.float32)
        result = augment(erp, prob=1.0, rng=np.random.default_rng(0))
        assert result.shape == erp.shape

    def test_augment_no_inplace_modification(self) -> None:
        """augment must not modify the input array."""
        rng = np.random.default_rng(11)
        erp = rng.random((8, 32, 64)).astype(np.float32)
        erp_copy = erp.copy()
        augment(erp, prob=1.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(erp, erp_copy)

    def test_augment_prob_zero_is_identity(self) -> None:
        """With prob=0, augment must return the input unchanged."""
        rng = np.random.default_rng(12)
        erp = rng.random((8, 32, 64)).astype(np.float32)
        result = augment(erp, prob=0.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(result, erp)

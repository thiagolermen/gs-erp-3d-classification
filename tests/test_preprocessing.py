"""
Unit tests for the ERP preprocessing pipeline.

Minimum required tests (per CLAUDE.md):
    - HSDC ERP output shape
    - SWHDC ERP output shape
    - Depth normalization: all values in [0, 1]
    - Zero-hit pixels: all channels are 0
    - SWHDC weight sum = 1 per row (tested in test_models; placeholder here)
    - ERP generation pixel count

All tests that require mesh data use a synthetic trimesh.Trimesh (a unit cube)
so that they run without downloading ModelNet.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from src.preprocessing.ray_casting import (
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
    HSDC_CHANNELS,
    SWHDC_CHANNELS,
)
from src.preprocessing.augmentation import (
    rotate_erp_3d,
    gaussian_blur_erp,
    gaussian_noise_erp,
    augment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def unit_cube() -> trimesh.Trimesh:
    """A unit cube centred at the origin — simple closed mesh for testing."""
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


@pytest.fixture(scope="module")
def intersection_data(unit_cube: trimesh.Trimesh) -> IntersectionData:
    """Pre-computed IntersectionData for the unit cube (full ERP resolution)."""
    centroid = compute_centroid(unit_cube)
    _, _, directions = generate_ray_directions(width=512, height=256)
    return cast_rays(unit_cube, centroid, directions, width=512, height=256)


@pytest.fixture(scope="module")
def hsdc_erp(intersection_data: IntersectionData) -> np.ndarray:
    return build_hsdc_erp(intersection_data)


@pytest.fixture(scope="module")
def swhdc_erp(intersection_data: IntersectionData) -> np.ndarray:
    return build_swhdc_erp(intersection_data)


# ---------------------------------------------------------------------------
# Ray direction tests
# ---------------------------------------------------------------------------

class TestRayDirections:
    def test_pixel_count(self) -> None:
        """ERP generation must produce exactly w × h ray directions."""
        W, H = 512, 256
        _, _, directions = generate_ray_directions(W, H)
        assert directions.shape == (W * H, 3), (
            f"Expected {W * H} rays, got {directions.shape[0]}"
        )

    def test_unit_length(self) -> None:
        """All ray directions must be unit vectors."""
        _, _, directions = generate_ray_directions(512, 256)
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_theta_range(self) -> None:
        """θ must be in [0, 2π)."""
        theta, _, _ = generate_ray_directions(512, 256)
        assert theta.min() >= 0.0
        assert theta.max() < 2.0 * np.pi

    def test_phi_range(self) -> None:
        """φ must be in (0, π)."""
        _, phi, _ = generate_ray_directions(512, 256)
        assert phi.min() > 0.0
        assert phi.max() < np.pi


# ---------------------------------------------------------------------------
# Centroid tests
# ---------------------------------------------------------------------------

class TestCentroid:
    def test_inside_bounding_box(self, unit_cube: trimesh.Trimesh) -> None:
        """Centroid must lie inside the mesh bounding box."""
        c = compute_centroid(unit_cube)
        lo, hi = unit_cube.bounds
        assert np.all(c >= lo - 1e-6)
        assert np.all(c <= hi + 1e-6)

    def test_unit_cube_centroid_near_origin(self, unit_cube: trimesh.Trimesh) -> None:
        """The unit cube's centroid should be very close to (0, 0, 0)."""
        c = compute_centroid(unit_cube)
        np.testing.assert_allclose(c, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# HSDC ERP tests
# ---------------------------------------------------------------------------

class TestHSDCERP:
    def test_output_shape(self, hsdc_erp: np.ndarray) -> None:
        """HSDC ERP shape must be (12, 256, 512)."""
        assert hsdc_erp.shape == (HSDC_CHANNELS, 256, 512), (
            f"Expected (12, 256, 512), got {hsdc_erp.shape}"
        )

    def test_dtype(self, hsdc_erp: np.ndarray) -> None:
        """ERP must be float32."""
        assert hsdc_erp.dtype == np.float32

    def test_depth_channels_normalized(self, hsdc_erp: np.ndarray) -> None:
        """Depth channels (0 and 6) must be in [0, 1]."""
        for ch in (0, 6):
            assert hsdc_erp[ch].min() >= 0.0, f"Channel {ch}: negative depth"
            assert hsdc_erp[ch].max() <= 1.0 + 1e-6, f"Channel {ch}: depth > 1"

    def test_zero_hit_pixels_all_zero(
        self, hsdc_erp: np.ndarray, intersection_data: IntersectionData
    ) -> None:
        """Zero-hit pixels must be 0 across all channels."""
        no_hit = ~intersection_data.hit_mask  # (H, W)
        if no_hit.any():
            zero_block = hsdc_erp[:, no_hit]  # (12, N_zero)
            np.testing.assert_array_equal(
                zero_block, 0.0, err_msg="Non-zero value at zero-hit pixel"
            )

    def test_alignment_range(self, hsdc_erp: np.ndarray) -> None:
        """Alignment channels (4, 10) must be in [-1, 1]."""
        for ch in (4, 10):
            assert hsdc_erp[ch].min() >= -1.0 - 1e-6
            assert hsdc_erp[ch].max() <=  1.0 + 1e-6

    def test_gradient_nonnegative(self, hsdc_erp: np.ndarray) -> None:
        """Gradient magnitude channels (5, 11) must be non-negative."""
        for ch in (5, 11):
            assert hsdc_erp[ch].min() >= 0.0

    def test_channel_count(self) -> None:
        """HSDC_CHANNELS constant must equal 12."""
        assert HSDC_CHANNELS == 12


# ---------------------------------------------------------------------------
# SWHDC ERP tests
# ---------------------------------------------------------------------------

class TestSWHDCERP:
    def test_output_shape(self, swhdc_erp: np.ndarray) -> None:
        """SWHDC ERP shape must be (1, 256, 512)."""
        assert swhdc_erp.shape == (SWHDC_CHANNELS, 256, 512), (
            f"Expected (1, 256, 512), got {swhdc_erp.shape}"
        )

    def test_dtype(self, swhdc_erp: np.ndarray) -> None:
        assert swhdc_erp.dtype == np.float32

    def test_depth_normalized(self, swhdc_erp: np.ndarray) -> None:
        """Depth values must be in [0, 1]."""
        assert swhdc_erp.min() >= 0.0
        assert swhdc_erp.max() <= 1.0 + 1e-6

    def test_zero_hit_pixels(
        self, swhdc_erp: np.ndarray, intersection_data: IntersectionData
    ) -> None:
        """Zero-hit pixels must be 0."""
        no_hit = ~intersection_data.hit_mask
        if no_hit.any():
            np.testing.assert_array_equal(swhdc_erp[0, no_hit], 0.0)

    def test_channel_count(self) -> None:
        """SWHDC_CHANNELS constant must equal 1."""
        assert SWHDC_CHANNELS == 1


# ---------------------------------------------------------------------------
# Gradient magnitude tests
# ---------------------------------------------------------------------------

class TestGradientMagnitude:
    def test_nonnegative(self) -> None:
        rng   = np.random.default_rng(0)
        depth = rng.random((256, 512)).astype(np.float32)
        grad  = compute_gradient_magnitude(depth, sigma=2.0)
        assert grad.min() >= 0.0

    def test_shape_preserved(self) -> None:
        depth = np.zeros((256, 512), dtype=np.float32)
        grad  = compute_gradient_magnitude(depth)
        assert grad.shape == depth.shape

    def test_flat_image_zero_gradient(self) -> None:
        """Constant depth map should have near-zero gradient."""
        depth = np.ones((64, 128), dtype=np.float32)
        grad  = compute_gradient_magnitude(depth, sigma=2.0)
        np.testing.assert_allclose(grad, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------

class TestAugmentation:
    """Tests for the three augmentation primitives and the composite function."""

    _ERP_SHAPE = (12, 256, 512)

    def _make_erp(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.random(self._ERP_SHAPE).astype(np.float32)

    # --- rotate_erp_3d ---

    def test_rotation_shape_preserved(self) -> None:
        erp     = self._make_erp()
        rotated = rotate_erp_3d(erp, 5.0, 10.0, 30.0)
        assert rotated.shape == erp.shape

    def test_rotation_dtype(self) -> None:
        erp     = self._make_erp()
        rotated = rotate_erp_3d(erp, 0.0, 0.0, 0.0)
        assert rotated.dtype == np.float32

    def test_zero_rotation_identity(self) -> None:
        """Zero rotation should leave the ERP essentially unchanged."""
        erp     = self._make_erp()
        rotated = rotate_erp_3d(erp, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(rotated, erp, atol=1e-4)

    # --- gaussian_blur_erp ---

    def test_blur_shape_preserved(self) -> None:
        erp    = self._make_erp()
        blurred = gaussian_blur_erp(erp, sigma=1.0)
        assert blurred.shape == erp.shape

    def test_blur_dtype(self) -> None:
        erp    = self._make_erp()
        blurred = gaussian_blur_erp(erp, sigma=1.0)
        assert blurred.dtype == np.float32

    # --- gaussian_noise_erp ---

    def test_noise_shape_preserved(self) -> None:
        erp   = self._make_erp()
        noisy = gaussian_noise_erp(erp, mean=0.0, std=0.01)
        assert noisy.shape == erp.shape

    def test_noise_changes_values(self) -> None:
        erp   = self._make_erp()
        noisy = gaussian_noise_erp(erp, mean=0.0, std=0.01, rng=np.random.default_rng(1))
        assert not np.allclose(erp, noisy)

    # --- augment (composite) ---

    def test_augment_shape_preserved(self) -> None:
        erp    = self._make_erp()
        result = augment(erp, prob=1.0, rng=np.random.default_rng(0))
        assert result.shape == erp.shape

    def test_augment_no_inplace_modification(self) -> None:
        """augment() must not modify the input array."""
        erp    = self._make_erp()
        erp_copy = erp.copy()
        augment(erp, prob=1.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(erp, erp_copy)

    def test_augment_prob_zero_is_identity(self) -> None:
        """With prob=0, augment must return the input unchanged."""
        erp    = self._make_erp()
        result = augment(erp, prob=0.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(result, erp)

    def test_augment_not_applied_when_prob_zero(self) -> None:
        """Sanity-check: prob=0 truly fires no primitive."""
        rng  = np.random.default_rng(0)
        erp  = self._make_erp()
        out  = augment(erp, prob=0.0, rng=rng)
        assert np.array_equal(out, erp)

    def test_augment_swhdc_single_channel(self) -> None:
        """augment must work on 1-channel SWHDC ERPs as well."""
        rng    = np.random.default_rng(7)
        erp_1c = rng.random((1, 256, 512)).astype(np.float32)
        result = augment(erp_1c, prob=1.0, rng=rng)
        assert result.shape == (1, 256, 512)


# ---------------------------------------------------------------------------
# mesh_to_erp integration test
# ---------------------------------------------------------------------------

class TestMeshToERP:
    def test_hsdc_pipeline(self, tmp_path: Path, unit_cube: trimesh.Trimesh) -> None:
        """mesh_to_erp with pipeline='hsdc' returns (12, 256, 512) float32."""
        mesh_file = tmp_path / "cube.obj"
        unit_cube.export(str(mesh_file))

        erp = mesh_to_erp(mesh_file, pipeline="hsdc", width=512, height=256)
        assert erp.shape == (12, 256, 512)
        assert erp.dtype == np.float32

    def test_swhdc_pipeline(self, tmp_path: Path, unit_cube: trimesh.Trimesh) -> None:
        """mesh_to_erp with pipeline='swhdc' returns (1, 256, 512) float32."""
        mesh_file = tmp_path / "cube.obj"
        unit_cube.export(str(mesh_file))

        erp = mesh_to_erp(mesh_file, pipeline="swhdc", width=512, height=256)
        assert erp.shape == (1, 256, 512)
        assert erp.dtype == np.float32

    def test_invalid_pipeline_raises(
        self, tmp_path: Path, unit_cube: trimesh.Trimesh
    ) -> None:
        mesh_file = tmp_path / "cube.obj"
        unit_cube.export(str(mesh_file))
        with pytest.raises(ValueError, match="pipeline"):
            mesh_to_erp(mesh_file, pipeline="invalid")

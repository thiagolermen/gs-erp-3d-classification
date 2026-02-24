"""
Unit tests for the neural network architecture components.

Covers the verification checklist from the AI Architect agent spec (CLAUDE.md):
    - HSDCBlock output shape: (B, 4C, H, W) given (B, C, H, W)
    - SWHDCBlock output shape: (B, C, H, W) given (B, C, H, W)
    - SWHDC weight rows sum to 1.0 (per-row Σ_n W_n = 1)
    - SWHDC adds exactly 0 extra trainable parameters vs a standard Conv2d
    - HSDCNet (ResNet-34 + HSDC) parameter count ≈ 5.3 M
    - SWHDCResNet (ResNet-50 + SWHDC) parameter count ≈ 25.5 M
    - All backbones accept the correct input channel count (12 or 1)

All tests are pure-PyTorch and do NOT require a GPU or ModelNet data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.blocks.hsdc import HSDCBlock
from src.models.blocks.swhdc import SWHDCBlock, compute_swhdc_weights
from src.models.classifier import ClassificationHead
from src.models.backbones.resnet_hsdc import HSDCNet, SWHDCResNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_params(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def _rand(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(*shape)


# ---------------------------------------------------------------------------
# HSDCBlock tests
# ---------------------------------------------------------------------------

class TestHSDCBlock:
    """Output shape, dtype, and channel-count checks for the HSDC block."""

    @pytest.mark.parametrize("C_in,C_out", [(1, 4), (12, 48), (64, 64)])
    def test_output_shape(self, C_in: int, C_out: int) -> None:
        """HSDCBlock output must be (B, C_out, H, W) with C_out = 4 × C_per."""
        block = HSDCBlock(C_in, C_out, kernel_size=3)
        x = _rand((2, C_in, 32, 64))
        y = block(x)
        assert y.shape == (2, C_out, 32, 64), (
            f"Expected (2, {C_out}, 32, 64), got {y.shape}"
        )

    def test_output_channels_equals_4x_C_per(self) -> None:
        """Output channels must equal 4 × C_per (shared-weight concatenation)."""
        C_in, C_out = 16, 64
        block = HSDCBlock(C_in, C_out)
        assert block.out_channels == C_out
        assert block._C_per == C_out // 4

    def test_stride_halves_spatial(self) -> None:
        """stride=2 must halve both H and W."""
        block = HSDCBlock(4, 16, kernel_size=3, stride=2)
        x = _rand((1, 4, 32, 64))
        y = block(x)
        assert y.shape == (1, 16, 16, 32)

    def test_out_channels_not_divisible_by_4_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible by 4"):
            HSDCBlock(4, 6)

    def test_dtype_preserved(self) -> None:
        block = HSDCBlock(4, 16)
        x = _rand((1, 4, 16, 32))
        y = block(x)
        assert y.dtype == torch.float32

    def test_activate_false_no_relu(self) -> None:
        """With activate=False, output should differ from activate=True version."""
        x = _rand((1, 4, 16, 32))
        block_act   = HSDCBlock(4, 16, activate=True)
        block_noact = HSDCBlock(4, 16, activate=False)
        # Copy weights so only activation differs
        block_noact.load_state_dict(block_act.state_dict())
        y_act   = block_act(x)
        y_noact = block_noact(x)
        # activate=True clips negatives to 0; activate=False does not
        assert (y_noact < 0).any(), "activate=False should allow negative values"

    def test_7x7_stem_shape(self) -> None:
        """Stem HSDCBlock (7×7, 12→64) must produce correct shape."""
        block = HSDCBlock(12, 64, kernel_size=7, stride=2)
        x = _rand((2, 12, 256, 512))
        y = block(x)
        assert y.shape == (2, 64, 128, 256)

    def test_weights_are_shared(self) -> None:
        """All 4 branches must use the exact same weight tensor."""
        block = HSDCBlock(4, 16, kernel_size=3)
        # The shared conv is self.conv; we verify there is only ONE set of conv weights
        conv_params = [
            p for name, p in block.named_parameters()
            if "conv.weight" in name
        ]
        assert len(conv_params) == 1, "Expected exactly one shared weight tensor"


# ---------------------------------------------------------------------------
# SWHDCBlock tests
# ---------------------------------------------------------------------------

class TestSWHDCBlock:
    """Output shape, parameter count, and weight-sum checks for SWHDC block."""

    def test_output_shape_same_channels(self) -> None:
        """SWHDCBlock must preserve channel count (B, C, H, W) → (B, C, H, W)."""
        block = SWHDCBlock(8, erp_height=64)
        x = _rand((2, 8, 64, 128))
        y = block(x)
        assert y.shape == (2, 8, 64, 128)

    @pytest.mark.parametrize("C", [1, 4, 16, 64])
    def test_output_channels_preserved(self, C: int) -> None:
        block = SWHDCBlock(C, erp_height=32)
        x = _rand((1, C, 32, 64))
        y = block(x)
        assert y.shape[1] == C

    def test_stride_halves_spatial(self) -> None:
        block = SWHDCBlock(4, stride=2, erp_height=32)
        x = _rand((1, 4, 32, 64))
        y = block(x)
        assert y.shape == (1, 4, 16, 32)

    def test_zero_extra_trainable_params(self) -> None:
        """SWHDC block must have the same trainable parameters as a plain Conv2d.

        The latitude weights W_n(φ) are registered as buffers, not nn.Parameters,
        so they do NOT contribute to the trainable parameter count.
        """
        C = 16
        k = 3
        swhdc = SWHDCBlock(C, kernel_size=k, erp_height=64)
        plain = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=k, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )
        assert _count_params(swhdc) == _count_params(plain), (
            f"SWHDC has {_count_params(swhdc)} trainable params, "
            f"plain Conv2d has {_count_params(plain)}"
        )

    def test_swhdc_weights_not_in_parameters(self) -> None:
        """swhdc_weights must be a buffer (not a trainable parameter)."""
        block = SWHDCBlock(4, erp_height=64)
        param_names = {name for name, _ in block.named_parameters()}
        assert "swhdc_weights" not in param_names

    def test_swhdc_weights_in_buffers(self) -> None:
        """swhdc_weights must appear in the module's buffer registry."""
        block = SWHDCBlock(4, erp_height=64)
        buffer_names = {name for name, _ in block.named_buffers()}
        assert "swhdc_weights" in buffer_names

    def test_dtype_preserved(self) -> None:
        block = SWHDCBlock(4, erp_height=16)
        x = _rand((1, 4, 16, 32))
        y = block(x)
        assert y.dtype == torch.float32


# ---------------------------------------------------------------------------
# SWHDC weight tests (Eq. 3–4 correctness)
# ---------------------------------------------------------------------------

class TestSWHDCWeights:
    """Verify that the pre-computed latitude weights satisfy Eq. 3–4."""

    @pytest.mark.parametrize("H", [16, 64, 256])
    def test_weight_sum_per_row_equals_1(self, H: int) -> None:
        """Σ_n W_n(φ(y)) must equal 1.0 for every ERP row y.

        This verifies the linear interpolation in SWHDC Eq. 4 produces a
        valid probability distribution over dilation rates.
        """
        weights = compute_swhdc_weights(H, N=4)   # (4, H)
        row_sums = weights.sum(dim=0)              # (H,)
        np.testing.assert_allclose(
            row_sums.numpy(), 1.0, atol=1e-6,
            err_msg="Weight rows must sum to 1 (SWHDC Eq. 4)"
        )

    def test_weight_shape(self) -> None:
        weights = compute_swhdc_weights(256, N=4)
        assert weights.shape == (4, 256)

    def test_weights_nonnegative(self) -> None:
        weights = compute_swhdc_weights(256, N=4)
        assert weights.min() >= 0.0

    def test_weights_at_equator(self) -> None:
        """At the equator (φ ≈ π/2, sin(φ) ≈ 1) → R_φ ≈ 1 → W_1 = 1."""
        H = 256
        equator_row = H // 2   # row closest to φ = π/2
        weights = compute_swhdc_weights(H, N=4)  # (4, H)
        # At equator, dilation 1 (index 0) should have the dominant weight
        assert weights[0, equator_row].item() > 0.9, (
            "Equator row should be dominated by dilation-1 weight"
        )

    def test_weights_at_pole(self) -> None:
        """Near poles (φ → 0 or π, sin(φ) → 0) → R_φ = N → W_N = 1."""
        H = 256
        N = 4
        weights = compute_swhdc_weights(H, N=N)  # (N, H)
        # Top row (y=0): φ = π/H ≈ small, sin(φ) ≈ small → R_φ → N
        top_row_weight_N = weights[N - 1, 0].item()
        assert top_row_weight_N > 0.9, (
            "Top-row (near pole) should be dominated by dilation-N weight"
        )

    def test_registered_buffer_weight_sum(self) -> None:
        """The swhdc_weights buffer on the module must also sum to 1 per row."""
        H = 64
        block = SWHDCBlock(4, erp_height=H)
        row_sums = block.swhdc_weights.sum(dim=0)   # (H,)
        np.testing.assert_allclose(
            row_sums.numpy(), 1.0, atol=1e-6,
            err_msg="Module buffer weight rows must sum to 1"
        )


# ---------------------------------------------------------------------------
# ClassificationHead tests
# ---------------------------------------------------------------------------

class TestClassificationHead:
    def test_4d_input(self) -> None:
        head = ClassificationHead(512, 10)
        x = _rand((4, 512, 8, 16))
        y = head(x)
        assert y.shape == (4, 10)

    def test_2d_input(self) -> None:
        """Transformer backbones output (B, C) already; head must accept it."""
        head = ClassificationHead(768, 40)
        x = _rand((4, 768))
        y = head(x)
        assert y.shape == (4, 40)


# ---------------------------------------------------------------------------
# HSDCNet (ResNet-34 + HSDC) integration tests
# ---------------------------------------------------------------------------

class TestHSDCNet:
    """Shape, dtype, and approximate parameter-count checks for HSDCNet."""

    def test_output_shape_mn10(self) -> None:
        """HSDCNet must map (B, 12, 256, 512) → (B, 10)."""
        model = HSDCNet(in_channels=12, num_classes=10)
        model.eval()
        with torch.no_grad():
            x = _rand((2, 12, 256, 512))
            y = model(x)
        assert y.shape == (2, 10)

    def test_output_shape_mn40(self) -> None:
        model = HSDCNet(in_channels=12, num_classes=40)
        model.eval()
        with torch.no_grad():
            y = model(_rand((1, 12, 256, 512)))
        assert y.shape == (1, 40)

    def test_input_channel_count(self) -> None:
        """HSDCNet must reject inputs with wrong channel count."""
        model = HSDCNet(in_channels=12, num_classes=10)
        model.eval()
        with pytest.raises(RuntimeError):
            with torch.no_grad():
                model(_rand((1, 3, 256, 512)))   # wrong: 3 instead of 12

    def test_param_count_approx_5_3m(self) -> None:
        """HSDCNet trainable parameters must be approximately 5.3 M.

        The original paper (HSDC §II-C, Table 1) reports 5.3 M parameters.
        We allow ±15% tolerance to account for minor implementation differences.
        """
        model = HSDCNet(in_channels=12, num_classes=10)
        n_params = _count_params(model)
        target   = 5_300_000
        assert abs(n_params - target) / target < 0.15, (
            f"HSDCNet has {n_params:,} params; expected ≈ {target:,} (±15%)"
        )

    def test_output_dtype(self) -> None:
        model = HSDCNet(in_channels=12, num_classes=10)
        model.eval()
        with torch.no_grad():
            y = model(_rand((1, 12, 32, 64)))   # small input for speed
        assert y.dtype == torch.float32


# ---------------------------------------------------------------------------
# SWHDCResNet (ResNet-50 + SWHDC) integration tests
# ---------------------------------------------------------------------------

class TestSWHDCResNet:
    """Shape and parameter-count checks for SWHDCResNet."""

    def test_output_shape_mn10(self) -> None:
        """SWHDCResNet must map (B, 1, 256, 512) → (B, 10)."""
        model = SWHDCResNet(in_channels=1, num_classes=10)
        model.eval()
        with torch.no_grad():
            y = model(_rand((2, 1, 256, 512)))
        assert y.shape == (2, 10)

    def test_input_channel_count(self) -> None:
        """SWHDCResNet must reject inputs with wrong channel count."""
        model = SWHDCResNet(in_channels=1, num_classes=10)
        model.eval()
        with pytest.raises(RuntimeError):
            with torch.no_grad():
                model(_rand((1, 3, 256, 512)))   # wrong: 3 instead of 1

    def test_param_count_approx_25_5m(self) -> None:
        """SWHDCResNet trainable parameters must be approximately 25.5 M.

        SWHDC adds 0 extra parameters (weights are non-trainable buffers), so
        the count equals standard ResNet-50 (≈ 25.5 M).  We allow ±10%.
        """
        model = SWHDCResNet(in_channels=1, num_classes=10)
        n_params = _count_params(model)
        target   = 25_500_000
        assert abs(n_params - target) / target < 0.10, (
            f"SWHDCResNet has {n_params:,} params; expected ≈ {target:,} (±10%)"
        )

    def test_swhdc_buffers_not_trainable(self) -> None:
        """SWHDC latitude weights must NOT appear in trainable parameters."""
        model = SWHDCResNet(in_channels=1, num_classes=10)
        param_names = {name for name, _ in model.named_parameters()}
        swhdc_weight_params = [n for n in param_names if "swhdc_weights" in n]
        assert len(swhdc_weight_params) == 0, (
            f"SWHDC weights found in parameters: {swhdc_weight_params}"
        )

"""
ResNet-34 + HSDC (HSDCNet) and ResNet-50 + SWHDC baseline architectures.

HSDCNet (HSDC paper §II-C, Table 1)
------------------------------------
ResNet-34 with every 3×3 convolutional layer replaced by an HSDCBlock.
The 7×7 stem conv is also replaced by an HSDCBlock.  Input channels are
adapted to 12 (HSDC ERP) instead of the standard 3 (RGB).

Architecture (Table 1):
    Conv 1  : 1× HSDCBlock(12→64, 7×7, stride=2) + MaxPool(3, s=2)
    Conv 2–5: residual stages with HSDCBasicBlock, matching ResNet-34 depths
              [3, 4, 6, 3] blocks and channel widths [64, 128, 256, 512].
    Head    : GlobalAvgPool → Linear(512, num_classes)

SWHDCResNet (SWHDC paper §III-B, §IV-A)
-----------------------------------------
ResNet-50 with the 3×3 convolution in every Bottleneck block replaced by a
SWHDCBlock.  The 1×1 reduction and expansion convolutions are left unchanged
(SWHDC requires in_channels == out_channels, which is satisfied by all 3×3
convs in the bottleneck mid-layer).  This replacement adds ZERO extra
parameters; total parameter count matches standard ResNet-50 (≈25.5 M).

Input channels: 1 (SWHDC single-channel depth ERP).

References:
    HSDC paper §II-C, Table 1 — Stringhini et al., IEEE ICIP 2024
    SWHDC paper §III-B, §IV-A  — Stringhini et al., SIBGRAPI 2024
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.models.blocks.hsdc import HSDCBlock
from src.models.blocks.swhdc import SWHDCBlock
from src.models.classifier import ClassificationHead


# ===========================================================================
# HSDCNet — ResNet-34 with HSDC blocks
# ===========================================================================

class _HSDCBasicBlock(nn.Module):
    """ResNet-34 BasicBlock with both 3×3 convolutions replaced by HSDCBlock.

    Structure:
        HSDCBlock(in_ch → out_ch, stride=stride, activate=True)   # conv1 + BN + ReLU
        HSDCBlock(out_ch → out_ch, stride=1,      activate=False)  # conv2 + BN
        + shortcut (1×1 Conv + BN if dimensions change)
        → ReLU

    The second HSDC block omits its internal ReLU so that activation is
    applied AFTER the residual addition (matching standard ResNet convention).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()

        # First HSDC: full block with BN + ReLU
        self.hsdc1 = HSDCBlock(
            in_ch, out_ch, kernel_size=3, stride=stride, activate=True
        )
        # Second HSDC: BN only, no ReLU — activation after residual addition
        self.hsdc2 = HSDCBlock(
            out_ch, out_ch, kernel_size=3, stride=1, activate=False
        )
        self.relu = nn.ReLU(inplace=True)

        # Shortcut projection (only when spatial size or channels change)
        if stride != 1 or in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.hsdc1(x)
        out = self.hsdc2(out)
        out = self.relu(out + self.shortcut(x))
        return out


class HSDCNet(nn.Module):
    """ResNet-34 baseline with all convolutions replaced by HSDC blocks.

    Implements HSDCNet from HSDC paper §II-C and Table 1.  Every standard
    convolutional layer (including the 7×7 stem) is replaced by an HSDCBlock.
    The shortcut projections remain standard 1×1 Conv2d (dilation is
    ineffective on 1×1 kernels, and the paper focuses on replacing spatial
    convolutions).

    Args:
        in_channels:  ERP input channels (12 for HSDC pipeline, default 12).
        num_classes:  Number of output classes (10 for MN10, 40 for MN40).

    Input:   ``(B, 12, 256, 512)``
    Output:  ``(B, num_classes)`` logits

    References:
        HSDC paper §II-C, Table 1
    """

    # ResNet-34 block counts per stage — HSDC paper Table 1
    _LAYERS: tuple[int, ...] = (3, 4, 6, 3)

    def __init__(self, in_channels: int = 12, num_classes: int = 10) -> None:
        super().__init__()

        # Conv 1: 1× HSDC block (7×7, 64 out-channels) — Table 1
        # stride=2 halves spatial size: (B, 12, 256, 512) → (B, 64, 128, 256)
        self.stem = nn.Sequential(
            HSDCBlock(in_channels, 64, kernel_size=7, stride=2, activate=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # After MaxPool: (B, 64, 64, 128)  ← Table 1 "64 × 128"
        )

        # Conv 2–5: residual stages — Table 1
        self.layer1 = self._make_stage(64,  64,  self._LAYERS[0], stride=1)
        self.layer2 = self._make_stage(64,  128, self._LAYERS[1], stride=2)
        self.layer3 = self._make_stage(128, 256, self._LAYERS[2], stride=2)
        self.layer4 = self._make_stage(256, 512, self._LAYERS[3], stride=2)

        # Global-average-pool → FC → num_classes
        self.head = ClassificationHead(512, num_classes)

        self._init_weights()

    # ------------------------------------------------------------------
    # Layer builders
    # ------------------------------------------------------------------

    @staticmethod
    def _make_stage(
        in_ch: int,
        out_ch: int,
        n_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build a residual stage with n_blocks HSDCBasicBlocks."""
        blocks: list[nn.Module] = [
            _HSDCBasicBlock(in_ch, out_ch, stride=stride)
        ]
        for _ in range(1, n_blocks):
            blocks.append(_HSDCBasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        """Kaiming-normal initialisation for Conv2d; constant init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


# ===========================================================================
# SWHDCResNet — ResNet-50 with SWHDC blocks
# ===========================================================================

class _SWHDCBottleneck(nn.Module):
    """ResNet-50 Bottleneck with the 3×3 conv replaced by SWHDCBlock.

    Standard Bottleneck layout:
        conv1: 1×1, in_ch → mid_ch          (channel reduction)
        conv2: 3×3, mid_ch → mid_ch  ← SWHDC replaces this conv
        conv3: 1×1, mid_ch → out_ch         (channel expansion)
        + shortcut (1×1 Conv + BN if dimensions change)
        → ReLU

    SWHDC is applicable to conv2 because in_channels == out_channels
    (mid_ch → mid_ch), satisfying the zero-extra-parameter constraint.

    Args:
        in_ch:      Input channels (from previous block or stem).
        mid_ch:     Mid-layer (bottleneck) channels (= out_ch // 4 in ResNet-50).
        out_ch:     Output channels (= mid_ch × 4).
        stride:     Spatial stride applied to conv2 (default 1).
        erp_height: Output ERP height fed into SWHDCBlock weight pre-computation.
    """

    expansion: int = 4

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        stride: int = 1,
        erp_height: int = 256,
    ) -> None:
        super().__init__()

        # conv1: 1×1 reduction (standard — dilation is ineffective on 1×1)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_ch)
        self.relu1 = nn.ReLU(inplace=True)

        # conv2: 3×3 REPLACED BY SWHDC — SWHDC paper §III-B
        # mid_ch → mid_ch preserves channel count, enabling 0-parameter replacement
        self.conv2 = SWHDCBlock(
            mid_ch,
            kernel_size=3,
            stride=stride,
            activate=True,
            erp_height=erp_height,
        )

        # conv3: 1×1 expansion (standard)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.relu3 = nn.ReLU(inplace=True)

        # Shortcut projection
        if stride != 1 or in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)                      # SWHDCBlock includes BN + ReLU
        out = self.bn3(self.conv3(out))
        out = self.relu3(out + self.shortcut(x))
        return out


class SWHDCResNet(nn.Module):
    """ResNet-50 baseline with SWHDC replacing every bottleneck 3×3 convolution.

    Implements the SWHDC baseline from SWHDC paper §IV-A.  The 3×3 mid-layer
    conv in each Bottleneck block is replaced by a SWHDCBlock; all other layers
    (1×1 convolutions, stem, shortcut projections) remain unchanged.

    Because SWHDCBlock has in_channels == out_channels and zero extra trainable
    parameters, total model size matches standard ResNet-50 (≈ 25.5 M params).

    Args:
        in_channels:  ERP input channels (1 for SWHDC depth pipeline).
        num_classes:  Number of output classes.

    Input:   ``(B, 1, 256, 512)``
    Output:  ``(B, num_classes)`` logits

    References:
        SWHDC paper §III-B, §IV-A
    """

    # ResNet-50: [3, 4, 6, 3] Bottleneck blocks; channels [64, 128, 256, 512]
    _CFG: tuple[tuple[int, int, int], ...] = (
        (64,  64,  256),   # stage 1: mid=64,  out=256
        (256, 128, 512),   # stage 2: mid=128, out=512
        (512, 256, 1024),  # stage 3: mid=256, out=1024
        (1024, 512, 2048), # stage 4: mid=512, out=2048
    )
    _DEPTHS: tuple[int, ...] = (3, 4, 6, 3)

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()

        # Standard ResNet-50 stem (7×7 conv, BN, ReLU, MaxPool)
        # stride=2 twice: (B, 1, 256, 512) → (B, 64, 64, 128)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Spatial dimensions after stem: H=64, W=128
        # Each stride-2 stage halves H.
        heights = [64, 32, 16, 8]   # output height of each stage's SWHDC conv2

        self.layer1 = self._make_stage(
            self._CFG[0], self._DEPTHS[0], stride=1, erp_height=heights[0]
        )
        self.layer2 = self._make_stage(
            self._CFG[1], self._DEPTHS[1], stride=2, erp_height=heights[1]
        )
        self.layer3 = self._make_stage(
            self._CFG[2], self._DEPTHS[2], stride=2, erp_height=heights[2]
        )
        self.layer4 = self._make_stage(
            self._CFG[3], self._DEPTHS[3], stride=2, erp_height=heights[3]
        )

        self.head = ClassificationHead(2048, num_classes)

        self._init_weights()

    # ------------------------------------------------------------------
    # Layer builders
    # ------------------------------------------------------------------

    @staticmethod
    def _make_stage(
        cfg: tuple[int, int, int],
        n_blocks: int,
        stride: int,
        erp_height: int,
    ) -> nn.Sequential:
        in_ch, mid_ch, out_ch = cfg
        blocks: list[nn.Module] = [
            _SWHDCBottleneck(in_ch, mid_ch, out_ch, stride=stride, erp_height=erp_height)
        ]
        for _ in range(1, n_blocks):
            blocks.append(
                _SWHDCBottleneck(out_ch, mid_ch, out_ch, stride=1, erp_height=erp_height)
            )
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)

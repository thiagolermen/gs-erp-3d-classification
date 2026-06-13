"""
Plain ResNet-34 and ResNet-50 baselines (no distortion-correction block).

These are the *ablation baselines* for the TCC: structurally identical to
:class:`HSDCNet` and :class:`SWHDCResNet` (same per-channel input normalisation,
same stem layout, same stage depths/widths, same classification head), but with
every spatial convolution left as a standard zero-padded ``nn.Conv2d`` instead
of an HSDC / SWHDC block.  Training one of these from scratch isolates the
contribution of the distortion-correction blocks: any accuracy gap between
``ResNet34Baseline`` and ``HSDCNet`` (or ``ResNet50Baseline`` and
``SWHDCResNet``) is attributable to the block, not to the backbone, input
pipeline, or training protocol.

Both models accept an ``(N_shells, H, W)`` radiance-field ERP as input, with
``in_channels`` configurable (default 8) to match the cascading-sphere ERP.

ResNet34Baseline
----------------
Standard ResNet-34: 7×7 stem, BasicBlock stages ``[3, 4, 6, 3]`` with channel
widths ``[64, 128, 256, 512]``, GlobalAvgPool → Linear head.  ≈ 21.3 M params
(much larger than HSDCNet's 5.3 M — HSDC's shared dilated branches are the
source of that efficiency, which the baseline deliberately lacks).

ResNet50Baseline
----------------
Standard ResNet-50: 7×7 stem, Bottleneck stages ``[3, 4, 6, 3]``, GlobalAvgPool
→ Linear head.  ≈ 23.5 M params (parameter-comparable to SWHDCResNet's 25.5 M,
since SWHDC adds zero trainable parameters).

References:
    He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
    Trained from scratch (no ImageNet pretraining) — CLAUDE.md hard constraint.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.classifier import ClassificationHead


# ===========================================================================
# ResNet-34 baseline — standard BasicBlock
# ===========================================================================

class _BasicBlock(nn.Module):
    """Standard ResNet BasicBlock: two 3×3 convolutions + identity shortcut.

    Layout (He et al., 2016, Fig. 2):
        Conv 3×3 (in→out, stride) → BN → ReLU
        Conv 3×3 (out→out, 1)     → BN
        + shortcut (1×1 Conv + BN when shape changes)
        → ReLU

    This is the direct counterpart of :class:`_HSDCBasicBlock`, with the two
    HSDC blocks replaced by zero-padded standard convolutions.
    """

    expansion: int = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        return out


class ResNet34Baseline(nn.Module):
    """Plain ResNet-34 baseline for radiance-field ERP classification.

    Direct ablation counterpart of :class:`HSDCNet` — same per-channel input
    normalisation, stem geometry, stage depths/widths, and classification head,
    but every spatial convolution is a standard zero-padded ``nn.Conv2d``.

    Args:
        in_channels:  Number of input ERP channels (default 8 = N_shells=8).
        num_classes:  Number of output classes (10 for MN10, 40 for MN40).
        dropout:      Dropout probability before the final FC layer.

    Input:   ``(B, in_channels, 256, 512)``
    Output:  ``(B, num_classes)`` logits
    """

    _LAYERS: tuple[int, ...] = (3, 4, 6, 3)   # ResNet-34 block counts per stage

    def __init__(self, in_channels: int = 8, num_classes: int = 10, dropout: float = 0.0) -> None:
        super().__init__()

        # Per-channel instance normalisation — identical to HSDCNet so the only
        # difference between the two models is the convolutional block itself.
        self.input_norm = nn.InstanceNorm2d(in_channels, affine=True)

        # Standard ResNet stem: 7×7 stride-2 conv + 3×3 stride-2 max-pool.
        # (B, in_channels, 256, 512) → (B, 64, 64, 128)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_stage(64,  64,  self._LAYERS[0], stride=1)
        self.layer2 = self._make_stage(64,  128, self._LAYERS[1], stride=2)
        self.layer3 = self._make_stage(128, 256, self._LAYERS[2], stride=2)
        self.layer4 = self._make_stage(256, 512, self._LAYERS[3], stride=2)

        self.head = ClassificationHead(512, num_classes, dropout=dropout)

        self._init_weights()

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        blocks: list[nn.Module] = [_BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            blocks.append(_BasicBlock(out_ch, out_ch, stride=1))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


# ===========================================================================
# ResNet-50 baseline — standard Bottleneck
# ===========================================================================

class _Bottleneck(nn.Module):
    """Standard ResNet Bottleneck: 1×1 → 3×3 → 1×1 + identity shortcut.

    Direct counterpart of :class:`_SWHDCBottleneck`, with the 3×3 mid-layer
    convolution left as a standard zero-padded ``nn.Conv2d`` instead of a SWHDC
    block.
    """

    expansion: int = 4

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.relu(out + self.shortcut(x))
        return out


class ResNet50Baseline(nn.Module):
    """Plain ResNet-50 baseline for radiance-field ERP classification.

    Direct ablation counterpart of :class:`SWHDCResNet` — same per-channel input
    normalisation, stem, stage depths/widths, and classification head, but every
    3×3 bottleneck convolution is a standard zero-padded ``nn.Conv2d``.

    Args:
        in_channels:  Number of input ERP channels (default 8 = N_shells=8).
        num_classes:  Number of output classes.
        dropout:      Dropout probability before the final FC layer.

    Input:   ``(B, in_channels, 256, 512)``
    Output:  ``(B, num_classes)`` logits
    """

    # (in_ch, mid_ch, out_ch) per stage — matches SWHDCResNet._CFG
    _CFG: tuple[tuple[int, int, int], ...] = (
        (64,   64,  256),
        (256,  128, 512),
        (512,  256, 1024),
        (1024, 512, 2048),
    )
    _DEPTHS: tuple[int, ...] = (3, 4, 6, 3)

    def __init__(self, in_channels: int = 8, num_classes: int = 10, dropout: float = 0.0) -> None:
        super().__init__()

        self.input_norm = nn.InstanceNorm2d(in_channels, affine=True)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_stage(self._CFG[0], self._DEPTHS[0], stride=1)
        self.layer2 = self._make_stage(self._CFG[1], self._DEPTHS[1], stride=2)
        self.layer3 = self._make_stage(self._CFG[2], self._DEPTHS[2], stride=2)
        self.layer4 = self._make_stage(self._CFG[3], self._DEPTHS[3], stride=2)

        self.head = ClassificationHead(2048, num_classes, dropout=dropout)

        self._init_weights()

    @staticmethod
    def _make_stage(cfg: tuple[int, int, int], n_blocks: int, stride: int) -> nn.Sequential:
        in_ch, mid_ch, out_ch = cfg
        blocks: list[nn.Module] = [_Bottleneck(in_ch, mid_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            blocks.append(_Bottleneck(out_ch, mid_ch, out_ch, stride=1))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)

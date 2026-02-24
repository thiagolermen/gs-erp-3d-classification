"""
EfficientNetV2-S + HSDC / SWHDC proposed architectures.

Integration strategy (CLAUDE.md §Agent 2):
    The HSDC or SWHDC block is placed BEFORE the EfficientNetV2-S feature
    extraction pipeline, operating at full ERP pixel resolution.  Distortion
    correction is applied before the EfficientNetV2-S convolutional stem
    processes the feature map.

    1. HSDC + EfficientNetV2-S:
       Input  (B, 12, H, W)
       → HSDCBlock(12→48, 3×3) [full-resolution distortion correction]
       → EfficientNetV2-S with in_chans=48
       → ClassificationHead

    2. SWHDC + EfficientNetV2-S:
       Input  (B, 1, H, W)
       → SWHDCBlock(1→1, 3×3) [full-resolution distortion correction]
       → EfficientNetV2-S with in_chans=1
       → ClassificationHead

    The EfficientNetV2-S backbone is created via ``timm.create_model`` with
    ``num_classes=0`` to remove the default head.

References:
    HSDC paper §II-C — Stringhini et al., IEEE ICIP 2024
    SWHDC paper §III-B — Stringhini et al., SIBGRAPI 2024
    EfficientNetV2 — Tan & Le, ICML 2021
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for EffNetV2HSDCNet. Install with: pip install timm"
    ) from e

from src.models.blocks.hsdc import HSDCBlock
from src.models.blocks.swhdc import SWHDCBlock
from src.models.classifier import ClassificationHead

# timm model name for EfficientNetV2-S
_EFFNETV2_S = "efficientnetv2_s"


class EffNetV2HSDCNet(nn.Module):
    """EfficientNetV2-S backbone preceded by an HSDC or SWHDC stem.

    The HSDC/SWHDC block corrects spherical ERP distortion at full pixel
    resolution before the EfficientNetV2-S stem processes the feature map.

    Args:
        pipeline:     ``'hsdc'`` (12-channel input) or ``'swhdc'`` (1-channel input).
        num_classes:  Number of output classes (10 or 40).
        erp_height:   ERP image height for SWHDCBlock weight pre-computation.
        pretrained:   Load timm ImageNet weights (default False; ERP inputs
                      differ from RGB, and input channels differ).

    Input:
        HSDC pipeline:  ``(B, 12, H, W)``
        SWHDC pipeline: ``(B, 1, H, W)``

    Output:  ``(B, num_classes)`` logits
    """

    def __init__(
        self,
        pipeline: str = "hsdc",
        num_classes: int = 10,
        erp_height: int = 256,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        if pipeline not in ("hsdc", "swhdc"):
            raise ValueError(f"pipeline must be 'hsdc' or 'swhdc', got '{pipeline}'")

        self.pipeline = pipeline

        # --- Distortion-correction stem ---
        if pipeline == "hsdc":
            # HSDC: 12 → 48 channels (4 × 12) — HSDC paper §II-C
            self.dc_stem: nn.Module = HSDCBlock(
                in_channels=12,
                out_channels=48,
                kernel_size=3,
                stride=1,
                activate=True,
            )
            backbone_in_chans = 48
        else:
            # SWHDC: 1 → 1 channel (no expansion) — SWHDC paper §III-B
            self.dc_stem = SWHDCBlock(
                in_channels=1,
                kernel_size=3,
                stride=1,
                activate=True,
                erp_height=erp_height,
            )
            backbone_in_chans = 1

        # --- EfficientNetV2-S backbone ---
        # timm's `in_chans` replaces the first-layer input channels
        self.backbone = timm.create_model(
            _EFFNETV2_S,
            pretrained=pretrained,
            in_chans=backbone_in_chans,
            num_classes=0,       # strip default head
            global_pool="avg",   # pool to (B, num_features)
        )

        # --- Classification head ---
        backbone_features = self.backbone.num_features
        self.head = ClassificationHead(backbone_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Distortion correction at full ERP resolution
        x = self.dc_stem(x)
        # EfficientNetV2-S: (B, C_dc, H, W) → (B, num_features)
        x = self.backbone(x)
        # Classification: (B, num_features) → (B, num_classes)
        return self.head(x)

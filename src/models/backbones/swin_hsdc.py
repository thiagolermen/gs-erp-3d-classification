"""
Swin Transformer (Swin-T) + HSDC / SWHDC proposed architectures.

Integration strategy (CLAUDE.md §Agent 2):
    The HSDC or SWHDC block is placed BEFORE the Swin-T patch-embedding layer,
    operating at the full ERP pixel resolution.  Distortion correction is thus
    applied to the raw feature map before patch tokenisation.

    1. HSDC + Swin-T:
       Input  (B, 12, H, W)
       → HSDCBlock(12→48, 7×7) [full-resolution distortion correction]
       → Swin-T with in_chans=48 (patch embedding adapted)
       → ClassificationHead

    2. SWHDC + Swin-T:
       Input  (B, 1, H, W)
       → SWHDCBlock(1→1, 3×3) [full-resolution distortion correction]
       → Swin-T with in_chans=1 (patch embedding adapted)
       → ClassificationHead

    The Swin-T backbone is created via ``timm.create_model`` with
    ``num_classes=0`` to strip the default head (replaced by ClassificationHead).

References:
    HSDC paper §II-C — Stringhini et al., IEEE ICIP 2024
    SWHDC paper §III-B — Stringhini et al., SIBGRAPI 2024
    Swin Transformer §3 — Liu et al., ICCV 2021
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for SwinHSDCNet. Install with: pip install timm"
    ) from e

from src.models.blocks.hsdc import HSDCBlock
from src.models.blocks.swhdc import SWHDCBlock
from src.models.classifier import ClassificationHead

# timm model name for Swin Transformer Tiny
_SWIN_TINY = "swin_tiny_patch4_window7_224"


class SwinHSDCNet(nn.Module):
    """Swin-T backbone preceded by an HSDC or SWHDC distortion-correction stem.

    The HSDC/SWHDC block operates at full ERP resolution (pixel-level), correcting
    spherical distortion before the Swin-T patch embedding tokenises the image.

    Args:
        pipeline:     ``'hsdc'`` (12-channel input, 48-channel HSDC output) or
                      ``'swhdc'`` (1-channel input, 1-channel SWHDC output).
        num_classes:  Number of output classes (10 or 40).
        erp_height:   ERP image height, used by SWHDCBlock weight pre-computation.
        pretrained:   Load timm ImageNet weights for the Swin backbone (default False;
                      ERP inputs differ from RGB images so pretrained weights need
                      fine-tuning or are incompatible due to channel mismatch).

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
            # HSDC: 12-channel ERP → 48 channels (4 × 12)
            # 7×7 kernel matches the original HSDCNet stem — HSDC paper §II-C
            self.dc_stem: nn.Module = HSDCBlock(
                in_channels=12,
                out_channels=48,
                kernel_size=7,
                stride=1,
                activate=True,
            )
            backbone_in_chans = 48
        else:
            # SWHDC: 1-channel depth ERP → 1 channel (same, no expansion)
            self.dc_stem = SWHDCBlock(
                in_channels=1,
                kernel_size=3,
                stride=1,
                activate=True,
                erp_height=erp_height,
            )
            backbone_in_chans = 1

        # --- Swin-T backbone (patch embedding adapted to new channel count) ---
        # timm's `in_chans` replaces the default 3-channel patch embedding projection
        self.backbone = timm.create_model(
            _SWIN_TINY,
            pretrained=pretrained,
            in_chans=backbone_in_chans,
            num_classes=0,           # strip default classification head
            global_pool="avg",       # pool to (B, embed_dim) before head
        )

        # --- Classification head ---
        backbone_features = self.backbone.num_features
        self.head = ClassificationHead(backbone_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C_in, H, W) → distortion correction → (B, C_dc, H, W)
        x = self.dc_stem(x)
        # Swin-T backbone: (B, C_dc, H, W) → (B, num_features) after avg pool
        x = self.backbone(x)
        # Classification head: (B, num_features) → (B, num_classes)
        return self.head(x)

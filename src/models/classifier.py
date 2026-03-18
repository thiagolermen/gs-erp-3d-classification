"""
Classification head for ERP-ViT 3D Classification models.

Implements a Global Average Pooling → Linear → (Softmax at inference) head
compatible with both CNN feature maps ``(B, C, H, W)`` and Transformer
sequence outputs ``(B, C)`` (already pooled by the backbone).

References:
    HSDC paper §II-C, Table 1 — classification layer
    SWHDC paper §IV-A — classification layer
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Global Average Pooling + fully-connected classification head.

    Accepts either:
    - 4-D CNN feature maps ``(B, C, H, W)`` — applies AdaptiveAvgPool2d(1).
    - 2-D Transformer outputs ``(B, C)`` — fed directly to the linear layer.

    The raw logits are returned; pass through ``nn.Softmax(dim=1)`` at
    inference time or use ``nn.CrossEntropyLoss`` during training.

    Args:
        in_features:  Number of input feature channels C.
        num_classes:  Number of target classes (10 for ModelNet10, 40 for MN40).

    Input:   ``(B, C, H, W)`` or ``(B, C)``
    Output:  ``(B, num_classes)`` logits
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = self.gap(x).flatten(1)   # (B, C, H, W) → (B, C)
        x = self.dropout(x)
        return self.fc(x)                # (B, num_classes)

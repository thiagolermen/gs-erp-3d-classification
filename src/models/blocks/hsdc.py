"""
HSDC (Horizontally Dilated Spherical Distortion Correction) block.

Implements the 4-branch horizontally dilated convolution with shared weights
described in HSDC paper §II-C (Fig. 2).  Four convolutions apply the SAME
weight tensor with horizontal dilation rates 1, 2, 3, 4.  Their outputs are
concatenated along the channel dimension and passed through BatchNorm + ReLU.

The horizontal axis uses circular padding to exploit the wrap-around property
of equirectangular projection images.  The vertical axis uses replicate padding.

References:
    HSDC paper §II-C, Fig. 2 — Stringhini et al., IEEE ICIP 2024
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HSDCBlock(nn.Module):
    """Horizontally Dilated Spherical Distortion Correction block.

    Applies 4 dilated convolutions (horizontal dilation rates 1–4) to the
    same input using SHARED weights, as described in HSDC paper §II-C.
    Outputs are concatenated along the channel dimension, producing
    ``4 × C_per_branch`` output channels, then passed through BatchNorm.
    An optional ReLU activation is applied afterwards.

    Shared-weight mechanism: a single ``nn.Conv2d(C_in, C_per, k)`` is defined.
    During ``forward``, ``F.conv2d`` is called 4 times reusing ``self.conv.weight``
    with dilation = (1, d) for d ∈ {1, 2, 3, 4}  (vertical dilation is always 1).

    Args:
        in_channels:   Number of input channels.
        out_channels:  Total output channels (must be divisible by 4).
                       Each of the 4 branches produces ``out_channels // 4`` channels.
        kernel_size:   Spatial kernel size (default 3; the stem uses 7).
        stride:        Convolution stride applied to ALL branches (default 1).
        bias:          Add learnable bias to the shared conv (default False;
                       BatchNorm makes the bias redundant).
        activate:      Whether to apply ReLU after BatchNorm (default True).
                       Set to False for the second conv in residual blocks so that
                       ReLU is applied after the residual addition.

    Input:   ``(B, C_in, H, W)``
    Output:  ``(B, out_channels, H // stride, W // stride)``

    References:
        HSDC paper §II-C, Fig. 2
    """

    DILATION_RATES: tuple[int, ...] = (1, 2, 3, 4)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False,
        activate: bool = True,
    ) -> None:
        super().__init__()

        if out_channels % 4 != 0:
            raise ValueError(
                f"out_channels must be divisible by 4, got {out_channels}"
            )

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.activate     = activate
        self._C_per       = out_channels // 4  # channels per branch

        # Single shared weight — HSDC paper §II-C: "all 4 convolutions share weights"
        self.conv = nn.Conv2d(
            in_channels,
            self._C_per,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,   # manual padding applied in forward()
            bias=bias,
        )

        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 4 shared-weight dilated convolutions and concatenate.

        For each dilation rate ``d`` in {1, 2, 3, 4}:
          1. Pad left/right with **circular** mode (ERP wrap-around) by
             ``(kernel_size - 1) // 2 * d`` pixels per side.
          2. Pad top/bottom with **replicate** mode (pole boundary) by
             ``(kernel_size - 1) // 2`` pixels per side.
          3. Call ``F.conv2d`` with ``dilation=(1, d)`` and the shared weight,
             producing ``C_per`` output channels.

        The 4 branches are concatenated → ``(B, 4·C_per, H', W')`` → BN → ReLU.

        Args:
            x: ``(B, C_in, H, W)`` input tensor.

        Returns:
            ``(B, out_channels, H // stride, W // stride)`` output tensor.
        """
        k = self.kernel_size
        branches: list[torch.Tensor] = []

        for d in self.DILATION_RATES:
            # Padding to maintain spatial size under dilation (1, d)
            # Vertical: (k-1)//2   (vertical dilation = 1)
            # Horizontal: (k-1)//2 * d  (horizontal dilation = d)
            pad_v = (k - 1) // 2
            pad_h = (k - 1) // 2 * d

            # Circular padding on horizontal axis — ERP wrap-around, HSDC §II-C
            x_pad = F.pad(x, (pad_h, pad_h, 0, 0), mode="circular")
            # Replicate padding on vertical axis (poles)
            x_pad = F.pad(x_pad, (0, 0, pad_v, pad_v), mode="replicate")

            # Shared-weight conv with horizontal dilation d — HSDC paper §II-C
            branch = F.conv2d(
                x_pad,
                self.conv.weight,
                self.conv.bias,
                stride=self.stride,
                padding=0,
                dilation=(1, d),
            )
            branches.append(branch)

        # Concatenate branches — HSDC paper §II-C: output = 4 × input channels
        out = torch.cat(branches, dim=1)   # (B, 4·C_per, H', W')
        out = self.bn(out)
        if self.activate:
            out = self.relu(out)
        return out

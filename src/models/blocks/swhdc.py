"""
SWHDC (Spherically Weighted Horizontally Dilated Convolution) block.

Implements the latitude-weighted dilated convolution described in SWHDC
paper §III-B (Eq. 3–5, Fig. 4).  Four shared-weight convolutions with
horizontal dilation rates 1–4 are computed.  Their outputs are combined
with pre-computed, latitude-dependent mixing weights W_n(φ):

    F* = Σ_n W_n(φ) ⊙ F_n                                    [SWHDC Eq. 5]

Key properties:
- Output channels **equal** input channels (no expansion).
- W_n weights are registered as **non-trainable buffers**; this block adds
  ZERO extra trainable parameters compared to a standard convolution.
- N=4 dilation rates cover 96.85% of the spherical surface (SWHDC §III-B).

References:
    SWHDC paper §III-B, Eq. 3–5, Fig. 4 — Stringhini et al., SIBGRAPI 2024
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weight pre-computation
# ---------------------------------------------------------------------------

def compute_swhdc_weights(height: int, N: int = 4) -> torch.Tensor:
    """Pre-compute the latitude-dependent mixing weights W_n(φ).

    For ERP row y (φ(y) = π(y + 0.5)/H):

        R_φ = min(N, 1 / sin(φ(y)))                           [SWHDC Eq. 3]

    If R_φ is integer n₀: W_n = 1 if n = n₀, else 0.
    If R_φ is between integers n1 and n2 = n1+1:
        W_n1 = n2 − R_φ   (weight of lower bracket)
        W_n2 = R_φ − n1   (weight of upper bracket)           [SWHDC Eq. 4]

    This guarantees Σ_n W_n(φ) = 1 for every row.

    Args:
        height: ERP image height H.
        N:      Number of dilation rates (fixed at 4 in SWHDC paper §III-B).

    Returns:
        weights: ``(N, H)`` float32 tensor — mixing weights per dilation × row.
    """
    y       = torch.arange(height, dtype=torch.float64)
    phi     = math.pi * (y + 0.5) / height           # (H,) in (0, π)

    # R_φ = min(N, 1/sin(φ)) — SWHDC paper Eq. 3
    sin_phi = torch.sin(phi).clamp(min=1e-8)          # guard against pole singularity
    R       = torch.clamp(1.0 / sin_phi, max=float(N))  # (H,)

    r_floor = torch.floor(R)                           # (H,)
    r_ceil  = torch.ceil(R)                            # (H,)
    frac    = R - r_floor                              # (H,) fractional part

    weights = torch.zeros(N, height, dtype=torch.float64)

    for n_idx in range(N):
        n = float(n_idx + 1)   # dilation indices are 1-based (1, 2, 3, 4)
        # Lower bracket contribution: (1 − frac) when floor(R) == n
        lower = (r_floor == n).double() * (1.0 - frac)
        # Upper bracket contribution: frac when ceil(R) == n
        upper = (r_ceil == n).double() * frac
        # Exact integer match overrides interpolation with weight 1
        exact = torch.abs(R - n) < 1e-9
        weights[n_idx] = torch.where(exact, torch.ones_like(R), lower + upper)

    return weights.float()   # (N, H)


# ---------------------------------------------------------------------------
# SWHDC Block
# ---------------------------------------------------------------------------

class SWHDCBlock(nn.Module):
    """Spherically Weighted Horizontally Dilated Convolution block.

    Applies ``N`` (default 4) horizontally dilated convolutions with SHARED
    weights to the input.  Their outputs are linearly combined row-by-row
    using pre-computed latitude weights W_n(φ):

        F* = Σ_n W_n(φ) ⊙ F_n                                [SWHDC Eq. 5]

    This block:
    - Preserves the channel count (``out = in``).
    - Adds **zero** trainable parameters relative to a standard 3×3 conv.
    - Registers ``swhdc_weights`` as a non-trainable buffer (``register_buffer``).

    Args:
        in_channels:  Number of input (and output) channels C.
        kernel_size:  Spatial kernel size (default 3).
        stride:       Convolution stride (default 1).
        bias:         Add learnable bias to the shared conv (default False).
        activate:     Whether to apply ReLU after BatchNorm (default True).
                      Set to False when placed before a residual addition.
        erp_height:   ERP image height used to pre-compute W_n buffers (default 256).
                      If the actual input height differs at runtime, weights are
                      recomputed on-the-fly (not stored permanently).
        N:            Number of dilation rates (fixed at 4 per SWHDC paper §III-B).

    Input:   ``(B, C, H, W)``
    Output:  ``(B, C, H // stride, W // stride)``

    References:
        SWHDC paper §III-B, Eq. 3–5, Fig. 4
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False,
        activate: bool = True,
        erp_height: int = 256,
        N: int = 4,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride      = stride
        self.activate    = activate
        self.N           = N

        # Shared weight: same in/out channels — SWHDC adds 0 extra parameters
        # SWHDC paper §III-B: "all convolutions share the same weights"
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,   # output channels == input channels (no expansion)
            kernel_size=kernel_size,
            stride=stride,
            padding=0,     # manual padding in forward()
            bias=bias,
        )

        self.bn   = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Pre-computed latitude weights — NON-TRAINABLE (SWHDC paper §III-B)
        # Registered as a buffer so they appear in state_dict and move with .to()
        weights = compute_swhdc_weights(erp_height, N)   # (N, H)
        self.register_buffer("swhdc_weights", weights)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_weights(self, out_h: int, device: torch.device) -> torch.Tensor:
        """Return latitude weights for the given output height.

        Uses the pre-computed buffer if its height matches; otherwise recomputes
        on-the-fly (each block has a fixed spatial position in the network, so
        this branch is only taken when the model is called with an unexpected
        input resolution).
        """
        if self.swhdc_weights.shape[1] == out_h:
            return self.swhdc_weights
        return compute_swhdc_weights(out_h, self.N).to(device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply N shared-weight dilated convolutions weighted by latitude.

        For each dilation rate ``d`` in {1, …, N}:
          1. Circular-pad horizontally by ``(k-1)//2 * d`` per side.
          2. Replicate-pad vertically by ``(k-1)//2`` per side.
          3. Apply shared conv with ``dilation=(1, d)``.
          4. Multiply by the latitude weight W_n(φ), broadcast across width.

        The N weighted branches are summed: F* = Σ_n W_n(φ) ⊙ F_n.

        Args:
            x: ``(B, C, H, W)`` input tensor.

        Returns:
            ``(B, C, H // stride, W // stride)`` latitude-corrected tensor.
        """
        k     = self.kernel_size
        out_h = x.shape[2] // self.stride
        weights = self._get_weights(out_h, x.device)   # (N, H')

        result: torch.Tensor | None = None

        for i in range(self.N):
            d     = i + 1   # dilation rate: 1, 2, 3, 4
            pad_v = (k - 1) // 2
            pad_h = (k - 1) // 2 * d

            # Circular horizontal padding — ERP wrap-around, SWHDC paper §III-B
            x_pad = F.pad(x, (pad_h, pad_h, 0, 0), mode="circular")
            # Replicate vertical padding (poles)
            x_pad = F.pad(x_pad, (0, 0, pad_v, pad_v), mode="replicate")

            branch = F.conv2d(
                x_pad,
                self.conv.weight,
                self.conv.bias,
                stride=self.stride,
                padding=0,
                dilation=(1, d),
            )  # (B, C, H', W')

            # Broadcast W_n(φ) over batch and width — SWHDC paper Eq. 5
            w      = weights[i].view(1, 1, out_h, 1)   # (1, 1, H', 1)
            result = w * branch if result is None else result + w * branch

        out = self.bn(result)
        if self.activate:
            out = self.relu(out)
        return out

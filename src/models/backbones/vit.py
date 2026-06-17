"""
Vision Transformer (ViT) backbone for radiance-field ERP 3D classification.

This is the *Transformer baseline* for the TCC: a plain, distortion-agnostic ViT
applied to the **exact same** N-channel radiance-field ERP fed to the
:class:`HSDCNet` / :class:`SWHDCResNet` CNNs.  It plays the same role the
"PanoFormer ViT (from scratch)" row plays in the reference papers — if a vanilla
ViT on the identical RF-3DGS ERP underperforms the distortion-aware CNNs, that
gap is attributable to the HSDC / SWHDC blocks, not to the input pipeline.

The model follows the original ViT recipe (Dosovitskiy et al., ICLR 2021) with
the DeiT-style from-scratch regularisation (stochastic depth / drop-path —
Huang et al., ECCV 2016; Touvron et al., ICML 2021), since ModelNet10/40 from
3DGS is small (~4-12 k objects) and ViTs overfit badly from scratch.  Default
hyper-parameters are **ViT-Tiny** (embed_dim=192, depth=12, heads=3, ≈ 5.5 M
params) — closely matching HSDCNet's 5.3 M for a fair comparison.

The ERP is non-square (256×512); a 1-D learnable positional embedding over the
flattened ``(H/patch)×(W/patch)`` patch grid handles this naturally.

Trained from scratch (no ImageNet pretraining) — CLAUDE.md hard constraint.

References:
    Dosovitskiy et al., "An Image is Worth 16×16 Words", ICLR 2021.
    Touvron et al., "Training data-efficient image transformers (DeiT)", ICML 2021.
    Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016 (drop-path).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.classifier import ClassificationHead


# ===========================================================================
# Stochastic depth (drop-path)
# ===========================================================================

def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Per-sample stochastic depth (Huang et al., ECCV 2016).

    Randomly zeroes the *entire* residual branch for a subset of samples in the
    batch, rescaling the survivors by ``1/(1-drop_prob)`` to keep the expectation
    unchanged.  A no-op at inference or when ``drop_prob == 0``.

    Args:
        x:          Residual-branch output ``(B, ...)``.
        drop_prob:  Probability of dropping the branch for a given sample.
        training:   Whether the module is in training mode.

    Returns:
        ``x`` with whole-sample paths randomly dropped (training) or unchanged.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # Broadcast mask over all non-batch dims: shape (B, 1, 1, ...).
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    return x / keep_prob * mask


class DropPath(nn.Module):
    """Module wrapper around :func:`drop_path` (stochastic depth)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ===========================================================================
# Patch embedding
# ===========================================================================

class PatchEmbed(nn.Module):
    """Split an ERP into non-overlapping patches and linearly project them.

    Implemented as a strided convolution (the standard ViT trick): a
    ``patch×patch`` kernel with ``stride = patch`` produces one ``embed_dim``
    token per patch.

    Args:
        in_channels: Number of ERP input channels (8 shells + derived = 10).
        embed_dim:   Token embedding dimension.
        img_size:    ``(H, W)`` of the ERP (default 256×512).
        patch_size:  Side length of a square patch.

    Input:   ``(B, in_channels, H, W)``
    Output:  ``(B, num_patches, embed_dim)`` with ``num_patches = (H/p)·(W/p)``
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        img_size: tuple[int, int] = (256, 512),
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        h, w = img_size
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(
                f"ERP size {img_size} must be divisible by patch_size {patch_size}."
            )
        self.grid_size = (h // patch_size, w // patch_size)      # (16, 32) for default
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 512 for default
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)              # (B, embed_dim, H/p, W/p)
        x = x.flatten(2)              # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)         # (B, num_patches, embed_dim)
        return x


# ===========================================================================
# Transformer encoder block
# ===========================================================================

class _MLP(nn.Module):
    """Position-wise feed-forward network: Linear → GELU → Dropout → Linear → Dropout."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1  = nn.Linear(dim, hidden_dim)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class _EncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block with multi-head self-attention.

    Layout (pre-norm, as in ViT/DeiT):
        x = x + DropPath(MHSA(LN(x)))
        x = x + DropPath(MLP(LN(x)))

    Args:
        dim:          Token embedding dimension.
        num_heads:    Number of attention heads.
        mlp_ratio:    Hidden-dim multiplier for the feed-forward network.
        dropout:      Dropout for projections and the MLP.
        attn_dropout: Dropout on attention weights.
        drop_path:    Stochastic-depth drop probability for this block.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# ===========================================================================
# Full ViT model
# ===========================================================================

class ERPViT(nn.Module):
    """Vision Transformer over a radiance-field ERP (RF-3DGS Transformer baseline).

    Pipeline:
        InstanceNorm2d → PatchEmbed → prepend CLS token → + pos-embed → dropout
        → ``depth`` pre-norm encoder blocks → final LayerNorm → CLS token
        → :class:`ClassificationHead`.

    The leading per-channel :class:`~torch.nn.InstanceNorm2d` mirrors the ResNet
    stems (``resnet_hsdc.py`` / ``resnet_baseline.py``): each density shell has a
    different magnitude, so normalising per channel keeps the only difference
    between models the architecture itself, not the input scaling.

    Args:
        in_channels:    Number of input ERP channels (default 10 = 8 shells + 2 derived).
        num_classes:    Number of output classes (10 for MN10, 40 for MN40).
        img_size:       ``(H, W)`` of the ERP (default 256×512).
        patch_size:     Side length of a square patch (default 16 → 16×32 = 512 tokens).
        embed_dim:      Token embedding dimension (default 192 = ViT-Tiny).
        depth:          Number of Transformer encoder blocks (default 12).
        num_heads:      Number of attention heads (default 3 = ViT-Tiny).
        mlp_ratio:      Feed-forward hidden-dim multiplier (default 4.0).
        dropout:        Dropout for the embedding, MLPs, and classification head.
        attn_dropout:   Dropout on attention weights.
        drop_path_rate: Max stochastic-depth rate; scaled linearly 0→rate over depth.

    Input:   ``(B, in_channels, H, W)``
    Output:  ``(B, num_classes)`` logits
    """

    def __init__(
        self,
        in_channels: int = 10,
        num_classes: int = 10,
        img_size: tuple[int, int] = (256, 512),
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        # Per-channel normalisation — identical to the ResNet backbones so the
        # only difference between models is the architecture, not input scaling.
        self.input_norm = nn.InstanceNorm2d(in_channels, affine=True)

        self.patch_embed = PatchEmbed(in_channels, embed_dim, img_size, patch_size)
        num_patches = self.patch_embed.num_patches

        # Learnable class token + 1-D positional embedding (handles the
        # non-square 16×32 grid by operating on the flattened token sequence).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Linearly-spaced stochastic-depth schedule across blocks (0 → rate).
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            _EncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr[i],
            )
            for i in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = ClassificationHead(embed_dim, num_classes, dropout=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Truncated-normal init for pos-embed / cls-token, Xavier for linears."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)                       # (B, C, H, W)
        x = self.patch_embed(x)                      # (B, num_patches, embed_dim)

        cls = self.cls_token.expand(x.shape[0], -1, -1)   # (B, 1, embed_dim)
        x = torch.cat((cls, x), dim=1)               # (B, num_patches+1, embed_dim)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        cls_out = x[:, 0]                            # (B, embed_dim) — CLS token
        return self.head(cls_out)                   # (B, num_classes)

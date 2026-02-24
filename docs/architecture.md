# Architecture Documentation

## Overview

Four model configurations are implemented, all operating on Equirectangular
Projection (ERP) images produced by `src/preprocessing/`:

| Model | Pipeline | Input channels | Backbone | Params |
|---|---|---|---|---|
| HSDCNet | HSDC | 12 | ResNet-34 | ≈ 5.3 M |
| SWHDCResNet | SWHDC | 1 | ResNet-50 | ≈ 25.5 M |
| SwinHSDCNet | HSDC/SWHDC | 12 / 1 | Swin-T | ≈ 28–30 M |
| EffNetV2HSDCNet | HSDC/SWHDC | 12 / 1 | EfficientNetV2-S | ≈ 21–24 M |

---

## Distortion-Correction Blocks

### HSDC Block (`src/models/blocks/hsdc.py`)

> HSDC paper §II-C, Fig. 2

```
Input: (B, C_in, H, W)

                  ┌──────────────────────┐
                  │  dilation (1, 1)     │──→ (B, C_per, H, W)
                  ├──────────────────────┤
x ──→ shared ────│  dilation (1, 2)     │──→ (B, C_per, H, W)  ──→ cat ──→ BN ──→ ReLU
      weight     ├──────────────────────┤
                  │  dilation (1, 3)     │──→ (B, C_per, H, W)
                  ├──────────────────────┤
                  │  dilation (1, 4)     │──→ (B, C_per, H, W)
                  └──────────────────────┘

Output: (B, 4·C_per, H, W)   where C_per = C_out // 4
```

- **Shared weights**: a single `nn.Conv2d(C_in, C_per, k)` weight tensor is
  reused via `F.conv2d` with horizontal dilation rates 1–4.
- **Circular horizontal padding**: exploits ERP wrap-around.
- **Replicate vertical padding**: handles poles.
- **Output channels = 4 × input channels** when `C_out = 4·C_in`.

### SWHDC Block (`src/models/blocks/swhdc.py`)

> SWHDC paper §III-B, Eq. 3–5, Fig. 4

```
Input: (B, C, H, W)

                  ┌─────────────────────────────────────────┐
                  │  dilation (1, 1)   → F₁ × W₁(φ)        │
                  ├─────────────────────────────────────────┤
x ──→ shared ────│  dilation (1, 2)   → F₂ × W₂(φ)        │──→ Σ ──→ BN ──→ ReLU
      weight     ├─────────────────────────────────────────┤
                  │  dilation (1, 3)   → F₃ × W₃(φ)        │
                  ├─────────────────────────────────────────┤
                  │  dilation (1, 4)   → F₄ × W₄(φ)        │
                  └─────────────────────────────────────────┘
                         ↑
               W_n(φ) pre-computed, non-trainable

Output: (B, C, H, W)   (channels unchanged, 0 extra parameters)
```

**Latitude weight computation** (Eq. 3–4):
```
φ(y) = π(y + 0.5) / H
R_φ  = min(4, 1 / sin(φ(y)))

If R_φ = integer n₀:   W_n = 1 if n=n₀, else 0
Else (between n₁, n₂): W_n₁ = n₂ - R_φ,  W_n₂ = R_φ - n₁
```

Near the **equator** (sin(φ)≈1 → R_φ≈1): W₁=1, dilation 1 is used.
Near the **poles** (sin(φ)≈0 → R_φ=4): W₄=1, dilation 4 is used.

---

## HSDCNet (ResNet-34 + HSDC)

> HSDC paper §II-C, Table 1

```
Input: (B, 12, 256, 512)

┌─────────────────────────────────────────────────────────┐
│ Stem                                                    │
│   HSDCBlock(12→64, 7×7, stride=2)  → (B, 64, 128, 256) │
│   MaxPool(3×3, stride=2)           → (B, 64,  64, 128) │
└─────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 1 (Conv 2): 3 × HSDCBasicBlock(64→64,   stride=1)               │
│                         → (B, 64,  64, 128)                           │
│ Layer 2 (Conv 3): 4 × HSDCBasicBlock(64→128,  stride=2 for first)     │
│                         → (B, 128, 32,  64)                           │
│ Layer 3 (Conv 4): 6 × HSDCBasicBlock(128→256, stride=2 for first)     │
│                         → (B, 256, 16,  32)                           │
│ Layer 4 (Conv 5): 3 × HSDCBasicBlock(256→512, stride=2 for first)     │
│                         → (B, 512,  8,  16)                           │
└────────────────────────────────────────────────────────────────────────┘
   GlobalAvgPool → Linear(512, num_classes) → logits
```

**HSDCBasicBlock** (replaces ResNet-34 BasicBlock):
```
x ──→ HSDCBlock(in→out, stride, activate=True)   [BN+ReLU inside]
   ──→ HSDCBlock(out→out, stride=1, activate=False) [BN only]
   + shortcut (1×1 Conv if channels/stride change)
   ──→ ReLU
```

**Parameter count**: ≈ 5.3 M (matches HSDC paper Table 1).

---

## SWHDCResNet (ResNet-50 + SWHDC)

> SWHDC paper §III-B, §IV-A

```
Input: (B, 1, 256, 512)

┌──────────────────────────────────────────────────┐
│ Stem (standard)                                  │
│   Conv(1→64, 7×7, stride=2) + BN + ReLU         │
│   MaxPool(3×3, stride=2) → (B, 64, 64, 128)     │
└──────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 1: 3 × SWHDCBottleneck(64→256,   mid=64,  stride=1, H=64)       │
│ Layer 2: 4 × SWHDCBottleneck(256→512,  mid=128, stride=2, H=32)       │
│ Layer 3: 6 × SWHDCBottleneck(512→1024, mid=256, stride=2, H=16)       │
│ Layer 4: 3 × SWHDCBottleneck(1024→2048,mid=512, stride=2, H=8)        │
└────────────────────────────────────────────────────────────────────────┘
   GlobalAvgPool → Linear(2048, num_classes) → logits
```

**SWHDCBottleneck** (replaces ResNet-50 Bottleneck):
```
x ──→ Conv1x1(in→mid) + BN + ReLU              [channel reduction]
   ──→ SWHDCBlock(mid→mid)                      [spatial + latitude correction]
   ──→ Conv1x1(mid→out) + BN                   [channel expansion]
   + shortcut
   ──→ ReLU
```

SWHDC replaces only the 3×3 middle conv because it requires `C_in == C_out`.
All 1×1 convolutions remain standard.

**Parameter count**: ≈ 25.5 M (= standard ResNet-50; SWHDC adds 0 params).

---

## SwinHSDCNet (Swin-T + HSDC or SWHDC)

> HSDC paper §II-C / SWHDC paper §III-B; Swin Transformer §3

```
HSDC variant:                          SWHDC variant:
(B, 12, H, W)                          (B, 1, H, W)
      │                                       │
HSDCBlock(12→48, 7×7)                  SWHDCBlock(1→1, 3×3)
      │                                       │
(B, 48, H, W)                          (B, 1, H, W)
      │                                       │
Swin-T(in_chans=48)                    Swin-T(in_chans=1)
      │                                       │
(B, 768)                                (B, 768)
      │                                       │
ClassificationHead(768, num_classes)   ClassificationHead(768, num_classes)
```

The HSDC/SWHDC block applies distortion correction at **full ERP resolution**
before patch tokenisation, preserving per-pixel latitude information.

---

## EffNetV2HSDCNet (EfficientNetV2-S + HSDC or SWHDC)

```
HSDC variant:                          SWHDC variant:
(B, 12, H, W)                          (B, 1, H, W)
      │                                       │
HSDCBlock(12→48, 3×3)                  SWHDCBlock(1→1, 3×3)
      │                                       │
(B, 48, H, W)                          (B, 1, H, W)
      │                                       │
EfficientNetV2-S(in_chans=48)          EfficientNetV2-S(in_chans=1)
      │                                       │
(B, 1280)                               (B, 1280)
      │                                       │
ClassificationHead(1280, num_classes)  ClassificationHead(1280, num_classes)
```

---

## Classification Head

```python
# src/models/classifier.py
Input (B, C, H, W) → AdaptiveAvgPool2d(1) → flatten → (B, C)
Input (B, C)        → (no pooling needed)
                    → Linear(C, num_classes) → (B, num_classes)
```

---

## Parameter Count Summary

| Model | Config | Trainable params |
|---|---|---|
| HSDCNet | ResNet-34 + HSDC, in=12, MN10 | ≈ 5.3 M |
| SWHDCResNet | ResNet-50 + SWHDC, in=1, MN10 | ≈ 25.5 M |
| SwinHSDCNet (HSDC) | Swin-T, in=48 after HSDC | ≈ 28 M |
| SwinHSDCNet (SWHDC) | Swin-T, in=1 after SWHDC | ≈ 28 M |
| EffNetV2HSDCNet (HSDC) | EffNetV2-S, in=48 | ≈ 22 M |
| EffNetV2HSDCNet (SWHDC) | EffNetV2-S, in=1 | ≈ 22 M |

> SWHDC blocks register `swhdc_weights` as **non-trainable buffers** (via
> `register_buffer`), so they do NOT appear in trainable parameter counts.
> This is verified by `tests/test_models.py::TestSWHDCBlock::test_zero_extra_trainable_params`.

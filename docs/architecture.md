# Architecture

Two model configurations, both operating on radiance field ERP tensors
produced from 3D Gaussian Splat PLY files.

The default input has **10 channels**: 8 density shells + pseudo-depth + MIP
(maximum intensity projection). The raw 8-shell density ERP is transformed
via log1p and augmented with derived feature channels before being fed to
the model. Both models accept arbitrary `in_channels` via config.

| Model | Input | Backbone | Block | Params |
|---|---|---|---|---|
| HSDCNet | (B, 10, 256, 512) | ResNet-34 | HSDC | ≈ 5.5 M |
| SWHDCResNet | (B, 10, 256, 512) | ResNet-50 | SWHDC | ≈ 23.6 M |

---

## HSDC Block (`src/models/blocks/hsdc.py`)

> HSDC paper §II-C, Fig. 2

4 dilated convolutions with **shared weights**, dilation rates (1, 2, 3, 4).
Outputs are **concatenated** → output channels = 4 × input channels per block.

```
Input: (B, C_in, H, W)

                 ┌── dilation (1,1) ──→ (B, C_per, H, W)
                 ├── dilation (1,2) ──→ (B, C_per, H, W)
x → shared W ───┤                                         → cat → BN → ReLU
                 ├── dilation (1,3) ──→ (B, C_per, H, W)
                 └── dilation (1,4) ──→ (B, C_per, H, W)

Output: (B, 4·C_per, H, W)   where C_per = C_out // 4
```

- Circular horizontal padding (exploits ERP longitude wrap-around)
- Replicate vertical padding (handles poles)

---

## SWHDC Block (`src/models/blocks/swhdc.py`)

> SWHDC paper §III-B, Eq. 3–5, Fig. 4

4 dilated convolutions with **shared weights**. Outputs combined via a
**latitude-dependent weighted sum** → output channels = input channels, 0 extra params.

```
Input: (B, C, H, W)

                 ┌── dilation (1,1) → F₁ × W₁(φ)
                 ├── dilation (1,2) → F₂ × W₂(φ)
x → shared W ───┤                               → Σ → BN → ReLU
                 ├── dilation (1,3) → F₃ × W₃(φ)
                 └── dilation (1,4) → F₄ × W₄(φ)
                        ↑
              W_n(φ) — non-trainable buffer

Output: (B, C, H, W)
```

**Latitude weights** (Eq. 3–4):
```
φ(y) = π(y + 0.5) / H
R_φ  = min(4, 1 / sin(φ(y)))

Linear interpolation between ⌊R_φ⌋ and ⌈R_φ⌉ → Σ W_n = 1 per row
```

Equator (sin(φ)≈1 → R_φ≈1): W₁=1. Poles (sin(φ)≈0 → R_φ=4): W₄=1.

---

## HSDCNet (ResNet-34 + HSDC)

> HSDC paper §II-C, Table 1 — `src/models/backbones/resnet_hsdc.py`

```
Input: (B, 10, 256, 512)

Stem:    HSDCBlock(10→64, 7×7, stride=2) → (B, 64, 128, 256)
         MaxPool(3×3, stride=2)           → (B, 64,  64, 128)

Layer 1: 3× HSDCBasicBlock(64→64)         → (B,  64, 64, 128)
Layer 2: 4× HSDCBasicBlock(64→128)        → (B, 128, 32,  64)
Layer 3: 6× HSDCBasicBlock(128→256)       → (B, 256, 16,  32)
Layer 4: 3× HSDCBasicBlock(256→512)       → (B, 512,  8,  16)

GlobalAvgPool → Linear(512, num_classes)
```

**HSDCBasicBlock:**
```
x → HSDCBlock(in→out, activate=True)
  → HSDCBlock(out→out, activate=False)
  + shortcut (1×1 Conv if channels change)
  → ReLU
```

Parameters: ≈ 5.3 M

---

## SWHDCResNet (ResNet-50 + SWHDC)

> SWHDC paper §III-B, §IV-A — `src/models/backbones/resnet_hsdc.py`

```
Input: (B, 10, 256, 512)

Stem:    Conv(10→64, 7×7, stride=2) + BN + ReLU
         MaxPool(3×3, stride=2)     → (B, 64, 64, 128)

Layer 1: 3× SWHDCBottleneck(64→256,   mid=64,  H=64)
Layer 2: 4× SWHDCBottleneck(256→512,  mid=128, H=32)
Layer 3: 6× SWHDCBottleneck(512→1024, mid=256, H=16)
Layer 4: 3× SWHDCBottleneck(1024→2048,mid=512, H=8)

GlobalAvgPool → Linear(2048, num_classes)
```

**SWHDCBottleneck:**
```
x → Conv1×1(in→mid) + BN + ReLU      [channel reduction]
  → SWHDCBlock(mid→mid)               [latitude-corrected spatial conv]
  → Conv1×1(mid→out) + BN             [channel expansion]
  + shortcut → ReLU
```

Parameters: ≈ 25.5 M (SWHDC adds 0 params — weights are buffers)

---

## Classification Head (`src/models/classifier.py`)

```
(B, C, H, W) → AdaptiveAvgPool2d(1) → flatten → Linear(C, num_classes)
(B, C)       →                                 → Linear(C, num_classes)
```

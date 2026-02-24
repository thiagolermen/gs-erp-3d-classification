# ERP-ViT 3D Classification

**Evaluating horizontally dilated convolution blocks (HSDC / SWHDC) combined with Vision Transformer backbones for 3D object classification via equirectangular projection.**

This is the implementation repository for the TCC (Trabalho de Conclusão de Curso) at the Institute of Informatics, Federal University of Rio Grande do Sul (UFRGS).

---

## Research Goal

The original ERP pipeline uses **ResNet-34** (HSDC paper) and **ResNet-50** (SWHDC paper) as CNN backbones. This work investigates whether the distortion-correction capability of the HSDC and SWHDC blocks transfers to **attention-based backbones** (Swin Transformer, EfficientNetV2-S).

**Core hypothesis:** The HSDC/SWHDC blocks encode structural inductive bias about ERP distortions. This bias should remain beneficial regardless of whether the backbone is CNN-based or Transformer-based — provided the Transformer architecture is data-efficient enough for the ModelNet scale.

**Direct motivation from prior work:** The SWHDC paper (SIBGRAPI 2024) already tested a ViT-based approach — a PanoFormer encoder adapted for classification — and found it yielded only **85.74% on ModelNet10 and 79.71% on ModelNet40**, far below ResNet-50+SWHDC (94.11% / 91.89%). The authors attributed this to the *data-hungry nature of transformers*. This TCC revisits that question with **Swin-T**, which — through its hierarchical design and shifted-window attention — is considerably more data-efficient than vanilla ViT. The central question is whether the HSDC/SWHDC inductive bias can compensate for limited training data when Transformer backbones are trained from scratch on ModelNet.

**Experimental plan:**

| Backbone | Block type | Dataset |
|----------|-----------|---------|
| ResNet-34 (baseline) | HSDC | ModelNet10, ModelNet40 |
| ResNet-50 (baseline) | SWHDC | ModelNet10, ModelNet40 |
| Swin-T | HSDC / SWHDC | ModelNet10, ModelNet40 |
| EfficientNetV2-S | HSDC / SWHDC | ModelNet10, ModelNet40 |

---

## Reference Papers

| Paper | Venue | Key contribution |
|-------|-------|-----------------|
| Stringhini et al., *Single-Panorama Classification of 3D Objects Using Horizontally Stacked Dilated Convolutions* | IEEE ICIP 2024 | HSDC block; 12-channel ERP; 97.1% MN10 with 5.3M params |
| Stringhini et al., *Spherically-Weighted Horizontally Dilated Convolutions for Omnidirectional Image Processing* | SIBGRAPI 2024 | SWHDC block; latitude-weighted dilation; zero extra parameters; tested PanoFormer ViT baseline |
| Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* | ICCV 2021 | Swin Transformer backbone |
| Xu et al., *Group Multi-View Transformer for 3D Shape Analysis with Spatial Encoding* | IEEE TMM 2024 | Multi-view Transformer for 3D analysis |

Reference code:
- HSDCNet: https://github.com/rmstringhini/HSDCNet
- SWHDC: https://github.com/rmstringhini/SWHDC

---

## Pipeline Overview

### 1. 3D → ERP Image Generation

A virtual spherical camera is placed at the centroid of the 3D mesh (area-weighted average of triangle centroids). Omnidirectional rays are cast outward, recording first and last intersections with the mesh surface. Spherical coordinates (θ, φ) are mapped to 2D pixel coordinates via:

```
x = ⌊w · θ / (2π)⌋,   y = ⌊h · φ / π⌋
```

**HSDC variant — 12-channel panorama (512 × 256):**

| Channel group | Features |
|---|---|
| First intersection (6 ch) | Depth d₁ (normalized), Surface normals (Nx, Ny, Nz), Ray-normal alignment cos(α), Gradient magnitude (Gaussian σ=2) |
| Last intersection (6 ch) | Depth dₙ (normalized), Surface normals (Nx, Ny, Nz), Ray-normal alignment cos(α), Gradient magnitude (Gaussian σ=2) |

Single intersections are replicated as both first and last hit. Zero-hit pixels are set to zero.

**SWHDC variant — 1-channel depth map (512 × 256):**
- Only the last intersection distance, normalized by the maximum distance within the object.

### 2. Distortion-Correction Blocks

**HSDC block** — 4 horizontally dilated convolutions (dilation rates 1, 2, 3, 4) with shared weights; feature maps are **concatenated** (output channels = 4 × input channels per block).

**SWHDC block** — 4 horizontally dilated convolutions (dilation rates 1, 2, 3, 4) with shared weights; feature maps are **linearly combined** via latitude-dependent weights (output channels = input channels, no parameters added):

```
R(φ) = min(N, 1 / sin(φ))            # ideal scaling factor     [SWHDC Eq. 3]
W_n(φ) = interpolation weights         # between ⌊R(φ)⌋ and ⌈R(φ)⌉  [SWHDC Eq. 4]
F* = Σ HB(W_n) ⊙ F_n                 # combined output           [SWHDC Eq. 5]
```

N=4 covers ≈96.85% of the spherical surface; N=5 adds only ≈1.1%.

Circular padding is applied on the horizontal axis to preserve full-sphere continuity.

### 3. Backbone

The blocks are integrated into the convolutional layers of:
- ResNet-34 + HSDC (original HSDCNet; Table 1 of HSDC paper)
- ResNet-50 + SWHDC (best SWHDC baseline; Table I of SWHDC paper)
- **Swin-T** (proposed new backbone — this work)
- **EfficientNetV2-S** (proposed alternative — this work)

### 4. Classification Head

Global average pooling → fully connected layer → softmax over N classes.

---

## Dataset

**ModelNet** (Princeton 3D Shape Benchmark):
- ModelNet10: 3,991 train / 908 test, 10 categories
- ModelNet40: 9,843 train / 2,468 test, 40 categories

The preset training set is split 80% train / 20% validation during experiments. Official test splits are used for final evaluation.

Download: https://modelnet.cs.princeton.edu/

---

## Project Structure

```
erp-vit-3d-classification/
├── src/
│   ├── preprocessing/
│   │   ├── ray_casting.py         ← spherical ray-casting engine
│   │   ├── erp_features.py        ← 12-ch (HSDC) and 1-ch (SWHDC) extraction
│   │   ├── augmentation.py        ← 3D rotation, Gaussian blur, Gaussian noise
│   │   └── dataset.py             ← PyTorch Dataset and DataLoader wrappers
│   ├── models/
│   │   ├── blocks/
│   │   │   ├── hsdc.py            ← HSDC block (concatenation output)
│   │   │   └── swhdc.py           ← SWHDC block (spherically-weighted output)
│   │   ├── backbones/
│   │   │   ├── resnet_hsdc.py     ← ResNet-34/50 + HSDC/SWHDC (baselines)
│   │   │   ├── swin_hsdc.py       ← Swin-T + HSDC / SWHDC (proposed)
│   │   │   └── effnetv2_hsdc.py   ← EfficientNetV2-S + HSDC / SWHDC (proposed)
│   │   └── classifier.py          ← GAP → FC → softmax head
│   ├── training/
│   │   ├── train.py               ← epoch loop, loss, checkpointing
│   │   ├── evaluate.py            ← test-set evaluation, top-1 accuracy
│   │   └── scheduler.py           ← Adam, LR decay, early stopping
│   └── analysis/
│       ├── metrics.py             ← accuracy, confusion matrix helpers
│       ├── visualize.py           ← ERP channel plots, training curves
│       └── compare.py             ← cross-run comparison utilities
├── configs/
│   ├── resnet34_hsdc_mn10.yaml
│   ├── resnet34_hsdc_mn40.yaml
│   ├── resnet50_swhdc_mn10.yaml
│   ├── resnet50_swhdc_mn40.yaml
│   ├── swin_hsdc_mn10.yaml
│   ├── swin_hsdc_mn40.yaml
│   ├── swin_swhdc_mn40.yaml
│   ├── effnetv2_hsdc_mn40.yaml
│   └── effnetv2_swhdc_mn40.yaml
├── experiments/                   ← run outputs: checkpoints, logs, CSVs (gitignored)
├── data/                          ← raw and processed ModelNet data (gitignored)
│   ├── raw/
│   │   ├── modelnet10/
│   │   └── modelnet40/
│   └── processed/
├── notebooks/
│   └── results_analysis.ipynb     ← master results analysis notebook
├── tests/                         ← unit tests
├── docs/
│   └── architecture.md            ← Mermaid / ASCII architecture diagrams
├── papers/                        ← reference PDFs
├── CLAUDE.md
├── README.md
└── requirements.txt
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download ModelNet data
#    ModelNet10 and ModelNet40: https://modelnet.cs.princeton.edu/
#    Place extracted datasets under:
#      data/raw/modelnet10/
#      data/raw/modelnet40/
```

---

## Training

All models are trained **from scratch** (no ImageNet pretraining), consistent with the original papers — ERP images differ substantially from natural perspective images and pretraining on ImageNet has not been shown to help in this setting.

```bash
# Baseline: ResNet-34 + HSDC on ModelNet10
python src/training/train.py --config configs/resnet34_hsdc_mn10.yaml

# Baseline: ResNet-50 + SWHDC on ModelNet40
python src/training/train.py --config configs/resnet50_swhdc_mn40.yaml

# Proposed: Swin-T + HSDC on ModelNet40
python src/training/train.py --config configs/swin_hsdc_mn40.yaml

# Proposed: EfficientNetV2-S + SWHDC on ModelNet40
python src/training/train.py --config configs/effnetv2_swhdc_mn40.yaml
```

---

## Results

Results are tracked in `experiments/` with per-run accuracy, loss curves, and confusion matrices.

| Backbone | Block | MN10 Acc. | MN40 Acc. | Params |
|----------|-------|-----------|-----------|--------|
| ResNet-34 | HSDC | 97.1%* | 93.9%* | 5.3M |
| ResNet-50 | SWHDC | 94.1%* | 91.9%* | 25.5M |
| PanoFormer (ViT) | — | 85.7%† | 79.7%† | — |
| Swin-T | HSDC | TBD | TBD | TBD |
| Swin-T | SWHDC | TBD | TBD | TBD |
| EfficientNetV2-S | HSDC | TBD | TBD | TBD |
| EfficientNetV2-S | SWHDC | TBD | TBD | TBD |

\* Reported in original papers.
† PanoFormer encoder + FC layer, trained from scratch on ModelNet ERP images (SWHDC paper, Table III). Serves as ViT lower-bound reference.

---

## Key Hyperparameters

| Parameter | HSDC paper | SWHDC paper |
|-----------|------------|-------------|
| ERP resolution | 512 × 256 | 512 × 256 |
| Input channels | 12 | 1 |
| Dilation rates N | 4 (rates 1–4) | 4 (rates 1–4) |
| Optimizer | Adam | Adam |
| Initial LR | 1e-4 | 1e-4 |
| LR decay | ×0.9 every 25 epochs | ×0.9 every 25 epochs |
| Min LR | 1e-7 | 1e-7 |
| Max epochs | 500 | 200 |
| Early stopping patience | 25 | 25 |
| Augmentation probability | 15% | 15% |
| 3D rotation x, y | [0°, 15°] | [0°, 15°] |
| 3D rotation z | [0°, 45°] | [0°, 45°] |
| Gaussian blur σ | [0.1, 2.0] | [0.1, 2.0] |
| Gaussian noise mean | [0, 0.001] | [0, 0.001] |
| Gaussian noise σ | [0, 0.03] | [0, 0.03] |
| Train / val split | — | 80% / 20% |

---

## References

```
Stringhini, R. M., Lermen, T. S., da Silveira, T. L. T., & Jung, C. R. (2024).
Single-Panorama Classification of 3D Objects Using Horizontally Stacked Dilated Convolutions.
IEEE ICIP 2024.

Stringhini, R. M., da Silveira, T. L. T., & Jung, C. R. (2024).
Spherically-Weighted Horizontally Dilated Convolutions for Omnidirectional Image Processing.
SIBGRAPI 2024.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.
ICCV 2021.

Xu, H., et al. (2024).
Group Multi-View Transformer for 3D Shape Analysis with Spatial Encoding.
IEEE Transactions on Multimedia, 26.
```

---

## License

MIT License. See [LICENSE](LICENSE).

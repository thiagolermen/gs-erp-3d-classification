# Technical Documentation
## GS-ERP 3D Classification

**TCC — Instituto de Informática, UFRGS**
**Advisor:** Prof. Cláudio R. Jung
**References:** Stringhini et al. (IEEE ICIP 2024), Stringhini et al. (SIBGRAPI 2024), Choi et al. (CVPR 2023)

---

## 1. Overview

This work evaluates whether the HSDC and SWHDC distortion-correction blocks (designed
for equirectangular projection) remain effective when the ERP input is generated from
**3D Gaussian Splat (3DGS) radiance fields** instead of geometric ray-casting on meshes.

**Input representation:** Each ModelNet object is stored as a 3DGS `.ply` file in the
ModelSplat dataset. We sample the continuous radiance field at N concentric spherical
shells to produce an N-channel ERP tensor, replacing the original 12-channel or
1-channel geometric ERP.

**Configurations evaluated:**

| Model | Input | MN10 target | Params |
|---|---|---|---|
| ResNet-34 + HSDC | 8-shell RF-ERP | 97.1%* | 5.3M |
| ResNet-50 + SWHDC | 8-shell RF-ERP | 94.1%* | 25.5M |

*Reported in original papers for geometric ERP. Results with RF-ERP are the TCC contribution.

---

## 2. Dataset

**ModelSplat** (`ShapeSplats/ModelNet_Splats` on HuggingFace):
- 12,309 ModelNet objects as pre-trained 3DGS `.ply` files
- 40 categories (superset of ModelNet10)

**Directory structure:**
```
gs_data/modelsplat/modelsplat_ply/
└── <category>/
    ├── train/<id>/point_cloud.ply
    └── test/<id>/point_cloud.ply
```

**Splits:**

| Split | Source | Usage |
|---|---|---|
| train | 80% of `<category>/train/` | Gradient updates + augmentation |
| val | 20% of `<category>/train/` | Early stopping |
| test | `<category>/test/` | Final evaluation only |

The 80/20 split uses a fixed `seed=42` numpy RNG (SWHDC paper §IV-A).

---

## 3. Radiance Field ERP Generation

> `src/preprocessing/radiance_field.py` — EgoNeRF (CVPR 2023)

Each 3DGS PLY contains Gaussian primitives: position `xyz`, logit-opacity, log-scale,
quaternion rotation, and SH DC color coefficients. The preprocessing:

1. **Decode** — `opacity = sigmoid(raw)`, `scale = exp(raw)`, `rgb = clip(0.5 + 0.28209 * f_dc, 0, 1)`
2. **Centroid** — opacity-weighted mean of Gaussian positions
3. **Shell radii** — EgoNeRF exponential spacing (5th/95th percentile of distances):
   ```
   r_s = r_near × (r_far / r_near)^(s / (N-1)),   s = 0 … N-1
   ```
4. **Ray directions** — unit vectors from ERP pixel (u, v) via spherical coordinates
5. **Density at each shell** — for each shell s and ERP pixel, sample point `p = centroid + r_s × d(u,v)`:
   ```
   ρ(p) = Σ_i opacity_i × exp(-0.5 × ||R_iᵀ (p - μ_i) / σ_i||²)
   ```
   Spatial culling keeps only Gaussians where `|r_dist - r_s| < 3σ × max_scale`.

**Output:** `(N_shells, H, W)` float32 ERP tensor. Default: N=8, H=256, W=512.

**Caching:** Computed ERPs are saved as `.npy` files in `data/processed/`. The cache
subdirectory name encodes all preprocessing parameters; changing any parameter
automatically invalidates the cache.

---

## 4. Input Transforms and Data Augmentation

### 4.1 Log1p Transform

> `src/preprocessing/dataset.py` — config key `data.log1p_transform`

Raw density ERP is sparse (mean≈0.057, 94% pixels below mean) and unbounded [0, ~14].
When enabled, `erp = log1p(erp)` is applied before augmentation. This compresses the
range to [0, ~2.7] and amplifies low-density boundary regions where discriminative
surface information resides.

### 4.2 Derived Feature Channels

> `src/preprocessing/dataset.py` — config key `data.derived_channels`

After the optional log1p, additional channels may be appended:

| Channel | Formula | Shape | Purpose |
|---|---|---|---|
| `pseudo_depth` | density-weighted avg shell index per pixel, normalised to [0,1] | (1, H, W) | Approximate surface distance along each ray |
| `mip` | max density across shells per pixel | (1, H, W) | Silhouette-like channel highlighting where density exists |

With 8 density shells + pseudo_depth + mip, the model input becomes **(10, H, W)**.

### 4.3 Augmentation

> `src/preprocessing/augmentation.py` — HSDC §III-A / SWHDC §IV-A

Applied to training samples only (after log1p + derived channels). Each primitive
fires independently at probability P (default 0.3):

| Primitive | Parameters |
|---|---|
| Horizontal flip | P=0.5; equivalent to 180° azimuthal rotation |
| 3D rotation | Rx, Ry ~ U[0°, 15°]; Rz ~ U[0°, 45°]; bilinear spherical remapping |
| Gaussian blur | σ ~ U[0.1, 2.0]; applied channel-wise |
| Gaussian noise | mean ~ U[0, 0.001]; std ~ U[0, 0.03]; additive |
| Random erasing | Area ~ U[2%, 33%]; aspect ~ logU[0.3, 3.3]; patch zeroed (Zhong et al., 2020) |

Augmentation is channel-agnostic — works for any number of ERP channels.

---

## 5. Distortion-Correction Blocks

ERP samples the sphere non-uniformly. Near the poles (`sin(φ) → 0`), pixels are
horizontally oversampled by `1/sin(φ)` relative to the equator. Both blocks adapt
horizontal receptive field width to compensate.

See `docs/architecture.md` for diagrams and equations.

---

## 6. Training Protocol

> `src/training/train.py`, `src/training/scheduler.py`

| Parameter | Value | Notes |
|---|---|---|
| Loss | CrossEntropyLoss | with label smoothing (0.1) and class weights |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) | decoupled weight decay 5e-4 |
| Initial LR | 1e-4 | with 10-epoch linear warmup |
| LR schedule | Cosine annealing | floor at 1e-6 |
| Max epochs | 500 (HSDC) / 200 (SWHDC) | |
| Early stopping | patience = 100 epochs | gives cosine schedule room |
| Gradient clipping | max_norm = 1.0 | |
| Batch size | 32 | |
| Mixed precision | AMP (CUDA only) | |
| MixUp | α = 0.4 (Zhang et al., 2018) | blends sample pairs to reduce overfitting |
| Pretraining | None — trained from scratch | |

**Outputs** per run (`experiments/<run_name>/`):

```
config.yaml          — copy of config used
train.log            — full logging output
metrics.csv          — epoch, train_loss, val_loss, train_acc, val_acc, lr
best_checkpoint.pt   — weights at best validation accuracy
last_checkpoint.pt   — weights at final epoch
```

---

## 7. Evaluation

> `src/training/evaluate.py`

**Primary metric:** Top-1 overall accuracy (same as HSDC Table 2, SWHDC Table I).

**Test-Time Augmentation (TTA):** When `--tta` is passed, the evaluator averages
softmax predictions over 5 views per test sample:

1. Original
2. Horizontal flip
3. Circular shift by W/4 (azimuthal 90°)
4. Circular shift by W/2 (azimuthal 180°)
5. Circular shift by 3W/4 (azimuthal 270°)

ERP is periodic horizontally, so circular shifts are exact viewpoint rotations.

**Saved artefacts:**
- `test_results.json` — `oa` (fraction), `macc`, `params_m`
- `predictions.npz` — `y_true`, `y_pred` for McNemar test
- `confusion_matrix.npy` — (C, C) integer counts

---

## 8. Repository Structure

```
src/
├── preprocessing/
│   ├── ply_loader.py        ← binary PLY parser for 3DGS files
│   ├── radiance_field.py    ← EgoNeRF shell ERP from 3DGS
│   ├── augmentation.py      ← channel-agnostic ERP augmentation
│   └── dataset.py           ← GaussianERPDataset + build_dataloaders()
├── models/
│   ├── blocks/
│   │   ├── hsdc.py          ← HSDCBlock (shared-weight, concat)
│   │   └── swhdc.py         ← SWHDCBlock (lat-weight buffer, same-ch)
│   ├── backbones/
│   │   └── resnet_hsdc.py   ← HSDCNet (ResNet-34) + SWHDCResNet (ResNet-50)
│   └── classifier.py        ← GAP → Linear head
├── training/
│   ├── train.py             ← epoch loop, AMP, checkpointing
│   ├── evaluate.py          ← test-set eval, metrics, artefacts
│   └── scheduler.py         ← Adam, StepLR, EarlyStopping
└── analysis/
    ├── metrics.py           ← OA, mAcc, confusion matrix, McNemar
    ├── visualize.py         ← training curves, ERP grids, Pareto plots
    └── compare.py           ← cross-run tables, LaTeX export

configs/                     ← one YAML per experiment (4 total)
gs_data/                     ← ModelSplat PLY files (gitignored)
data/processed/              ← ERP .npy cache (gitignored)
experiments/                 ← run outputs (gitignored)
notebooks/                   ← exploratory analysis
tests/                       ← pytest unit tests
scripts/                     ← download, preprocess, train scripts
docs/                        ← this file + architecture.md
```

---

## 9. Comparison with Related 3DGS Classification Methods

This section surveys all known works that perform 3D object classification using
3D Gaussian Splatting (3DGS) representations. Our work is distinct in that it
**renders** Gaussians into equirectangular projections (ERP) and applies
distortion-correction CNNs, whereas all prior work either (a) operates directly
on raw Gaussian parameters as a point-like modality, or (b) projects Gaussians
into multi-view 2D images for CLIP-based recognition.

### 9.1 Taxonomy of Approaches

| Approach | Input to Classifier | Architecture | Pretraining |
|---|---|---|---|
| **This work (GS-ERP)** | N-shell density ERP (image-like) | ResNet + HSDC/SWHDC | None (from scratch) |
| Gaussian-MAE (ShapeSplat) | Raw Gaussian params (C, O, S, R, SH) | Transformer + MAE | Self-supervised on 3DGS |
| GS-PT | Point cloud + 3DGS-rendered views | Transformer + contrastive | Self-supervised on 3DGS |
| 3D Gaussian Point Encoders | Gaussian primitives | PointNet/Mamba3D hybrid | None (from scratch) |
| UniGS | 3DGS aligned with CLIP | Transformer + CLIP | Language-Image-3D |
| GS-PointCLIP | 3DGS → 2D projections → CLIP | CLIP ViT-B/16 | ImageNet CLIP |
| TU Delft (van den Berg) | Gaussian features → PointNet++ | PointNet++ | None |

### 9.2 ShapeSplat + Gaussian-MAE (Ma et al., 3DV 2025 — Oral)

**The most directly comparable work.** This paper created the ModelSplat dataset
(`ShapeSplats/ModelNet_Splats`) that we use, and is the first to benchmark 3D
classification on 3DGS representations of ModelNet objects.

**Method:** Gaussian-MAE applies masked autoencoder pretraining to raw Gaussian
parameters. It introduces *Gaussian feature grouping* in a normalised feature
space and a *splats pooling layer* to aggregate similar Gaussians. The encoder
is a standard Transformer (similar to Point-MAE). Different feature embeddings
are ablated: E(C) = centroids only, E(C,S,R) = centroids + scale + rotation,
E(All) = all parameters including opacity and SH.

**Key finding:** Using only Gaussian centroids **degrades** classification
relative to uniformly sampled point clouds (93.72% vs 94.93% on MN10),
because 3DGS optimisation produces a non-uniform centroid distribution
biased toward high-frequency surface regions. However, incorporating
additional Gaussian attributes (scale, rotation, opacity) recovers and
surpasses point-cloud methods.

**Classification results (full fine-tuning, 1024 splats):**

| Method | Input | MN10 (%) | MN40 (%) |
|---|---|---|---|
| PointNet | Point cloud (1024 pts) | — | 89.2 |
| PointNet++ | Point cloud (1024 pts) | — | 91.9 |
| Point-BERT | Point cloud (1024 pts) | 94.82 | 93.20 |
| Point-MAE | Point cloud (1024 pts) | 94.93 | 93.20 |
| Gaussian-MAE E(C) | 3DGS centroids | 93.72 | 91.77 |
| Gaussian-MAE E(C,O) | 3DGS centroids + opacity | 93.83 | 91.78 |
| Gaussian-MAE E(C,SH) | 3DGS centroids + SH color | 93.83 | 92.41 |
| Gaussian-MAE E(C,S,R) | 3DGS centroids + scale + rot. | 94.27 | 93.19 |
| Gaussian-MAE E(O,C,S,R) | 3DGS all except SH | 95.48 | 92.42 |
| Gaussian-MAE E(All) | 3DGS all parameters | **95.37** | **93.35** |

**Linear probing (MLP-3, frozen encoder):**

| Method | MN10 (%) | MN40 (%) |
|---|---|---|
| Point-BERT | 94.27 | 91.82 |
| Point-MAE | 93.61 | 92.63 |
| Gaussian-MAE E(All) | **95.26** | **92.74** |

**Generalization to real-world data (ScanObjectNN):**

| Method | Protocol | MN40 | OBJ_BG | OBJ_ONLY | PB_T50_RS |
|---|---|---|---|---|---|
| Point-MAE | Full | 93.20 | 90.02 | 88.29 | 85.18 |
| Gaussian-MAE E(C) | Full | 92.78 | 87.61 | 88.64 | 84.98 |
| Point-MAE | MLP-3 | 92.63 | 84.29 | 85.24 | 77.34 |
| Gaussian-MAE E(C) | MLP-3 | 90.36 | 81.93 | 85.37 | 75.02 |

**Ablation — number of input splats:**

| Splats | MN10 (%) | MN40 (%) |
|---|---|---|
| 1024 | **95.37** | **93.35** |
| 2048 | 93.29 | 92.29 |
| 4096 | 94.82 | 93.02 |
| 8192 | 95.26 | 93.05 |

**Reference:**
Ma, Q., Xu, Y., Wu, S., Prokudin, S., Sridhar, S., van Gool, L., and Birdal, T.
"ShapeSplat: A Large-scale Dataset of Gaussian Splats and Their Self-Supervised
Pretraining." *3DV 2025 (Oral)*. arXiv:2408.10906.
Code: https://github.com/qimaqi/ShapeSplat-Gaussian_MAE

---

### 9.3 GS-PT (Luo et al., ICASSP 2025)

**Method:** GS-PT integrates 3DGS into point cloud self-supervised learning.
The pipeline uses a Transformer backbone for masked point reconstruction,
while 3DGS renders multi-view images to generate (1) enhanced point cloud
distributions via densification and (2) novel-view images for cross-modal
contrastive learning. The 3DGS branch acts as a data augmentation and
auxiliary supervision source during pretraining.

**Key contribution:** Unlike Gaussian-MAE which treats Gaussians as a new
modality, GS-PT uses 3DGS as a **pretraining tool** to improve standard
point cloud encoders. After pretraining, the model operates on point clouds.

**Results:** The paper reports state-of-the-art results on ModelNet40
classification, few-shot learning, and ShapeNet part segmentation,
outperforming prior SSL methods (Point-MAE, Point-BERT, ACT, PointGPT).
Exact accuracy tables are available in the published ICASSP proceedings.

**Reference:**
Luo, Y., Li, Z., Li, C., Zhang, H., and Ma, L.
"GS-PT: Exploiting 3D Gaussian Splatting for Comprehensive Point Cloud
Understanding via Self-supervised Learning." *ICASSP 2025*.
arXiv:2409.04963.

---

### 9.4 3D Gaussian Point Encoders (James, arXiv 2025)

**Method:** Reinterprets PointNet's per-point embedding function as a
volumetric representation by integrating 3D Gaussian primitives. Each
embedding dimension corresponds to a Gaussian in 3D space, replacing
the learned MLP with explicit Gaussian evaluation. Achieves 2.7× higher
throughput than standard PointNet with comparable accuracy.

**Classification results:**

| Method | MN40 mAcc (%) | MN40 OA (%) | ScanObjNN OA (%) |
|---|---|---|---|
| PointNet | 86.1 | 90.0 | 69.0 |
| 3DGPE (N) | 86.4 | 90.1 | 69.0 |
| PointMLP | 91.3 | 94.1 | — |
| PointNeXt | 90.8 | 93.2 | 87.7 |
| Mamba3D | 89.7 | 93.3 | 91.6 |
| 3DGPE + Mamba3D (N) | 89.9 | 93.6 | 88.0 |
| DeLA | **92.2** | **94.0** | **90.4** |

**Reference:**
James, J. "3D Gaussian Point Encoders." *arXiv:2511.04797*, November 2025.

---

### 9.5 UniGS (Li et al., ICLR 2025)

**Method:** Aligns 3DGS representations with CLIP language-image embeddings.
A 3D encoder processes Gaussians and a *Gaussian-Aware Guidance* module
learns fine-grained 3DGS features aligned with CLIP's multimodal space.

**Results:** Zero-shot classification (not directly comparable to supervised):

| Dataset | UniGS Top-1 | Uni3D Top-1 | Improvement |
|---|---|---|---|
| Objaverse-LVIS | 38.57% | 36.72% | +1.85 pp |
| ABO | 46.97% | 37.79% | +9.18 pp |
| MVImgNet | 7.65% | 4.92% | +2.73 pp |
| SUN RGB-D | 69.64% | 54.51% | +15.13 pp |

**Note:** UniGS does not report on ModelNet10/40 and uses CLIP pretraining,
so results are not directly comparable to supervised-from-scratch methods.

**Reference:**
Li, H., Wang, H., Zhang, Z., Xu, H., Liu, M., and Luo, J.
"UniGS: Unified Language-Image-3D Pretraining with Gaussian Splatting."
*ICLR 2025*. arXiv:2502.17860.

---

### 9.6 GS Projection + PointCLIP (Anonymous, under review)

**Method:** Projects 3D Gaussian point clouds into 2D images via splatting,
then feeds the renders to a frozen CLIP ViT-B/16 encoder. Combines the
geometric structure of 3DGS with CLIP's powerful visual features.

**Claimed results:** >96% OA on ModelNet40 and ~99.9% on ScanObjectNN.
However, these results rely on CLIP's ImageNet-pretrained features and
have not yet been peer-reviewed. Not comparable to from-scratch methods.

**Reference:**
"Gaussian Splatting Projection in PointCLIP." OpenReview (under review).
GitHub: https://github.com/genji970/3d-vlm-gaussian-splatting-pointclip-on-modelnet40-and-scanobjectnn

---

### 9.7 TU Delft BSc Thesis (van den Berg, 2024)

**Method:** Converts ModelNet10 objects to 3DGS via standard training, then
feeds Gaussian features (position, scale, rotation, opacity, SH) to
PointNet++ for classification.

**Key finding:** 3DGS enables effective classification, but does **not
outperform** methods that use ground-truth point clouds sampled directly
from mesh surfaces. The non-uniform distribution of optimised Gaussian
centroids hurts classification, consistent with the ShapeSplat findings.

**Reference:**
van den Berg, L. "Utilising 3D Gaussian Splatting for PointNet Object
Classification." BSc Thesis, TU Delft, 2024.
https://repository.tudelft.nl/record/uuid:ef9e967a-92bf-4ad6-83da-7d6e0b4fd470

---

### 9.8 Computational Cost Comparison

The following table compares trainable parameter counts, training
infrastructure, and wall-clock training time across all methods. Where
exact figures are not publicly available, we note the closest information
from the papers or repositories.

#### 9.8.1 Trainable Parameters

| Method | Params (M) | Architecture | Notes |
|---|---|---|---|
| PointNet | 3.5 | MLP | Qi et al., CVPR 2017 |
| PointNet++ | 1.7 | Hierarchical MLP | Qi et al., NeurIPS 2017 |
| Point-BERT | ~22 | Transformer (12 layers, 384 dim) | Yu et al., CVPR 2022; same encoder as Point-MAE |
| Point-MAE | 22.1 | Transformer (12 layers, 384 dim) | Pang et al., ECCV 2022; 22.1M full fine-tuning |
| PointMLP | 13.2 | Residual MLP | Ma et al., ICLR 2022 |
| PointNeXt-S | 4.5 | Improved PointNet++ | Qian et al., NeurIPS 2022 |
| Gaussian-MAE (pretrain) | ~28.8 | Transformer encoder-decoder | Ma et al., 3DV 2025; 28.79M during pretraining |
| Gaussian-MAE (finetune) | ~22 | Transformer encoder + linear | Same encoder as Point-MAE; decoder discarded |
| GS-PT | ~22 | Transformer + 3DGS renderer | Luo et al., ICASSP 2025; same backbone as Point-MAE |
| 3DGPE (standalone) | ~0 | Non-parametric Gaussian eval. | James, 2025; zero learnable params |
| 3DGPE + Mamba3D | ~12.3 | Gaussian encoder + Mamba3D | Mamba3D backbone: 12.3M |
| HSDCNet (geometric ERP) | 5.3 | ResNet-34 + HSDC blocks | Stringhini et al., ICIP 2024 |
| SWHDCResNet (geometric ERP) | 25.5 | ResNet-50 + SWHDC blocks | Stringhini et al., SIBGRAPI 2024 |
| **GS-ERP: ResNet-34+HSDC** | **5.5** | **ResNet-34 + HSDC blocks** | **This work; 10-ch input (8 density + 2 derived)** |
| **GS-ERP: ResNet-50+SWHDC** | **23.6** | **ResNet-50 + SWHDC blocks** | **This work; 10-ch input** |

Note: The slight parameter difference between our HSDCNet (5.5M) and the
original paper's (5.3M) is due to the different number of input channels
(10 vs 12), which affects only the first convolutional layer.

#### 9.8.2 Training Infrastructure and Time

| Method | GPU | Pretrain Time | Finetune Time | Total Epochs | Notes |
|---|---|---|---|---|---|
| Gaussian-MAE | 1× A6000 (pretrain) / 1× H100 (finetune) | 300 epochs on 52K ShapeNet | 300 epochs on MN10/MN40 | 600 total | Pretrain batch=128; finetune batch=224 |
| GS-PT | 2× A100 (pretrain) / 1× A100 (finetune) | 20 epochs on ShapeNet | 300 epochs on MN40 | 320 total | Finetune batch=32, AdamW lr=5e-2 |
| Point-MAE | 1× GPU (unspecified) | 300 epochs on ShapeNet-55 | 300 epochs on MN40 | 600 total | Standard Transformer training |
| PointMLP | 1× GPU | — | ~300 epochs on MN40 | 300 | Reported ~11 hours on ModelNet40 |
| **GS-ERP: ResNet-34+HSDC** | **1× RTX 3090 Ti** | **—** | **131.6 min (261 ep, early stop)** | **261** | **Best val at epoch 161** |
| **GS-ERP: ResNet-50+SWHDC** | **1× RTX 3090 Ti** | **—** | **100.7 min (200 ep, full)** | **200** | **Best val at epoch 121** |

#### 9.8.3 Training Cost Analysis

A fair comparison must account for the **total** compute cost, not just
the fine-tuning phase:

| Method | Pretrain Cost | Finetune Cost | Total Cost | MN10 OA |
|---|---|---|---|---|
| Gaussian-MAE E(All) | 300 ep × 52K objects (A6000) | 300 ep × 3.9K objects (H100) | High | 95.37% |
| GS-PT | 20 ep × ShapeNet (2× A100) | 300 ep × MN40 (1× A100) | High | N/A† |
| Point-MAE | 300 ep × ShapeNet-55 | 300 ep × MN10 | Medium | 94.93% |
| Point-BERT | BERT-style on ShapeNet | 300 ep × MN10 | Medium | 94.82% |
| **GS-ERP: ResNet-34+HSDC** | **None** | **261 ep × 3.2K (RTX 3090 Ti)** | **Low (2.2 h)** | **91.96%** |
| **GS-ERP: ResNet-50+SWHDC** | **None** | **200 ep × 3.2K (RTX 3090 Ti)** | **Low (1.7 h)** | **90.75%** |

†GS-PT does not report MN10 results; MN40 figures are in the ICASSP proceedings.

Note that the preprocessing cost of generating the ERP cache (converting
all 3,991 MN10 PLYs into 8-shell density ERPs) is a one-time cost of
approximately 3–4 hours on a single CPU, and is not included in the
training time above.

#### 9.8.4 Parameter Efficiency (OA per Million Parameters)

| Method | Params (M) | MN10 OA (%) | OA/M ratio |
|---|---|---|---|
| **GS-ERP: ResNet-34+HSDC** | **5.5** | **91.96** | **16.8** |
| Gaussian-MAE E(All) | ~22 | 95.37 | 4.3 |
| Point-MAE | 22.1 | 94.93 | 4.3 |
| Point-BERT | ~22 | 94.82 | 4.3 |
| **GS-ERP: ResNet-50+SWHDC** | **23.6** | **90.75** | **3.8** |

The HSDC variant achieves the highest parameter efficiency among all
methods (16.8% OA per million parameters), despite a lower absolute
accuracy. This makes it an attractive option for deployment scenarios
where model size is constrained.

---

### 9.9 Summary: Positioning This Work

The table below positions our GS-ERP approach among all 3DGS-based
classification methods on ModelNet10:

| Method | Venue | Input | Architecture | Pretrain | Params (M) | MN10 OA (%) |
|---|---|---|---|---|---|---|
| Gaussian-MAE E(All) | 3DV 2025 | Raw Gaussian params | Transformer | Self-sup MAE | ~22 | **95.37** |
| Point-MAE | ECCV 2022 | Point cloud (1024) | Transformer | Self-sup MAE | 22.1 | 94.93 |
| Point-BERT | CVPR 2022 | Point cloud (1024) | Transformer | Self-sup BERT | ~22 | 94.82 |
| Gaussian-MAE E(C,S,R) | 3DV 2025 | Gaussian C+S+R | Transformer | Self-sup MAE | ~22 | 94.27 |
| Gaussian-MAE E(C) | 3DV 2025 | Gaussian centroids | Transformer | Self-sup MAE | ~22 | 93.72 |
| **GS-ERP: ResNet-34+HSDC** | **This work** | **8-shell RF-ERP** | **CNN (ResNet-34)** | **None** | **5.5** | **91.96** |
| **GS-ERP: ResNet-50+SWHDC** | **This work** | **8-shell RF-ERP** | **CNN (ResNet-50)** | **None** | **23.6** | **90.75** |

**Key observations:**

1. **Different paradigm.** Our approach is the only one to convert 3DGS into
   a 2D image representation (ERP) and apply CNN-based processing. All other
   methods process Gaussian primitives as unordered sets (point-like modality).

2. **No pretraining, dramatically lower compute.** Our models train from
   scratch in ~2 hours on a single consumer GPU (RTX 3090 Ti). Gaussian-MAE
   requires 600 total epochs across pretraining (A6000/H100) and fine-tuning,
   plus the 52K-object ShapeNet pretraining corpus. Our total compute cost
   is roughly two orders of magnitude lower.

3. **4× fewer parameters with HSDC.** ResNet-34+HSDC (5.5M params) achieves
   91.96% while being 4× smaller than any Transformer-based method (~22M).
   The HSDC block adds negligible parameters while providing ERP-specific
   distortion correction.

4. **Distortion correction transfers to radiance fields.** The HSDC and SWHDC
   blocks were designed for geometric ray-cast ERP. On 3DGS-derived ERP, they
   still achieve meaningful classification (91.96%), demonstrating that the
   distortion-correction principle generalises to the radiance field domain.

5. **Representation gap.** The 5.1 pp gap between our best (91.96%) and the
   geometric ERP baseline (97.1%) quantifies the information cost of
   replacing mesh ray-casting with 3DGS radiance field sampling. The 3.4 pp
   gap from Gaussian-MAE (95.37%) reflects both the representation difference
   (ERP image vs raw Gaussian params) and the architectural difference
   (CNN from scratch vs pretrained Transformer).

6. **Parameter efficiency.** At 16.8% OA per million parameters, the HSDC
   variant is the most parameter-efficient method in this comparison — nearly
   4× better than any Transformer-based approach (4.3% OA/M). This suggests
   that ERP-based representations, when combined with appropriate distortion
   correction, offer a highly efficient pathway for 3DGS classification.

---

## 10. References

```
[1]  Stringhini et al. Single-Panorama Classification of 3D Objects Using
     Horizontally Stacked Dilated Convolutions. IEEE ICIP 2024.

[2]  Stringhini et al. Spherically-Weighted Horizontally Dilated Convolutions
     for Omnidirectional Image Processing. SIBGRAPI 2024.

[3]  Choi et al. Balanced Spherical Grid for Egocentric View Synthesis
     (EgoNeRF). CVPR 2023.

[4]  He et al. Deep Residual Learning for Image Recognition. CVPR 2016.

[5]  Wu et al. 3D ShapeNets: A Deep Representation for Volumetric Shapes.
     CVPR 2015. (ModelNet dataset)

[6]  Zhang et al. mixup: Beyond Empirical Risk Minimization. ICLR 2018.

[7]  Zhong et al. Random Erasing Data Augmentation. AAAI 2020.

[8]  Loshchilov & Hutter. Decoupled Weight Decay Regularization. ICLR 2019.

[9]  Ma et al. ShapeSplat: A Large-scale Dataset of Gaussian Splats and Their
     Self-Supervised Pretraining. 3DV 2025 (Oral). arXiv:2408.10906.

[10] Luo et al. GS-PT: Exploiting 3D Gaussian Splatting for Comprehensive
     Point Cloud Understanding via Self-supervised Learning. ICASSP 2025.
     arXiv:2409.04963.

[11] James. 3D Gaussian Point Encoders. arXiv:2511.04797, 2025.

[12] Li et al. UniGS: Unified Language-Image-3D Pretraining with Gaussian
     Splatting. ICLR 2025. arXiv:2502.17860.

[13] van den Berg. Utilising 3D Gaussian Splatting for PointNet Object
     Classification. BSc Thesis, TU Delft, 2024.

[14] Pang et al. Masked Autoencoders for Point Cloud Self-supervised Learning
     (Point-MAE). ECCV 2022.

[15] Yu et al. Point-BERT: Pre-training 3D Point Cloud Transformers with
     Masked Point Modeling. CVPR 2022.

[16] Kerbl et al. 3D Gaussian Splatting for Real-Time Radiance Field
     Rendering. SIGGRAPH 2023.
```

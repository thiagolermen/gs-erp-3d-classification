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

## 9. References

```
[1] Stringhini et al. Single-Panorama Classification of 3D Objects Using
    Horizontally Stacked Dilated Convolutions. IEEE ICIP 2024.

[2] Stringhini et al. Spherically-Weighted Horizontally Dilated Convolutions
    for Omnidirectional Image Processing. SIBGRAPI 2024.

[3] Choi et al. Balanced Spherical Grid for Egocentric View Synthesis
    (EgoNeRF). CVPR 2023.

[4] He et al. Deep Residual Learning for Image Recognition. CVPR 2016.

[5] Wu et al. 3D ShapeNets: A Deep Representation for Volumetric Shapes.
    CVPR 2015. (ModelNet dataset)

[6] Zhang et al. mixup: Beyond Empirical Risk Minimization. ICLR 2018.
    (MixUp augmentation)

[7] Zhong et al. Random Erasing Data Augmentation. AAAI 2020.
    (Random erasing augmentation)

[8] Loshchilov & Hutter. Decoupled Weight Decay Regularization. ICLR 2019.
    (AdamW optimizer, cosine annealing)
```

# GS-ERP 3D Classification

**3D object classification via Radiance Field Equirectangular Projection from 3D Gaussian Splats, with HSDC and SWHDC distortion-correction blocks.**

This is the implementation repository for the TCC (Trabalho de Conclusão de Curso) at the Institute of Informatics, Federal University of Rio Grande do Sul (UFRGS).

---

## Research Goal

Prior ERP-based classification work (HSDC / SWHDC papers) generates ERP images by geometric ray-casting on raw 3D mesh files. This work replaces geometric ray-casting with **volumetric density sampling through a pre-trained 3D Gaussian Splat (3DGS)**, producing a multi-channel radiance field ERP that encodes the continuous density of the object at concentric spherical shells.

**Core contribution:** Evaluate whether the HSDC and SWHDC distortion-correction blocks remain effective when applied to a new input representation — the N-shell radiance field ERP derived from 3DGS — using the same CNN backbones (ResNet-34, ResNet-50) and training protocol from the original papers.

**Dataset:** ModelSplat (ShapeSplats/ModelNet_Splats) — 12,309 ModelNet objects as pre-trained 3DGS `.ply` files covering all 40 ModelNet40 categories.

**Experimental plan:**

| Backbone | Block type | Dataset |
|----------|-----------|---------|
| ResNet-34 | HSDC | ModelNet10, ModelNet40 |
| ResNet-50 | SWHDC | ModelNet10, ModelNet40 |

---

## Reference Papers

| Paper | Venue | Key contribution |
|-------|-------|-----------------|
| Stringhini et al., *Single-Panorama Classification of 3D Objects Using Horizontally Stacked Dilated Convolutions* | IEEE ICIP 2024 | HSDC block; ERP-based classification; 97.1% MN10 |
| Stringhini et al., *Spherically-Weighted Horizontally Dilated Convolutions for Omnidirectional Image Processing* | SIBGRAPI 2024 | SWHDC block; latitude-weighted dilation; zero extra params |
| Choi et al., *EgoNeRF: Egocentric Neural Radiance Fields* | CVPR 2023 | Exponential shell spacing for omnidirectional radiance fields |

Reference code:
- HSDCNet: https://github.com/rmstringhini/HSDCNet
- SWHDC: https://github.com/rmstringhini/SWHDC

---

## Pipeline Overview

### 1. 3DGS PLY → Radiance Field ERP

Each 3D object in the ModelSplat dataset is stored as a `.ply` file containing 3D Gaussian primitives (position, opacity, scale, rotation, SH color coefficients). We place a virtual camera at the **opacity-weighted centroid** of the Gaussians and sample the radiance field at **N concentric spherical shells** (radii r₁…rₙ) using EgoNeRF exponential spacing:

```
r_s = r_near × (r_far / r_near)^(s / (N-1)),   s = 0 … N-1
```

For each shell s and each ERP pixel (u, v), the sample point is:

```
p = centroid + r_s × d(u, v)
```

where `d(u, v)` is the unit ray direction from spherical coordinates. The density at p is the sum of Gaussian contributions evaluated via the full Mahalanobis distance:

```
ρ(p) = Σ_i opacity_i × exp(-0.5 × ||R_iᵀ (p - μ_i) / s_i||²)
```

Spatial culling keeps only Gaussians within 3σ of the shell radius to make evaluation tractable.

The result is an **N-channel ERP** (default N=8, resolution 512×256), where each channel encodes radiance field density at one concentric shell.

### 2. Distortion-Correction Blocks

**HSDC block** — 4 horizontally dilated convolutions (dilation rates 1, 2, 3, 4) with shared weights; feature maps are **concatenated** (output channels = 4 × input channels per block).

**SWHDC block** — 4 horizontally dilated convolutions (dilation rates 1, 2, 3, 4) with shared weights; feature maps are **linearly combined** via latitude-dependent weights (output channels = input channels, no parameters added):

```
R(φ) = min(N, 1 / sin(φ))             # ideal scaling factor     [SWHDC Eq. 3]
W_n(φ) = interpolation weights          # between ⌊R(φ)⌋ and ⌈R(φ)⌉  [SWHDC Eq. 4]
F* = Σ HB(W_n) ⊙ F_n                  # combined output           [SWHDC Eq. 5]
```

Circular padding is applied on the horizontal axis to preserve full-sphere continuity.

### 3. Backbone + Classification Head

The blocks are integrated into the convolutional stem of:
- **ResNet-34 + HSDC** (HSDCNet architecture, HSDC paper Table 1)
- **ResNet-50 + SWHDC** (best SWHDC baseline, SWHDC paper Table I)

Global average pooling → fully connected layer → softmax over N classes.

All models are trained **from scratch** (no ImageNet pretraining).

---

## Dataset

**ModelSplat** (ShapeSplats/ModelNet_Splats on HuggingFace):
- 12,309 ModelNet objects as pre-trained 3DGS `.ply` files
- Covers all 40 ModelNet40 categories (superset of ModelNet10)
- ModelNet10: 10 categories; ModelNet40: 40 categories
- Train/test split follows the standard ModelNet convention

Download:
```bash
python scripts/download_modelsplat.py --token <HF_TOKEN> --mn10-only
python scripts/download_modelsplat.py --token <HF_TOKEN>  # full MN40 (~40 GB)
```

---

## Local Machine Setup

> **The local machine is used for data download, preprocessing, and notebook exploration.**
> Training and evaluation run on the lab machine via Docker (see [docs/DOCKER_SETUP.md](docs/DOCKER_SETUP.md)).

### 1. Create a Python virtual environment

```bash
python -m venv .venv
```

Activate:

```bash
# Windows — Command Prompt
.venv\Scripts\activate.bat

# Windows — PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
# Step 1 — PyTorch CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Step 2 — Remaining dependencies
pip install -r requirements-local.txt

# Step 3 — Install project as editable package
pip install -e . --no-deps
```

### 3. Download ModelSplat dataset

```bash
# ModelNet10 only (~15 GB)
python scripts/download_modelsplat.py --token <HF_TOKEN> --mn10-only

# Full ModelNet40 (~40 GB) — use --dest D:/gs_data/modelsplat for other drives
python scripts/download_modelsplat.py --token <HF_TOKEN>
```

Extracted PLY files land in `gs_data/modelsplat/modelsplat_ply/<category>/train|test/<id>/point_cloud.ply`.

### 4. Generate ERP cache

```bash
# Generates radiance field ERP cache (8-shell, 512×256)
bash scripts/preprocess_all.sh
```

Or for a single dataset:

```bash
python -m src.preprocessing.dataset \
    --data_root  gs_data/modelsplat/modelsplat_ply \
    --cache_dir  data/processed/modelnet10/radiance_field \
    --pipeline   radiance_field \
    --dataset    modelnet10 \
    --n_shells   8
```

### 5. Explore with notebooks

```bash
jupyter notebook notebooks/
```

Key notebooks:
- `notebooks/modelsplat_visualization.ipynb` — 3DGS property distributions, point clouds, shell assignment, ERP maps
- `notebooks/radiance_field_erp.ipynb` — radiance field ERP generation with full Mahalanobis distance evaluation
- `notebooks/results_analysis.ipynb` — training results analysis and comparison

---

## Project Structure

```
gs-erp-3d-classification/
├── src/
│   ├── preprocessing/
│   │   ├── ply_loader.py          <- binary PLY parser for 3DGS files
│   │   ├── radiance_field.py      <- EgoNeRF exponential shell ERP generation
│   │   ├── augmentation.py        <- channel-agnostic ERP augmentation
│   │   └── dataset.py             <- PyTorch Dataset and DataLoader wrappers
│   ├── models/
│   │   ├── blocks/
│   │   │   ├── hsdc.py            <- HSDC block (concatenation output)
│   │   │   └── swhdc.py           <- SWHDC block (spherically-weighted output)
│   │   ├── backbones/
│   │   │   └── resnet_hsdc.py     <- ResNet-34/50 + HSDC/SWHDC
│   │   └── classifier.py          <- GAP -> FC -> softmax head
│   ├── training/
│   │   ├── train.py               <- epoch loop, AMP, checkpointing
│   │   ├── evaluate.py            <- test-set evaluation, top-1 accuracy
│   │   └── scheduler.py           <- Adam, StepLR, early stopping
│   └── analysis/
│       ├── metrics.py             <- accuracy, confusion matrix helpers
│       ├── visualize.py           <- ERP channel plots, training curves
│       └── compare.py             <- cross-run comparison, LaTeX tables
├── configs/                       <- one YAML per experiment (4 total)
│   ├── resnet34_hsdc_mn10.yaml
│   ├── resnet34_hsdc_mn40.yaml
│   ├── resnet50_swhdc_mn10.yaml
│   └── resnet50_swhdc_mn40.yaml
├── gs_data/                       <- ModelSplat PLY files (gitignored)
│   └── modelsplat/modelsplat_ply/<category>/train|test/<id>/point_cloud.ply
├── data/processed/                <- ERP cache (gitignored)
│   ├── modelnet10/radiance_field/
│   └── modelnet40/radiance_field/
├── experiments/                   <- run outputs: checkpoints, logs, CSVs (gitignored)
├── notebooks/
│   ├── modelsplat_visualization.ipynb
│   ├── radiance_field_erp.ipynb
│   └── results_analysis.ipynb
├── scripts/
│   ├── download_modelsplat.py     <- HuggingFace download script
│   ├── preprocess_all.sh          <- generate ERP cache (run inside container)
│   └── run_baselines.sh           <- train all four experiments
├── tests/                         <- unit tests (pytest)
├── docs/
│   ├── technical_documentation.md
│   ├── architecture.md
│   └── DOCKER_SETUP.md
├── papers/                        <- reference PDFs
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt               <- full deps (Docker / lab machine)
├── requirements-local.txt         <- CPU-only deps (local + notebooks)
└── README.md
```

---

## Lab Machine Setup (Docker — training and evaluation)

> Full guide: [`docs/DOCKER_SETUP.md`](docs/DOCKER_SETUP.md)
>
> Lab machine: `lermen@anubis` — Ubuntu 22.04.1 LTS, Docker 26.0.0

```bash
# 1. Clone and transfer data (already done on anubis)
git clone https://github.com/thiagolermen/gs-erp-3d-classification.git
cd gs-erp-3d-classification
rsync -avz data/processed/ lermen@anubis:~/DEV_ENV/gs-erp-3d-classification/data/processed/

# 2. Build and verify GPU
make build
make check-gpu

# 3. Train all experiments
tmux new-session -s training
make baselines-all
```

---

## Training Details

| Parameter | HSDC (ResNet-34) | SWHDC (ResNet-50) |
|-----------|------------------|-------------------|
| ERP resolution | 512 x 256 | 512 x 256 |
| Input channels | 10 (8 shells + pseudo_depth + mip) | 10 |
| Dilation rates N | 4 (rates 1-4) | 4 (rates 1-4) |
| Optimizer | AdamW (weight decay 5e-4) | AdamW (weight decay 5e-4) |
| Initial LR | 1e-4 | 1e-4 |
| LR warmup | 10 epochs (linear) | 10 epochs (linear) |
| LR schedule | Cosine annealing | Cosine annealing |
| Min LR | 1e-6 | 1e-6 |
| Max epochs | 500 | 400 |
| Early stopping patience | 100 | 150 |
| Label smoothing | 0.1 | 0.1 |
| MixUp / CutMix | alpha 0.4 / alpha 0.4 (50-50) | alpha 0.4 / alpha 0.4 (50-50) |
| Augmentation probability | 30% | 30% |
| Train / val split | 80% / 20% | 80% / 20% |

---

## References

```
Stringhini, R. M., Lermen, T. S., da Silveira, T. L. T., & Jung, C. R. (2024).
Single-Panorama Classification of 3D Objects Using Horizontally Stacked Dilated Convolutions.
IEEE ICIP 2024.

Stringhini, R. M., da Silveira, T. L. T., & Jung, C. R. (2024).
Spherically-Weighted Horizontally Dilated Convolutions for Omnidirectional Image Processing.
SIBGRAPI 2024.

Choi, J., Lee, J., Shin, H., & Kim, Y. M. (2023).
EgoNeRF: Egocentric Neural Radiance Fields.
CVPR 2023.
```

---

## License

MIT License. See [LICENSE](LICENSE).

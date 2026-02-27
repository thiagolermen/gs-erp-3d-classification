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

## Local Machine Setup

> **The local machine is used for data preprocessing and notebook exploration only.**
> Training and evaluation run on the lab machine via Docker (see [docs/DOCKER_SETUP.md](docs/DOCKER_SETUP.md)).

### 1. Create a Python virtual environment

```bash
python -m venv .venv
```

Activate it:

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
# Step 1 — PyTorch CPU-only (much smaller than the CUDA build; no GPU needed locally)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Step 2 — Remaining preprocessing + notebook dependencies
pip install -r requirements-local.txt

# Step 3 — Install the project as an editable package
pip install -e . --no-deps
```

### 3. Download ModelNet

Download from [https://modelnet.cs.princeton.edu/](https://modelnet.cs.princeton.edu/) and extract so the directory layout is:

```
data/raw/modelnet10/<class>/train/*.off
data/raw/modelnet10/<class>/test/*.off
data/raw/modelnet40/<class>/train/*.off
data/raw/modelnet40/<class>/test/*.off
```

### 4. Run preprocessing

Preprocessing converts raw 3D meshes (`.off`) into ERP image caches (`.npy`).  This is a **one-time step** — the cache is reused for all training runs.  Existing files are skipped automatically, so it is safe to interrupt and resume.

Expected time on a modern laptop (single CPU core): ~1–2 h for ModelNet10, ~5–7 h for ModelNet40.

```bash
# 12-channel ERP cache (HSDC pipeline) — ModelNet10
python scripts/preprocess_dataset.py \
    --data_root data/raw/modelnet10 \
    --cache_dir data/processed/modelnet10/hsdc \
    --pipeline  hsdc

# 12-channel ERP cache (HSDC pipeline) — ModelNet40
python scripts/preprocess_dataset.py \
    --data_root data/raw/modelnet40 \
    --cache_dir data/processed/modelnet40/hsdc \
    --pipeline  hsdc

# 1-channel depth ERP cache (SWHDC pipeline) — ModelNet10
python scripts/preprocess_dataset.py \
    --data_root data/raw/modelnet10 \
    --cache_dir data/processed/modelnet10/swhdc \
    --pipeline  swhdc

# 1-channel depth ERP cache (SWHDC pipeline) — ModelNet40
python scripts/preprocess_dataset.py \
    --data_root data/raw/modelnet40 \
    --cache_dir data/processed/modelnet40/swhdc \
    --pipeline  swhdc
```

### 5. Verify the output

```bash
# Expected file counts (one .npy per mesh)
python -c "import pathlib; print(len(list(pathlib.Path('data/processed/modelnet10/hsdc').rglob('*.npy'))))"
# ~4 899  (3 991 train + 908 test)

python -c "import pathlib; print(len(list(pathlib.Path('data/processed/modelnet40/hsdc').rglob('*.npy'))))"
# ~12 311  (9 843 train + 2 468 test)
```

Run unit tests to verify shapes and channel statistics:

```bash
python -m pytest tests/ -v -k preprocess
```

### 6. Transfer the cache to the lab machine

Once preprocessing is complete, copy the `data/processed/` directory to the lab machine:

```bash
# rsync (recommended — skips files already transferred)
rsync -avz --progress data/processed/ user@lab-machine:~/erp-vit-3d-classification/data/processed/

# scp (simpler, no incremental support)
scp -r data/processed user@lab-machine:~/erp-vit-3d-classification/data/
```

### 7. Launch Jupyter notebooks locally

```bash
# Make sure the venv is active
jupyter notebook notebooks/

# Or launch JupyterLab
jupyter lab notebooks/
```

This opens the browser at `http://localhost:8888`. No token is required (the local server runs with token disabled by default) — if prompted, use the token printed in the terminal.

The main analysis notebook is `notebooks/results_analysis.ipynb`. It reads CSVs from `experiments/` — copy those from the lab machine if you want to inspect results locally:

```bash
# On local machine — pull results from lab
rsync -avz user@lab-machine:~/erp-vit-3d-classification/experiments/ experiments/
```

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
│   │   ├── train.py               ← epoch loop, AMP, checkpointing
│   │   ├── evaluate.py            ← test-set evaluation, top-1 accuracy
│   │   └── scheduler.py           ← Adam/AdamW, StepLR, early stopping
│   └── analysis/
│       ├── metrics.py             ← accuracy, confusion matrix helpers
│       ├── visualize.py           ← ERP channel plots, training curves
│       └── compare.py             ← cross-run comparison utilities
├── configs/                       ← one YAML per experiment (10 total)
│   ├── resnet34_hsdc_mn10.yaml
│   ├── resnet34_hsdc_mn40.yaml
│   ├── resnet50_swhdc_mn10.yaml
│   ├── resnet50_swhdc_mn40.yaml
│   ├── swin_hsdc_mn10.yaml  /  swin_hsdc_mn40.yaml
│   ├── swin_swhdc_mn10.yaml  /  swin_swhdc_mn40.yaml
│   ├── effnetv2_hsdc_mn40.yaml
│   └── effnetv2_swhdc_mn40.yaml
├── experiments/                   ← run outputs: checkpoints, logs, CSVs (gitignored)
├── data/                          ← raw and processed ModelNet data (gitignored)
│   ├── raw/modelnet10/  /  raw/modelnet40/
│   └── processed/modelnet10/  /  processed/modelnet40/
├── notebooks/
│   └── results_analysis.ipynb     ← master results analysis notebook
├── scripts/
│   ├── check_gpu.py               ← CUDA + GPU sanity check
│   ├── preprocess_all.sh          ← generate all ERP caches (run inside container)
│   └── run_baselines.sh           ← train all four baseline experiments
├── tests/                         ← unit tests (pytest)
├── docs/
│   ├── technical_documentation.md ← full method description
│   ├── architecture.md            ← Mermaid architecture diagrams
│   └── DOCKER_SETUP.md            ← Docker + SSH remote-training guide
├── Dockerfile
├── docker-compose.yml
├── Makefile                       ← all operations as make targets (lab machine)
├── pyproject.toml
├── requirements.txt               ← full deps (Docker / lab machine)
├── requirements-local.txt         ← CPU-only deps (local preprocessing + notebooks)
├── papers/                        ← reference PDFs
└── README.md
```

---

## Lab Machine Setup (Docker — training and evaluation)

> The lab machine is **only** used for training and evaluation.
> Data preprocessing happens on the local machine — see [Local Machine Setup](#local-machine-setup).
>
> **Full guide:** [`docs/DOCKER_SETUP.md`](docs/DOCKER_SETUP.md)

### Requirements on the lab machine

- Docker Engine ≥ 24 with `docker compose` plugin (v2)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA driver ≥ 525.60 (for CUDA 12.1)
- Git configured: `git config --global user.name/user.email`
- SSH key added to GitHub (for `git push` from inside the container)

### Step-by-step

```bash
# ── On the lab machine ────────────────────────────────────────────────────────

# 1. Clone the repo
git clone git@github.com:<user>/erp-vit-3d-classification.git
cd erp-vit-3d-classification

# 2. Transfer the preprocessed ERP cache from your local machine
#    (run this on your LOCAL machine after preprocessing)
rsync -avz --progress data/processed/ user@lab-machine:~/erp-vit-3d-classification/data/processed/

# 3. Build the Docker image (first time, ~5 min)
make build

# 4. Verify GPU access
make check-gpu

# 5. Reproduce baselines — use tmux so the session survives SSH disconnects
tmux new-session -s training
make baselines-all          # ~10 h total on a single RTX 3090
# Ctrl+B D  (detach — training continues after disconnect)

# 6. Run proposed experiments
make train CONFIG=configs/swin_hsdc_mn40.yaml
make train CONFIG=configs/swin_swhdc_mn40.yaml
make train CONFIG=configs/effnetv2_hsdc_mn40.yaml
make train CONFIG=configs/effnetv2_swhdc_mn40.yaml

# 7. Evaluate best checkpoint
make evaluate \
  CONFIG=configs/resnet34_hsdc_mn10.yaml \
  CHECKPOINT=experiments/resnet34_hsdc_mn10_seed42/best_checkpoint.pt
```

### Running from home via SSH

```bash
# Reattach to watch progress
ssh user@lab-machine
tmux attach -t training

# Or stream the log directly (no tmux needed):
make logs RUN_NAME=swin_t_hsdc_mn40_seed42
```

### Jupyter notebook from home (lab machine)

```bash
# On the lab machine — start Jupyter inside the container
make jupyter

# On your local machine — open SSH tunnel (in a new terminal)
ssh -L 8888:localhost:8888 -N user@lab-machine

# Open in browser:  http://localhost:8888   |   Token: erp-vit
```

### Git push from inside the container

The project directory (including `.git/`) and `~/.ssh` keys are mounted into the
container, so `git push` works exactly like on the host:

```bash
make shell                          # open interactive GPU session

# Inside the container
git add src/ configs/
git commit -m "feat: update Swin-T HSDC integration"
git push origin main
exit
```

See [docs/DOCKER_SETUP.md — Git Push section](docs/DOCKER_SETUP.md) for SSH key setup and troubleshooting.

### All `make` targets

```
make build                          # build Docker image
make shell                          # interactive GPU bash session
make check-gpu                      # verify CUDA + GPU
make test                           # run pytest suite
make jupyter                        # Jupyter on port 8888 (SSH-tunnel ready)

make train CONFIG=<yaml>            # train one experiment
make baselines-mn10                 # ResNet-34+HSDC and ResNet-50+SWHDC on MN10
make baselines-mn40                 # same on MN40
make baselines-all                  # all four baseline experiments (sequential)

make evaluate CONFIG=<yaml> CHECKPOINT=<pt>

make logs RUN_NAME=<run>            # tail -f train.log
make clean                          # remove experiments/ (asks confirmation)
make clean-cache                    # remove data/processed/ (asks confirmation)
```

---

## Training

All models are trained **from scratch** (no ImageNet pretraining), consistent with the original papers — ERP images differ substantially from natural perspective images and pretraining on ImageNet has not been shown to help in this setting.

Training runs on the **lab machine** via Docker:

```bash
# Train one experiment
make train CONFIG=configs/resnet34_hsdc_mn10.yaml

# Equivalent (inside the container shell)
python -m src.training.train --config configs/resnet34_hsdc_mn10.yaml
```

Each run produces under `experiments/<run_name>/`:

| File | Contents |
|---|---|
| `config.yaml` | Copy of the YAML config used |
| `train.log` | Python logging output |
| `metrics.csv` | epoch, train\_loss, val\_loss, train\_acc, val\_acc, lr |
| `best_checkpoint.pt` | Model weights at best validation accuracy |
| `last_checkpoint.pt` | Model weights at final epoch |

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

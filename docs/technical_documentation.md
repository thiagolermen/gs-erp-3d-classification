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

| Model | Input | MN10 (RF-ERP) | MN40 (RF-ERP) | MN10 geometric* | MN40 geometric* | Params |
|---|---|---|---|---|---|---|
| ResNet-34 + HSDC | 10-ch density ERP | pending | pending | 97.1% | 93.9% | 5.5M |
| ResNet-50 + SWHDC | 10-ch density ERP | pending | pending | 94.1% | 91.9% | 23.6M |

*Reported in original papers for geometric ray-cast ERP on raw meshes. RF-ERP results are the TCC contribution.
Input: 8 density shells + pseudo_depth + mip (geometry-only, no colour).

---

## 2. Dataset: ModelSplat

> **Source code:** `src/preprocessing/ply_loader.py`
> **References:** Ma et al. [9]; Wu et al. [5]; Kerbl et al. [16]

### 2.1 Origin and Scope

The **ModelSplat** dataset (`ShapeSplats/ModelNet_Splats` on HuggingFace,
released alongside Ma et al. [9]) provides a 3D Gaussian Splat representation
of every CAD model from the original ModelNet40 benchmark of Wu et al. [5].
Each mesh was converted into a 3DGS reconstruction by training a standard
3DGS optimiser [16] from a set of multi-view renders of the mesh until the
synthesised views match the renders. The resulting Gaussian cloud occupies
roughly the same spatial region as the original mesh, and approximates the
mesh's surface up to the densification budget used during training.

**Headline statistics:**

| Item | Value |
|---|---|
| Total objects | 12,309 |
| Categories | 40 (ModelNet40 superset of ModelNet10) |
| Format | Binary little-endian PLY (`point_cloud.ply`) |
| Average Gaussians per object | $N_i \approx 10^4$–$10^5$ (variable per object) |
| Source mesh dataset | ModelNet [5] |
| Training pipeline | Vanilla 3DGS [16] |

### 2.2 Directory Layout

```
gs_data/modelsplat/modelsplat_ply/
└── <category>/                          (e.g. bathtub, chair, …)
    ├── train/<object_id>/point_cloud.ply
    └── test/<object_id>/point_cloud.ply
```

The `train` and `test` sub-directories preserve the official ModelNet split
of Wu et al. [5]. The total number of objects per split is reported in
Table 1 of [9].

### 2.3 PLY File Contents

A 3DGS PLY file stores a single `vertex` element of $N_i$ primitives. Each
primitive carries the raw parameters of one 3D Gaussian in the form they
were optimised by the 3DGS pipeline of Kerbl et al. [16, eq. 2-4]. These
raw values use unbounded internal parameterisations (log-space for scale,
logit-space for opacity) to ensure unconstrained gradient updates during
training; they must be decoded back into their geometric meaning before any
downstream computation.

**Property layout (in storage order, all `float32`):**

```
x, y, z,                  # Gaussian centre position
nx, ny, nz,               # (unused — left over from PLY format)
f_dc_0, f_dc_1, f_dc_2,   # SH degree-0 (DC) colour coefficients
f_rest_*,                 # SH degree-1..3 coefficients (variable count)
opacity,                  # logit-space opacity (raw)
scale_0, scale_1, scale_2,# log-space scale per local axis (raw)
rot_0, rot_1, rot_2, rot_3 # quaternion (w, x, y, z), unnormalised
```

The number of `f_rest_*` properties depends on the spherical-harmonic degree
$\ell$ used during 3DGS training; the parser detects this automatically.

### 2.4 Decoding Raw Parameters

For every Gaussian $i \in \{1, \ldots, N\}$ the loader produces the
following physically meaningful quantities (matching Kerbl et al. [16] and
the 3DGS reference implementation):

| Quantity | Symbol | Formula | Domain |
|---|---|---|---|
| Centre | $\boldsymbol{\mu}_i$ | $(x_i, y_i, z_i)$ | $\mathbb{R}^3$ |
| Opacity | $\alpha_i$ | $\sigma(\text{opacity}_i) = \dfrac{1}{1 + e^{-\text{opacity}_i}}$ | $[0, 1]$ |
| Scale (per axis) | $\mathbf{s}_i$ | $\exp(\text{scale}_i)$ | $\mathbb{R}_{>0}^3$ |
| Rotation | $\mathbf{q}_i$ | $(w, x, y, z)_i / \|(w, x, y, z)_i\|$ | unit quaternion |
| Albedo | $\mathbf{c}_i$ | $\mathrm{clip}\!\left(0.5 + Y_0^0 \cdot \mathbf{f}^{dc}_i,\ 0,\ 1\right)$ | $[0, 1]^3$ |

where $Y_0^0 = \frac{1}{2\sqrt{\pi}} \approx 0.28209479$ is the zeroth-order
spherical-harmonic basis function, so that the DC SH coefficient maps to a
view-independent RGB albedo. Higher-order SH coefficients
(`f_rest_*`) are not used by the ERP pipeline; we only ever need the
geometric extent of each Gaussian for density evaluation, not its
view-dependent colour.

Each Gaussian therefore defines a 3D anisotropic kernel with covariance

$$\boldsymbol{\Sigma}_i = \mathbf{R}_i\,\mathrm{diag}(\mathbf{s}_i)^2\,\mathbf{R}_i^{\top},$$

where $\mathbf{R}_i$ is the rotation matrix induced by the quaternion
$\mathbf{q}_i$. This anisotropic covariance is what makes 3DGS a more
expressive representation than an isotropic point cloud, and our preprocessing
fully accounts for it (Section 3.5).

### 2.5 Splits

| Split | Source | Usage |
|---|---|---|
| `train` | 80 % of `<category>/train/` | Gradient updates + augmentation |
| `val`   | 20 % of `<category>/train/` | Early stopping, no augmentation |
| `test`  | `<category>/test/`          | Final evaluation only |

The 80/20 train/val split is drawn once by a fixed-seed `numpy.random.default_rng(seed=42)`
permutation of the preset training pool, matching the protocol of Stringhini
et al. [2, §IV-A]. The official ModelNet `test` directory is held out and
touched only for the final-epoch evaluation reported in Section 7.

---

## 3. Radiance Field ERP Generation

> **Source code:** `src/preprocessing/radiance_field.py`
> **Pipeline overview:** PLY $\rightarrow$ centroid $\rightarrow$ shell radii
> $\rightarrow$ per-pixel ray directions $\rightarrow$ per-shell radial integral
> of the 3DGS density field $\rightarrow$ $(N_{\text{shells}}, H, W)$ tensor.
> **References:** Kerbl et al. [16]; Choi et al. [3]; Stringhini et al. [1, 2].

Whereas the original HSDC/SWHDC pipelines [1, 2] obtained the ERP by
ray-casting a triangle mesh, our pipeline samples a continuous radiance
field over a family of concentric spheres centred on the object. This
section formalises every step.

### 3.1 Spherical View Centre — Opacity-Weighted Centroid

The single free parameter common to every ERP camera model is the location
of its optical centre. We choose the **opacity-weighted centroid** of the
Gaussian cloud:

$$\mathbf{C} \;=\; \frac{\displaystyle\sum_{i=1}^{N} \alpha_i \, \boldsymbol{\mu}_i}{\displaystyle\sum_{i=1}^{N} \alpha_i}.$$

This is the natural analogue of the area-weighted mesh centroid used by
Stringhini et al. [1, §II-A]: $\alpha_i$ acts as the perceptual "mass" of a
Gaussian, since low-opacity primitives contribute proportionally less to
the rendered radiance field. If $\sum_i \alpha_i$ is numerically zero the
implementation falls back to the unweighted mean to avoid a degenerate
camera (`compute_centroid`, `radiance_field.py:62`). For a typical
ModelSplat object, $\mathbf{C}$ lies inside the convex hull of the surface
Gaussians, which is the geometric prerequisite for the spherical
panorama to capture the full object surface.

### 3.2 ERP Camera Model — Per-Pixel Ray Directions

The output ERP has resolution $H \times W$ (default $256 \times 512$). For
each pixel $(u, v)$ with $u \in \{0, \ldots, W-1\}$, $v \in \{0, \ldots, H-1\}$,
the ERP camera model defines spherical coordinates

$$\theta(u) \;=\; \frac{u}{W}\cdot 2\pi - \pi
\quad\in\quad [-\pi, \pi),$$

$$\varphi(v) \;=\; \frac{\pi}{2} - \frac{v}{H}\cdot\pi
\quad\in\quad \left[-\tfrac{\pi}{2}, \tfrac{\pi}{2}\right],$$

and a unit ray direction

$$\mathbf{d}(u, v) \;=\; \bigl(\cos\varphi\cos\theta,\; \cos\varphi\sin\theta,\; \sin\varphi\bigr)^{\top}.$$

This matches the ERP convention used in the HSDC paper [1, eq. 1-2]: row 0
of the image is the north pole ($\varphi = +\pi/2$); the bottom row is the
south pole; the centre column corresponds to azimuth $\theta = 0$. The set
$\{\mathbf{d}(u, v)\}$ is precomputed once per output resolution
(`build_ray_directions`, `radiance_field.py:90`).

### 3.3 Concentric Shell Radii — EgoNeRF Spacing

Inspired by Choi et al. [3, §3.2], we sample the density field at $N$
concentric spheres of radii

$$r_s \;=\; r_{\text{near}} \cdot \left(\frac{r_{\text{far}}}{r_{\text{near}}}\right)^{\frac{s}{N-1}},
\qquad s = 0, 1, \ldots, N-1.$$

The exponential spacing places more shells near the surface (which usually
lies close to $r_{\text{near}}$) and fewer in the empty exterior, mirroring
the radiance-field intuition that the most informative signal is
concentrated near the object boundary.

**Dynamic radius selection.** The two free parameters $r_{\text{near}}$ and
$r_{\text{far}}$ are derived from the distribution of Gaussian centre
distances

$$d_i \;=\; \|\boldsymbol{\mu}_i - \mathbf{C}\|_2.$$

Specifically, $r_{\text{near}}$ and $r_{\text{far}}$ are the
$p_{\text{near}}$-th and $p_{\text{far}}$-th percentiles of $\{d_i\}_{i=1}^N$,
with defaults $p_{\text{near}} = 5$, $p_{\text{far}} = 95$
(`compute_shell_radii`, `radiance_field.py:138`). The percentiles are
preferred over $\min_i d_i$ and $\max_i d_i$ because they reject **floater
Gaussians**: low-opacity primitives that 3DGS optimisation occasionally
parks far away from the object surface. Floaters would otherwise inflate
$r_{\text{far}}$ and waste outer shells on empty space. A degenerate case
($r_{\text{near}} = r_{\text{far}}$, e.g. a planar object) falls back to a
linspace around $r_{\text{near}}$.

### 3.4 Per-Shell Radial Integration

A pure point sample at $r_s$ would miss density that lives between adjacent
shells. We therefore associate each shell $s$ with a radial interval
$[r_s^{-}, r_s^{+}]$ defined by midpoints between consecutive shell centres
(`compute_shell_bounds`, `radiance_field.py:189`):

$$r_s^{\pm} \;=\; \frac{r_s + r_{s\pm 1}}{2}.$$

The endpoints $r_0^{-}$ and $r_{N-1}^{+}$ extend symmetrically by half the
adjacent inter-shell gap so that the shells cover the full radial range.

Within $[r_s^{-}, r_s^{+}]$ we ray-march at $K$ uniformly spaced sample
radii

$$t_k \;=\; r_s^{-} + \left(k + \tfrac{1}{2}\right)\frac{r_s^{+} - r_s^{-}}{K},
\qquad k = 0, 1, \ldots, K-1,$$

and accumulate the **arithmetic mean** of the densities at the samples
into channel $s$ of the ERP. With $K = 1$ this reduces to a point sample
at the shell centre; with $K \geq 4$ (configuration default) it
approximates a Riemann sum over the radial extent of the shell,
representing each shell by a representative density integral rather than a
slice. The value $K$ is exposed as `n_steps_per_shell` in the YAML
configs.

### 3.5 3DGS Density Evaluation

At every sample point $\mathbf{p} = \mathbf{C} + t_k\,\mathbf{d}(u, v)$ the
volumetric density of the radiance field is the standard
3DGS expression of Kerbl et al. [16, eq. 3] summed over all primitives:

$$\rho(\mathbf{p}) \;=\; \sum_{i=1}^{N} \alpha_i \, \exp\!\left(-\tfrac{1}{2}\, D_i^2(\mathbf{p})\right),$$

where $D_i^2$ is the squared Mahalanobis distance from $\mathbf{p}$ to
Gaussian $i$:

$$D_i^2(\mathbf{p}) \;=\; (\mathbf{p} - \boldsymbol{\mu}_i)^{\top}\,\boldsymbol{\Sigma}_i^{-1}\,(\mathbf{p} - \boldsymbol{\mu}_i).$$

Substituting $\boldsymbol{\Sigma}_i = \mathbf{R}_i\,\mathrm{diag}(\mathbf{s}_i)^2\,\mathbf{R}_i^{\top}$ gives the
factorised form

$$D_i^2(\mathbf{p}) \;=\; \left\|\,\mathrm{diag}(\mathbf{s}_i)^{-1}\,\mathbf{R}_i^{\top}\,(\mathbf{p} - \boldsymbol{\mu}_i)\,\right\|_2^{2}.$$

The implementation precomputes the **pre-scaled inverse rotation**
$\widetilde{\mathbf{R}}_i \in \mathbb{R}^{3\times 3}$ defined by
$\widetilde{\mathbf{R}}_i[k, :] = \mathbf{R}_i^{\top}[k, :] / s_{i,k}$
(`precompute_gaussian_params`, `radiance_field.py:257`), so that

$$D_i^2(\mathbf{p}) \;=\; \|\widetilde{\mathbf{R}}_i\,(\mathbf{p} - \boldsymbol{\mu}_i)\|_2^{2}.$$

This formulation allows the per-shell computation to be expressed as a
single batched `einsum` over (Gaussians $\times$ pixels), which is the
bottleneck of the pipeline (lines 479-525 in numpy; 581-627 in torch).
The quaternion-to-rotation conversion uses the standard formula
(`quaternions_to_rotation_matrices`, `radiance_field.py:214`):

$$\mathbf{R}(\mathbf{q}) \;=\; \begin{pmatrix} 1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\ 2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\ 2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2) \end{pmatrix},$$

with the quaternion normalised to unit length first.

### 3.6 Spatial Culling

A naive evaluation of $\rho$ at every $(s, u, v, k)$ would touch every one
of the $N \sim 10^5$ Gaussians for every one of the $K \cdot N_{\text{shells}} \cdot H \cdot W \sim 10^7$
sample points. We exploit two facts to cull most contributions:

1. The kernel $\exp(-\tfrac{1}{2} D_i^2)$ falls below numerical precision
   for $D_i > 3$ ("3σ cutoff"), as the contribution drops by
   $e^{-4.5} \approx 0.011$ at $D = 3$ and faster thereafter.
2. The Mahalanobis distance from any point on the sphere of radius $r$ to
   Gaussian $i$ is bounded below by
   $\bigl(|d_i - r| - 3\,\max_k s_{i,k}\bigr) / \max_k s_{i,k}$.

Therefore, for each shell $s$ with radial extent $[r_s^{-}, r_s^{+}]$ we
keep only the Gaussians whose centre-to-centroid distance lies inside
$[r_s^{-} - 3\,s_{\max,i},\ r_s^{+} + 3\,s_{\max,i}]$, where
$s_{\max,i} = \max_k s_{i,k}$ (`radiance_field.py:491-494`). Inside the
inner loop a second `where(D^2 < (3\sigma)^2, ..., 0)` mask drops any pixel
whose Mahalanobis distance still exceeds the threshold. In practice
fewer than 5 % of Gaussians survive culling per shell, giving the pipeline
a roughly $20\times$ speedup over the unculled evaluation.

### 3.7 Output Tensor

After all shells have been integrated the pipeline returns a contiguous
`float32` tensor of shape

$$\mathrm{ERP} \in \mathbb{R}_{\geq 0}^{N_{\text{shells}} \times H \times W}, \qquad N_{\text{shells}} = 8,\ H = 256,\ W = 512$$

whose entry at $(s, v, u)$ is the mean density inside shell $s$ along the
ray $\mathbf{d}(u, v)$. Density values are unbounded above (typical max
$\approx 14$ on ModelSplat) and the distribution is heavily right-skewed
because most pixels look into empty regions of the radiance field. This
skew motivates the log-compression transform of Section 4.1.

### 3.8 Caching

The radiance-field evaluation is the most expensive step in the pipeline
(roughly 1-2 s per object on CPU, ms-scale on GPU). To amortise this cost
across training epochs and across experiments, every computed ERP is
serialised to `data/processed/<dataset>/radiance_field/<param_hash>/<category>/<split>/<id>.npy`,
where `<param_hash>` encodes every preprocessing hyperparameter
($N_{\text{shells}}$, $H$, $W$, cutoff $\sigma$, percentile bounds, ray-march
steps, colour flag). Changing any preprocessing parameter automatically
invalidates the cache, eliminating a class of silent reproducibility bugs
(`_cache_subdir`, `dataset.py:358`).

---

## 4. Classifier Input — Transforms and Augmentation

> **Source code:** `src/preprocessing/dataset.py`, `src/preprocessing/augmentation.py`
> **References:** Stringhini et al. [1, 2]; Zhang et al. [6]; Zhong et al. [7]; Yun et al. [17].

The cached ERP from Section 3 is not fed directly to the network. Inside
`GaussianERPDataset.__getitem__` (`dataset.py:223`) four transforms run in
order: log-compression, derived-channel augmentation of the channel
dimension, single-sample geometric/photometric augmentation, and finally
tensor conversion. A separate batch-level transform (MixUp/CutMix) is
performed in the training loop.

### 4.1 Log-Compression

The raw shell density $\rho \in [0, \rho_{\max}]$ is heavy-tailed (the
99 th percentile is roughly an order of magnitude smaller than the
maximum). To stabilise downstream optimisation and to amplify
low-density boundary signal where the discriminative content of an
object lives, we apply the elementwise

$$\tilde{\rho} \;=\; \log\!\bigl(1 + \rho\bigr),$$

i.e. `numpy.log1p`, before any further processing
(`dataset.py:236`; activated by `data.log1p_transform: true` in the YAML).
This bounded-derivative function compresses $\rho \in [0, 14]$ to
$\tilde\rho \in [0, 2.71]$ while preserving the partial ordering of
densities.

### 4.2 Derived Feature Channels

After log1p, two derived scalar fields are concatenated as additional
channels (`_compute_derived_channels`, `dataset.py:257`).

**Pseudo-depth** is the density-weighted average shell index, normalised
to $[0, 1]$:

$$\mathrm{depth}(u, v) \;=\; \frac{1}{N_{\text{shells}} - 1} \cdot \frac{\sum_{s=0}^{N_{\text{shells}}-1} s\,\tilde\rho_s(u, v)}{\sum_{s=0}^{N_{\text{shells}}-1} \tilde\rho_s(u, v) + \epsilon}.$$

It encodes which shell along the ray contains the dominant surface, i.e.
a coarse depth-from-camera map equivalent to the geometric depth channel
of HSDC [1] but computed from the radiance field.

**Maximum intensity projection (MIP)** records the strongest density
along the ray:

$$\mathrm{mip}(u, v) \;=\; \max_{s=0, \ldots, N_{\text{shells}}-1} \tilde\rho_s(u, v).$$

It behaves as a silhouette mask: pixels for which every shell is empty
become zero, while pixels that intersect any part of the object are
positive.

Together these channels embed two scalar summaries of the radial axis
into the spatial-channel encoding, providing the convolutional backbone
with explicit depth and silhouette cues that would otherwise have to be
learned from the 8 raw shells. With 8 density shells + pseudo-depth +
MIP, the final classifier input is

$$\mathrm{ERP}^{\,\text{clf}} \in \mathbb{R}_{\geq 0}^{10 \times 256 \times 512}.$$

### 4.3 Single-Sample Augmentation

> `src/preprocessing/augmentation.py`

All augmentations operate on `(C, H, W)` float32 arrays and are agnostic
to the channel count $C$. Augmentation is only applied to the training
split.

| Primitive | Probability | Parameters | Notes |
|---|---|---|---|
| Horizontal flip | 0.5 | — | Exact azimuthal $180^{\circ}$ rotation (ERP is $2\pi$-periodic along $u$) |
| 3-D rotation | 0.3 | $\theta_x, \theta_y \sim \mathcal{U}[0^{\circ}, 15^{\circ}]$, $\theta_z \sim \mathcal{U}[0^{\circ}, 45^{\circ}]$ | Spherical remapping with bilinear sampling, circular along $u$ |
| Gaussian blur | 0.3 | $\sigma \sim \mathcal{U}[0.1, 2.0]$ | Applied channel-wise |
| Gaussian noise | 0.3 | $\mu \sim \mathcal{U}[0, 10^{-3}]$, $\sigma \sim \mathcal{U}[0, 0.03]$ | Independent per channel |
| Random erasing [7] | 0.3 | area $\sim \mathcal{U}[2\%, 33\%]$, aspect $\sim \log\mathcal{U}[0.3, 3.3]$ | Sets a rectangular patch to zero |

The 3-D rotation deserves special mention because it is non-trivial to
implement correctly on an ERP. For every output pixel $(u, v)$ we
compute its output direction $\mathbf{d}_{\text{out}}$ as in Section 3.2,
apply the **inverse** rotation $\mathbf{d}_{\text{src}} = \mathbf{R}^{-1}\,\mathbf{d}_{\text{out}}$,
convert back to ERP coordinates, and bilinearly sample the input ERP at
$(u_{\text{src}}, v_{\text{src}})$ with circular wrap on $u$
(`rotate_erp_3d`, `augmentation.py:45`). This procedure is rigid on the
sphere and therefore preserves the geometric correctness of the ERP
camera model, in contrast to a naive 2-D image rotation which would
distort the spherical geometry.

The angle ranges and probabilities match the original HSDC and SWHDC
augmentation recipes of Stringhini et al. [1, §III-A; 2, §IV-A], except
that we use probability 0.3 instead of the paper's 0.15 to compensate for
the smaller effective dataset size when training from scratch on
ModelSplat. Random erasing [7] is not in the original recipe and is
added because the radiance-field ERP tends to contain large
low-information regions that the network can otherwise memorise.

### 4.4 Sample-Pair Augmentation (MixUp and CutMix)

In the training loop (`src/training/train.py`) we further combine pairs
of samples using **MixUp** [6] with $\alpha = 0.4$ and **CutMix** [17]
adapted for ERPs.

**MixUp** linearly blends two samples and their one-hot labels:

$$\tilde{\mathbf{x}} = \lambda\,\mathbf{x}_a + (1 - \lambda)\,\mathbf{x}_b,
\qquad \tilde{\mathbf{y}} = \lambda\,\mathbf{y}_a + (1 - \lambda)\,\mathbf{y}_b,
\qquad \lambda \sim \mathrm{Beta}(0.4, 0.4).$$

**CutMix** replaces a rectangular crop of $\mathbf{x}_a$ with the
corresponding crop from $\mathbf{x}_b$ and adjusts the label by the
fraction of pixels retained. Our implementation
(`cutmix_erp`, `augmentation.py:272`) wraps the crop horizontally so
that ERP periodicity is preserved (`cx` is sampled in $[0, W)$ and the
patch wraps across the seam if necessary). The two primitives alternate
50/50 per batch.

### 4.5 Final Tensor Hand-Off

After all transforms the per-sample tensor delivered to the training loop
is

$$\mathbf{x} \in \mathbb{R}^{C\,\times\,H\,\times\,W},
\qquad C = N_{\text{shells}} + n_{\text{derived}} = 10,\ H = 256,\ W = 512,$$

with corresponding label $y \in \{0, \ldots, K-1\}$ ($K = 10$ for
ModelNet10, $K = 40$ for ModelNet40). This is the contract every
classifier in Section 5 consumes.

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
| Max epochs | 500 (HSDC) / 400 (SWHDC) | |
| Early stopping | patience = 100 (HSDC) / 150 (SWHDC) | gives cosine schedule room |
| Gradient clipping | max_norm = 1.0 | |
| Batch size | 32 | |
| Mixed precision | AMP (CUDA only) | |
| MixUp | α = 0.4 (Zhang et al., 2018) | blends sample pairs to reduce overfitting |
| CutMix | α = 0.4, 50/50 with MixUp | rectangular region swap between samples |
| Augmentation prob | 30% per transform | flip, rotation, blur, noise, random erasing |
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

| Method | Dataset | GPU | Pretrain Time | Finetune Time | Total Epochs | Notes |
|---|---|---|---|---|---|---|
| Gaussian-MAE | MN10/MN40 | 1× A6000 (pretrain) / 1× H100 (finetune) | 300 epochs on 52K ShapeNet | 300 epochs | 600 total | Pretrain batch=128; finetune batch=224 |
| GS-PT | MN40 | 2× A100 (pretrain) / 1× A100 (finetune) | 20 epochs on ShapeNet | 300 epochs | 320 total | Finetune batch=32, AdamW lr=5e-2 |
| Point-MAE | MN40 | 1× GPU (unspecified) | 300 epochs on ShapeNet-55 | 300 epochs | 600 total | Standard Transformer training |
| PointMLP | MN40 | 1× GPU | — | ~300 epochs | 300 | Reported ~11 hours on ModelNet40 |
| **GS-ERP: ResNet-34+HSDC** | **MN10** | **1× RTX 3090 Ti** | **—** | **131.6 min (261 ep, early stop)** | **261** | **Best val at epoch 161** |
| **GS-ERP: ResNet-50+SWHDC** | **MN10** | **1× RTX 3090 Ti** | **—** | **100.7 min (200 ep, full)** | **200** | **Best val at epoch 121** |
| **GS-ERP: ResNet-34+HSDC** | **MN40** | **1× RTX 3090 Ti** | **—** | **491.5 min (403 ep, early stop)** | **403** | **Best val at epoch 303** |
| **GS-ERP: ResNet-50+SWHDC** | **MN40** | **1× RTX 3090 Ti** | **—** | **240.2 min (200 ep, full)** | **200** | **Best val at epoch 101** |

#### 9.8.3 Training Cost Analysis

A fair comparison must account for the **total** compute cost, not just
the fine-tuning phase:

| Method | Pretrain Cost | Finetune Cost | Total Cost | MN10 OA | MN40 OA |
|---|---|---|---|---|---|
| Gaussian-MAE E(All) | 300 ep × 52K objects (A6000) | 300 ep × 3.9K objects (H100) | High | 95.37% | 93.35% |
| GS-PT | 20 ep × ShapeNet (2× A100) | 300 ep × MN40 (1× A100) | High | N/A† | — |
| Point-MAE | 300 ep × ShapeNet-55 | 300 ep × MN10/MN40 | Medium | 94.93% | 93.20% |
| Point-BERT | BERT-style on ShapeNet | 300 ep × MN10/MN40 | Medium | 94.82% | 93.20% |
| **GS-ERP: ResNet-34+HSDC** | **None** | **261 ep × 3.2K (RTX 3090 Ti)** | **Low (2.2 h)** | **91.96%** | — |
| **GS-ERP: ResNet-50+SWHDC** | **None** | **200 ep × 3.2K (RTX 3090 Ti)** | **Low (1.7 h)** | **90.75%** | — |
| **GS-ERP: ResNet-34+HSDC** | **None** | **403 ep × 9.8K (RTX 3090 Ti)** | **Low (8.2 h)** | — | **87.72%** |
| **GS-ERP: ResNet-50+SWHDC** | **None** | **200 ep × 9.8K (RTX 3090 Ti)** | **Low (4.0 h)** | — | **87.19%** |

†GS-PT does not report MN10 results; MN40 figures are in the ICASSP proceedings.

Note that the preprocessing cost of generating the ERP cache (converting
all 3,991 MN10 PLYs into 8-shell density ERPs) is a one-time cost of
approximately 3–4 hours on a single CPU, and is not included in the
training time above.

#### 9.8.4 Parameter Efficiency (OA per Million Parameters)

**ModelNet10:**

| Method | Params (M) | MN10 OA (%) | OA/M ratio |
|---|---|---|---|
| **GS-ERP: ResNet-34+HSDC** | **5.5** | **91.96** | **16.8** |
| Gaussian-MAE E(All) | ~22 | 95.37 | 4.3 |
| Point-MAE | 22.1 | 94.93 | 4.3 |
| Point-BERT | ~22 | 94.82 | 4.3 |
| **GS-ERP: ResNet-50+SWHDC** | **23.6** | **90.75** | **3.8** |

**ModelNet40:**

| Method | Params (M) | MN40 OA (%) | OA/M ratio |
|---|---|---|---|
| **GS-ERP: ResNet-34+HSDC** | **5.5** | **87.72** | **15.9** |
| Gaussian-MAE E(All) | ~22 | 93.35 | 4.2 |
| Point-MAE | 22.1 | 93.20 | 4.2 |
| Point-BERT | ~22 | 93.20 | 4.2 |
| **GS-ERP: ResNet-50+SWHDC** | **23.6** | **87.19** | **3.7** |

The HSDC variant achieves the highest parameter efficiency among all
methods on both datasets (~16× better OA/M than Transformer methods),
despite a lower absolute accuracy. This makes it an attractive option
for deployment scenarios where model size is constrained.

---

### 9.9 Summary: Positioning This Work

The tables below position our GS-ERP approach among all 3DGS-based
classification methods on ModelNet10 and ModelNet40.

**ModelNet10:**

| Method | Venue | Input | Architecture | Pretrain | Params (M) | MN10 OA (%) |
|---|---|---|---|---|---|---|
| Gaussian-MAE E(All) | 3DV 2025 | Raw Gaussian params | Transformer | Self-sup MAE | ~22 | **95.37** |
| Point-MAE | ECCV 2022 | Point cloud (1024) | Transformer | Self-sup MAE | 22.1 | 94.93 |
| Point-BERT | CVPR 2022 | Point cloud (1024) | Transformer | Self-sup BERT | ~22 | 94.82 |
| Gaussian-MAE E(C,S,R) | 3DV 2025 | Gaussian C+S+R | Transformer | Self-sup MAE | ~22 | 94.27 |
| Gaussian-MAE E(C) | 3DV 2025 | Gaussian centroids | Transformer | Self-sup MAE | ~22 | 93.72 |
| **GS-ERP: ResNet-34+HSDC** | **This work** | **8-shell RF-ERP** | **CNN (ResNet-34)** | **None** | **5.5** | **91.96** |
| **GS-ERP: ResNet-50+SWHDC** | **This work** | **8-shell RF-ERP** | **CNN (ResNet-50)** | **None** | **23.6** | **90.75** |

**ModelNet40:**

| Method | Venue | Input | Architecture | Pretrain | Params (M) | MN40 OA (%) | MN40 mAcc (%) |
|---|---|---|---|---|---|---|---|
| Gaussian-MAE E(All) | 3DV 2025 | Raw Gaussian params | Transformer | Self-sup MAE | ~22 | **93.35** | — |
| Gaussian-MAE E(C,S,R) | 3DV 2025 | Gaussian C+S+R | Transformer | Self-sup MAE | ~22 | 93.19 | — |
| Point-MAE | ECCV 2022 | Point cloud (1024) | Transformer | Self-sup MAE | 22.1 | 93.20 | — |
| Point-BERT | CVPR 2022 | Point cloud (1024) | Transformer | Self-sup BERT | ~22 | 93.20 | — |
| Gaussian-MAE E(C) | 3DV 2025 | Gaussian centroids | Transformer | Self-sup MAE | ~22 | 91.77 | — |
| PointNet++ | NeurIPS 2017 | Point cloud (1024) | Hierarchical MLP | None | 1.7 | 91.9 | — |
| **GS-ERP: ResNet-34+HSDC** | **This work** | **8-shell RF-ERP** | **CNN (ResNet-34)** | **None** | **5.5** | **87.72** | **83.99** |
| **GS-ERP: ResNet-50+SWHDC** | **This work** | **8-shell RF-ERP** | **CNN (ResNet-50)** | **None** | **23.6** | **87.19** | **83.04** |

**Key observations:**

1. **Different paradigm.** Our approach is the only one to convert 3DGS into
   a 2D image representation (ERP) and apply CNN-based processing. All other
   methods process Gaussian primitives as unordered sets (point-like modality).

2. **No pretraining, dramatically lower compute.** Our MN10 models train from
   scratch in ~2 hours; MN40 in 4–8 hours — all on a single consumer GPU
   (RTX 3090 Ti). Gaussian-MAE requires 600 total epochs across pretraining
   (A6000/H100) plus the 52K-object ShapeNet pretraining corpus. Our total
   compute cost is roughly two orders of magnitude lower.

3. **4× fewer parameters with HSDC.** ResNet-34+HSDC (5.5M params) achieves
   91.96% on MN10 and 87.72% on MN40 while being 4× smaller than any
   Transformer-based method (~22M). The HSDC block adds negligible parameters
   while providing ERP-specific distortion correction.

4. **Distortion correction transfers to radiance fields.** The HSDC and SWHDC
   blocks were designed for geometric ray-cast ERP. On 3DGS-derived ERP, they
   still achieve meaningful classification on both benchmarks, demonstrating
   that the distortion-correction principle generalises to the radiance field
   domain.

5. **Representation gap.** On MN10, the 5.1 pp gap between our best (91.96%)
   and the geometric ERP baseline (97.1%) quantifies the information cost of
   replacing mesh ray-casting with 3DGS radiance field sampling. On MN40,
   the gap to the geometric baseline (93.9%) is 6.2 pp (HSDC) and 6.7 pp
   (SWHDC). The gaps to Gaussian-MAE (93.35%) on MN40 are 5.6 pp and 6.2 pp,
   reflecting both the representation difference (ERP image vs raw Gaussian
   params) and the architectural difference (CNN from scratch vs pretrained
   Transformer).

6. **Parameter efficiency.** On MN10, the HSDC variant achieves 16.8 OA/M vs
   4.3 OA/M for Transformer methods. On MN40, HSDC achieves 15.9 OA/M vs
   4.2 OA/M — consistently ~4× more parameter-efficient. This suggests that
   ERP-based representations, when combined with appropriate distortion
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

[17] Yun et al. CutMix: Regularization Strategy to Train Strong Classifiers
     with Localizable Features. ICCV 2019.

[18] Shoemake. Animating Rotation with Quaternion Curves. SIGGRAPH 1985.
     (Standard quaternion-to-rotation-matrix formula used in §3.5)
```

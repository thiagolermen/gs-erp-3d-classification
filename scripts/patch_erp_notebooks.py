"""
Patch erp_cache_visualization.ipynb and radiance_field_erp.ipynb to:

1. Support the new N_shells=12 + RGB color ERP + new cache path.
2. Add car, airplane, flower_pot to all gallery / comparison sections.
3. Add a new Section 10 in erp_cache_visualization: PLY-based visualization
   for the 3 extra classes (no cache dependency).
4. Add a new Section 6 in radiance_field_erp: gallery for car/airplane/flower_pot.

Run from project root:
    python scripts/patch_erp_notebooks.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_ERP_CACHE = ROOT / "notebooks" / "erp_cache_visualization.ipynb"
NB_RF        = ROOT / "notebooks" / "radiance_field_erp.ipynb"


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source],
    }


def make_md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [source],
    }


# ===========================================================================
# 1. Patch erp_cache_visualization.ipynb
# ===========================================================================

with open(NB_ERP_CACHE, encoding="utf-8") as f:
    nb_cache = json.load(f)

cells = nb_cache["cells"]

# --- Cell 4: update config to support new cache dir + dynamic N_SHELLS ---
NEW_CONFIG_CELL = """\
# ── cache root — tries new n12_rgb cache first, falls back to legacy n8 ────────
PROJECT_ROOT = Path('..').resolve()
CACHE_CANDIDATES = [
    PROJECT_ROOT / 'data' / 'processed' / 'modelnet10' / 'radiance_field_n12_rgb',
    PROJECT_ROOT / 'data' / 'processed' / 'modelnet40' / 'radiance_field_n12_rgb',
    PROJECT_ROOT / 'data' / 'processed' / 'modelnet10' / 'radiance_field',
    PROJECT_ROOT / 'data' / 'processed' / 'modelnet40' / 'radiance_field',
]

CACHE_ROOT = next((c for c in CACHE_CANDIDATES if c.exists()), None)
if CACHE_ROOT is None:
    raise FileNotFoundError(
        'No ERP cache found.  Run:\\n'
        '  python scripts/preprocess_radiance_field.py --dataset modelnet10 --add_color'
    )

param_dirs = [d for d in sorted(CACHE_ROOT.iterdir()) if d.is_dir()]
if not param_dirs:
    raise FileNotFoundError(f'No params subdirectory found under {CACHE_ROOT}')
PARAM_DIR = param_dirs[0]
print(f'Cache root    : {CACHE_ROOT.name}')
print(f'Cache params  : {PARAM_DIR.name}')

# Detect N_SHELLS from the params dir name (ns<N>_...)
import re as _re
_ns_match = _re.search(r'ns(\\d+)', PARAM_DIR.name)
N_SHELLS = int(_ns_match.group(1)) if _ns_match else 8
HAS_COLOR = '_rgb' in PARAM_DIR.name
print(f'N_SHELLS      : {N_SHELLS}')
print(f'Color channels: {HAS_COLOR}')

# Standard MN10 classes + 3 extra classes downloaded for analysis
from src.preprocessing.dataset import MODELNET10_CATEGORIES
MN10_CLASSES   = list(MODELNET10_CATEGORIES)
EXTRA_CLASSES  = ['car', 'airplane', 'flower_pot']
ALL_CLASSES    = MN10_CLASSES + [c for c in EXTRA_CLASSES if c not in MN10_CLASSES]

DATA_ROOT = PROJECT_ROOT / 'gs_data' / 'modelsplat' / 'modelsplat_ply'

# ── helpers ───────────────────────────────────────────────────────────────────
def list_npys(category: str, split: str = 'train') -> list[Path]:
    d = PARAM_DIR / category / split
    return sorted(d.glob('*.npy')) if d.exists() else []

def load_erp(path: Path) -> np.ndarray:
    \"\"\"Load a single (N_shells[+3], H, W) float32 ERP tensor.\"\"\"
    return np.load(path).astype(np.float32)

def normalise(arr: np.ndarray) -> np.ndarray:
    \"\"\"Min-max normalise to [0, 1].\"\"\"
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-12)

# pick a reference sample
FOCUS_CLASS = 'desk'
sample_paths = list_npys(FOCUS_CLASS, 'train')
if not sample_paths:
    FOCUS_CLASS = MN10_CLASSES[0]
    sample_paths = list_npys(FOCUS_CLASS, 'train')
    if not sample_paths:
        FOCUS_CLASS = next((c for c in ALL_CLASSES if list_npys(c)), None)
        if FOCUS_CLASS:
            sample_paths = list_npys(FOCUS_CLASS, 'train')

if not sample_paths:
    raise FileNotFoundError('No cached .npy files found. Run the preprocessing script first.')

REF_PATH = sample_paths[0]
erp = load_erp(REF_PATH)
H, W = erp.shape[1], erp.shape[2]

print(f'Reference class  : {FOCUS_CLASS}')
print(f'Sample file      : {REF_PATH.name}')
print(f'ERP shape        : {erp.shape}   (N_channels={erp.shape[0]}, H={H}, W={W})')
print(f'Value range      : [{erp.min():.4f}, {erp.max():.4f}]')
print(f'Available classes: {[c for c in ALL_CLASSES if list_npys(c)]}')
"""
cells[4]["source"] = [NEW_CONFIG_CELL]

# --- Cell 22: update gallery to use ALL_CLASSES (includes extra classes) ---
NEW_GALLERY_CELL = """\
# Gallery: one RGB composite per class for all available classes
# Includes standard MN10 classes + extra classes (car, airplane, flower_pot)
available = [(cls, list_npys(cls, 'train')) for cls in ALL_CLASSES]
available = [(cls, paths) for cls, paths in available if paths]

if not available:
    print('No cached classes found — skipping gallery.')
else:
    n_cls = len(available)
    ncols = 5
    nrows = (n_cls + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows),
                              gridspec_kw={'hspace': 0.5, 'wspace': 0.25})
    axes_flat = list(axes.flat) if hasattr(axes, 'flat') else [axes]

    # Use first N_SHELLS density channels for RGB composite (channels 0,3,N_SHELLS-1)
    r_ch = 0
    g_ch = min(3, N_SHELLS - 1)
    b_ch = N_SHELLS - 1

    for ax, (cls, paths) in zip(axes_flat, available):
        erp_cls = load_erp(paths[0])[:N_SHELLS]   # density channels only
        rgb_img = make_rgb_composite(erp_cls, r_ch, g_ch, b_ch)
        ax.imshow(rgb_img, aspect='auto')
        tag = ' *' if cls in EXTRA_CLASSES else ''
        ax.set_title(f'{cls}{tag}', fontsize=10, pad=4)
        ax.axis('off')

    for ax in axes_flat[n_cls:]:
        ax.axis('off')

    fig.text(0.5, -0.01,
             f'RGB composite: S1→Red (inner)  |  S{g_ch+1}→Green (mid)  |  S{N_SHELLS}→Blue (outer)'
             '   (* = extra classes, not in MN10 training set)',
             ha='center', fontsize=9, style='italic')

    fig.suptitle(f'Class Gallery — Radiance Field ERP Composites (N_shells={N_SHELLS})',
                 fontsize=13, y=1.01)
    fig.savefig(FIG_DIR / '9_class_gallery.pdf', bbox_inches='tight')
    plt.show()
    print(f'Rendered {n_cls} classes ({len([c for c in ALL_CLASSES if list_npys(c)])} cached).')
"""
cells[22]["source"] = [NEW_GALLERY_CELL]

# --- Append new Section 10: extra classes from PLY (no cache needed) ---
NEW_SEC10_MD = """\
---
## Section 10 — Extra Classes from PLY: Car / Airplane / Flower Pot

These three classes are not part of ModelNet10 training but are analysed here
to understand how the radiance field ERP represents:
- **car**: complex metallic surface, many thin structures
- **airplane**: long thin fuselage, wings, engines
- **flower_pot**: organic multi-scale geometry (worst performer at 5% SWHDC)

Each row shows all N density shells + the RGB color ERP (last 3 channels when
`add_color=True`).  If the PLY is not downloaded the cell is skipped silently.
"""

NEW_SEC10_CODE = """\
from src.preprocessing.ply_loader import load_gaussian_ply
from src.preprocessing.radiance_field import (
    compute_centroid, build_ray_directions, compute_shell_radii,
    precompute_gaussian_params, compute_radiance_field_erp, gaussian_ply_to_erp,
)
from scipy.ndimage import gaussian_filter as _gf

# Configuration — use same params as new configs
_N   = N_SHELLS if 'N_SHELLS' in dir() else 12
_H, _W = 64, 128      # reduced for fast rendering

def _find_ply(category: str, split: str = 'train', idx: int = 0) -> Path | None:
    split_dir = DATA_ROOT / category / split
    if not split_dir.exists():
        return None
    objs = sorted(split_dir.iterdir())
    if idx >= len(objs):
        return None
    p = objs[idx] / 'point_cloud.ply'
    return p if p.exists() else None

extra_erps = {}   # category -> (density_erp, color_erp | None)
for _cat in EXTRA_CLASSES:
    _ply = _find_ply(_cat)
    if _ply is None:
        print(f'  {_cat}: PLY not found — skipping (run download_modelsplat.py first)')
        continue
    print(f'  Processing {_cat} ...', end=' ', flush=True)
    import time as _t
    _t0 = _t.time()
    try:
        _gs  = load_gaussian_ply(_ply)
        _cen = compute_centroid(_gs['xyz'], _gs['opacity'])
        _gp  = precompute_gaussian_params(_gs, _cen)
        _sr  = compute_shell_radii(_gp['r_dist'], n_shells=_N, r_near_pct=10.0, r_far_pct=90.0)
        _ray = build_ray_directions(_H, _W)
        _d   = compute_radiance_field_erp(
            _gp, _cen, _ray, _sr, _H, _W,
            cutoff_sigma=3.0, batch_size=2048, add_color=True,
        )
        _dens = _d[:_N]
        _col  = _d[_N:] if _d.shape[0] > _N else None
        extra_erps[_cat] = (_dens, _col)
        print(f'done ({_t.time()-_t0:.1f}s, {_gs["n_gaussians"]:,} Gaussians)')
    except Exception as _e:
        print(f'ERROR: {_e}')

print(f'\\nLoaded {len(extra_erps)}/{len(EXTRA_CLASSES)} extra classes.')
"""

NEW_SEC10_VIZ = """\
# Visualise each extra class: all N density shells + RGB color (if available)
for _cat, (_dens, _col) in extra_erps.items():
    _N_plot = _dens.shape[0]
    _has_col = _col is not None
    _ncols = 4
    _nrows_dens = (_N_plot + _ncols - 1) // _ncols
    _extra_rows = 1 if _has_col else 0
    _nrows = _nrows_dens + _extra_rows

    fig, axes = plt.subplots(_nrows, _ncols,
                              figsize=(16, 3.5 * _nrows),
                              gridspec_kw={'hspace': 0.5, 'wspace': 0.25})
    axes = np.array(axes).reshape(_nrows, _ncols)

    # Density shells
    for _s in range(_N_plot):
        _r, _c = divmod(_s, _ncols)
        _ax = axes[_r, _c]
        _d  = _gf(_dens[_s].astype(float), sigma=1)
        _im = _ax.imshow(_d, cmap='hot', aspect='auto', interpolation='bilinear')
        plt.colorbar(_im, ax=_ax, fraction=0.03, pad=0.03)
        _ax.set_title(f'Shell {_s+1}', fontsize=9)
        _ax.axis('off')

    # Hide unused density axes
    for _s in range(_N_plot, _nrows_dens * _ncols):
        _r, _c = divmod(_s, _ncols)
        axes[_r, _c].set_visible(False)

    # RGB color ERP row
    if _has_col:
        _col_norm = (_col - _col.min()) / (_col.max() - _col.min() + 1e-9)
        _rgb_img  = np.clip(_col_norm.transpose(1, 2, 0), 0, 1)   # (H, W, 3)
        axes[_nrows_dens, 0].imshow(_rgb_img, aspect='auto')
        axes[_nrows_dens, 0].set_title('RGB color ERP\\n(opacity-weighted)', fontsize=9)
        axes[_nrows_dens, 0].axis('off')
        for _c in range(1, _ncols):
            axes[_nrows_dens, _c].set_visible(False)

    fig.suptitle(
        f'Radiance Field ERP — {_cat} (N={_N_plot} shells, H={_dens.shape[1]}, W={_dens.shape[2]})',
        fontsize=13, fontweight='bold',
    )
    fig.savefig(FIG_DIR / f'10_extra_{_cat}.pdf', bbox_inches='tight')
    plt.show()
"""

nb_cache["cells"].extend([
    make_md_cell(NEW_SEC10_MD),
    make_code_cell(NEW_SEC10_CODE),
    make_code_cell(NEW_SEC10_VIZ),
])

with open(NB_ERP_CACHE, "w", encoding="utf-8") as f:
    json.dump(nb_cache, f, indent=1, ensure_ascii=False)

print(f"Patched: {NB_ERP_CACHE.name}  ({len(nb_cache['cells'])} cells)")


# ===========================================================================
# 2. Patch radiance_field_erp.ipynb
# ===========================================================================

with open(NB_RF, encoding="utf-8") as f:
    nb_rf = json.load(f)

# --- Update Cell 2 (imports/config): update N_SHELLS to 12, add EXTRA_CLASSES ---
OLD_CONFIG_SNIP = "N_SHELLS     = 8"
NEW_CONFIG_SNIP = "N_SHELLS     = 12    # updated from 8 — finer grid for thin-walled objects"

for cell in nb_rf["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if OLD_CONFIG_SNIP in src:
            src = src.replace(OLD_CONFIG_SNIP, NEW_CONFIG_SNIP)
            # Also add EXTRA_CLASSES after MN10_CATEGORIES line
            src = src.replace(
                "MN10_CATEGORIES = list(MODELNET10_CATEGORIES)",
                "MN10_CATEGORIES = list(MODELNET10_CATEGORIES)\n"
                "EXTRA_CLASSES   = ['car', 'airplane', 'flower_pot']",
            )
            # Update default percentiles to 10/90
            src = src.replace(
                "np.random.seed(42)",
                "R_NEAR_PCT  = 10.0   # updated from 5.0 — tighter, filters floaters\n"
                "R_FAR_PCT   = 90.0   # updated from 95.0\n"
                "MIN_OPACITY = 0.05   # filter background floater Gaussians\n"
                "np.random.seed(42)",
            )
            cell["source"] = [src]
            break

# --- Append new Section 6: multi-class gallery for extra classes ---
NEW_SEC6_MD = """\
## 6. Extra-Class Gallery: Car, Airplane, Flower Pot

These three classes are analysed to understand radiance field quality for:
- **car**: dense metallic surface with thin structures; good 3DGS fidelity
- **airplane**: elongated thin fuselage and wings; sparse shell density at many shells
- **flower_pot**: organic multi-scale geometry (soil+leaves+stems); worst SWHDC performer
  at 5% due to floaters and poor 3DGS reconstruction of organic shapes

The gallery uses the new params (N_shells=12, min_opacity=0.05, r_near/far pct=10/90)
and shows: middle density shell, RGB color ERP, XZ cross-section with shell circles.
"""

NEW_SEC6_CODE = """\
# ── Extra class gallery at gallery resolution ──────────────────────────────────
_H_EX, _W_EX = H_GAL, W_GAL
_ray_ex = build_ray_directions(_H_EX, _W_EX)

extra_gallery = {}   # category -> dict with keys: density, color, shell_radii, gs

for _cat in EXTRA_CLASSES:
    _split_dir = DATA_ROOT / _cat / 'train'
    if not _split_dir.exists():
        print(f'  {_cat:<15} NOT FOUND (run download_modelsplat.py --categories {_cat})')
        continue
    _candidates = sorted(_split_dir.iterdir())
    if not _candidates:
        print(f'  {_cat:<15} empty directory')
        continue
    _ply = _candidates[0] / 'point_cloud.ply'
    if not _ply.exists():
        print(f'  {_cat:<15} no point_cloud.ply')
        continue

    import time as _t
    _t0 = _t.time()
    try:
        _gs  = load_gaussian_ply(_ply)

        # Apply opacity filter (same as new preprocessing config)
        _mask = _gs['opacity'] > MIN_OPACITY
        _gs_f = {k: v[_mask] for k, v in _gs.items() if isinstance(v, np.ndarray)}
        _gs_f['n_gaussians'] = int(_mask.sum())

        _cen  = compute_centroid(_gs_f['xyz'], _gs_f['opacity'])
        _gp   = precompute_gaussian_params(_gs_f, _cen)
        _sr   = compute_shell_radii(_gp['r_dist'], n_shells=N_SHELLS,
                                    r_near_pct=R_NEAR_PCT, r_far_pct=R_FAR_PCT)
        _erp  = compute_radiance_field_erp(
            _gp, _cen, _ray_ex, _sr, _H_EX, _W_EX,
            cutoff_sigma=CUTOFF_SIGMA, batch_size=BATCH_SIZE, add_color=True,
        )
        extra_gallery[_cat] = {
            'density':     _erp[:N_SHELLS],
            'color':       _erp[N_SHELLS:] if _erp.shape[0] > N_SHELLS else None,
            'shell_radii': _sr,
            'gs':          _gs_f,
            'centroid':    _cen,
        }
        print(f'  {_cat:<15} done  ({_t.time()-_t0:.1f}s, {_gs_f["n_gaussians"]:,} Gaussians '
              f'[{_gs["n_gaussians"]-_gs_f["n_gaussians"]:,} floaters removed])')
    except Exception as _ex:
        print(f'  {_cat:<15} ERROR: {_ex}')

print(f'\\nExtra gallery: {len(extra_gallery)}/{len(EXTRA_CLASSES)} classes loaded.')
"""

NEW_SEC6_VIZ = """\
# ── Figure: side-by-side comparison for car / airplane / flower_pot ────────────
if not extra_gallery:
    print('No extra class data available.')
else:
    _mid = N_SHELLS // 2
    _cats_loaded = list(extra_gallery.keys())

    fig, axes = plt.subplots(len(_cats_loaded), 3,
                              figsize=(18, 5 * len(_cats_loaded)),
                              gridspec_kw={'hspace': 0.5, 'wspace': 0.3})
    if len(_cats_loaded) == 1:
        axes = axes[np.newaxis, :]

    for _ri, _cat in enumerate(_cats_loaded):
        _data = extra_gallery[_cat]
        _dens = _data['density']
        _col  = _data['color']
        _sr   = _data['shell_radii']
        _gs   = _data['gs']
        _cen  = _data['centroid']

        # Column 0: middle density shell
        _ax0 = axes[_ri, 0]
        _d   = gaussian_filter(_dens[_mid].astype(float), sigma=1)
        _im  = _ax0.imshow(_d, cmap='hot', aspect='auto', interpolation='bilinear')
        plt.colorbar(_im, ax=_ax0, fraction=0.025)
        _ax0.set_title(f'{_cat} — Shell {_mid+1}/{N_SHELLS}\\n(r={_sr[_mid]:.4f})', fontsize=10)
        _ax0.axis('off')

        # Column 1: RGB color ERP
        _ax1 = axes[_ri, 1]
        if _col is not None:
            _col_n = (_col - _col.min()) / (_col.max() - _col.min() + 1e-9)
            _ax1.imshow(np.clip(_col_n.transpose(1, 2, 0), 0, 1), aspect='auto')
            _ax1.set_title(f'{_cat} — RGB color ERP\\n(opacity-weighted mean)', fontsize=10)
        else:
            _ax1.text(0.5, 0.5, 'No color\\n(add_color=False)', ha='center', va='center',
                      transform=_ax1.transAxes, fontsize=11)
        _ax1.axis('off')

        # Column 2: XZ cross-section + shell circles
        _ax2 = axes[_ri, 2]
        _thickness = 0.08 * (_gs['xyz'].max() - _gs['xyz'].min())
        _xz_mask   = np.abs(_gs['xyz'][:, 1] - _cen[1]) < _thickness
        _xyz_xz    = _gs['xyz'][_xz_mask]
        _rgb_xz    = _gs['rgb'][_xz_mask]
        _opa_xz    = _gs['opacity'][_xz_mask]
        if len(_xyz_xz) > 0:
            _pt   = (_opa_xz / (_opa_xz.max() + 1e-8) * 12).clip(1, 12)
            _ax2.scatter(_xyz_xz[:, 0], _xyz_xz[:, 2], c=_rgb_xz, s=_pt,
                         alpha=0.7, linewidths=0)
        _ang = np.linspace(0, 2 * np.pi, 300)
        for _r_s in _sr:
            _ax2.plot(_cen[0] + _r_s * np.cos(_ang),
                      _cen[2] + _r_s * np.sin(_ang),
                      'r--', linewidth=0.8, alpha=0.6)
        _ax2.plot([], [], 'r--', linewidth=1, label=f'{N_SHELLS} shells (exp)')
        _ax2.scatter([_cen[0]], [_cen[2]], c='red', s=120, marker='*', zorder=10, label='centroid')
        _ax2.set_title(f'{_cat} — XZ cross-section', fontsize=10)
        _ax2.set_aspect('equal')
        _ax2.legend(fontsize=8)

    fig.suptitle(
        f'Extra-Class Gallery: Car / Airplane / Flower Pot\\n'
        f'Radiance Field ERP (N={N_SHELLS}, r_pct={R_NEAR_PCT}/{R_FAR_PCT}, '
        f'min_opacity={MIN_OPACITY})',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.show()
"""

nb_rf["cells"].extend([
    make_md_cell(NEW_SEC6_MD),
    make_code_cell(NEW_SEC6_CODE),
    make_code_cell(NEW_SEC6_VIZ),
])

with open(NB_RF, "w", encoding="utf-8") as f:
    json.dump(nb_rf, f, indent=1, ensure_ascii=False)

print(f"Patched: {NB_RF.name}  ({len(nb_rf['cells'])} cells)")
print("\nDone. Run the notebooks to visualize the new data.")

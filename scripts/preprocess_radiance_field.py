"""
Standalone radiance field ERP cache builder for the 3DGS pipeline.

Walks a ModelSplat PLY directory (gs_data/modelsplat/modelsplat_ply/) and
pre-computes the multi-channel radiance field ERP for every point_cloud.ply
file.  Results are saved as .npy files under the cache directory, mirroring
the subdirectory structure used by GaussianERPDataset so that training picks
them up automatically without re-computing.

Existing cache files are skipped — safe to resume after interruption.

Usage (radiance field, new default params — n_shells=12, RGB color channels):

    python scripts/preprocess_radiance_field.py \\
        --data_root gs_data/modelsplat/modelsplat_ply \\
        --cache_dir data/processed/modelnet10/radiance_field_n12_rgb \\
        --dataset   modelnet10 \\
        --n_shells  12 \\
        --add_color

    # Old cache (n_shells=8, no color) — reproduce previous results:
    python scripts/preprocess_radiance_field.py \\
        --data_root gs_data/modelsplat/modelsplat_ply \\
        --cache_dir data/processed/modelnet10/radiance_field \\
        --dataset   modelnet10 \\
        --n_shells  8

    # Specific categories only (e.g. newly downloaded classes):
    python scripts/preprocess_radiance_field.py \\
        --data_root gs_data/modelsplat/modelsplat_ply \\
        --cache_dir data/processed/modelnet10/radiance_field_n12_rgb \\
        --categories car airplane flower_pot \\
        --n_shells  12 \\
        --add_color

The cache subdirectory layout is identical to GaussianERPDataset._cache_subdir():
    <cache_dir> / ns{n}_{H}x{W}_c{cutoff}_p{near}-{far}_op{opacity}[_rgb] /
                 <category> / <split> / <object_id>.npy

References:
    EgoNeRF (Choi et al., CVPR 2023) — exponential shell spacing
    Kerbl et al. (SIGGRAPH 2023) — 3D Gaussian Splatting
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Make src.* importable without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.dataset import MODELNET10_CATEGORIES, MODELNET40_CATEGORIES
from src.preprocessing.radiance_field import gaussian_ply_to_erp


# ---------------------------------------------------------------------------
# Cache subdir helper — must match GaussianERPDataset._cache_subdir() exactly
# ---------------------------------------------------------------------------

def _cache_subdir_name(
    n_shells: int,
    H: int,
    W: int,
    cutoff_sigma: float,
    r_near_pct: float,
    r_far_pct: float,
    min_opacity: float,
    add_color: bool,
) -> str:
    name = (
        f"ns{n_shells}_{H}x{W}"
        f"_c{cutoff_sigma:.1f}"
        f"_p{r_near_pct:.1f}-{r_far_pct:.1f}"
        f"_op{min_opacity}"
    )
    if add_color:
        name += "_rgb"
    return name


# ---------------------------------------------------------------------------
# Main preprocessing loop
# ---------------------------------------------------------------------------

def precompute(
    data_root: Path,
    cache_dir: Path,
    categories: list[str],
    n_shells: int,
    H: int,
    W: int,
    cutoff_sigma: float,
    r_near_pct: float,
    r_far_pct: float,
    min_opacity: float,
    add_color: bool,
    batch_size: int,
    device: str | None,
) -> None:
    # Build the params subdirectory (matches dataset.py)
    subdir_name = _cache_subdir_name(
        n_shells, H, W, cutoff_sigma, r_near_pct, r_far_pct, min_opacity, add_color
    )
    param_cache_dir = cache_dir / subdir_name
    param_cache_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Cache dir     : %s", param_cache_dir)
    logging.info("N shells      : %d", n_shells)
    logging.info("ERP size      : %dx%d", H, W)
    logging.info("r_near/far pct: %.1f / %.1f", r_near_pct, r_far_pct)
    logging.info("Min opacity   : %.3f", min_opacity)
    logging.info("Add color     : %s", add_color)
    logging.info("Categories    : %s", categories)

    # Collect all PLY paths for the target categories
    ply_paths: list[Path] = []
    for cat in categories:
        for split in ("train", "test"):
            split_dir = data_root / cat / split
            if not split_dir.exists():
                continue
            for obj_dir in sorted(split_dir.iterdir()):
                ply = obj_dir / "point_cloud.ply"
                if ply.exists():
                    ply_paths.append(ply)

    if not ply_paths:
        raise FileNotFoundError(
            f"No point_cloud.ply files found under '{data_root}' "
            f"for categories: {categories}"
        )

    logging.info("PLY files     : %d", len(ply_paths))

    total     = len(ply_paths)
    skipped   = 0
    processed = 0
    failed    = 0
    t0        = time.time()

    pbar = tqdm(ply_paths, unit="ply", desc="ERP-RF", dynamic_ncols=True)
    for ply_path in pbar:
        # Mirror the path relative to data_root into cache
        try:
            rel = ply_path.parent.relative_to(data_root)   # <cat>/<split>/<obj_id>
        except ValueError:
            rel = Path(ply_path.parent.name)

        cache_path = param_cache_dir / rel.with_suffix(".npy")

        if cache_path.exists():
            skipped += 1
            pbar.set_postfix(done=processed, skip=skipped, fail=failed)
            continue

        try:
            erp = gaussian_ply_to_erp(
                ply_path=ply_path,
                n_shells=n_shells,
                H=H,
                W=W,
                cutoff_sigma=cutoff_sigma,
                r_near_pct=r_near_pct,
                r_far_pct=r_far_pct,
                min_opacity=min_opacity,
                add_color=add_color,
                batch_size=batch_size,
                device=device,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), erp)
            processed += 1
        except Exception as exc:  # noqa: BLE001
            logging.warning("FAILED %s — %s", ply_path, exc)
            failed += 1

        elapsed = time.time() - t0
        rate    = processed / elapsed if elapsed > 0 else 0
        eta     = (total - processed - skipped - failed) / rate if rate > 0 else float("inf")
        pbar.set_postfix(
            done=processed, skip=skipped, fail=failed,
            rate=f"{rate:.1f}/s", eta=f"{eta/60:.0f}min",
        )

    elapsed_min = (time.time() - t0) / 60
    logging.info(
        "Finished | processed=%d  skipped=%d  failed=%d  time=%.1fmin",
        processed, skipped, failed, elapsed_min,
    )
    logging.info("Cache written to: %s", param_cache_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Pre-compute radiance field ERP cache from 3DGS PLY files."
    )
    parser.add_argument(
        "--data_root", type=Path,
        default=Path("gs_data/modelsplat/modelsplat_ply"),
        help="Root directory containing <category>/<split>/<obj_id>/point_cloud.ply files.",
    )
    parser.add_argument(
        "--cache_dir", type=Path,
        default=Path("data/processed/modelnet10/radiance_field_n12_rgb"),
        help="Output cache directory (params subdirectory created automatically).",
    )
    parser.add_argument(
        "--dataset", choices=["modelnet10", "modelnet40", "custom"], default="modelnet10",
        help="Preset category list (modelnet10 or modelnet40). Ignored if --categories given.",
    )
    parser.add_argument(
        "--categories", nargs="+", metavar="CAT", default=None,
        help="Specific categories to process (overrides --dataset).",
    )
    parser.add_argument("--n_shells",    type=int,   default=12)
    parser.add_argument("--height",      type=int,   default=256)
    parser.add_argument("--width",       type=int,   default=512)
    parser.add_argument("--cutoff_sigma", type=float, default=3.0)
    parser.add_argument("--r_near_pct",  type=float, default=10.0)
    parser.add_argument("--r_far_pct",   type=float, default=90.0)
    parser.add_argument("--min_opacity", type=float, default=0.05,
                        help="Remove Gaussians with opacity below this value (floater filter).")
    parser.add_argument("--add_color",   action="store_true",
                        help="Append 3 opacity-weighted RGB channels to the density shells.")
    parser.add_argument("--batch_size",  type=int,   default=4096)
    parser.add_argument("--device",      type=str,   default=None,
                        help="Torch device string, e.g. 'cuda:0' or 'cpu'. Auto-detected if omitted.")
    args = parser.parse_args()

    # Determine category list
    if args.categories:
        cats = args.categories
    elif args.dataset == "modelnet10":
        cats = list(MODELNET10_CATEGORIES)
    elif args.dataset == "modelnet40":
        cats = list(MODELNET40_CATEGORIES)
    else:
        raise ValueError("--dataset=custom requires --categories to be specified.")

    project_root = Path(__file__).resolve().parent.parent

    data_root = args.data_root
    if not data_root.is_absolute():
        data_root = project_root / data_root

    cache_dir = args.cache_dir
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir

    precompute(
        data_root    = data_root,
        cache_dir    = cache_dir,
        categories   = cats,
        n_shells     = args.n_shells,
        H            = args.height,
        W            = args.width,
        cutoff_sigma = args.cutoff_sigma,
        r_near_pct   = args.r_near_pct,
        r_far_pct    = args.r_far_pct,
        min_opacity  = args.min_opacity,
        add_color    = args.add_color,
        batch_size   = args.batch_size,
        device       = args.device,
    )

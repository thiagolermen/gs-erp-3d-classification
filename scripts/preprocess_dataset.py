"""
Standalone ERP preprocessing runner — no torch required.

Generates .npy ERP caches for every mesh in a ModelNet dataset.
Existing cache files are skipped (safe to resume after interruption).

Usage:
    python scripts/preprocess_dataset.py \\
        --data_root data/raw/modelnet10 \\
        --cache_dir data/processed/modelnet10/hsdc \\
        --pipeline  hsdc

    python scripts/preprocess_dataset.py \\
        --data_root data/raw/modelnet10 \\
        --cache_dir data/processed/modelnet10/swhdc \\
        --pipeline  swhdc
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── adjust sys.path so src.* is importable without `pip install -e .` ─────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.erp_features import mesh_to_erp


def precompute(
    data_root: Path,
    cache_dir: Path,
    pipeline: str,
    width: int = 512,
    height: int = 256,
    batch_size: int = 32_768,
) -> None:
    mesh_paths = sorted(data_root.rglob("*.off")) + sorted(data_root.rglob("*.obj"))

    if not mesh_paths:
        raise FileNotFoundError(f"No .off / .obj meshes under '{data_root}'")

    total     = len(mesh_paths)
    skipped   = 0
    processed = 0
    failed    = 0
    t0        = time.time()

    logging.info("Pipeline : %s", pipeline)
    logging.info("Data root: %s", data_root)
    logging.info("Cache dir: %s", cache_dir)
    logging.info("Meshes   : %d", total)

    with tqdm(mesh_paths, unit="mesh", desc=f"ERP-{pipeline.upper()}") as bar:
        for mesh_path in bar:
            try:
                rel        = mesh_path.relative_to(data_root)
            except ValueError:
                rel        = Path(mesh_path.name)
            cache_path = cache_dir / rel.with_suffix(".npy")

            if cache_path.exists():
                skipped += 1
                bar.set_postfix(done=processed, skip=skipped, fail=failed)
                continue

            try:
                erp = mesh_to_erp(mesh_path, pipeline, width, height, batch_size)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(cache_path), erp)
                processed += 1
            except Exception as exc:       # noqa: BLE001
                logging.warning("FAILED %s — %s", mesh_path.name, exc)
                failed += 1

            elapsed = time.time() - t0
            rate    = processed / elapsed if elapsed > 0 else 0
            eta     = (total - processed - skipped - failed) / rate if rate > 0 else float("inf")
            bar.set_postfix(done=processed, skip=skipped, fail=failed,
                            rate=f"{rate:.1f}/s", eta=f"{eta/60:.0f}min")

    elapsed_min = (time.time() - t0) / 60
    logging.info(
        "Finished | processed=%d  skipped=%d  failed=%d  time=%.1fmin",
        processed, skipped, failed, elapsed_min,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Pre-compute ERP image cache for a ModelNet dataset (no torch required)."
    )
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--pipeline",  choices=["hsdc", "swhdc"], required=True)
    parser.add_argument("--width",      type=int, default=512)
    parser.add_argument("--height",     type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32_768)
    args = parser.parse_args()

    precompute(
        data_root  = args.data_root,
        cache_dir  = args.cache_dir,
        pipeline   = args.pipeline,
        width      = args.width,
        height     = args.height,
        batch_size = args.batch_size,
    )

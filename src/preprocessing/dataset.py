"""
PyTorch Dataset and DataLoader wrappers for the ModelNet benchmark.

Workflow
--------
1. **Pre-computation** (run once):
       python -m src.preprocessing.dataset \\
           --data_root data/raw/modelnet10 \\
           --cache_dir data/processed/modelnet10 \\
           --pipeline  hsdc

   This generates a .npy ERP cache file for every mesh in the dataset.

2. **Training / evaluation**:
   The ERPDataset loads cached .npy files and applies augmentation on-the-fly
   for the training split only.

Train / validation split
------------------------
The 80% / 20% split (SWHDC paper §IV-A) is applied to the preset "train"
directory of ModelNet.  A fixed seed ensures reproducibility.  The official
"test" directory is used exclusively for final evaluation.

ModelNet directory structure expected on disk:
    <data_root>/
    ├── bathtub/
    │   ├── train/
    │   │   ├── bathtub_0001.off
    │   │   └── ...
    │   └── test/
    │       └── ...
    └── ...

References:
    HSDC paper §III-A  — Stringhini et al., IEEE ICIP 2024
    SWHDC paper §IV-A  — Stringhini et al., SIBGRAPI 2024
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.preprocessing.augmentation import augment
from src.preprocessing.erp_features import mesh_to_erp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ERPDataset(Dataset):
    """ERP-based ModelNet dataset with optional pre-computed cache.

    Args:
        data_root:        Root directory of the ModelNet dataset
                          (e.g. ``data/raw/modelnet10``).
        cache_dir:        Directory where pre-computed .npy ERP files are stored
                          (e.g. ``data/processed/modelnet10/hsdc``).
        split:            One of ``'train'``, ``'val'``, or ``'test'``.
        pipeline:         ``'hsdc'`` (12-channel) or ``'swhdc'`` (1-channel).
        width:            ERP width in pixels (default 512).
        height:           ERP height in pixels (default 256).
        train_val_split:  Fraction of the preset training set used for training
                          (default 0.8). The rest becomes validation.
        seed:             Random seed for the train/val split.
        transform:        Optional callable applied to the ERP tensor after
                          augmentation (e.g. torchvision normalise).
    """

    def __init__(
        self,
        data_root: Path,
        cache_dir: Path,
        split: str,
        pipeline: str,
        width: int = 512,
        height: int = 256,
        train_val_split: float = 0.8,
        seed: int = 42,
        transform: Callable | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")
        if pipeline not in ("hsdc", "swhdc"):
            raise ValueError(f"pipeline must be 'hsdc' or 'swhdc'; got '{pipeline}'")

        self.data_root       = Path(data_root)
        self.cache_dir       = Path(cache_dir)
        self.split           = split
        self.pipeline        = pipeline
        self.width           = width
        self.height          = height
        self.train_val_split = train_val_split
        self.seed            = seed
        self.transform       = transform
        self._apply_augment  = split == "train"

        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_sample_list()

        # Per-dataset RNG for augmentation (seeded per-sample for reproducibility)
        self._rng = np.random.default_rng(seed)

        logger.info(
            "ERPDataset | split=%s  pipeline=%s  classes=%d  samples=%d",
            split,
            pipeline,
            len(self.classes),
            len(self.samples),
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        mesh_path, label = self.samples[idx]

        erp = self._load_or_generate(mesh_path)

        # Augmentation on training split only
        if self._apply_augment:
            # Use a per-sample seed derived from the dataset seed and index so
            # that results are reproducible when workers are used.
            sample_rng = np.random.default_rng(self.seed + idx)
            erp = augment(erp, prob=0.15, rng=sample_rng)

        tensor = torch.from_numpy(erp.copy())  # (C, H, W) float32

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_classes(self) -> tuple[list[str], dict[str, int]]:
        """Scan *data_root* for class sub-directories, sorted alphabetically."""
        classes = sorted(
            p.name for p in self.data_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
        if not classes:
            raise FileNotFoundError(
                f"No class directories found under '{self.data_root}'."
            )
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx

    def _make_sample_list(self) -> list[tuple[Path, int]]:
        """Build a list of (mesh_path, label) for the requested split.

        - ``split='test'``: all files in ``<class>/test/``.
        - ``split='train'`` or ``split='val'``: files from ``<class>/train/``
          are split 80/20 with a fixed seed (SWHDC paper §IV-A).
        """
        all_samples: list[tuple[Path, int]] = []

        if self.split == "test":
            # Official test set — never used for early stopping
            for cls in self.classes:
                label    = self.class_to_idx[cls]
                test_dir = self.data_root / cls / "test"
                if not test_dir.exists():
                    logger.warning("Missing test directory: %s", test_dir)
                    continue
                for f in sorted(test_dir.glob("*.off")):
                    all_samples.append((f, label))
                for f in sorted(test_dir.glob("*.obj")):
                    all_samples.append((f, label))

        else:
            # Collect all preset training files per class
            preset_train: list[tuple[Path, int]] = []
            for cls in self.classes:
                label     = self.class_to_idx[cls]
                train_dir = self.data_root / cls / "train"
                if not train_dir.exists():
                    logger.warning("Missing train directory: %s", train_dir)
                    continue
                for f in sorted(train_dir.glob("*.off")):
                    preset_train.append((f, label))
                for f in sorted(train_dir.glob("*.obj")):
                    preset_train.append((f, label))

            # Deterministic 80/20 train/val split — SWHDC paper §IV-A
            rng     = np.random.default_rng(self.seed)
            indices = np.arange(len(preset_train))
            rng.shuffle(indices)
            n_train = int(len(indices) * self.train_val_split)
            train_idx = indices[:n_train]
            val_idx   = indices[n_train:]

            chosen = train_idx if self.split == "train" else val_idx
            all_samples = [preset_train[i] for i in sorted(chosen)]

        return all_samples

    def _cache_path(self, mesh_path: Path) -> Path:
        """Return the .npy cache path corresponding to *mesh_path*."""
        # Mirror the raw directory structure under cache_dir
        try:
            rel = mesh_path.relative_to(self.data_root)
        except ValueError:
            rel = Path(mesh_path.name)
        return self.cache_dir / rel.with_suffix(".npy")

    def _load_or_generate(self, mesh_path: Path) -> np.ndarray:
        """Load a cached ERP, or generate and cache it if absent."""
        cache_path = self._cache_path(mesh_path)

        if cache_path.exists():
            return np.load(str(cache_path))

        # Generate and save
        erp = mesh_to_erp(mesh_path, self.pipeline, self.width, self.height)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), erp)
        return erp


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_root: Path,
    cache_dir: Path,
    pipeline: str,
    batch_size: int = 32,
    num_workers: int = 4,
    width: int = 512,
    height: int = 256,
    train_val_split: float = 0.8,
    seed: int = 42,
    transform: Callable | None = None,
) -> dict[str, DataLoader]:
    """Build train, val, and test DataLoaders for a ModelNet dataset.

    Args:
        data_root:        Path to the raw ModelNet directory.
        cache_dir:        Path to the pre-computed ERP cache directory.
        pipeline:         ``'hsdc'`` or ``'swhdc'``.
        batch_size:       Samples per batch (default 32).
        num_workers:      DataLoader worker processes (default 4).
        width:            ERP width (default 512).
        height:           ERP height (default 256).
        train_val_split:  Train fraction of the preset training set (default 0.8).
        seed:             Random seed for split and augmentation RNG.
        transform:        Optional transform applied after augmentation.

    Returns:
        A dict with keys ``'train'``, ``'val'``, ``'test'``, each mapping to
        the corresponding DataLoader.
    """
    loaders: dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        ds = ERPDataset(
            data_root=data_root,
            cache_dir=cache_dir,
            split=split,
            pipeline=pipeline,
            width=width,
            height=height,
            train_val_split=train_val_split,
            seed=seed,
            transform=transform,
        )
        shuffle = split == "train"
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,
        )

    return loaders


# ---------------------------------------------------------------------------
# Pre-computation script
# ---------------------------------------------------------------------------

def precompute_dataset(
    data_root: Path,
    cache_dir: Path,
    pipeline: str,
    width: int = 512,
    height: int = 256,
    batch_size: int = 32_768,
) -> None:
    """Pre-generate and cache ERP images for every mesh in *data_root*.

    Existing cache files are skipped.  Progress is displayed with tqdm.

    Args:
        data_root:  Path to the raw ModelNet directory.
        cache_dir:  Destination directory for .npy cache files.
        pipeline:   ``'hsdc'`` or ``'swhdc'``.
        width:      ERP width (default 512).
        height:     ERP height (default 256).
        batch_size: Ray-casting batch size per mesh.
    """
    mesh_paths = sorted(data_root.rglob("*.off")) + sorted(data_root.rglob("*.obj"))

    if not mesh_paths:
        raise FileNotFoundError(
            f"No .off or .obj meshes found under '{data_root}'."
        )

    logger.info(
        "Pre-computing %d meshes → %s  (pipeline=%s)",
        len(mesh_paths),
        cache_dir,
        pipeline,
    )

    for mesh_path in tqdm(mesh_paths, desc=f"ERP ({pipeline})", unit="mesh"):
        try:
            rel        = mesh_path.relative_to(data_root)
        except ValueError:
            rel        = Path(mesh_path.name)
        cache_path = cache_dir / rel.with_suffix(".npy")

        if cache_path.exists():
            continue

        try:
            erp = mesh_to_erp(mesh_path, pipeline, width, height, batch_size)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), erp)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to process '%s': %s", mesh_path, exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Pre-compute ERP image cache for a ModelNet dataset."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Path to the raw ModelNet directory (e.g. data/raw/modelnet10).",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        required=True,
        help="Output directory for .npy cache files.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["hsdc", "swhdc"],
        required=True,
        help="'hsdc' for 12-channel ERP; 'swhdc' for 1-channel depth ERP.",
    )
    parser.add_argument("--width",  type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32_768,
        help="Ray-casting batch size per mesh.",
    )

    args = parser.parse_args()
    precompute_dataset(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        pipeline=args.pipeline,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
    )

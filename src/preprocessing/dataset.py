"""
Dataset and DataLoader for ModelSplats radiance field ERP classification.

Data source:
    gs_data/modelsplat/modelsplat_ply/<category>/train|test/<id>/point_cloud.ply

Each PLY file is a 3D Gaussian Splat (3DGS) representation of a ModelNet
object produced by the ShapeSplats pipeline (HuggingFace: ShapeSplats/
ModelNet_Splats).  The preprocessing evaluates the continuous 3DGS density
field at N concentric exponentially-spaced sphere shells to produce a
(N_shells, H, W) ERP tensor per object.

Train / validation split:
    An 80% / 20% split (SWHDC paper §IV-A) is applied to the preset "train"
    directory.  A fixed seed ensures reproducibility.  The official "test"
    directory is used exclusively for final evaluation.

Caching:
    Computed ERP tensors are saved as .npy files under cache_dir to avoid
    re-running the expensive radiance field computation.  The cache sub-
    directory encodes all preprocessing hyperparameters so that changing any
    parameter automatically invalidates old cached files.

References:
    ShapeSplats/ModelNet_Splats (HuggingFace, 2024)
    Choi et al., "Balanced Spherical Grid for Egocentric View Synthesis",
    CVPR 2023 (EgoNeRF exponential shell spacing)
    SWHDC paper §IV-A — train/val split protocol
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.augmentation import augment
from src.preprocessing.radiance_field import gaussian_ply_to_erp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category lists
# ---------------------------------------------------------------------------

MODELNET10_CATEGORIES: list[str] = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]

MODELNET40_CATEGORIES: list[str] = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GaussianERPDataset(Dataset):
    """ModelNet 3DGS Radiance Field ERP Dataset.

    Loads 3D Gaussian Splat PLY files from the ModelSplats directory structure
    and converts each object to a (N_shells, H, W) radiance field ERP tensor
    on-the-fly (or from cache).

    Expected directory structure under data_root:
        <data_root>/
        ├── bathtub/
        │   ├── train/
        │   │   ├── bathtub_0001/
        │   │   │   └── point_cloud.ply
        │   │   └── ...
        │   └── test/
        │       └── ...
        └── ...

    Args:
        data_root:      Path to the modelsplat_ply root directory.
        categories:     Ordered list of category names.  The index in this list
                        becomes the integer class label.
        split:          One of 'train', 'val', or 'test'.
        n_shells:       Number of ERP shells / input channels (default 8).
        H:              ERP height in pixels (default 256).
        W:              ERP width  in pixels (default 512).
        cutoff_sigma:   Gaussian truncation in max_scale units (default 3.0).
        batch_size_rf:  Batch size for radiance field computation (default 128).
        r_near_pct:     Percentile for inner shell radius (default 5.0).
        r_far_pct:      Percentile for outer shell radius (default 95.0).
        cache_dir:      If given, .npy files are saved/loaded here to avoid
                        recomputing.  Sub-directory encodes all preprocessing
                        parameters; changing any parameter uses a fresh cache.
        augment_train:  Apply augmentation to the training split (default True).
        val_fraction:   Fraction of training set reserved for validation
                        (default 0.2; SWHDC paper §IV-A).
        seed:           Random seed for the train/val split (default 42).
        transform:      Optional callable applied to the ERP tensor after
                        augmentation (e.g. normalisation).
    """

    def __init__(
        self,
        data_root: Path,
        categories: list[str],
        split: str,
        n_shells: int = 8,
        H: int = 256,
        W: int = 512,
        cutoff_sigma: float = 3.0,
        batch_size_rf: int = 128,
        r_near_pct: float = 5.0,
        r_far_pct: float = 95.0,
        cache_dir: Path | None = None,
        augment_train: bool = True,
        val_fraction: float = 0.2,
        seed: int = 42,
        transform: Callable | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

        self.data_root    = Path(data_root)
        self.categories   = list(categories)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.categories)}
        self.split        = split
        self.n_shells     = n_shells
        self.H            = H
        self.W            = W
        self.cutoff_sigma = cutoff_sigma
        self.batch_size_rf = batch_size_rf
        self.r_near_pct   = r_near_pct
        self.r_far_pct    = r_far_pct
        self.cache_dir    = Path(cache_dir) if cache_dir is not None else None
        self._do_augment  = augment_train and (split == "train")
        self.val_fraction = val_fraction
        self.seed         = seed
        self.transform    = transform

        self.samples = self._make_sample_list()

        logger.info(
            "GaussianERPDataset | split=%-5s  categories=%d  samples=%d  "
            "n_shells=%d  H=%d  W=%d",
            split,
            len(self.categories),
            len(self.samples),
            n_shells,
            H,
            W,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return (erp_tensor, label) for sample *idx*.

        erp_tensor: (N_shells, H, W) float32 torch.Tensor.
        label:      int — class index.
        """
        ply_path, label = self.samples[idx]

        erp = self._load_or_compute(ply_path)  # (N_shells, H, W) float32

        # Augmentation on training split only — HSDC §III-A / SWHDC §IV-A
        if self._do_augment:
            # Per-sample seed derived from dataset seed + index for reproducibility
            sample_rng = np.random.default_rng(self.seed + idx)
            erp = augment(erp, prob=0.15, rng=sample_rng)

        tensor = torch.from_numpy(erp.copy())  # (N_shells, H, W) float32

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_sample_list(self) -> list[tuple[Path, int]]:
        """Build the list of (ply_path, class_index) for this split.

        For 'test': all PLY files under <category>/test/<id>/point_cloud.ply.
        For 'train'/'val': all PLY files under <category>/train/ are split
            (1 - val_fraction) / val_fraction with a fixed seed
            (SWHDC paper §IV-A: 80% / 20%).
        """
        all_samples: list[tuple[Path, int]] = []

        if self.split == "test":
            for cls in self.categories:
                label    = self.class_to_idx[cls]
                test_dir = self.data_root / cls / "test"
                if not test_dir.exists():
                    logger.warning("Missing test directory: %s", test_dir)
                    continue
                for ply in sorted(test_dir.rglob("point_cloud.ply")):
                    all_samples.append((ply, label))

        else:
            # Collect all preset training PLY files
            preset_train: list[tuple[Path, int]] = []
            for cls in self.categories:
                label     = self.class_to_idx[cls]
                train_dir = self.data_root / cls / "train"
                if not train_dir.exists():
                    logger.warning("Missing train directory: %s", train_dir)
                    continue
                for ply in sorted(train_dir.rglob("point_cloud.ply")):
                    preset_train.append((ply, label))

            # Deterministic split — SWHDC paper §IV-A: 80% train / 20% val
            rng     = np.random.default_rng(self.seed)
            indices = np.arange(len(preset_train))
            rng.shuffle(indices)
            n_train = int(len(indices) * (1.0 - self.val_fraction))
            train_idx = indices[:n_train]
            val_idx   = indices[n_train:]

            chosen = train_idx if self.split == "train" else val_idx
            all_samples = [preset_train[i] for i in sorted(chosen)]

        if not all_samples:
            logger.warning(
                "GaussianERPDataset: no samples found for split='%s' under '%s'",
                self.split,
                self.data_root,
            )

        return all_samples

    def _cache_subdir(self) -> Path | None:
        """Return the parameter-specific cache subdirectory (or None if no cache)."""
        if self.cache_dir is None:
            return None
        # Encode all preprocessing parameters in the subdirectory name so that
        # changing any parameter does not silently reuse incompatible cached files.
        subdir_name = (
            f"ns{self.n_shells}_H{self.H}_W{self.W}"
            f"_c{self.cutoff_sigma}"
            f"_p{self.r_near_pct}-{self.r_far_pct}"
        )
        return self.cache_dir / subdir_name

    def _cache_path_for(self, ply_path: Path) -> Path | None:
        """Return the .npy cache path for *ply_path* (None if caching disabled).

        Mirrors the source directory structure so the cache is human-readable:
            <cache_subdir>/<category>/train|test/<id>.npy
        matching:
            <data_root>/<category>/train|test/<id>/point_cloud.ply
        """
        subdir = self._cache_subdir()
        if subdir is None:
            return None

        # rel = <category>/train|test/<id>/point_cloud.ply
        try:
            rel = ply_path.resolve().relative_to(self.data_root.resolve())
        except ValueError:
            # Fallback: use just the immediate parent name as the stem
            rel = Path(ply_path.parent.name) / ply_path.name

        # Drop the point_cloud.ply filename; use the object-id directory as stem
        # rel.parent = <category>/train|test/<id>  →  cache: <category>/train|test/<id>.npy
        return subdir / rel.parent.with_suffix(".npy")

    def _load_or_compute(self, ply_path: Path) -> np.ndarray:
        """Load a cached ERP, or compute and cache it if absent.

        Returns:
            erp: (N_shells, H, W) float32 numpy array.
        """
        cache_path = self._cache_path_for(ply_path)

        if cache_path is not None and cache_path.exists():
            erp = np.load(str(cache_path))
            # Validate cached shape matches current parameters
            expected = (self.n_shells, self.H, self.W)
            if erp.shape == expected:
                return erp
            logger.warning(
                "Cache shape mismatch for '%s': expected %s, got %s — recomputing.",
                ply_path,
                expected,
                erp.shape,
            )

        # Compute radiance field ERP
        erp = gaussian_ply_to_erp(
            ply_path=ply_path,
            n_shells=self.n_shells,
            H=self.H,
            W=self.W,
            cutoff_sigma=self.cutoff_sigma,
            batch_size=self.batch_size_rf,
            r_near_pct=self.r_near_pct,
            r_far_pct=self.r_far_pct,
            device=None,  # auto-detect: CUDA if available, else CPU
        )

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), erp)

        return erp


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(config: dict) -> dict[str, DataLoader]:
    """Build train, val, and test DataLoaders from a YAML config dict.

    The config dict is expected to contain a 'data' sub-dict (as loaded from
    a YAML experiment file).  All keys are read from config['data'].

    Config keys:
        data_root     : str/Path — path to the modelsplat_ply root directory.
        dataset       : 'mn10' or 'mn40' — selects category list
                        (ignored if 'categories' is also given).
        categories    : list[str] — explicit category list (overrides 'dataset').
        n_shells      : int  — number of ERP shells (default 8).
        erp_height    : int  — ERP height (default 256).
        erp_width     : int  — ERP width  (default 512).
        cutoff_sigma  : float — Gaussian truncation (default 3.0).
        r_near_pct    : float — inner shell percentile (default 5.0).
        r_far_pct     : float — outer shell percentile (default 95.0).
        cache_dir     : str/Path or None — cache directory (default None).
        batch_size    : int  — samples per batch (default 32).
        num_workers   : int  — DataLoader worker processes (default 4).
        val_fraction  : float — validation fraction (default 0.2).
        seed          : int  — random seed (default 42).

    Args:
        config: Dict loaded from a YAML experiment configuration file.

    Returns:
        A dict with keys 'train', 'val', 'test', each a DataLoader.
    """
    data_cfg = config.get("data", config)  # support flat config or nested under 'data'

    # Data root
    data_root = Path(data_cfg["data_root"])

    # Category list
    if "categories" in data_cfg:
        categories = list(data_cfg["categories"])
    else:
        dataset_name = data_cfg.get("dataset", "mn10").lower()
        if dataset_name in ("mn10", "modelnet10"):
            categories = MODELNET10_CATEGORIES
        elif dataset_name in ("mn40", "modelnet40"):
            categories = MODELNET40_CATEGORIES
        else:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'; use 'mn10' or 'mn40', "
                "or provide an explicit 'categories' list."
            )

    # Preprocessing parameters
    n_shells     = int(data_cfg.get("n_shells",    8))
    H            = int(data_cfg.get("erp_height", 256))
    W            = int(data_cfg.get("erp_width",  512))
    cutoff_sigma = float(data_cfg.get("cutoff_sigma", 3.0))
    r_near_pct   = float(data_cfg.get("r_near_pct",  5.0))
    r_far_pct    = float(data_cfg.get("r_far_pct",  95.0))
    batch_size_rf = int(data_cfg.get("batch_size_rf", 4096))

    cache_dir_raw = data_cfg.get("cache_dir", None)
    cache_dir = Path(cache_dir_raw) if cache_dir_raw is not None else None

    # DataLoader parameters
    batch_size   = int(data_cfg.get("batch_size",  32))
    num_workers  = int(data_cfg.get("num_workers",  4))

    # Support both val_fraction (0.2) and train_val_split (0.8) keys
    if "val_fraction" in data_cfg:
        val_fraction = float(data_cfg["val_fraction"])
    elif "train_val_split" in data_cfg:
        val_fraction = 1.0 - float(data_cfg["train_val_split"])
    else:
        val_fraction = 0.2

    # Seed may live at the top-level config or inside the data section
    seed = int(data_cfg.get("seed", config.get("seed", 42)))

    loaders: dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        ds = GaussianERPDataset(
            data_root=data_root,
            categories=categories,
            split=split,
            n_shells=n_shells,
            H=H,
            W=W,
            cutoff_sigma=cutoff_sigma,
            batch_size_rf=batch_size_rf,
            r_near_pct=r_near_pct,
            r_far_pct=r_far_pct,
            cache_dir=cache_dir,
            augment_train=True,
            val_fraction=val_fraction,
            seed=seed,
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
# CLI entry point — used by scripts/preprocess_all.sh
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Pre-compute and cache radiance field ERP tensors for all PLY files "
            "in the ModelSplat dataset.  Run this once before training."
        )
    )
    parser.add_argument("--data_root",  type=Path, required=True,
                        help="Path to modelsplat_ply root directory.")
    parser.add_argument("--cache_dir",  type=Path, required=True,
                        help="Directory to write .npy ERP cache files.")
    parser.add_argument("--dataset",    type=str,  default="modelnet10",
                        choices=["modelnet10", "modelnet40", "mn10", "mn40"],
                        help="Dataset variant (default: modelnet10).")
    parser.add_argument("--n_shells",   type=int,  default=8,
                        help="Number of radiance field shells (default: 8).")
    parser.add_argument("--erp_height", type=int,  default=256,
                        help="ERP height in pixels (default: 256).")
    parser.add_argument("--erp_width",  type=int,  default=512,
                        help="ERP width in pixels (default: 512).")
    parser.add_argument("--pipeline",   type=str,  default="radiance_field",
                        help="Pipeline type (informational, default: radiance_field).")
    parser.add_argument("--r_near_pct", type=float, default=5.0)
    parser.add_argument("--r_far_pct",  type=float, default=95.0)
    parser.add_argument("--cutoff_sigma", type=float, default=3.0)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers for cache warm-up (default: 0).")
    args = parser.parse_args()

    cfg = {
        "data": {
            "data_root":    str(args.data_root),
            "cache_dir":    str(args.cache_dir),
            "dataset":      args.dataset,
            "n_shells":     args.n_shells,
            "erp_height":   args.erp_height,
            "erp_width":    args.erp_width,
            "r_near_pct":   args.r_near_pct,
            "r_far_pct":    args.r_far_pct,
            "cutoff_sigma": args.cutoff_sigma,
            "batch_size":   1,
            "num_workers":  args.num_workers,
            "val_fraction": 0.2,
            "seed":         42,
        }
    }

    logger.info("Pre-computing ERP cache for dataset=%s  n_shells=%d  H=%d  W=%d",
                args.dataset, args.n_shells, args.erp_height, args.erp_width)
    logger.info("data_root : %s", args.data_root)
    logger.info("cache_dir : %s", args.cache_dir)

    loaders = build_dataloaders(cfg)

    for split, loader in loaders.items():
        total = len(loader.dataset)
        logger.info("Warming cache for split='%s' (%d samples)...", split, total)
        for i, _ in enumerate(loader.dataset):
            if (i + 1) % 100 == 0 or (i + 1) == total:
                logger.info("  %d / %d", i + 1, total)

    logger.info("Cache warm-up complete.")

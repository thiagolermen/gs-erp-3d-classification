"""
Download, extract, and configure the ModelSplat dataset (ShapeSplats/ModelNet_Splats).

Downloads per-category zip files from HuggingFace, extracts them into
gs_data/modelsplat/modelsplat_ply/, and writes dataset configs for ModelNet10/40.

Usage:
    # Download everything (MN40, ~40 GB)
    python scripts/download_modelsplat.py --token <HF_TOKEN>

    # Download only ModelNet10 categories (~15 GB)
    python scripts/download_modelsplat.py --token <HF_TOKEN> --mn10-only

    # Download specific categories
    python scripts/download_modelsplat.py --token <HF_TOKEN> --categories sofa table toilet

    # Skip download, only extract already-present zips
    python scripts/download_modelsplat.py --token <HF_TOKEN> --skip-download

    # Use a different destination drive
    python scripts/download_modelsplat.py --token <HF_TOKEN> --dest D:/gs_data/modelsplat
"""

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


HF_REPO_ID = "ShapeSplats/ModelNet_Splats"
HF_REPO_TYPE = "dataset"

# All 40 ModelNet40 categories
MN40_CATEGORIES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
]

# ModelNet10 is a subset of ModelNet40
MN10_CATEGORIES = [
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
]


def download_categories(token: str, dest: Path, categories: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(categories)} categories from {HF_REPO_ID} -> {dest}")
    print(f"Categories: {', '.join(categories)}\n")

    for cat in categories:
        filename = f"{cat}.zip"
        target = dest / filename
        if target.exists():
            print(f"  [skip] {filename} already present ({target.stat().st_size // 1_048_576} MB).")
            continue
        print(f"  Downloading {filename} ...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=filename,
            token=token,
            local_dir=str(dest),
        )
        size_mb = (dest / filename).stat().st_size // 1_048_576
        print(f"  Done: {filename} ({size_mb} MB)")

    print("\nAll downloads complete.")


def extract(src: Path, out: Path, categories: list[str] | None = None) -> None:
    if categories:
        zips = sorted(src / f"{cat}.zip" for cat in categories if (src / f"{cat}.zip").exists())
    else:
        zips = sorted(src.glob("*.zip"))

    if not zips:
        print(f"No .zip files found in {src}. Skipping extraction.")
        return

    out.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {len(zips)} zip file(s) -> {out}")

    for zf_path in zips:
        target = out / zf_path.stem
        if target.exists():
            print(f"  [skip] {zf_path.name} already extracted.")
            continue
        print(f"  Extracting {zf_path.name} ...")
        with zipfile.ZipFile(zf_path, "r") as zf:
            zf.extractall(target)
        print(f"  Done: {zf_path.stem}/")

    print("Extraction complete.")


def print_next_steps(ply_dir: Path) -> None:
    """Print instructions for the next preprocessing step."""
    print("\nNext step: generate radiance field ERP cache:")
    print(f"  bash scripts/preprocess_all.sh")
    print(f"  # or:")
    print(f"  python -m src.preprocessing.dataset \\")
    print(f"      --data_root  {ply_dir} \\")
    print(f"      --cache_dir  data/processed/modelnet10/radiance_field \\")
    print(f"      --pipeline   radiance_field \\")
    print(f"      --dataset    modelnet10 \\")
    print(f"      --n_shells   8")


def summarise(src: Path, ply_dir: Path) -> None:
    zips = sorted(src.glob("*.zip"))
    total_zip_mb = sum(z.stat().st_size for z in zips) // 1_048_576

    mn10_present = [c for c in MN10_CATEGORIES if (src / f"{c}.zip").exists()]
    mn40_present = [c for c in MN40_CATEGORIES if (src / f"{c}.zip").exists()]

    print(f"\n--- Summary ---")
    print(f"  Zips downloaded : {len(zips)} ({total_zip_mb} MB)")
    print(f"  MN10 complete   : {len(mn10_present)}/10  {mn10_present}")
    print(f"  MN40 complete   : {len(mn40_present)}/40")

    mn10_missing = [c for c in MN10_CATEGORIES if c not in mn10_present]
    mn40_missing = [c for c in MN40_CATEGORIES if c not in mn40_present]
    if mn10_missing:
        print(f"  MN10 missing    : {mn10_missing}")
    if mn40_missing:
        print(f"  MN40 missing    : {mn40_missing}")

    if ply_dir.exists():
        total_plys = sum(1 for _ in ply_dir.rglob("point_cloud.ply"))
        print(f"  point_cloud.ply : {total_plys}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and set up the ModelSplat dataset.")
    parser.add_argument("--token", required=True, help="HuggingFace access token")
    parser.add_argument(
        "--dest",
        default="gs_data/modelsplat",
        help="Destination directory for raw zip downloads (default: gs_data/modelsplat). "
             "Can be an absolute path on another drive, e.g. D:/gs_data/modelsplat.",
    )
    parser.add_argument(
        "--mn10-only",
        action="store_true",
        help="Download only the 10 ModelNet10 categories.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        metavar="CAT",
        help="Download only specific categories (e.g. --categories sofa table toilet).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download; only extract zips already present in --dest.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction; only download.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # Resolve dest: support both relative (to project root) and absolute paths
    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = project_root / dest
    ply_dir = dest / "modelsplat_ply"

    # Determine target categories
    if args.categories:
        categories = args.categories
    elif args.mn10_only:
        categories = MN10_CATEGORIES
    else:
        categories = MN40_CATEGORIES

    if not args.skip_download:
        download_categories(args.token, dest, categories)

    if not args.skip_extract:
        extract(dest, ply_dir, categories)

    summarise(dest, ply_dir)
    print_next_steps(ply_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Quick sanity check: Python version, PyTorch, CUDA, and GPU properties.

Run inside the Docker container:
    python scripts/check_gpu.py
    # or via make:
    make check-gpu
"""

import sys
import torch


def main() -> None:
    print("=" * 55)
    print("  ERP-ViT — Environment Check")
    print("=" * 55)
    print(f"  Python       : {sys.version.split()[0]}")
    print(f"  PyTorch      : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"  CUDA         : {'available ✓' if cuda_ok else 'NOT available ✗'}")

    if not cuda_ok:
        print("\n  ✗ No CUDA device found.")
        print("    Check that nvidia-container-toolkit is installed on the host")
        print("    and that the container was started with --gpus all.")
        sys.exit(1)

    print(f"  CUDA version : {torch.version.cuda}")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  GPU count    : {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(
            f"  GPU {i}        : {props.name}"
            f"  |  {total // 1024**2:,} MB total"
            f"  |  {free  // 1024**2:,} MB free"
        )

    # Functional test: allocate a tiny tensor on GPU 0
    x = torch.ones(4, 4, device="cuda:0")
    assert x.sum().item() == 16.0
    print("\n  Tensor test  : OK ✓")

    # Report visible devices from environment
    import os
    vis = os.environ.get("NVIDIA_VISIBLE_DEVICES", "not set")
    print(f"  VISIBLE_DEVS : {vis}")

    print("=" * 55)
    print("  All checks passed — ready to train.")
    print("=" * 55)


if __name__ == "__main__":
    main()

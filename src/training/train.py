"""Main training loop for ERP 3D Classification experiments.

Implements the training protocol from the HSDC and SWHDC papers:

  - CrossEntropyLoss
  - Adam / AdamW optimiser, StepLR LR scheduler
  - Gradient clipping (max_norm = 1.0)
  - Mixed-precision training via torch.cuda.amp (CUDA only)
  - Early stopping with patience = 25 epochs
  - Per-epoch metrics logged to CSV and Python logging
  - Best and last checkpoints saved per run
  - Multi-GPU support via torch.nn.DataParallel

Supported backbones:
  - resnet34 + hsdc  → HSDCNet  (N_shells-channel radiance-field ERP input)
  - resnet50 + swhdc → SWHDCResNet (N_shells-channel radiance-field ERP input)

Entry point::

    python -m src.training.train --config configs/resnet34_hsdc_mn10.yaml

Output (under ``experiments/<run_name>/``):

    config.yaml          — copy of the config used
    train.log            — Python logging output
    metrics.csv          — epoch, train_loss, val_loss, train_acc, val_acc, lr
    best_checkpoint.pt   — state dict at best val accuracy
    last_checkpoint.pt   — state dict at final epoch

References:
    HSDC paper §III-A  — Stringhini et al., IEEE ICIP 2024
    SWHDC paper §IV-A  — Stringhini et al., SIBGRAPI 2024
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.backbones.resnet_hsdc import HSDCNet, SWHDCResNet
from src.preprocessing.augmentation import cutmix_erp
from src.preprocessing.dataset import build_dataloaders
from src.training.scheduler import EarlyStopping, build_optimizer, build_lr_scheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed all RNGs for fully reproducible training runs.

    Sets ``random``, ``numpy``, ``torch``, and CUDA seeds, and enables
    deterministic cuDNN mode.

    Args:
        seed: Integer seed; stored in every experiment config.

    References:
        HSDC paper §III-A; SWHDC paper §IV-A (fixed-seed reproducibility)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_model(cfg: dict) -> nn.Module:
    """Instantiate a model from an experiment config dict.

    Supported ``backbone + block`` combinations:

    - ``resnet34 + hsdc``  → :class:`HSDCNet`
    - ``resnet50 + swhdc`` → :class:`SWHDCResNet`

    The number of input channels is read from ``cfg['model']['in_channels']``
    (default 8, matching the N_shells=8 cascading-sphere radiance-field ERP).

    All models are initialised from scratch — ``pretrained=False`` is a
    hard constraint for fair comparison with the papers (CLAUDE.md §Rules).

    Args:
        cfg: Full experiment config dict (reads ``cfg['model']``).

    Returns:
        Uninitialised (randomly-weighted) :class:`torch.nn.Module`.

    Raises:
        ValueError: If the backbone/block combination is unknown.
    """
    model_cfg   = cfg["model"]
    backbone    = str(model_cfg["backbone"]).lower()
    block       = str(model_cfg["block"]).lower()
    num_classes = int(model_cfg["num_classes"])
    in_channels = int(model_cfg.get("in_channels", 8))
    dropout     = float(model_cfg.get("dropout", 0.0))

    if backbone == "resnet34" and block == "hsdc":
        return HSDCNet(in_channels=in_channels, num_classes=num_classes, dropout=dropout)

    if backbone == "resnet50" and block == "swhdc":
        return SWHDCResNet(in_channels=in_channels, num_classes=num_classes, dropout=dropout)

    raise ValueError(
        f"Unsupported backbone+block combination: '{backbone}' + '{block}'. "
        "Valid options: resnet34+hsdc, resnet50+swhdc."
    )


# ---------------------------------------------------------------------------
# MixUp augmentation (Zhang et al., 2018)
# ---------------------------------------------------------------------------


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp augmentation to a batch (Zhang et al., 2018).

    Blends pairs of samples: ``x_mix = λ·x_i + (1-λ)·x_j`` where
    ``λ ~ Beta(alpha, alpha)``.  Directly attacks the generalization gap
    by encouraging linear behaviour between training examples.

    Args:
        x:     (B, C, H, W) input batch.
        y:     (B,) integer labels.
        alpha: Beta distribution parameter.  0 disables MixUp.

    Returns:
        (x_mixed, y_a, y_b, lam) — mixed inputs, two label vectors, and λ.
    """
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1.0 - lam) * x[index]
    return x_mixed, y, y[index], lam


# ---------------------------------------------------------------------------
# CutMix batch-level wrapper (Yun et al., ICCV 2019)
# ---------------------------------------------------------------------------


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Batch-level CutMix wrapper calling augmentation.cutmix_erp per sample pair.

    Pastes a rectangular crop from a randomly permuted batch into the primary
    batch.  The horizontal axis wraps circularly (valid ERP symmetry).  Returns
    the mixed batch and the mean kept-fraction lambda across the batch.

    Args:
        x:     (B, C, H, W) input batch.
        y:     (B,) integer labels.
        alpha: Beta distribution parameter passed to :func:`cutmix_erp`.

    Returns:
        (x_mixed, y_a, y_b, lam_avg) — mixed inputs, primary labels, donor
        labels, and mean fraction of the primary sample retained.

    References:
        Yun et al., "CutMix: Training Strategy that Makes Use of Sample
        Mixing for Strong Classifiers", ICCV 2019.
    """
    if alpha <= 0.0:
        return x, y, y, 1.0
    B = x.shape[0]
    index = torch.randperm(B, device=x.device)
    x_np = x.cpu().numpy()
    mixed_list: list[np.ndarray] = []
    lam_list:   list[float]      = []
    for i in range(B):
        mixed, lam = cutmix_erp(x_np[i], x_np[index[i].item()], alpha=alpha)
        mixed_list.append(mixed)
        lam_list.append(lam)
    lam_avg = float(np.mean(lam_list))
    x_mixed = torch.from_numpy(np.stack(mixed_list)).to(x.device)
    return x_mixed, y, y[index], lam_avg


# ---------------------------------------------------------------------------
# Training / validation epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_clip: float,
    use_amp: bool = True,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
) -> tuple[float, float]:
    """Run one full training epoch.

    Applies:
    - Optional MixUp / CutMix augmentation (50/50 alternation when both > 0)
    - Forward pass (with optional AMP autocast)
    - CrossEntropyLoss (with optional label smoothing already baked into
      *criterion*)
    - Backward pass scaled by *scaler* (identity transform when AMP disabled)
    - Gradient clipping to ``max_norm = grad_clip``
    - Optimizer step

    Args:
        model:        Model in ``train()`` mode.
        loader:       Training :class:`DataLoader`.
        criterion:    Loss function (e.g. ``nn.CrossEntropyLoss``).
        optimizer:    Configured optimizer.
        scaler:       AMP :class:`GradScaler` (no-op when ``enabled=False``).
        device:       Target device.
        grad_clip:    Max gradient norm for :func:`torch.nn.utils.clip_grad_norm_`.
        use_amp:      Enable FP16 mixed-precision (CUDA only).
        mixup_alpha:  Beta distribution parameter for MixUp.  0 disables MixUp.
        cutmix_alpha: Beta distribution parameter for CutMix.  0 disables CutMix.

    Returns:
        Tuple ``(mean_loss, accuracy_percent)`` over all batches.

    References:
        HSDC paper §III-A — gradient clipping, AMP
        Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
        Yun et al., "CutMix", ICCV 2019
    """
    model.train()
    total_loss    = 0.0
    total_correct = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for inputs, targets in pbar:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        use_mix = mixup_alpha > 0.0 or cutmix_alpha > 0.0
        if use_mix:
            if mixup_alpha > 0.0 and cutmix_alpha > 0.0:
                fn = mixup_data if random.random() < 0.5 else cutmix_data
                inputs, targets_a, targets_b, lam = fn(inputs, targets, mixup_alpha)
            elif mixup_alpha > 0.0:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
            else:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, cutmix_alpha)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(inputs)
                loss = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
            preds = logits.argmax(dim=1)
            total_correct += (lam * (preds == targets_a).float().sum().item()
                              + (1.0 - lam) * (preds == targets_b).float().sum().item())
        else:
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(inputs)
                loss   = criterion(logits, targets)
            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        b              = inputs.size(0)
        total_loss    += loss.item() * b
        total_samples += b

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    mean_loss = total_loss / max(total_samples, 1)
    accuracy  = 100.0 * total_correct / max(total_samples, 1)
    return mean_loss, accuracy


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one evaluation epoch (val or test set).

    Gradients are disabled throughout (``@torch.no_grad()``).

    Args:
        model:     Model in ``eval()`` mode.
        loader:    Validation / test :class:`DataLoader`.
        criterion: Loss function.
        device:    Target device.

    Returns:
        Tuple ``(mean_loss, accuracy_percent)`` over all batches.
    """
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss   = criterion(logits, targets)

        b              = inputs.size(0)
        total_loss    += loss.item() * b
        preds          = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += b

    mean_loss = total_loss / max(total_samples, 1)
    accuracy  = 100.0 * total_correct / max(total_samples, 1)
    return mean_loss, accuracy


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    early_stopping: EarlyStopping,
    epoch: int,
    val_acc: float,
) -> None:
    """Serialise model, optimizer, scheduler, and early-stopping state.

    Unwraps :class:`torch.nn.DataParallel` automatically so that checkpoints
    are portable to single-GPU and CPU environments.

    Args:
        path:           Destination ``.pt`` file.
        model:          Model (may be wrapped in ``DataParallel``).
        optimizer:      Current optimizer.
        scheduler:      Current LR scheduler.
        early_stopping: Early-stopping tracker.
        epoch:          Current epoch number.
        val_acc:        Validation accuracy at this epoch.
    """
    model_state = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    torch.save(
        {
            "epoch": epoch,
            "val_acc": val_acc,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "early_stopping_state": early_stopping.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    early_stopping: EarlyStopping | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Restore model (and optionally optimizer / scheduler) from a checkpoint.

    Handles :class:`torch.nn.DataParallel` wrappers transparently.

    Args:
        path:            Path to the ``.pt`` checkpoint file.
        model:           Model to restore weights into.
        optimizer:       Optional optimizer to restore (for resuming training).
        scheduler:       Optional LR scheduler to restore.
        early_stopping:  Optional early-stopping state to restore.
        device:          Device to map tensors to (default: CPU).

    Returns:
        The raw checkpoint dict (contains ``'epoch'``, ``'val_acc'``, etc.).
    """
    ckpt  = torch.load(path, map_location=device or torch.device("cpu"))
    state = ckpt["model_state_dict"]

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if early_stopping is not None and "early_stopping_state" in ckpt:
        early_stopping.load_state_dict(ckpt["early_stopping_state"])

    return ckpt


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


def _setup_logging(log_path: Path) -> None:
    """Add a file handler to the root logger (avoids duplicate handlers)."""
    fmt     = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    root    = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler — add once
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(ch)

    # File handler — always add (each run gets its own file)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(fh)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def run_training(config_path: Path) -> dict[str, Any]:
    """Run a complete training experiment from a YAML config file.

    Workflow:

    1. Load and parse the YAML config.
    2. Set all RNG seeds for reproducibility.
    3. Create the experiment output directory; copy config.
    4. Set up Python logging to file + console.
    5. Build train/val/test DataLoaders from the ERP cache.
    6. Instantiate the model (from scratch; no ImageNet pretraining).
    7. Wrap in ``DataParallel`` if multiple GPUs are available.
    8. Build optimizer, LR scheduler, and early-stopping tracker.
    9. Epoch loop: train → validate → scheduler.step → LR clamp → log.
    10. Save best and last checkpoints; write per-epoch CSV metrics.
    11. Return a summary dict.

    Args:
        config_path: Path to the YAML experiment config file.

    Returns:
        Summary dict with keys ``run_name``, ``best_val_acc``, ``best_epoch``,
        ``final_epoch``, ``experiment_dir``.

    References:
        HSDC paper §III-A; SWHDC paper §IV-A
    """
    # ------------------------------------------------------------------
    # 1. Config
    # ------------------------------------------------------------------
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_name = str(cfg["run_name"])
    seed     = int(cfg.get("seed", 42))
    set_seed(seed)

    # ------------------------------------------------------------------
    # 2. Experiment directory
    # ------------------------------------------------------------------
    exp_root = Path(cfg.get("output", {}).get("experiments_dir", "experiments"))
    exp_dir  = exp_root / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, exp_dir / "config.yaml")

    # ------------------------------------------------------------------
    # 3. Logging
    # ------------------------------------------------------------------
    _setup_logging(exp_dir / "train.log")
    logger.info("=" * 70)
    logger.info("Run       : %s", run_name)
    logger.info("Config    : %s", config_path)
    logger.info("Seed      : %d", seed)

    # ------------------------------------------------------------------
    # 4. Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device    : %s", device)
    if device.type == "cuda":
        logger.info("GPU       : %s", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------------
    # 5. Data
    # ------------------------------------------------------------------
    loaders = build_dataloaders(cfg)
    logger.info(
        "Splits    : train=%d  val=%d  test=%d",
        len(loaders["train"].dataset),
        len(loaders["val"].dataset),
        len(loaders["test"].dataset),
    )

    # ------------------------------------------------------------------
    # 6. Model
    # ------------------------------------------------------------------
    model = build_model(cfg)

    if torch.cuda.device_count() > 1:
        logger.info("DataParallel across %d GPUs", torch.cuda.device_count())
        model = nn.DataParallel(model)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %s trainable", f"{n_params:,}")

    # ------------------------------------------------------------------
    # 7. Loss
    # ------------------------------------------------------------------
    train_cfg = cfg["training"]
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))

    class_weight_tensor: torch.Tensor | None = None
    if bool(train_cfg.get("class_weighted_loss", False)):
        # Compute inverse-frequency weights from the training set label counts.
        # weight[c] = N_total / (N_classes * N_c)  — balances minority classes.
        train_labels = [label for _, label in loaders["train"].dataset.samples]
        num_cls      = int(cfg["model"]["num_classes"])
        counts       = torch.zeros(num_cls)
        for lbl in train_labels:
            counts[lbl] += 1
        # Guard against zero-count classes (shouldn't happen on well-formed data)
        counts = counts.clamp(min=1)
        class_weight_tensor = (len(train_labels) / (num_cls * counts)).to(device)
        logger.info("Class weights (min=%.3f  max=%.3f)", class_weight_tensor.min().item(),
                    class_weight_tensor.max().item())

    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
        label_smoothing=label_smoothing,
    )

    # ------------------------------------------------------------------
    # 8. Optimizer, LR scheduler, early stopping
    # ------------------------------------------------------------------
    optimizer      = build_optimizer(model, cfg)
    scheduler      = build_lr_scheduler(optimizer, cfg)
    early_stopping = EarlyStopping(patience=int(train_cfg.get("early_stopping_patience", 25)))

    lr_min      = float(train_cfg.get("lr_min", 1e-7))
    grad_clip   = float(train_cfg.get("gradient_clip_norm", 1.0))
    max_epochs  = int(train_cfg["max_epochs"])
    save_every  = int(cfg.get("output", {}).get("save_every_n_epochs", 0))
    mixup_alpha  = float(train_cfg.get("mixup_alpha", 0.0))
    cutmix_alpha = float(train_cfg.get("cutmix_alpha", 0.0))

    # ------------------------------------------------------------------
    # 9. AMP
    # ------------------------------------------------------------------
    use_amp = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger.info("AMP       : %s", use_amp)
    logger.info("Max epochs: %d  |  early-stop patience: %d", max_epochs, early_stopping.patience)

    # ------------------------------------------------------------------
    # 10. CSV metrics file
    # ------------------------------------------------------------------
    csv_path = exp_dir / "metrics.csv"
    csv_fh   = open(csv_path, "w", newline="", encoding="utf-8")
    writer   = csv.writer(csv_fh)
    writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"])

    best_val_acc   = -1.0
    best_epoch     = 0
    best_ckpt_path = exp_dir / "best_checkpoint.pt"
    last_ckpt_path = exp_dir / "last_checkpoint.pt"

    t_start = time.time()

    # ------------------------------------------------------------------
    # 11. Epoch loop
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    final_epoch = 0

    for epoch in range(1, max_epochs + 1):
        final_epoch = epoch

        # Train
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler,
            device, grad_clip, use_amp, mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
        )

        # Validate
        val_loss, val_acc = eval_one_epoch(model, loaders["val"], criterion, device)

        # Step LR scheduler, then enforce floor
        scheduler.step()
        for pg in optimizer.param_groups:
            if pg["lr"] < lr_min:
                pg["lr"] = lr_min
        current_lr = optimizer.param_groups[0]["lr"]

        # Persist to CSV
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                         f"{train_acc:.4f}", f"{val_acc:.4f}", f"{current_lr:.2e}"])
        csv_fh.flush()

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            save_checkpoint(
                best_ckpt_path, model, optimizer, scheduler, early_stopping, epoch, val_acc
            )
            logger.info("  → best_checkpoint.pt  (val_acc=%.2f%%)", val_acc)

        # Save last checkpoint every epoch (cheap; enables resume)
        save_checkpoint(
            last_ckpt_path, model, optimizer, scheduler, early_stopping, epoch, val_acc
        )

        # Optional periodic checkpoint
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                exp_dir / f"checkpoint_ep{epoch:04d}.pt",
                model, optimizer, scheduler, early_stopping, epoch, val_acc,
            )

        # Early stopping (step first so counter reflects current epoch in log)
        stop = early_stopping.step(val_acc)

        # Log
        logger.info(
            "Ep %4d/%d | "
            "tr_loss=%.4f  tr_acc=%6.2f%%  "
            "val_loss=%.4f  val_acc=%6.2f%%  "
            "lr=%.2e  pat=%d/%d",
            epoch, max_epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            current_lr,
            early_stopping.counter, early_stopping.patience,
        )

        if stop:
            logger.info(
                "Early stopping at epoch %d — patience=%d exhausted.",
                epoch, early_stopping.patience,
            )
            break

    # ------------------------------------------------------------------
    # 12. Wrap up
    # ------------------------------------------------------------------
    csv_fh.close()
    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info(
        "Done. best_val_acc=%.2f%% at epoch %d.  Total time: %.0f s (%.1f min).",
        best_val_acc, best_epoch, elapsed, elapsed / 60,
    )

    return {
        "run_name":       run_name,
        "best_val_acc":   best_val_acc,
        "best_epoch":     best_epoch,
        "final_epoch":    final_epoch,
        "experiment_dir": str(exp_dir),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Train an ERP 3D classification model.")
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to the YAML experiment config file.",
    )
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()

"""Learning-rate scheduling and early stopping for ERP-ViT training.

Implements the exact optimiser protocol from the HSDC and SWHDC papers:

  - Adam (β₁=0.9, β₂=0.999) or AdamW optimiser
  - StepLR: multiply LR by γ=0.9 every 25 epochs; hard floor at lr_min=1e-7
  - Early stopping: patience = 25 epochs without val-accuracy improvement

An optional cosine-annealing scheduler is provided for Transformer-backbone
experiments where the Swin-T paper's schedule is preferred.

References:
    HSDC paper §III-A  — Stringhini et al., IEEE ICIP 2024
    SWHDC paper §IV-A  — Stringhini et al., SIBGRAPI 2024
"""

from __future__ import annotations

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Monitors validation accuracy and signals when to stop training.

    Each epoch call :meth:`step` with the current validation accuracy.
    The method returns ``True`` when patience is exhausted — i.e. validation
    accuracy has not improved for ``patience`` consecutive epochs.

    Args:
        patience:   Epochs without improvement before signalling stop.
                    Both papers use ``patience = 25`` (HSDC §III-A / SWHDC §IV-A).
        min_delta:  Minimum absolute change to qualify as an improvement.
                    Default 0.0 means any strict increase counts.

    References:
        HSDC paper §III-A; SWHDC paper §IV-A
    """

    def __init__(self, patience: int = 25, min_delta: float = 0.0) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self._best_val_acc: float = -1.0
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def step(self, val_acc: float) -> bool:
        """Update state with the latest validation accuracy.

        Args:
            val_acc: Current epoch validation accuracy in percent (0–100).

        Returns:
            ``True`` if training should stop (patience exhausted),
            ``False`` otherwise.
        """
        if val_acc > self._best_val_acc + self.min_delta:
            self._best_val_acc = val_acc
            self._counter = 0
            return False
        self._counter += 1
        return self._counter >= self.patience

    def improved(self, val_acc: float) -> bool:
        """Return ``True`` if *val_acc* would set a new best.

        Does NOT modify internal state; safe to call before :meth:`step`.
        """
        return val_acc > self._best_val_acc + self.min_delta

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_val_acc(self) -> float:
        """Best validation accuracy seen so far."""
        return self._best_val_acc

    @property
    def counter(self) -> int:
        """Number of consecutive epochs without improvement."""
        return self._counter

    # ------------------------------------------------------------------
    # Serialisation (checkpoint round-trip)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"best_val_acc": self._best_val_acc, "counter": self._counter}

    def load_state_dict(self, state: dict) -> None:
        self._best_val_acc = state["best_val_acc"]
        self._counter = state["counter"]


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------


def build_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    """Build an Adam or AdamW optimiser from the experiment config.

    Configuration keys read from ``cfg['training']``:

    - ``optimizer``    : ``'adam'`` (default) or ``'adamw'``
    - ``lr``           : Initial learning rate (default ``1e-4``)
    - ``weight_decay`` : L2 regularisation coefficient (default ``0.0``)

    Args:
        model: The model whose parameters will be optimised.
        cfg:   Full experiment config dict.

    Returns:
        Configured :class:`torch.optim.Optimizer`.

    References:
        HSDC paper §III-A — Adam, lr=1e-4, β₁=0.9, β₂=0.999
    """
    train_cfg      = cfg["training"]
    lr             = float(train_cfg.get("lr", 1e-4))
    weight_decay   = float(train_cfg.get("weight_decay", 0.0))
    optimizer_name = str(train_cfg.get("optimizer", "adam")).lower()

    kwargs: dict = dict(lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)

    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), **kwargs)
    if optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), **kwargs)

    raise ValueError(
        f"Unknown optimizer '{optimizer_name}'. Supported: 'adam', 'adamw'."
    )


# ---------------------------------------------------------------------------
# LR-scheduler factory
# ---------------------------------------------------------------------------


def build_lr_scheduler(
    optimizer: optim.Optimizer,
    cfg: dict,
) -> lr_sched.LRScheduler:
    """Build a learning-rate scheduler from the experiment config.

    Two main modes, selected by ``cfg['training']['lr_scheduler']``:

    - ``'step'`` (default): :class:`StepLR` with decay factor γ every N epochs.
      PyTorch's ``StepLR`` has no built-in LR floor; the calling code must
      clamp ``optimizer.param_groups[*]['lr'] ≥ lr_min`` after each
      ``scheduler.step()`` call.

    - ``'cosine'``: :class:`CosineAnnealingLR` with ``eta_min = lr_min``.

    Optional linear warmup (``lr_warmup_epochs > 0``): ramps the LR from
    ``start_factor × base_lr`` to ``base_lr`` over the first N epochs using
    :class:`LinearLR`, then hands off to the main scheduler via
    :class:`SequentialLR`.  The main scheduler's epoch count is offset by
    ``lr_warmup_epochs`` so that, e.g., ``StepLR(step_size=50)`` fires at
    epoch 50 of the *main phase* (epoch 60 of total training).

    Configuration keys read from ``cfg['training']``:

    - ``lr_scheduler``     : ``'step'`` or ``'cosine'`` (default ``'step'``)
    - ``lr_step_size``     : Decay period in epochs (default ``50``)
    - ``lr_gamma``         : Decay factor (default ``0.5``)
    - ``lr_min``           : LR floor for cosine / clamp reference (default ``1e-7``)
    - ``max_epochs``       : Used as ``T_max`` for cosine annealing
    - ``lr_warmup_epochs`` : Linear warmup length in epochs (default ``0`` = off)

    References:
        HSDC paper §III-A — StepLR, step_size=25, gamma=0.9, lr_min=1e-7
    """
    train_cfg      = cfg["training"]
    scheduler_name = str(train_cfg.get("lr_scheduler", "step")).lower()
    lr_min         = float(train_cfg.get("lr_min", 1e-7))
    warmup_epochs  = int(train_cfg.get("lr_warmup_epochs", 0))

    # Build the main (post-warmup) scheduler
    if scheduler_name == "step":
        main_sched: lr_sched.LRScheduler = lr_sched.StepLR(
            optimizer,
            step_size=int(train_cfg.get("lr_step_size", 50)),
            gamma=float(train_cfg.get("lr_gamma", 0.5)),
        )
    elif scheduler_name == "cosine":
        # Cosine T_max counts only the non-warmup epochs
        t_max = int(train_cfg["max_epochs"]) - warmup_epochs
        main_sched = lr_sched.CosineAnnealingLR(optimizer, T_max=max(t_max, 1), eta_min=lr_min)
    else:
        raise ValueError(
            f"Unknown lr_scheduler '{scheduler_name}'. Supported: 'step', 'cosine'."
        )

    if warmup_epochs <= 0:
        return main_sched

    # Prepend a linear warmup phase (10% → 100% of base LR)
    warmup = lr_sched.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return lr_sched.SequentialLR(
        optimizer,
        schedulers=[warmup, main_sched],
        milestones=[warmup_epochs],
    )

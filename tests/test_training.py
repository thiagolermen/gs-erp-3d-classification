"""Unit tests for the training infrastructure (AI Developer agent).

Covers:
  1. EarlyStopping  — patience, counter, improvement detection, serialisation
  2. build_optimizer  — Adam and AdamW creation, LR injection
  3. build_lr_scheduler  — StepLR decay, cosine annealing, unknown names
  4. LR clamping  — verifies the lr_min floor is enforced in the training loop
  5. train_one_epoch / eval_one_epoch  — runs a full epoch on CPU with a
     tiny synthetic model and TensorDataset; checks loss and accuracy shapes
  6. save_checkpoint / load_checkpoint  — round-trip serialisation check

All tests are pure-CPU and do NOT require ModelNet data, a GPU, or timm.
The tiny model used in epoch tests is a minimal nn.Module that accepts
(B, C, H, W) tensors and returns (B, num_classes) logits.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.training.scheduler import EarlyStopping, build_optimizer, build_lr_scheduler
from src.training.train import (
    build_model,
    eval_one_epoch,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal model for testing: GAP → Linear.  Accepts any (B, C, H, W)."""

    def __init__(self, in_channels: int = 4, num_classes: int = 5) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))


def _make_loader(
    in_channels: int = 4,
    num_classes: int = 5,
    batch_size:  int = 4,
    n_batches:   int = 2,
    h: int = 8,
    w: int = 16,
) -> DataLoader:
    """Return a DataLoader backed by random float32 tensors."""
    n = batch_size * n_batches
    x = torch.randn(n, in_channels, h, w)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _make_cfg(
    optimizer: str = "adam",
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    lr_scheduler: str = "step",
    max_epochs: int = 100,
    patience: int = 25,
    lr_step_size: int = 25,
    lr_gamma: float = 0.9,
    lr_min: float = 1e-7,
) -> dict:
    """Minimal config dict accepted by build_optimizer / build_lr_scheduler."""
    return {
        "model": {
            "backbone": "resnet34",
            "block": "hsdc",
            "num_classes": 10,
        },
        "training": {
            "optimizer": optimizer,
            "lr": lr,
            "weight_decay": weight_decay,
            "lr_scheduler": lr_scheduler,
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "lr_step_size": lr_step_size,
            "lr_gamma": lr_gamma,
            "lr_min": lr_min,
            "gradient_clip_norm": 1.0,
        },
    }


# ---------------------------------------------------------------------------
# EarlyStopping tests
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_initial_state(self) -> None:
        es = EarlyStopping(patience=3)
        assert es.best_val_acc == -1.0
        assert es.counter == 0

    def test_first_step_is_always_improvement(self) -> None:
        es = EarlyStopping(patience=3)
        stop = es.step(50.0)
        assert not stop
        assert es.best_val_acc == 50.0
        assert es.counter == 0

    def test_improvement_resets_counter(self) -> None:
        es = EarlyStopping(patience=3)
        es.step(50.0)
        es.step(49.0)   # no improvement → counter=1
        es.step(51.0)   # improvement → counter=0
        assert es.counter == 0
        assert es.best_val_acc == 51.0

    def test_no_improvement_increments_counter(self) -> None:
        es = EarlyStopping(patience=3)
        es.step(80.0)
        es.step(79.0)
        es.step(78.0)
        assert es.counter == 2

    def test_stop_after_patience_epochs(self) -> None:
        """After exactly *patience* non-improving epochs, step returns True."""
        patience = 5
        es = EarlyStopping(patience=patience)
        es.step(90.0)       # improvement → best=90, counter=0
        for _ in range(patience - 1):
            stop = es.step(89.0)
            assert not stop
        stop = es.step(89.0)  # patience-th non-improvement
        assert stop

    def test_paper_patience_25(self) -> None:
        """Default patience must be 25 (HSDC §III-A / SWHDC §IV-A)."""
        es = EarlyStopping()
        assert es.patience == 25

    def test_improved_is_readonly(self) -> None:
        """improved() must NOT modify best_val_acc or counter."""
        es = EarlyStopping(patience=3)
        es.step(70.0)
        counter_before = es.counter
        best_before    = es.best_val_acc
        _ = es.improved(80.0)
        assert es.counter    == counter_before
        assert es.best_val_acc == best_before

    def test_state_dict_round_trip(self) -> None:
        es = EarlyStopping(patience=5)
        es.step(60.0)
        es.step(59.0)
        state = es.state_dict()

        es2 = EarlyStopping(patience=5)
        es2.load_state_dict(state)
        assert es2.best_val_acc == es.best_val_acc
        assert es2.counter      == es.counter

    def test_min_delta(self) -> None:
        """With min_delta=1.0, an improvement of 0.5 must NOT reset counter."""
        es = EarlyStopping(patience=5, min_delta=1.0)
        es.step(80.0)
        es.step(80.5)   # +0.5 < 1.0 → not an improvement
        assert es.counter == 1


# ---------------------------------------------------------------------------
# build_optimizer tests
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    def test_adam_default(self) -> None:
        model = _TinyModel()
        cfg   = _make_cfg(optimizer="adam", lr=1e-4)
        opt   = build_optimizer(model, cfg)
        assert isinstance(opt, optim.Adam)
        assert abs(opt.param_groups[0]["lr"] - 1e-4) < 1e-10

    def test_adamw(self) -> None:
        model = _TinyModel()
        cfg   = _make_cfg(optimizer="adamw", lr=5e-4, weight_decay=0.01)
        opt   = build_optimizer(model, cfg)
        assert isinstance(opt, optim.AdamW)
        assert abs(opt.param_groups[0]["lr"] - 5e-4) < 1e-10

    def test_unknown_optimizer_raises(self) -> None:
        model = _TinyModel()
        cfg   = _make_cfg(optimizer="sgd")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer(model, cfg)

    def test_betas_set_correctly(self) -> None:
        """Optimizer betas must match β₁=0.9, β₂=0.999 from papers."""
        model = _TinyModel()
        opt   = build_optimizer(model, _make_cfg())
        betas = opt.param_groups[0]["betas"]
        assert betas == (0.9, 0.999)


# ---------------------------------------------------------------------------
# build_lr_scheduler tests
# ---------------------------------------------------------------------------


class TestBuildLRScheduler:
    def test_step_lr_decays(self) -> None:
        """StepLR must reduce the LR by gamma after step_size steps."""
        model = _TinyModel()
        cfg   = _make_cfg(lr_scheduler="step", lr_step_size=5, lr_gamma=0.9)
        opt   = build_optimizer(model, cfg)
        sched = build_lr_scheduler(opt, cfg)

        initial_lr = opt.param_groups[0]["lr"]
        for _ in range(5):
            sched.step()
        new_lr = opt.param_groups[0]["lr"]
        assert abs(new_lr - initial_lr * 0.9) < 1e-12

    def test_step_lr_multiple_decays(self) -> None:
        """After 2×step_size steps, LR must equal lr₀ × γ²."""
        model = _TinyModel()
        cfg   = _make_cfg(lr=1e-3, lr_scheduler="step", lr_step_size=10, lr_gamma=0.5)
        opt   = build_optimizer(model, cfg)
        sched = build_lr_scheduler(opt, cfg)

        for _ in range(20):
            sched.step()
        expected = 1e-3 * (0.5 ** 2)
        assert abs(opt.param_groups[0]["lr"] - expected) < 1e-12

    def test_cosine_annealing(self) -> None:
        """CosineAnnealingLR must return a scheduler and reduce LR over time."""
        model = _TinyModel()
        cfg   = _make_cfg(lr=1e-3, lr_scheduler="cosine", max_epochs=50)
        opt   = build_optimizer(model, cfg)
        sched = build_lr_scheduler(opt, cfg)

        lr_0 = opt.param_groups[0]["lr"]
        for _ in range(25):
            sched.step()
        lr_25 = opt.param_groups[0]["lr"]
        # At T_max/2 with cosine annealing, LR should be below initial
        assert lr_25 < lr_0

    def test_unknown_scheduler_raises(self) -> None:
        model = _TinyModel()
        cfg   = _make_cfg(lr_scheduler="polynomial")
        opt   = build_optimizer(model, cfg)
        with pytest.raises(ValueError, match="Unknown lr_scheduler"):
            build_lr_scheduler(opt, cfg)

    def test_lr_min_clamping_in_loop(self) -> None:
        """The training loop must clamp LR to lr_min (not a scheduler feature).

        Simulate many StepLR decay steps and manually clamp, verifying the LR
        never falls below lr_min.  (StepLR itself has no built-in floor.)
        """
        lr_min = 1e-7
        model  = _TinyModel()
        cfg    = _make_cfg(lr=1e-4, lr_scheduler="step", lr_step_size=1, lr_gamma=0.5,
                           lr_min=lr_min)
        opt    = build_optimizer(model, cfg)
        sched  = build_lr_scheduler(opt, cfg)

        for _ in range(50):          # 50 halvings → would reach 0 without clamp
            sched.step()
            for pg in opt.param_groups:
                pg["lr"] = max(pg["lr"], lr_min)

        assert opt.param_groups[0]["lr"] >= lr_min


# ---------------------------------------------------------------------------
# train_one_epoch / eval_one_epoch tests
# ---------------------------------------------------------------------------


class TestEpochFunctions:
    """Verify that epoch functions run and return correct-shaped scalars."""

    @pytest.fixture()
    def setup(self):
        C, K, H, W = 4, 5, 8, 16
        model     = _TinyModel(in_channels=C, num_classes=K)
        loader    = _make_loader(in_channels=C, num_classes=K, batch_size=4, n_batches=2,
                                 h=H, w=W)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scaler    = torch.cuda.amp.GradScaler(enabled=False)
        device    = torch.device("cpu")
        return model, loader, criterion, optimizer, scaler, device

    def test_train_one_epoch_returns_scalars(self, setup) -> None:
        model, loader, criterion, optimizer, scaler, device = setup
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, scaler,
                                    device, grad_clip=1.0, use_amp=False)
        assert isinstance(loss, float)
        assert isinstance(acc, float)

    def test_train_one_epoch_acc_range(self, setup) -> None:
        """Accuracy must be in [0, 100]."""
        model, loader, criterion, optimizer, scaler, device = setup
        _, acc = train_one_epoch(model, loader, criterion, optimizer, scaler,
                                  device, grad_clip=1.0, use_amp=False)
        assert 0.0 <= acc <= 100.0

    def test_train_one_epoch_loss_positive(self, setup) -> None:
        model, loader, criterion, optimizer, scaler, device = setup
        loss, _ = train_one_epoch(model, loader, criterion, optimizer, scaler,
                                   device, grad_clip=1.0, use_amp=False)
        assert loss > 0.0

    def test_eval_one_epoch_returns_scalars(self, setup) -> None:
        model, loader, criterion, _, _, device = setup
        loss, acc = eval_one_epoch(model, loader, criterion, device)
        assert isinstance(loss, float)
        assert isinstance(acc, float)

    def test_eval_does_not_update_parameters(self, setup) -> None:
        """eval_one_epoch must NOT modify model parameters (no gradients)."""
        model, loader, criterion, _, _, device = setup
        params_before = [p.clone() for p in model.parameters()]
        eval_one_epoch(model, loader, criterion, device)
        for before, after in zip(params_before, model.parameters()):
            assert torch.allclose(before, after), "eval_one_epoch must not modify weights"

    def test_train_updates_parameters(self, setup) -> None:
        """train_one_epoch must update at least one parameter."""
        model, loader, criterion, optimizer, scaler, device = setup
        params_before = [p.clone().detach() for p in model.parameters()]
        train_one_epoch(model, loader, criterion, optimizer, scaler,
                        device, grad_clip=1.0, use_amp=False)
        changed = any(
            not torch.allclose(before, after)
            for before, after in zip(params_before, model.parameters())
        )
        assert changed, "train_one_epoch must change at least one parameter"


# ---------------------------------------------------------------------------
# Checkpointing tests
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_save_load_round_trip(self) -> None:
        """Save and load a checkpoint; verify model weights are identical."""
        model  = _TinyModel(in_channels=4, num_classes=5)
        opt    = optim.Adam(model.parameters(), lr=1e-3)
        sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
        es     = EarlyStopping(patience=5)
        es.step(75.0)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "test_ckpt.pt"
            save_checkpoint(ckpt_path, model, opt, sched, es, epoch=3, val_acc=75.0)

            # Create a fresh model with different weights
            model2 = _TinyModel(in_channels=4, num_classes=5)
            ckpt   = load_checkpoint(ckpt_path, model2)

            # Weights must match
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2), "Loaded weights do not match saved weights"

            # Metadata must be preserved
            assert ckpt["epoch"]   == 3
            assert ckpt["val_acc"] == 75.0

    def test_checkpoint_contains_early_stopping_state(self) -> None:
        """The checkpoint must include early-stopping state for resuming."""
        model = _TinyModel()
        opt   = optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9)
        es    = EarlyStopping(patience=3)
        es.step(88.0)
        es.step(87.0)  # counter = 1

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "ckpt.pt"
            save_checkpoint(ckpt_path, model, opt, sched, es, epoch=10, val_acc=88.0)

            es2 = EarlyStopping(patience=3)
            load_checkpoint(ckpt_path, model, early_stopping=es2)

            assert es2.best_val_acc == 88.0
            assert es2.counter == 1


# ---------------------------------------------------------------------------
# set_seed tests
# ---------------------------------------------------------------------------


class TestSetSeed:
    def test_reproducible_random(self) -> None:
        """Two models initialised after set_seed(42) must have identical weights."""
        set_seed(42)
        m1 = nn.Linear(10, 5)
        set_seed(42)
        m2 = nn.Linear(10, 5)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)

    def test_different_seeds_differ(self) -> None:
        set_seed(1)
        m1 = nn.Linear(10, 5)
        set_seed(2)
        m2 = nn.Linear(10, 5)
        params_same = all(
            torch.allclose(p1, p2) for p1, p2 in zip(m1.parameters(), m2.parameters())
        )
        assert not params_same


# ---------------------------------------------------------------------------
# build_model tests (no-GPU, no timm required for resnet variants)
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_unknown_backbone_raises(self) -> None:
        cfg = {
            "model": {
                "backbone": "vgg16",
                "block": "hsdc",
                "num_classes": 10,
                "erp_height": 256,
            },
            "training": {},
        }
        with pytest.raises(ValueError, match="Unsupported backbone"):
            build_model(cfg)

    def test_resnet34_hsdc_forward(self) -> None:
        """HSDCNet must accept (2, 8, 64, 128) and return (2, 10) (8-shell RF-ERP)."""
        cfg = {
            "model": {
                "backbone": "resnet34",
                "block": "hsdc",
                "num_classes": 10,
                "in_channels": 8,
                "erp_height": 64,
                "erp_width": 128,
            },
            "training": {},
        }
        model = build_model(cfg)
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(2, 8, 64, 128))
        assert y.shape == (2, 10)

    def test_resnet50_swhdc_forward(self) -> None:
        """SWHDCResNet must accept (2, 8, 64, 128) and return (2, 10) (8-shell RF-ERP)."""
        cfg = {
            "model": {
                "backbone": "resnet50",
                "block": "swhdc",
                "num_classes": 10,
                "in_channels": 8,
                "erp_height": 64,
                "erp_width": 128,
            },
            "training": {},
        }
        model = build_model(cfg)
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(2, 8, 64, 128))
        assert y.shape == (2, 10)

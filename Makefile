# =============================================================================
# ERP-ViT 3D Classification — Makefile
# =============================================================================
# Primary interface for building, preprocessing, training, and evaluation.
# Designed to run on the Linux lab machine (GPU host).
#
# All commands execute inside the Docker container via `docker compose run`.
#
# Quick reference:
#   make build                                  # build Docker image
#   make check-gpu                              # verify CUDA access
#   make preprocess-all                         # generate all ERP caches
#   make train CONFIG=configs/resnet34_hsdc_mn10.yaml
#   make evaluate CONFIG=<yaml> CHECKPOINT=<pt>
#   make jupyter                                # Jupyter on port 8888
#   make test                                   # run pytest suite
#   make logs RUN_NAME=resnet34_hsdc_mn10_seed42
# =============================================================================

.DEFAULT_GOAL := help
SHELL         := /bin/bash
COMPOSE       := docker compose
IMAGE         := erp-vit:latest

# ── User-overridable variables ────────────────────────────────────────────────
# Set from command line:
#   make train CONFIG=configs/resnet34_hsdc_mn10.yaml
#   make evaluate CONFIG=configs/resnet34_hsdc_mn10.yaml CHECKPOINT=experiments/.../best_checkpoint.pt
#   make logs RUN_NAME=resnet34_hsdc_mn10_seed42
CONFIG     ?=
CHECKPOINT ?=
RUN_NAME   ?=

# =============================================================================
# Help
# =============================================================================
.PHONY: help
help:
	@echo ""
	@echo "  ERP-ViT 3D Classification — available targets"
	@echo ""
	@printf "  \033[1m%-32s\033[0m %s\n" "TARGET" "DESCRIPTION"
	@printf "  %-32s %s\n"  "──────────────────────────────" "───────────────────────────────────────────────"
	@printf "  \033[33m%-32s\033[0m %s\n" "[ Docker ]" ""
	@printf "  %-32s %s\n"  "build"       "Build (or rebuild) the Docker image"
	@printf "  %-32s %s\n"  "shell"       "Open interactive bash session with GPU"
	@printf "  %-32s %s\n"  "check-gpu"   "Verify PyTorch + CUDA + GPU inside container"
	@printf "  %-32s %s\n"  "test"        "Run pytest unit test suite"
	@printf "  %-32s %s\n"  "jupyter"     "Start Jupyter on port 8888 (SSH-tunnel ready)"
	@echo ""
	@printf "  \033[33m%-32s\033[0m %s\n" "[ Preprocessing  (run once) ]" ""
	@printf "  %-32s %s\n"  "preprocess-mn10-hsdc"   "12-channel ERP cache — ModelNet10"
	@printf "  %-32s %s\n"  "preprocess-mn10-swhdc"  " 1-channel ERP cache — ModelNet10"
	@printf "  %-32s %s\n"  "preprocess-mn40-hsdc"   "12-channel ERP cache — ModelNet40"
	@printf "  %-32s %s\n"  "preprocess-mn40-swhdc"  " 1-channel ERP cache — ModelNet40"
	@printf "  %-32s %s\n"  "preprocess-all"         "All four above (sequential)"
	@echo ""
	@printf "  \033[33m%-32s\033[0m %s\n" "[ Training ]" ""
	@printf "  %-32s %s\n"  "train CONFIG=<yaml>"    "Run a single experiment"
	@printf "  %-32s %s\n"  "baselines-mn10"         "Run both baselines on ModelNet10"
	@printf "  %-32s %s\n"  "baselines-mn40"         "Run both baselines on ModelNet40"
	@printf "  %-32s %s\n"  "baselines-all"          "All four baseline experiments (sequential)"
	@echo ""
	@printf "  \033[33m%-32s\033[0m %s\n" "[ Evaluation ]" ""
	@printf "  %-32s %s\n"  "evaluate CONFIG=<yaml> CHECKPOINT=<pt>"  "Test-set evaluation"
	@echo ""
	@printf "  \033[33m%-32s\033[0m %s\n" "[ Monitoring ]" ""
	@printf "  %-32s %s\n"  "logs RUN_NAME=<name>"  "tail -f on the run's train.log"
	@echo ""
	@printf "  \033[33m%-32s\033[0m %s\n" "[ Cleanup ]" ""
	@printf "  %-32s %s\n"  "clean"        "Remove all experiment outputs (asks confirmation)"
	@printf "  %-32s %s\n"  "clean-cache"  "Remove all ERP cache files (asks confirmation)"
	@echo ""

# =============================================================================
# Docker image
# =============================================================================
.PHONY: build
build:
	$(COMPOSE) build

# =============================================================================
# Interactive / diagnostic
# =============================================================================
.PHONY: shell
shell:
	$(COMPOSE) run --rm erp-vit bash

.PHONY: check-gpu
check-gpu:
	$(COMPOSE) run --rm erp-vit python scripts/check_gpu.py

.PHONY: test
test:
	$(COMPOSE) run --rm erp-vit python -m pytest tests/ -v --tb=short

.PHONY: jupyter
jupyter:
	@echo "──────────────────────────────────────────────"
	@echo "  Jupyter starting on port 8888"
	@echo "  SSH tunnel from home:"
	@echo "    ssh -L 8888:localhost:8888 user@lab-machine"
	@echo "  Then open:  http://localhost:8888"
	@echo "  Token:      erp-vit"
	@echo "──────────────────────────────────────────────"
	$(COMPOSE) run --rm -p 8888:8888 erp-vit \
		jupyter notebook \
		  --ip=0.0.0.0 \
		  --port=8888 \
		  --no-browser \
		  --allow-root \
		  --NotebookApp.token='erp-vit' \
		  notebooks/

# =============================================================================
# Preprocessing
# =============================================================================
.PHONY: preprocess-mn10-hsdc
preprocess-mn10-hsdc:
	@echo "► Preprocessing ModelNet10 — HSDC (12-channel ERP)"
	$(COMPOSE) run --rm erp-vit \
		python -m src.preprocessing.dataset \
		  --data_root data/raw/modelnet10 \
		  --cache_dir data/processed/modelnet10/hsdc \
		  --pipeline  hsdc

.PHONY: preprocess-mn10-swhdc
preprocess-mn10-swhdc:
	@echo "► Preprocessing ModelNet10 — SWHDC (1-channel depth ERP)"
	$(COMPOSE) run --rm erp-vit \
		python -m src.preprocessing.dataset \
		  --data_root data/raw/modelnet10 \
		  --cache_dir data/processed/modelnet10/swhdc \
		  --pipeline  swhdc

.PHONY: preprocess-mn40-hsdc
preprocess-mn40-hsdc:
	@echo "► Preprocessing ModelNet40 — HSDC (12-channel ERP)"
	$(COMPOSE) run --rm erp-vit \
		python -m src.preprocessing.dataset \
		  --data_root data/raw/modelnet40 \
		  --cache_dir data/processed/modelnet40/hsdc \
		  --pipeline  hsdc

.PHONY: preprocess-mn40-swhdc
preprocess-mn40-swhdc:
	@echo "► Preprocessing ModelNet40 — SWHDC (1-channel depth ERP)"
	$(COMPOSE) run --rm erp-vit \
		python -m src.preprocessing.dataset \
		  --data_root data/raw/modelnet40 \
		  --cache_dir data/processed/modelnet40/swhdc \
		  --pipeline  swhdc

.PHONY: preprocess-all
preprocess-all: preprocess-mn10-hsdc preprocess-mn10-swhdc \
                preprocess-mn40-hsdc preprocess-mn40-swhdc
	@echo "✓ All ERP caches generated."

# =============================================================================
# Training
# =============================================================================

# Guard that CONFIG is set
.PHONY: _require-config
_require-config:
ifndef CONFIG
	$(error CONFIG is not set — usage: make train CONFIG=configs/resnet34_hsdc_mn10.yaml)
endif

.PHONY: train
train: _require-config
	@echo "► Training: $(CONFIG)"
	$(COMPOSE) run --rm erp-vit \
		python -m src.training.train --config $(CONFIG)

# Convenience targets for each baseline
.PHONY: baselines-mn10
baselines-mn10:
	$(MAKE) train CONFIG=configs/resnet34_hsdc_mn10.yaml
	$(MAKE) train CONFIG=configs/resnet50_swhdc_mn10.yaml

.PHONY: baselines-mn40
baselines-mn40:
	$(MAKE) train CONFIG=configs/resnet34_hsdc_mn40.yaml
	$(MAKE) train CONFIG=configs/resnet50_swhdc_mn40.yaml

.PHONY: baselines-all
baselines-all: baselines-mn10 baselines-mn40
	@echo "✓ All baseline experiments complete."

# =============================================================================
# Evaluation
# =============================================================================

.PHONY: _require-eval
_require-eval:
ifndef CONFIG
	$(error CONFIG is not set — usage: make evaluate CONFIG=<yaml> CHECKPOINT=<pt>)
endif
ifndef CHECKPOINT
	$(error CHECKPOINT is not set — usage: make evaluate CONFIG=<yaml> CHECKPOINT=<pt>)
endif

.PHONY: evaluate
evaluate: _require-eval
	@echo "► Evaluating: $(CHECKPOINT)"
	$(COMPOSE) run --rm erp-vit \
		python -m src.training.evaluate \
		  --config     $(CONFIG) \
		  --checkpoint $(CHECKPOINT)

# =============================================================================
# Monitoring
# =============================================================================
.PHONY: logs
logs:
ifndef RUN_NAME
	$(error RUN_NAME is not set — usage: make logs RUN_NAME=resnet34_hsdc_mn10_seed42)
endif
	@test -f experiments/$(RUN_NAME)/train.log \
	  || (echo "Log not found: experiments/$(RUN_NAME)/train.log" && exit 1)
	tail -f experiments/$(RUN_NAME)/train.log

# =============================================================================
# Cleanup  (both ask for confirmation before deleting)
# =============================================================================
.PHONY: clean
clean:
	@printf "Remove ALL experiment outputs? [y/N]: "; \
	read ans; \
	if [ "$$ans" = "y" ]; then \
	  rm -rf experiments/*/; \
	  echo "Done."; \
	else \
	  echo "Aborted."; \
	fi

.PHONY: clean-cache
clean-cache:
	@printf "Remove ALL ERP cache files in data/processed/? [y/N]: "; \
	read ans; \
	if [ "$$ans" = "y" ]; then \
	  rm -rf data/processed/; \
	  echo "Done."; \
	else \
	  echo "Aborted."; \
	fi

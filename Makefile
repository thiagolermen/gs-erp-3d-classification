# =============================================================================
# GS-ERP 3D Classification — Makefile
# =============================================================================
# All commands run inside the Docker container.
#
# Quick reference:
#   make build                                        # build image
#   make check-gpu                                    # verify CUDA
#   make download TOKEN=<HF_TOKEN> [MN10_ONLY=1]     # download ModelSplat
#   make preprocess [DATASET=mn10]                    # generate ERP cache
#   make train CONFIG=configs/resnet34_hsdc_mn10.yaml # train one experiment
#   make resume CONFIG=configs/resnet34_hsdc_mn10.yaml # resume from last_checkpoint.pt
#   make resume CONFIG=<yaml> CHECKPOINT=<pt>          # resume from specific checkpoint
#   make baselines-all                                # all 4 experiments
#   make evaluate CONFIG=<yaml> CHECKPOINT=<pt>       # test-set evaluation
#   make jupyter                                      # Jupyter on port 8888
#   make test                                         # run pytest
#   make logs RUN_NAME=<name>                         # tail train.log
# =============================================================================

.DEFAULT_GOAL := help
SHELL         := /bin/bash
COMPOSE       := docker compose
SVC           := gs-erp

CONFIG     ?=
CHECKPOINT ?=
RUN_NAME   ?=
TOKEN      ?=
DATASET    ?= mn10
MN10_ONLY  ?=
CATEGORIES ?=
N_SHELLS   ?= 12
ADD_COLOR  ?= 1

GS_ROOT    := gs_data/modelsplat/modelsplat_ply

# =============================================================================
# Help
# =============================================================================
.PHONY: help
help:
	@echo ""
	@echo "  GS-ERP 3D Classification"
	@echo ""
	@printf "  \033[33m%-38s\033[0m %s\n" "[ Setup ]" ""
	@printf "  %-38s %s\n" "build"                  "Build the Docker image"
	@printf "  %-38s %s\n" "shell"                  "Interactive bash session with GPU"
	@printf "  %-38s %s\n" "check-gpu"              "Verify PyTorch + CUDA inside container"
	@printf "  %-38s %s\n" "test"                   "Run pytest unit tests"
	@echo ""
	@printf "  \033[33m%-38s\033[0m %s\n" "[ Data ]" ""
	@printf "  %-38s %s\n" "download TOKEN=<token>" "Download all ModelSplat categories (~40 GB)"
	@printf "  %-38s %s\n" "download TOKEN=<token> MN10_ONLY=1" "ModelNet10 only (~15 GB)"
	@printf "  %-38s %s\n" "download-extra TOKEN=<token>" "Download car, airplane, flower_pot"
	@echo ""
	@printf "  \033[33m%-38s\033[0m %s\n" "[ Preprocessing ]" ""
	@printf "  %-38s %s\n" "preprocess"             "New cache: mn10+mn40, 12 shells, RGB color"
	@printf "  %-38s %s\n" "preprocess DATASET=mn10" "ModelNet10 only"
	@printf "  %-38s %s\n" "preprocess DATASET=mn40" "ModelNet40 only"
	@printf "  %-38s %s\n" "preprocess CATEGORIES='car airplane'" "Specific categories only"
	@printf "  %-38s %s\n" "preprocess N_SHELLS=8 ADD_COLOR=0" "Legacy 8-shell no-color cache"
	@echo ""
	@printf "  \033[33m%-38s\033[0m %s\n" "[ Training ]" ""
	@printf "  %-38s %s\n" "train CONFIG=<yaml>"    "Train a single experiment"
	@printf "  %-38s %s\n" "resume CONFIG=<yaml>"   "Resume from last_checkpoint.pt"
	@printf "  %-38s %s\n" "resume CONFIG=<yaml> CHECKPOINT=<pt>" "Resume from specific checkpoint"
	@printf "  %-38s %s\n" "baselines-mn10"         "ResNet-34+HSDC and ResNet-50+SWHDC on MN10"
	@printf "  %-38s %s\n" "baselines-mn40"         "Same on MN40"
	@printf "  %-38s %s\n" "baselines-all"          "All 4 experiments (sequential)"
	@echo ""
	@printf "  \033[33m%-38s\033[0m %s\n" "[ Evaluation & Analysis ]" ""
	@printf "  %-38s %s\n" "evaluate CONFIG=<yaml> CHECKPOINT=<pt>" "Test-set evaluation"
	@printf "  %-38s %s\n" "jupyter"                "Jupyter on port 8888 (SSH-tunnel ready)"
	@printf "  %-38s %s\n" "logs RUN_NAME=<name>"   "tail -f the run's train.log"
	@echo ""
	@printf "  \033[33m%-38s\033[0m %s\n" "[ Cleanup ]" ""
	@printf "  %-38s %s\n" "clean"                  "Remove all experiment outputs"
	@printf "  %-38s %s\n" "clean-cache"            "Remove ERP cache (data/processed/)"
	@printf "  %-38s %s\n" "clean-data"             "Remove downloaded PLY files (gs_data/)"
	@echo ""

# =============================================================================
# Setup
# =============================================================================
.PHONY: build
build:
	$(COMPOSE) build

.PHONY: shell
shell:
	$(COMPOSE) run --rm $(SVC) bash

.PHONY: check-gpu
check-gpu:
	$(COMPOSE) run --rm $(SVC) python -c "\
import torch, sys; \
print(f'  Python  : {sys.version.split()[0]}'); \
print(f'  PyTorch : {torch.__version__}'); \
avail = torch.cuda.is_available(); \
print(f'  CUDA    : {torch.version.cuda}  OK' if avail else '  CUDA    : NOT available'); \
[print(f'  GPU {i}   : {torch.cuda.get_device_properties(i).name}') for i in range(torch.cuda.device_count())]; \
sys.exit(0 if avail else 1)"

.PHONY: test
test:
	$(COMPOSE) run --rm $(SVC) python -m pytest tests/ -v --tb=short

# =============================================================================
# Data download
# =============================================================================
.PHONY: download
download:
ifndef TOKEN
	$(error TOKEN is required — usage: make download TOKEN=<huggingface_token>)
endif
	$(COMPOSE) run --rm $(SVC) python scripts/download_modelsplat.py \
		--token $(TOKEN) \
		$(if $(MN10_ONLY),--mn10-only,)

.PHONY: download-extra
download-extra:
ifndef TOKEN
	$(error TOKEN is required — usage: make download-extra TOKEN=<huggingface_token>)
endif
	$(COMPOSE) run --rm $(SVC) python scripts/download_modelsplat.py \
		--token $(TOKEN) \
		--categories car airplane flower_pot

# =============================================================================
# Preprocessing
# =============================================================================

# Internal helper: resolve cache dir name from N_SHELLS and ADD_COLOR
_CACHE_SUFFIX  = $(if $(filter 0,$(ADD_COLOR)),radiance_field,radiance_field_n$(N_SHELLS)_rgb)
_COLOR_FLAG    = $(if $(filter 0,$(ADD_COLOR)),,--add_color)
_CATS_FLAG     = $(if $(CATEGORIES),--categories $(CATEGORIES),--dataset $(DATASET))
_CACHE_MN10    = data/processed/modelnet10/$(_CACHE_SUFFIX)
_CACHE_MN40    = data/processed/modelnet40/$(_CACHE_SUFFIX)

.PHONY: preprocess
preprocess:
ifdef CATEGORIES
	@echo "Preprocessing categories: $(CATEGORIES)  (N_SHELLS=$(N_SHELLS), color=$(if $(filter 0,$(ADD_COLOR)),no,yes))"
	$(COMPOSE) run --rm $(SVC) \
	  python scripts/preprocess_radiance_field.py \
	    --data_root  $(GS_ROOT) \
	    --cache_dir  $(_CACHE_MN10) \
	    --categories $(CATEGORIES) \
	    --n_shells   $(N_SHELLS) \
	    $(_COLOR_FLAG)
else
	@if [ "$(DATASET)" = "mn10" ] || [ "$(DATASET)" = "all" ]; then \
	  echo "Preprocessing ModelNet10 (N_SHELLS=$(N_SHELLS), color=$(if $(filter 0,$(ADD_COLOR)),no,yes))..."; \
	  $(COMPOSE) run --rm $(SVC) \
	    python scripts/preprocess_radiance_field.py \
	      --data_root  $(GS_ROOT) \
	      --cache_dir  $(_CACHE_MN10) \
	      --dataset    modelnet10 \
	      --n_shells   $(N_SHELLS) \
	      $(_COLOR_FLAG); \
	fi
	@if [ "$(DATASET)" = "mn40" ] || [ "$(DATASET)" = "all" ]; then \
	  echo "Preprocessing ModelNet40 (N_SHELLS=$(N_SHELLS), color=$(if $(filter 0,$(ADD_COLOR)),no,yes))..."; \
	  $(COMPOSE) run --rm $(SVC) \
	    python scripts/preprocess_radiance_field.py \
	      --data_root  $(GS_ROOT) \
	      --cache_dir  $(_CACHE_MN40) \
	      --dataset    modelnet40 \
	      --n_shells   $(N_SHELLS) \
	      $(_COLOR_FLAG); \
	fi
	@if [ "$(DATASET)" != "mn10" ] && [ "$(DATASET)" != "mn40" ] && [ "$(DATASET)" != "all" ]; then \
	  echo "Unknown DATASET=$(DATASET). Use mn10, mn40, or all."; exit 1; \
	fi
endif

# =============================================================================
# Training
# =============================================================================
.PHONY: train
train:
ifndef CONFIG
	$(error CONFIG is required — usage: make train CONFIG=configs/resnet34_hsdc_mn10.yaml)
endif
	@echo "Training: $(CONFIG)"
	$(COMPOSE) run --rm $(SVC) python -m src.training.train --config $(CONFIG)

.PHONY: resume
resume:
ifndef CONFIG
	$(error CONFIG is required — usage: make resume CONFIG=configs/resnet34_hsdc_mn10.yaml [CHECKPOINT=experiments/<run>/last_checkpoint.pt])
endif
	$(eval _RUN  := $(shell grep -m1 '^run_name:' $(CONFIG) | sed 's/run_name:[[:space:]]*//'))
	$(eval _CKPT := $(if $(CHECKPOINT),$(CHECKPOINT),experiments/$(_RUN)/last_checkpoint.pt))
	@echo "Resuming: $(CONFIG)  checkpoint: $(_CKPT)"
	$(COMPOSE) run --rm $(SVC) python -m src.training.train \
		--config $(CONFIG) \
		--resume $(_CKPT)

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
	@echo "All baseline experiments complete."

# =============================================================================
# Evaluation
# =============================================================================
.PHONY: evaluate
evaluate:
ifndef CONFIG
	$(error CONFIG is required — usage: make evaluate CONFIG=<yaml> CHECKPOINT=<pt>)
endif
ifndef CHECKPOINT
	$(error CHECKPOINT is required — usage: make evaluate CONFIG=<yaml> CHECKPOINT=<pt>)
endif
	@echo "Evaluating: $(CHECKPOINT)"
	$(COMPOSE) run --rm $(SVC) python -m src.training.evaluate \
		--config     $(CONFIG) \
		--checkpoint $(CHECKPOINT)

# =============================================================================
# Jupyter
# =============================================================================
.PHONY: jupyter
jupyter:
	@echo "Jupyter starting on port 8888"
	@echo "SSH tunnel: ssh -L 8888:localhost:8888 lermen@anubis"
	@echo "Browser:    http://localhost:8888   token: erp-vit"
	$(COMPOSE) run --rm -p 8888:8888 $(SVC) \
		jupyter notebook \
		  --ip=0.0.0.0 --port=8888 --no-browser \
		  --allow-root --NotebookApp.token='erp-vit' \
		  notebooks/

# =============================================================================
# Monitoring
# =============================================================================
.PHONY: logs
logs:
ifndef RUN_NAME
	$(error RUN_NAME is required — usage: make logs RUN_NAME=resnet34_hsdc_mn10_seed42)
endif
	@test -f experiments/$(RUN_NAME)/train.log \
	  || { echo "Not found: experiments/$(RUN_NAME)/train.log"; exit 1; }
	tail -f experiments/$(RUN_NAME)/train.log

# =============================================================================
# Cleanup
# =============================================================================
.PHONY: clean
clean:
	@printf "Remove all experiment outputs in experiments/? [y/N]: "; \
	read ans; [ "$$ans" = "y" ] && rm -rf experiments/*/ && echo "Done." || echo "Aborted."

.PHONY: clean-cache
clean-cache:
	@printf "Remove ERP cache in data/processed/? [y/N]: "; \
	read ans; [ "$$ans" = "y" ] && rm -rf data/processed/ && echo "Done." || echo "Aborted."

.PHONY: clean-data
clean-data:
	@printf "Remove ModelSplat PLY files in gs_data/? [y/N]: "; \
	read ans; [ "$$ans" = "y" ] && rm -rf gs_data/ && echo "Done." || echo "Aborted."

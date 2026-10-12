# Shortcuts for common tasks. `make help` lists the targets.
.DEFAULT_GOAL := help
.PHONY: help setup setup-gpu check test test-slow test-all bench bench-paper gate table runs plot lint

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

# torch from PyPI defaults to the CUDA build (~2.5 GB); the variant is chosen explicitly
# by index so a CPU machine does not download CUDA and a GPU machine does not end up on CPU.
TORCH_CPU  := https://download.pytorch.org/whl/cpu
TORCH_CU   := https://download.pytorch.org/whl/cu121
PYG_CPU    := https://data.pyg.org/whl/torch-2.4.1+cpu.html
PYG_CU     := https://data.pyg.org/whl/torch-2.4.1+cu121.html

setup: ## Create venv (uv) + CPU dependencies + LibSignal submodule
	git submodule update --init --recursive
	uv venv --python 3.10 --clear   # always a fresh environment
	uv pip install -e ".[cpu,dev]" \
	  --extra-index-url $(TORCH_CPU) --find-links $(PYG_CPU)
	@echo "Done. Verify with: make check"

setup-gpu: ## Same, with torch/torch-scatter for CUDA 12.1
	git submodule update --init --recursive
	uv venv --python 3.10 --clear   # always a fresh environment
	uv pip install -e ".[gpu,dev]" \
	  --extra-index-url $(TORCH_CU) --find-links $(PYG_CU)
	@echo "Done. Verify with: make check"

check: ## Quick environment health check (~15 s, Efficient-MP on JiNan)
	.venv/bin/tsc doctor
	.venv/bin/tsc train -a efficientmp -n cityflow_jinan3x4_4ph --seed 0 --replace

test: ## Fast tests (no simulator, ~15 s)
	.venv/bin/python -m pytest

test-slow: ## Only tests that start a simulator, CityFlow or SUMO (~30 s)
	.venv/bin/python -m pytest -m slow

test-all: ## All tests
	.venv/bin/python -m pytest -m ''

bench: ## T1 suite, realistic mode (yellow=5): baselines without training
	.venv/bin/tsc suite run t1_baselines
	.venv/bin/tsc table

bench-paper: ## T1 suite, benchmark mode (yellow=0), as in the papers
	.venv/bin/tsc suite run t1_paper
	.venv/bin/tsc table

table: ## Rebuild tables from stored runs (no simulation)
	.venv/bin/tsc table

runs: ## List stored runs (runs/t1): status, parameters, ATT
	.venv/bin/tsc runs

plot: ## Learning curves from runs/ -> runs/plots/ (also for runs in progress)
	.venv/bin/tsc plot
	.venv/bin/tsc plot --split train

gate: ## Core vs Advanced-CoLight on the same networks and seeds
	.venv/bin/tsc suite run p1_reference
	.venv/bin/tsc suite run p1_gate
	.venv/bin/tsc suite status p1_gate

lint: ## ruff
	.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format --check src/ tests/

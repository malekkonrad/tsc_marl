# Skróty do najczęstszych operacji. `make help` listuje cele.
.DEFAULT_GOAL := help
.PHONY: help setup setup-gpu check bench bench-paper table lint hpc-build

help: ## Pokaż tę pomoc
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

# torch z PyPI to domyślnie build CUDA (~2,5 GB). Wariant wybieramy jawnie indeksem,
# żeby na maszynie bez GPU nie ściągać CUDA, a na maszynie z GPU nie wylądować na CPU.
TORCH_CPU  := https://download.pytorch.org/whl/cpu
TORCH_CU   := https://download.pytorch.org/whl/cu121
PYG_CPU    := https://data.pyg.org/whl/torch-2.4.1+cpu.html
PYG_CU     := https://data.pyg.org/whl/torch-2.4.1+cu121.html

setup: ## Utwórz venv (uv) + zależności CPU + submoduł LibSignal
	git submodule update --init --recursive
	uv venv --python 3.10 --clear   # setup = świeże środowisko od zera
	uv pip install -e ".[cpu,dev]" \
	  --extra-index-url $(TORCH_CPU) --find-links $(PYG_CPU)
	@echo "Gotowe. Sprawdź: make check"

setup-gpu: ## To samo, ale torch/torch-scatter w wariancie CUDA 12.1
	git submodule update --init --recursive
	uv venv --python 3.10 --clear   # setup = świeże środowisko od zera
	uv pip install -e ".[gpu,dev]" \
	  --extra-index-url $(TORCH_CU) --find-links $(PYG_CU)
	@echo "Gotowe. Sprawdź: make check"

check: ## Szybki test zdrowia środowiska (~15 s, Efficient-MP na JiNan)
	scripts/run.sh -a efficientmp -n cityflow_jinan3x4_4ph --seed 0

bench: ## Tabela T1, tryb realistyczny (yellow=5) — baseline'y bez treningu
	python -m tsc_marl.bench -m realistic

bench-paper: ## Tabela T1, tryb benchmark (yellow=0) — jak liczą papiery
	python -m tsc_marl.bench -m benchmark

table: ## Przelicz tabele z istniejących CSV, bez uruchamiania symulacji
	python -m tsc_marl.bench --table-only

lint: ## ruff
	ruff check src/ && ruff format --check src/

hpc-build: ## Obraz Apptainer pod Cyfronet (ścieżka zapasowa — patrz deploy/hpc/)
	SIF=tsc-marl.sif IMAGE=tsc-marl:latest ./deploy/hpc/build_sif.sh

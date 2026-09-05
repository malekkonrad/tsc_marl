# Skróty do najczęstszych operacji. `make help` listuje cele.
.DEFAULT_GOAL := help
.PHONY: help setup setup-gpu check test test-slow test-all bench bench-paper gate table runs plot lint hpc-build

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
	.venv/bin/tsc doctor
	.venv/bin/tsc train -a efficientmp -n cityflow_jinan3x4_4ph --seed 0 --replace

test: ## Testy jednostkowe (bez silnika, ~2 s)
	.venv/bin/python -m pytest

test-slow: ## Same testy wymagające CityFlow (stawia świat, ~30 s)
	.venv/bin/python -m pytest -m slow

test-all: ## Wszystkie testy naraz
	.venv/bin/python -m pytest -m ''

bench: ## Siatka T1, tryb realistyczny (yellow=5) — baseline'y bez treningu
	.venv/bin/tsc suite run t1_baselines
	.venv/bin/tsc table

bench-paper: ## Siatka T1, tryb benchmark (yellow=0) — jak liczą papiery
	.venv/bin/tsc suite run t1_paper
	.venv/bin/tsc table

table: ## Przelicz tabele z zapisanych biegów, bez uruchamiania symulacji
	.venv/bin/tsc table

runs: ## Wypisz policzone biegi (runs/t1) — status, parametry, ATT
	.venv/bin/tsc runs

plot: ## Krzywe uczenia z runs/ → runs/plots/ (działa też dla biegu w toku)
	.venv/bin/tsc plot
	.venv/bin/tsc plot --split train

gate: ## GATE Fazy 1 — rdzeń vs Advanced-CoLight na tych samych sieciach i seedach
	.venv/bin/tsc suite run p1_reference
	.venv/bin/tsc suite run p1_gate
	.venv/bin/tsc suite status p1_gate

lint: ## ruff
	.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format --check src/ tests/

hpc-build: ## Obraz Apptainer pod Cyfronet (ścieżka zapasowa — patrz deploy/hpc/)
	SIF=tsc-marl.sif IMAGE=tsc-marl:latest ./deploy/hpc/build_sif.sh

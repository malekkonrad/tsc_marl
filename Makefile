# Convenience wrapper. `make help` lists targets.
IMAGE ?= tsc-marl:latest
SIF   ?= tsc-marl.sif
WORK  ?= $(PWD)

.DEFAULT_GOAL := help

.PHONY: help lock build smoke smoke-gpu shell shell-gpu train-gpu sif clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

lock: ## Regenerate the hash-locked requirements.txt from requirements.in (needs uv on host)
	uv pip compile requirements.in -o requirements.txt \
	  --generate-hashes --python-version 3.10 --python-platform x86_64-unknown-linux-gnu

build: ## Build the Docker image (single source of truth)
	docker build -t $(IMAGE) .

smoke: ## Run the environment health check (CPU)
	docker run --rm -v $(WORK):/workspace $(IMAGE) python /opt/smoke_test.py

smoke-gpu: ## Run the health check with the GPU (needs NVIDIA Container Toolkit)
	docker run --rm --gpus all -v $(WORK):/workspace $(IMAGE) python /opt/smoke_test.py

shell: ## Interactive shell, CPU only (laptop)
	docker run --rm -it -v $(WORK):/workspace $(IMAGE) bash

shell-gpu: ## Interactive shell with GPU (desktop)
	docker run --rm -it --gpus all -v $(WORK):/workspace $(IMAGE) bash

train-gpu: ## Example: run training on the GPU desktop (edit the command inside)
	docker run --rm --gpus all -e LIBSUMO_AS_TRACI=1 -v $(WORK):/workspace $(IMAGE) \
	  python -m src.train --seed 1 --out /workspace/results

sif: ## Convert the Docker image to an Apptainer .sif for HPC
	SIF=$(SIF) IMAGE=$(IMAGE) ./scripts/build_sif.sh

clean: ## Remove the local .sif and any docker-save tarballs
	rm -f $(SIF) *.tar

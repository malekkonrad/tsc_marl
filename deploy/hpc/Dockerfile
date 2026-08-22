# syntax=docker/dockerfile:1
#
# Single source of truth for the MARL / Traffic Signal Control environment.
# Runs GPU-accelerated on a host with an NVIDIA GPU + `--gpus all`, and
# CPU-only (unchanged) when `--gpus` is omitted. Convert this same image to an
# Apptainer .sif for HPC (see scripts/build_sif.sh).
#
# Python packages are installed with `uv` (fast, from Astral) for a big build
# speedup and hash-verified installs. All versions are pinned. To bump anything,
# edit the ARGs / requirements.in (then regenerate requirements.txt) and rebuild.

# CUDA 12.1 runtime + cuDNN 8 on Ubuntu 22.04 (system Python 3.10).
# Works for A100 (Athena, sm_80) and V100 (Ares, sm_70) and consumer GPUs.
# If this exact tag is unavailable, 12.1.0-cudnn8-runtime-ubuntu22.04 also works.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="tsc-marl" \
      org.opencontainers.image.description="Portable MARL Traffic Signal Control env (SUMO + CityFlow + LibSignal + PyG)"

ARG DEBIAN_FRONTEND=noninteractive

# --- Pinned versions (single place to change them) -----------------------------
ARG SUMO_VERSION=1.27.1
ARG TORCH_VERSION=2.3.1
ARG TORCHVISION_VERSION=0.18.1
ARG CUDA_WHEEL=cu121
ARG PYG_VERSION=2.6.1
ARG UV_VERSION=0.11.28
# Pinned commits (verified via `git ls-remote`). Repos below have no PyPI package.
ARG CITYFLOW_REF=81ee0f47659ca66177a71f81676691c58ee89184
ARG LIBSIGNAL_REF=127af9f93902778e556de2eedb2b606c4c9447e6
ARG RESCO_REF=f1ed9a174f8de41fc9d8689373b836bc882570dc

# --- uv (fast Python package manager) ------------------------------------------
# Copy the standalone uv binary from its pinned official image. No pip bootstrap.
# ARG expansion in --from requires the BuildKit syntax directive at the top (set).
COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /uvx /usr/local/bin/
# Install into the system interpreter (no venv inside the container).
ENV UV_SYSTEM_PYTHON=1
ENV UV_NO_CACHE=1

# --- System dependencies -------------------------------------------------------
# build-essential + cmake + swig: needed to compile CityFlow's pybind11 extension.
# libxerces/fox/gdal/proj/gl2ps: SUMO runtime/build libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget curl ca-certificates \
        python3 python3-dev python3-pip python3-venv \
        swig \
        libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev \
        libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Make `python` -> python3. (pip/setuptools kept modern for CityFlow's source build.)
RUN ln -sf /usr/bin/python3 /usr/local/bin/python && \
    python -m pip install --no-cache-dir --upgrade "pip==24.2" "setuptools==75.1.0" "wheel==0.44.0"

# --- PyTorch (CUDA build; falls back to CPU automatically when no GPU) ----------
# --index-url replaces PyPI with the torch wheel index (mirrors needed deps).
RUN uv pip install --system \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"

# --- PyTorch Geometric ---------------------------------------------------------
# Core PyG is pure-Python and robust. The compiled companions only ACCELERATE
# some ops (PyG 2.5+ works without them), so install them best-effort: a
# wheel-index miss must never break the build. --find-links points at the PyG
# wheel index for this exact torch+cuda combo.
RUN uv pip install --system "torch_geometric==${PYG_VERSION}"
RUN uv pip install --system \
        torch_scatter torch_sparse torch_cluster torch_spline_conv \
        --find-links "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_WHEEL}.html" \
    || echo "WARNING: optional PyG C++ extensions skipped (PyG still works without them)."

# --- SUMO (self-contained wheels: binaries + tools + data all included) ---------
RUN uv pip install --system \
        "eclipse-sumo==${SUMO_VERSION}" \
        "libsumo==${SUMO_VERSION}" \
        "traci==${SUMO_VERSION}" \
        "sumolib==${SUMO_VERSION}"
# Locate the wheel's data dir (contains bin/ tools/ data/) and expose it at a
# stable path so SUMO_HOME is deterministic regardless of Python version.
RUN SUMO_PKG_DIR="$(python -c 'import sumo, os; print(os.path.dirname(sumo.__file__))')" && \
    ln -s "${SUMO_PKG_DIR}" /opt/sumo && \
    echo "SUMO_HOME -> ${SUMO_PKG_DIR}"
ENV SUMO_HOME=/opt/sumo
ENV PATH=/opt/sumo/bin:${PATH}

# --- CityFlow (fast large-scale simulator; compiled C++ extension) --------------
# Built from source via plain pip: this is the community-proven path for
# CityFlow's pybind11/cmake build. (If you prefer, `uv pip install --system .`
# also works, but pip is kept here as the low-risk default for the one package
# we compile from source.)
RUN git clone https://github.com/cityflow-project/CityFlow.git /opt/CityFlow && \
    cd /opt/CityFlow && git checkout ${CITYFLOW_REF} && \
    pip install --no-cache-dir .

# --- LibSignal (cross-simulator MARL framework) + RESCO (benchmark scenarios) ---
# Research repos (no pip package). Cloned to /opt and put on PYTHONPATH. You
# typically work inside LibSignal or import from it; keep your own code in the
# mounted /workspace so you can iterate without rebuilding the image.
RUN git clone https://github.com/DaRL-LibSignal/LibSignal.git /opt/LibSignal && \
    cd /opt/LibSignal && git checkout ${LIBSIGNAL_REF}
RUN git clone https://github.com/Pi-Star-Lab/RESCO.git /opt/RESCO && \
    cd /opt/RESCO && git checkout ${RESCO_REF}
ENV PYTHONPATH=/opt/LibSignal:${PYTHONPATH}

# --- Remaining Python deps (RL glue, benchmarks, utils) ------------------------
# requirements.txt is a fully-locked, hash-verified file generated from
# requirements.in with `uv pip compile ... --generate-hashes`. uv enforces the
# hashes on install. Regenerate after editing requirements.in (see README / make lock).
COPY deploy/hpc/requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt

# --- Environment hygiene -------------------------------------------------------
# Deterministic cuBLAS workspace (needed for torch.use_deterministic_algorithms).
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
# Set LIBSUMO_AS_TRACI=1 at RUN time for ~8x faster headless SUMO (no GUI/parallel).

WORKDIR /workspace
COPY deploy/hpc/smoke_test.py /opt/smoke_test.py

# Quick health check any time: `docker run --rm tsc-marl python /opt/smoke_test.py`
CMD ["/bin/bash"]

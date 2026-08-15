# tsc-marl — portable MARL Traffic Signal Control environment

A **build-once, run-everywhere** environment for multi-agent reinforcement
learning on traffic signal control, for the thesis *"Utilizing a Multi-Agent
Approach in the Traffic Signal Control Problem"*.

One Docker image is the single source of truth. It runs:

- **CPU-only on your laptop** (development, small scenarios) — same image, no GPU,
- **GPU-accelerated on your desktop** (`--gpus all`) — full training runs,
- **on Cyfronet HPC** (Ares / Athena / Helios) — converted to an Apptainer `.sif`
  and launched under SLURM.

### What's inside the image
| Component | Version / ref | Role |
|---|---|---|
| SUMO (`eclipse-sumo`, `libsumo`, `traci`, `sumolib`) | 1.27.1 | microscopic simulator + Python APIs |
| CityFlow | commit `81ee0f4` | fast large/city-scale simulator for RL |
| LibSignal (DaRL-LibSignal) | commit `127af9f` | cross-simulator MARL framework (SUMO **and** CityFlow) |
| RESCO | commit `f1ed9a1` | standardized benchmark scenarios/baselines |
| sumo-rl | 1.4.5 | Gymnasium/PettingZoo interface to SUMO |
| PyTorch + torchvision | 2.3.1 / 0.18.1 (cu121) | deep learning (GPU or CPU) |
| PyTorch Geometric | 2.6.1 (+ optional C++ ext) | GNN layers (for CoLight-style models) |
| numpy | 1.26.4 (pinned <2.0) | pinned for the compiled-wheel ABI |

> **Why this stack:** LibSignal runs the *same* agent code on both SUMO and
> CityFlow with a calibrated conversion, so you can benchmark a GNN policy across
> simulators. Use **CityFlow** for fast city-scale RL, **SUMO + sumo-rl/RESCO**
> for microscopic realism and standardized benchmarks (Cologne, Ingolstadt,
> grid4x4, arterial4x4), and **PyG** for the graph model.

---

## 0. Prerequisites

| Machine | Needs |
|---|---|
| Laptop (CPU) | Docker |
| Desktop (GPU) | Docker + NVIDIA driver + **NVIDIA Container Toolkit** (see §2) |
| Cyfronet HPC | a PLGrid account + grant; jobs use Apptainer + SLURM (see §3). You build the `.sif` on the desktop and copy it over. |

No Python/conda needed on the host — everything lives in the container.

---

## 1. Quickstart (laptop / desktop)

```bash
make build          # build the image (first time: ~15–25 min, downloads a lot)
make smoke          # health check (CPU): verifies SUMO, CityFlow, torch, PyG...
make shell          # drop into an interactive shell with your code mounted
```

On the GPU desktop, confirm the GPU is visible inside the container:

```bash
make smoke-gpu      # should report "CUDA available - 1 GPU(s): NVIDIA ..."
```

`make smoke` runs [`scripts/smoke_test.py`](scripts/smoke_test.py); a green run
means the environment is healthy. Your project directory is bind-mounted at
`/workspace`, so you edit code on the host and run it in the container without
rebuilding.

---

## 2. GPU desktop — NVIDIA Container Toolkit

Docker needs the toolkit to pass the GPU through. Install the **host NVIDIA
driver** only (no CUDA toolkit on the host — the image brings CUDA), then:

```bash
# Add NVIDIA's repo (see docs.nvidia.com for your distro), then:
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify the GPU reaches a container:
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

Then run anything with `--gpus all` (see the `*-gpu` Make targets).

**Compatibility rule:** the container's CUDA (12.1) must be **≤** the CUDA
version your host driver supports (`nvidia-smi` shows it, top-right). If the host
driver is too old you'll get *"could not select device driver with capabilities:
[[gpu]]"* or a silent fall-back to CPU — update the driver or lower the CUDA
wheel (`CUDA_WHEEL`/base image) in the `Dockerfile`.

---

## 3. Cyfronet HPC (Ares / Athena / Helios)

HPC clusters **forbid Docker** (it needs a root daemon) and use **Apptainer**.
The flow is: build the image on your desktop → convert to `.sif` → copy to the
cluster's `$SCRATCH` → submit a SLURM job.

### 3.1 Get access (PLGrid)
1. Log in to the **PLGrid Portal** (portal.plgrid.pl), create/join a team.
2. Apply for a **pilot grant** to start (approved within ~1 hour: 1000 CPU-hours,
   10 GB, 72 h max task) — enough to validate the pipeline.
3. Request **service access** for the cluster you want:
   - **Athena** — A100 GPUs, **GPU-only**.
   - **Ares** — CPU partitions + a V100 GPU partition.
4. For real experiments, move to a **proper grant** with a stated GPU-hour budget.

Check what you have on the cluster:
```bash
hpc-grants        # your grants and their resource allocations
hpc-fs            # your storage / quotas
module spider apptainer   # confirm the exact container-runtime module name
```

### 3.2 Build the `.sif` and copy it over
Recommended (registry) path — reproducible and avoids moving a big tarball:
```bash
# on the desktop, after `make build`:
docker tag tsc-marl:latest ghcr.io/YOURUSER/tsc-marl:v1
docker push ghcr.io/YOURUSER/tsc-marl:v1
# then, from anywhere (even a cluster login node — pulling is allowed):
REGISTRY_REF=ghcr.io/YOURUSER/tsc-marl:v1 ./scripts/build_sif.sh
```
Offline path — no registry:
```bash
make sif                                   # docker save + apptainer build -> tsc-marl.sif
./scripts/sync_to_cyfronet.sh push-sif     # scp the .sif to $SCRATCH
```
> Build the `.sif` where you have Apptainer + privileges (the desktop, or a
> compute node with `--fakeroot`). **Do not** build it on a login node.

### 3.3 Sync your code and submit a job
```bash
# edit LOGIN/CLUSTER in scripts/sync_to_cyfronet.sh, then:
./scripts/sync_to_cyfronet.sh push         # code -> $SCRATCH/tsc-marl

# on the cluster, verify the image first:
apptainer exec --nv $SCRATCH/tsc-marl.sif python /opt/smoke_test.py

# then submit (edit the account line first — see the gotcha below):
sbatch scripts/slurm/train_athena_a100.sh          # A100
sbatch scripts/slurm/train_ares_v100.sh            # V100
sbatch scripts/slurm/sweep_ares_cpu_array.sh       # 5 seeds in parallel (CPU)

# pull results back:
./scripts/sync_to_cyfronet.sh pull-results
```

### 3.4 Cyfronet gotchas (these bite everyone)
- **Account suffix is mandatory.** `sbatch -A grantname` alone is **rejected**.
  Append the partition suffix: `-gpu-a100` on Athena, `-gpu`/`-cpu` on Ares. The
  SLURM scripts have the right suffix — just replace `plgYOURGRANT`.
- **Athena is GPU-only** and **bills every job in GPU-hours**; don't run CPU-only
  jobs there (risks account suspension).
- **`$SCRATCH` auto-purges:** on Athena, job dirs older than 7 days and data
  older than 30 days are deleted. Keep the `.sif` and datasets on `$SCRATCH` for
  speed, but **pull results off promptly**. Never rely on `$HOME` for large data.
- **Modules:** use `module spider <name>` (some clusters' `module avail` hides
  things). Athena's login node lacks dev libraries — compile on a compute node.
- **Helios** uses ARM (Grace-Hopper); an x86_64 `.sif` won't run there. Build a
  separate arm64 image if you target Helios. Athena/Ares are x86_64 (this image).

---

## 4. Running experiments

The image installs the heavy, painful-to-build pieces. **Your agent/training
code lives in `src/`** (mounted at `/workspace`), so iterate freely.

Two ways to drive training:
- **Your own entry point** — e.g. `python -m src.train --seed 1 ...`. The SLURM
  scripts and `make train-gpu` call this shape; wire it to your GNN/MARL loop.
- **LibSignal's runner** — it lives at `/opt/LibSignal` in the image with its own
  YAML configs (`CoLight.yml`, `FRAP.yml`, `MPLight.yml`, ...). Switch simulator
  (SUMO/CityFlow) and dataset in the config. Example inside the container:
  ```bash
  cd /opt/LibSignal && python run.py --thread_num 8
  ```
  Copy a LibSignal config into your repo and edit it there to keep runs tracked.

**Speed:** set `LIBSUMO_AS_TRACI=1` (already set in the Make/SLURM/compose
targets) for ~8× faster headless SUMO. Trade-off: no `sumo-gui` and no parallel
SUMO instances in one process — unset it (and use TraCI) when you need the GUI to
debug. CityFlow accepts `thread_num` for multithreaded speedups.

---

## 5. Reproducibility

- **Versions** are pinned in the `Dockerfile` (ARGs) and `requirements.txt`; the
  image + these files fully define the environment.
- **Seeds:** call `set_global_seed(seed)` from [`src/seeding.py`](src/seeding.py)
  at the start of every run, and pass the *same* seed to the simulator
  (SUMO: `--seed`; CityFlow: `"seed"` in the engine config). Run several seeds
  (e.g. 1–5, see the array job) and report mean ± std.
- **Determinism caveat:** PyTorch does not guarantee bit-identical results across
  versions/hardware or CPU-vs-GPU. Rely on the pinned image + fixed seeds +
  multi-seed reporting, not on bit-exactness.

---

## 6. Project layout

```
tsc-marl/
├── Dockerfile              # single source of truth (all pins here); installs via uv
├── requirements.in         # human-edited direct deps (edit this)
├── requirements.txt        # uv-generated, fully locked + hash-verified (don't hand-edit)
├── apptainer.def           # records the Docker -> .sif conversion recipe
├── docker-compose.yml      # local dev convenience (CPU + GPU services)
├── Makefile                # build / smoke / shell / sif shortcuts
├── scripts/
│   ├── smoke_test.py        # environment health check (run this first)
│   ├── build_sif.sh         # Docker image -> Apptainer .sif (registry/offline)
│   ├── sync_to_cyfronet.sh  # rsync/scp code, image, and results
│   └── slurm/
│       ├── train_athena_a100.sh     # A100 GPU job
│       ├── train_ares_v100.sh       # V100 GPU job
│       └── sweep_ares_cpu_array.sh  # multi-seed CPU array job
├── src/
│   ├── seeding.py           # reproducible seeding utility
│   └── __init__.py          # <- put your GNN/MARL agents + train.py here
├── configs/                 # your experiment YAMLs
└── nets/                    # your SUMO/CityFlow scenarios (see nets/README.md)
```

---

## 7. Changing versions (and the `uv` workflow)

Python packages are installed with **`uv`** inside the image (fast, hash-verified).
The dependency set is split by necessity:

- **torch / torchvision / PyG** and **SUMO** are pinned as explicit `ARG`s in the
  `Dockerfile` because they come from custom indexes (`download.pytorch.org`,
  `data.pyg.org`) — each is a separate, isolated `uv pip install` step.
- **Everything else** lives in `requirements.in` (direct deps) and is compiled
  into a fully locked, hash-pinned `requirements.txt` (35 packages incl.
  transitive) that `uv` installs and verifies.

To change the app deps: edit `requirements.in`, then regenerate the lock:
```bash
make lock        # = uv pip compile requirements.in -o requirements.txt --generate-hashes ...
make build
```
To change torch/PyG/SUMO: edit the `ARG`s at the top of the `Dockerfile`, then
`make build`. If you bump `torch`, the optional PyG C++ extensions must match a
wheel at `data.pyg.org/whl/torch-<TORCH>+<CUDA>.html` — if none exists, the build
skips them (PyG still works) or you can switch the base image to the CUDA `devel`
variant to compile them from source.

> **Why `uv` and not plain pip?** Two wins that matter for a thesis: (1) much
> faster image builds (parallel resolver + downloads), and (2) `requirements.txt`
> is now transitively locked with SHA256 hashes, so a rebuild months later
> resolves to the exact same bytes. The file is still a normal requirements file
> — plain `pip install -r requirements.txt` works as a fallback. `uv` does **not**
> replace the Dockerfile's orchestration of the git-cloned/compiled pieces
> (CityFlow's source build, LibSignal/RESCO clones) — those aren't PyPI packages.
> CityFlow is intentionally built with plain `pip` as the low-risk path for the
> one package compiled from source.
>
> Want to use `uv` on the **host** too (for editing/linting outside Docker)?
> `uv venv && uv pip install -r requirements.txt` gives you a matching venv — but
> the native bits (SUMO binaries, CityFlow's C++ engine) still want the container.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `make smoke-gpu` shows CPU-only on the desktop | Install NVIDIA Container Toolkit (§2); check host driver CUDA ≥ 12.1. |
| `could not select device driver ... [[gpu]]` | Toolkit not configured / daemon not restarted; re-run `nvidia-ctk runtime configure` + `systemctl restart docker`. |
| CityFlow build fails during `docker build` | Almost always a toolchain/Python mismatch; building inside this image (fixed GCC + Python 3.10) is the known-good path. Re-run `make build` from a clean context. |
| `SUMO_HOME` errors at runtime | Handled by the image (symlink `/opt/sumo`); if you customized the base image, ensure `import sumo` resolves and `$SUMO_HOME/{bin,tools,data}` exist. |
| Need the SUMO GUI but it won't open | You have `LIBSUMO_AS_TRACI=1` set — unset it and use TraCI; Libsumo has no GUI. |
| `sbatch` rejected on Cyfronet | Add the account suffix (`-gpu-a100` / `-gpu` / `-cpu`); replace `plgYOURGRANT`. |
| Results vanished on the cluster | `$SCRATCH` auto-purge — pull results after each job; don't leave them there. |
| PyG optional ext missing after a torch bump | Expected if no matching wheel; PyG runs without them, or use the CUDA `devel` base to build from source. |
| `uv` hash-mismatch / "hash does not match" on build | You hand-edited `requirements.txt`. Edit `requirements.in` instead and run `make lock` to regenerate the hashes. |

---

*Performance claims (CityFlow >20× vs SUMO; Libsumo ~8× vs TraCI) come from the
respective projects and depend on hardware/scenario — benchmark on your own
machines. Verify the Apptainer module name and current partition limits on
Cyfronet with `module spider` / `sinfo`, as cluster configs change over time.*

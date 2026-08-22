#!/bin/bash -l
# =====================================================================
# Cyfronet ATHENA - GPU training (NVIDIA A100). Athena is GPU-ONLY.
# Submit from a login node:   sbatch scripts/slurm/train_athena_a100.sh
# Override the grant/seed:    SEED=3 sbatch --account=plgYOURGRANT-gpu-a100 ...
# =====================================================================
#SBATCH --job-name=tsc-marl
#SBATCH --partition=plgrid-gpu-a100
# !!! IMPORTANT: on Athena the account MUST carry the -gpu-a100 suffix.
#     `sbatch -A grantname` alone will be REJECTED. Replace plgYOURGRANT below.
#SBATCH --account=plgYOURGRANT-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Load the container runtime. Confirm the exact module name first with:
#   module spider apptainer      (or: module spider singularity)
module load apptainer

# ---- Paths: keep large files on $SCRATCH, NOT $HOME ------------------
#   Athena auto-purge: job dirs older than 7 days, data older than 30 days.
#   Pull results off $SCRATCH promptly.
SIF="${SIF:-$SCRATCH/tsc-marl.sif}"
PROJECT="${PROJECT:-$SCRATCH/tsc-marl}"          # your synced repo (see sync_to_cyfronet.sh)
RESULTS="$SCRATCH/results/${SLURM_JOB_ID}"
mkdir -p "$RESULTS" logs

export APPTAINER_CACHEDIR="$SCRATCH/.apptainer/cache"
export APPTAINER_TMPDIR="$SCRATCH/.apptainer/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

SEED="${SEED:-1}"
echo "Job $SLURM_JOB_ID on $(hostname) | GPU: $CUDA_VISIBLE_DEVICES | seed=$SEED"

# ---- Run training inside the container -------------------------------
#   --nv          : expose the GPU to the container
#   --bind        : mount your code and a results dir
#   LIBSUMO_AS_TRACI=1 : ~8x faster headless SUMO (no GUI, no parallel sims)
#
# EDIT the python command to launch YOUR experiment. Two common shapes:
#   (a) your own entry point in the mounted repo:
#         python -m src.train --config configs/your_experiment.yml --seed $SEED
#   (b) LibSignal's runner (lives at /opt/LibSignal inside the image):
#         cd /opt/LibSignal && python run.py --thread_num $SLURM_CPUS_PER_TASK
srun apptainer exec --nv \
    --env LIBSUMO_AS_TRACI=1 \
    --bind "$PROJECT":/workspace \
    --bind "$RESULTS":/workspace/results \
    "$SIF" \
    python -m src.train --seed "$SEED" --out /workspace/results

echo "Done. Results in $RESULTS"

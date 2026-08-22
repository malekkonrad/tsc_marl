#!/bin/bash -l
# =====================================================================
# Cyfronet ARES - GPU training (NVIDIA V100).
# Submit:  sbatch scripts/slurm/train_ares_v100.sh
# =====================================================================
#SBATCH --job-name=tsc-marl
#SBATCH --partition=plgrid-gpu-v100
# On Ares GPU partitions the account carries the -gpu suffix.
#SBATCH --account=plgYOURGRANT-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
module load apptainer   # verify: module spider apptainer

SIF="${SIF:-$SCRATCH/tsc-marl.sif}"
PROJECT="${PROJECT:-$SCRATCH/tsc-marl}"
RESULTS="$SCRATCH/results/${SLURM_JOB_ID}"
mkdir -p "$RESULTS" logs
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer/cache"
export APPTAINER_TMPDIR="$SCRATCH/.apptainer/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

SEED="${SEED:-1}"
echo "Job $SLURM_JOB_ID on $(hostname) | GPU: $CUDA_VISIBLE_DEVICES | seed=$SEED"

srun apptainer exec --nv \
    --env LIBSUMO_AS_TRACI=1 \
    --bind "$PROJECT":/workspace \
    --bind "$RESULTS":/workspace/results \
    "$SIF" \
    python -m src.train --seed "$SEED" --out /workspace/results

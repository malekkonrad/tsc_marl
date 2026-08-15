#!/bin/bash -l
# =====================================================================
# Cyfronet ARES - CPU array job. Good for CityFlow (CPU sim) or SUMO
# sweeps where you launch one seed per array task in parallel.
# Submit:  sbatch scripts/slurm/sweep_ares_cpu_array.sh
# The %A_%a in filenames expands to jobID_arrayIndex.
# =====================================================================
#SBATCH --job-name=tsc-sweep
#SBATCH --partition=plgrid
# Ares CPU partitions carry the -cpu suffix.
#SBATCH --account=plgYOURGRANT-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=1-5          # 5 seeds -> 5 parallel tasks (seeds 1..5)
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
module load apptainer

SIF="${SIF:-$SCRATCH/tsc-marl.sif}"
PROJECT="${PROJECT:-$SCRATCH/tsc-marl}"
SEED="$SLURM_ARRAY_TASK_ID"                       # this task's seed
RESULTS="$SCRATCH/results/${SLURM_ARRAY_JOB_ID}/seed_${SEED}"
mkdir -p "$RESULTS" logs
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer/cache"
export APPTAINER_TMPDIR="$SCRATCH/.apptainer/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "Array task $SLURM_ARRAY_TASK_ID on $(hostname) | seed=$SEED"

# CPU-only: no --nv. LIBSUMO_AS_TRACI still speeds up SUMO on CPU.
srun apptainer exec \
    --env LIBSUMO_AS_TRACI=1 \
    --bind "$PROJECT":/workspace \
    --bind "$RESULTS":/workspace/results \
    "$SIF" \
    python -m src.train --seed "$SEED" --out /workspace/results

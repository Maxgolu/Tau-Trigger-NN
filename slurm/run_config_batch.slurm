#!/bin/bash
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --constraint=titan_xp
#SBATCH --job-name=tau-batch
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=1

set -euo pipefail

# Supply CONFIG_BATCH through sbatch; the same runner works for future sweeps.
: "${CONFIG_BATCH:?Set CONFIG_BATCH to a directory name under configs/}"
EXPERIMENT_BATCH="${EXPERIMENT_BATCH:-$CONFIG_BATCH}"
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/yandex/DLWorkShop2025b/maximg/envs/tau-trigger/bin/python}"

cd "$PROJECT_ROOT"

env PYTHONUNBUFFERED=1 "$PYTHON_BIN" src/train.py \
  --configs_dir "configs/$CONFIG_BATCH" \
  --data_dir . \
  --experiments_dir "experiments/$EXPERIMENT_BATCH" \
  --feature_cache_mb 512

env PYTHONUNBUFFERED=1 "$PYTHON_BIN" src/evaluate.py \
  --experiments_dir "experiments/$EXPERIMENT_BATCH" \
  --recalc \
  --num_bins 44 \
  --pt_min 10 \
  --pt_max 120 \
  --bin_var truth_pt

ARCHIVE="${EXPERIMENT_BATCH}-results-${SLURM_JOB_ID}.tar.gz"
tar -czf "$ARCHIVE" -C experiments "$EXPERIMENT_BATCH"
echo "Saved result archive: $PROJECT_ROOT/$ARCHIVE"

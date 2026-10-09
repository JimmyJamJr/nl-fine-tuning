#!/bin/bash
#SBATCH -J eval_nocurr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --partition=smallgpu
#SBATCH -A asaparov
#SBATCH -q normal

# Pass --export=ALL,TARGET_L=<L>,JOB_ID=<id> when sbatch'ing
set -uo pipefail
module load conda
# fa3_test (used for the May 2026 sweep) was purged and has no python binary.
conda activate "${NL_CONDA_ENV:-search}"
python3 -c "import torch" || { echo "FATAL: no torch in ${NL_CONDA_ENV:-search}"; exit 1; }

export SCRATCH=/scratch/gautschi/huan2073
export HF_HOME=$SCRATCH/model_cache
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR=$SCRATCH/triton_cache
export PYTHONUNBUFFERED=1

cd /home/huan2073/nl-fine-tuning/nl

L=${TARGET_L:?TARGET_L env var required}
J=${JOB_ID:?JOB_ID env var required}
OUT=${OUT_OVERRIDE:-/home/huan2073/nl-fine-tuning/nl/eval_at_L/eval_nocurr_L${L}_${J}.json}

# Final checkpoints are forced at the terminal step, which need not be a multiple
# of 500 (e.g. 16900, 27850), so the default interval silently skips them. Lower
# STEP_INTERVAL and set RESUME=1 to score only the stragglers.
STEP_INTERVAL=${STEP_INTERVAL:-500}
RESUME_FLAG=""
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"

echo "=== eval nocurr-L=${L} (job_${J}) at target L=${L} (interval=${STEP_INTERVAL} resume=${RESUME:-0}) ==="

# Eval HF rolling checkpoints + persistent_checkpoints
python3 eval_checkpoints.py \
    --job_id $J \
    --target_L $L \
    --max_input_size $((L * 6)) \
    --max_lookahead $L \
    --eval_samples 500 \
    --checkpoint_mode regular \
    --step_interval $STEP_INTERVAL \
    $RESUME_FLAG \
    --base_dir $SCRATCH/nl_output/search \
    --output $OUT \
    --model_name "Qwen/Qwen3-0.6B" \
    --cache_dir $HF_HOME

echo "DONE $(date)"

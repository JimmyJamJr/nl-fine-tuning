#!/bin/bash
#SBATCH -J eval_downstream
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=16:00:00
#SBATCH --partition=ai
#SBATCH -A asaparov
#SBATCH -q normal
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out

set -euo pipefail
module load conda
conda activate search

export SCRATCH="/scratch/gautschi/$USER"
export HF_HOME="$SCRATCH/model_cache"
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
mkdir -p "$SCRATCH/triton_cache" "$SCRATCH/model_cache"
export PYTHONUNBUFFERED=1
if [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
fi

cd /home/huan2073/nl-fine-tuning/nl/downstream

# Defaults: base + instruct + 6% low-L + 6% high-L on zebra + legal
MODELS="${MODELS:-base instruct_only 6pct_L16 6pct_L75}"
BENCHMARKS="${BENCHMARKS:-zebra_mc legal}"
# Full test set by default. Only cap the slow generation-heavy benchmarks.
# Override via env.
N_OVERRIDES="${N_OVERRIDES:-blocksworld=50 blocksworld_first=50 mystery_blocksworld=50 mystery_blocksworld_first=50 logistics=50 logistics_first=50 chess_mate=50 chess_mate_first=50 stepgame_gen=100 proofwriter_gen=200 proofwriter_cwa_gen=200}"
DEBUG_SAMPLES="${DEBUG_SAMPLES:-0}"  # set to e.g. 3 to print sample model outputs per benchmark
OUTPUT="${OUTPUT:-results/eval_${SLURM_JOB_ID}.json}"

# Cluster-specific paths (override via env if layout changes)
HF_CACHE="${HF_CACHE:-$SCRATCH/model_cache}"
PROMPTS_DIR="${PROMPTS_DIR:-$(dirname "$0")/prompts}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-$SCRATCH/nl_output/search}"
DATA_DIR="${DATA_DIR:-$SCRATCH/nl_eval}"

echo "Models:           $MODELS"
echo "Benchmarks:       $BENCHMARKS"
echo "n_overrides:      $N_OVERRIDES"
echo "Output:           $OUTPUT"
echo "hf_cache:         $HF_CACHE"
echo "prompts_dir:      $PROMPTS_DIR"
echo "checkpoints_root: $CHECKPOINTS_ROOT"
echo "data_dir:         $DATA_DIR"

python eval_downstream.py \
    --models $MODELS \
    --benchmarks $BENCHMARKS \
    --n-per-benchmark $N_OVERRIDES \
    --debug-samples "$DEBUG_SAMPLES" \
    --output "$OUTPUT" \
    --hf-cache "$HF_CACHE" \
    --prompts-dir "$PROMPTS_DIR" \
    --checkpoints-root "$CHECKPOINTS_ROOT" \
    --data-dir "$DATA_DIR"

echo "DONE $(date)"

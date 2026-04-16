#!/bin/bash
#SBATCH -J eval_downstream
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.out
#SBATCH --time=03:55:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --ntasks=1
#SBATCH -A asaparov
#SBATCH -p a30
#SBATCH -q standby

set -euo pipefail
module load conda
conda activate search

# Gilbreth-local scratch (Gautschi scratch is NOT shared across clusters)
export SCRATCH="/scratch/gilbreth/$USER"
export HF_HOME="$SCRATCH/model_cache"
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
mkdir -p slurm_logs "$SCRATCH/triton_cache" "$SCRATCH/model_cache"
export PYTHONUNBUFFERED=1
if [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
fi

cd "$SCRATCH/nl_eval"

# Default models use explicit 'name=path' aliases pointing at the Gilbreth-local
# copies that were rsync'd in from Gautschi. Override via env var.
MODELS="${MODELS:-base instruct_only=$SCRATCH/models/instruct_only 6pct_L16=$SCRATCH/models/6pct_L16 6pct_L75=$SCRATCH/models/6pct_L75}"
BENCHMARKS="${BENCHMARKS:-zebra_mc legal}"
# Full test set by default. Only cap the slow generation-heavy benchmarks.
N_OVERRIDES="${N_OVERRIDES:-game24=100 blocksworld=50 mystery_blocksworld=50 logistics=50 chess_mate=50 stepgame_gen=100 proofwriter_gen=200 proofwriter_cwa_gen=200}"
DEBUG_SAMPLES="${DEBUG_SAMPLES:-0}"  # set to e.g. 3 to print sample model outputs per benchmark
OUTPUT="${OUTPUT:-results/eval_gilbreth_${SLURM_JOB_ID}.json}"

# Cluster-specific paths
HF_CACHE="${HF_CACHE:-$SCRATCH/model_cache}"
PROMPTS_DIR="${PROMPTS_DIR:-$SCRATCH/nl_eval/prompts}"
# NOTE: MODEL_REGISTRY aliases rely on job_<id>/ dirs under this root. Gilbreth
# doesn't mount Gautschi's nl_output tree, so we use explicit name=/path aliases
# for MODELS instead (see above). CHECKPOINTS_ROOT is still passed in case a
# registry alias ever gets used here — point it at whatever exists locally.
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

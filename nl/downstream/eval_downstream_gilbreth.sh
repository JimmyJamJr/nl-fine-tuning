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

# Gilbreth launcher (A30 GPUs, standby QoS 4h walltime).
# Scratch is NOT shared with Gautschi — checkpoints must be rsync'd to Gilbreth
# separately, and model aliases use explicit `name=/scratch/gilbreth/...` paths
# (MODEL_REGISTRY's job_<id>/ references won't resolve here).

set -euo pipefail
module load conda
conda activate search

export SCRATCH="/scratch/gilbreth/$USER"
export HF_HOME="$SCRATCH/model_cache"
# Cache-only (pre-populated by cache_datasets.sh on Gilbreth).
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
mkdir -p slurm_logs "$SCRATCH/triton_cache" "$SCRATCH/model_cache"
export PYTHONUNBUFFERED=1

# Propagate HF auth so streaming datasets (Lichess/FOLIO gated) don't hit
# anonymous rate limits / auth errors.
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
if [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
fi

cd "$SCRATCH/nl_eval"

# Default models: explicit name=path aliases pointing at Gilbreth-local copies.
# Override via env var. Use the same alias names the paper tables use so output
# JSONs are directly comparable across clusters.
MODELS="${MODELS:-base instruct_only=$SCRATCH/models/instruct_only 6pct_L16=$SCRATCH/models/6pct_L16 6pct_L32=$SCRATCH/models/6pct_L32 6pct_L75=$SCRATCH/models/6pct_L75}"

# Default benchmarks: all 14 primary + 5 few-shot variants.
BENCHMARKS="${BENCHMARKS:-proofwriter proofwriter_cwa prontoqa_ood clutrr clutrr_fs stepgame folio logiqa logiqa_fs ruletaker ruletaker_fs logicbench_bqa logicbench_mcqa multilogieval multilogieval_fs nlgraph_gen legal zebra_mc zebra_mc_fs}"

N="${N:-1000}"
DEBUG_SAMPLES="${DEBUG_SAMPLES:-1}"
OUTPUT="${OUTPUT:-results/eval_gilbreth_${SLURM_JOB_ID}.json}"

HF_CACHE="${HF_CACHE:-$SCRATCH/model_cache}"
PROMPTS_DIR="${PROMPTS_DIR:-$SCRATCH/nl_eval/prompts}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-$SCRATCH/nl_output/search}"
DATA_DIR="${DATA_DIR:-$SCRATCH/nl_eval}"

echo "Models:      $MODELS"
echo "Benchmarks:  $BENCHMARKS"
echo "n:           $N"
echo "Output:      $OUTPUT"
echo "data_dir:    $DATA_DIR"

python eval_downstream.py \
    --models $MODELS \
    --benchmarks $BENCHMARKS \
    --n "$N" \
    --debug-samples "$DEBUG_SAMPLES" \
    --output "$OUTPUT" \
    --hf-cache "$HF_CACHE" \
    --prompts-dir "$PROMPTS_DIR" \
    --checkpoints-root "$CHECKPOINTS_ROOT" \
    --data-dir "$DATA_DIR"

echo "DONE $(date)"

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

export SCRATCH="${SCRATCH:-/scratch/gautschi/$USER}"
export HF_HOME="$SCRATCH/model_cache"
# Use cached datasets only (pre-populated by setup_cache_datasets.py).
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
mkdir -p "$SCRATCH/triton_cache" "$SCRATCH/model_cache"
export PYTHONUNBUFFERED=1
# Propagate HF auth so streaming datasets don't hit anonymous rate limits.
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
if [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
fi

cd /home/huan2073/nl-fine-tuning/nl/downstream

# Legacy Gautschi h100 launcher. For current runs prefer eval_full_n1000.sh
# (smallgpu, 1 GPU, 8h walltime, takes $MODEL env var, one model per job).
# This script is a multi-model runner kept for ad-hoc sweeps.

# Defaults: base + instruct + a curriculum-depth sweep
MODELS="${MODELS:-base instruct_only 6pct_L8 6pct_L16 6pct_L32 6pct_L48 6pct_L64 6pct_L75}"
# Default to the 14 primary benchmarks + 5 few-shot variants (19 total).
BENCHMARKS="${BENCHMARKS:-proofwriter proofwriter_cwa prontoqa_ood clutrr clutrr_fs stepgame folio logiqa logiqa_fs ruletaker ruletaker_fs logicbench_bqa logicbench_mcqa multilogieval multilogieval_fs nlgraph_gen legal zebra_mc zebra_mc_fs}"
N="${N:-1000}"
DEBUG_SAMPLES="${DEBUG_SAMPLES:-1}"  # print 1 sample per subtask for inspection
OUTPUT="${OUTPUT:-results/eval_${SLURM_JOB_ID}.json}"

# Cluster-specific paths (override via env if layout changes)
HF_CACHE="${HF_CACHE:-$SCRATCH/model_cache}"
PROMPTS_DIR="${PROMPTS_DIR:-/home/huan2073/nl-fine-tuning/nl/downstream/prompts}"
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
    --n "$N" \
    --debug-samples "$DEBUG_SAMPLES" \
    --output "$OUTPUT" \
    --hf-cache "$HF_CACHE" \
    --prompts-dir "$PROMPTS_DIR" \
    --checkpoints-root "$CHECKPOINTS_ROOT" \
    --data-dir "$DATA_DIR"

echo "DONE $(date)"

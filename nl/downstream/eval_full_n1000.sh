#!/bin/bash
#SBATCH -J eval_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --partition=smallgpu
#SBATCH -A asaparov
#SBATCH -q normal
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out

# Full downstream eval at n=1000. One model per job, 1 H100 preemptible.
# Expects MODEL env var set (e.g. MODEL=base) — the submit wrapper sets this.

set -euo pipefail
module load conda
conda activate search

export SCRATCH="${SCRATCH:-/scratch/gautschi/$USER}"
export HF_HOME="$SCRATCH/model_cache"
# Cache-only (pre-populated by setup_cache_datasets.py); fail loudly on miss.
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
mkdir -p slurm "$SCRATCH/triton_cache" "$SCRATCH/model_cache"
export PYTHONUNBUFFERED=1

if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

cd /home/huan2073/nl-fine-tuning/nl/downstream

: "${MODEL:?MODEL env var required. Accepted forms:
    - Registry key:        MODEL=base (see MODEL_REGISTRY in eval_downstream.py)
    - Alias + path:        MODEL=my_run=/scratch/path/to/checkpoint-123
    - Bare local path:     MODEL=/scratch/path/to/checkpoint-123
    - HuggingFace repo:    MODEL=Qwen/Qwen3-0.6B}"

# All 14 paper benchmarks (consolidated: each adapter returns both loglik+gen
# metrics in a single forward pass where applicable).
BENCHMARKS=(
    proofwriter proofwriter_cwa
    prontoqa_ood
    clutrr clutrr_fs
    stepgame
    # stepgame_gen — dropped; redundant with fused stepgame (log-lik + constrained-greedy gen
    # from one forward pass); was a pre-consolidation artifact, no canonical LLM method exists
    folio
    logiqa logiqa_fs
    ruletaker ruletaker_fs
    logicbench_bqa logicbench_mcqa
    multilogieval multilogieval_fs
    nlgraph_gen
    legal
    # chess_mate — dropped; 0% (exact + first_move) across all 12 models at 0.6-1.7B scale
    zebra_mc zebra_mc_fs
)

OUTPUT="results/eval_n1000_${MODEL}_${SLURM_JOB_ID}.json"
HF_CACHE="$SCRATCH/model_cache"
PROMPTS_DIR="/home/huan2073/nl-fine-tuning/nl/downstream/prompts"
CHECKPOINTS_ROOT="$SCRATCH/nl_output/search"
DATA_DIR="$SCRATCH/nl_eval"

echo "Model:       $MODEL"
echo "Benchmarks:  ${BENCHMARKS[*]}"
echo "n:           1000"
echo "Output:      $OUTPUT"
echo "data_dir:    $DATA_DIR"
echo

python eval_downstream.py \
    --models $MODEL \
    --benchmarks "${BENCHMARKS[@]}" \
    --n 1000 \
    --debug-samples 1 \
    --output "$OUTPUT" \
    --hf-cache "$HF_CACHE" \
    --prompts-dir "$PROMPTS_DIR" \
    --checkpoints-root "$CHECKPOINTS_ROOT" \
    --data-dir "$DATA_DIR"

echo "DONE $(date)"

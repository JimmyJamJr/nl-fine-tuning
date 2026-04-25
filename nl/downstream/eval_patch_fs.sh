#!/bin/bash
#SBATCH -J eval_patch_fs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --partition=smallgpu
#SBATCH -A asaparov
#SBATCH -q normal
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out

# Patch job: re-runs the 5 few-shot benchmark variants for a single model at
# n=1000. For backfilling the 0.6B suite (jobs 9522434-9522441) which were
# submitted before the _fs adapters were added.
#
# Usage:  MODEL=base sbatch eval_patch_fs.sh
# Output: results/eval_patch_fs_${MODEL}_${SLURM_JOB_ID}.json

set -euo pipefail
module load conda
conda activate search

export SCRATCH="${SCRATCH:-/scratch/gautschi/$USER}"
export HF_HOME="$SCRATCH/model_cache"
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

: "${MODEL:?MODEL env var required}"

OUTPUT="results/eval_patch_fs_${MODEL}_${SLURM_JOB_ID}.json"
HF_CACHE="$SCRATCH/model_cache"
PROMPTS_DIR="/home/huan2073/nl-fine-tuning/nl/downstream/prompts"
CHECKPOINTS_ROOT="$SCRATCH/nl_output/search"
DATA_DIR="$SCRATCH/nl_eval"

echo "Model:       $MODEL"
echo "Benchmarks:  clutrr_fs logiqa_fs ruletaker_fs multilogieval_fs zebra_mc_fs"
echo "Output:      $OUTPUT"
echo

python eval_downstream.py \
    --models $MODEL \
    --benchmarks clutrr_fs logiqa_fs ruletaker_fs multilogieval_fs zebra_mc_fs \
    --n 1000 \
    --debug-samples 1 \
    --output "$OUTPUT" \
    --hf-cache "$HF_CACHE" \
    --prompts-dir "$PROMPTS_DIR" \
    --checkpoints-root "$CHECKPOINTS_ROOT" \
    --data-dir "$DATA_DIR"

echo "DONE $(date)"

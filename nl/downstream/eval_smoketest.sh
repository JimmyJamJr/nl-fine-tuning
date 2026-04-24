#!/bin/bash
#SBATCH -J eval_smoke_qwen06b_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=04:00:00
#SBATCH --partition=ai
#SBATCH -A asaparov
#SBATCH -q preemptible
#SBATCH --requeue
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out

# Smoke test of the overhauled downstream eval: Qwen 0.6B base, n=20 per benchmark.
# Goal: confirm prompt formats, NLGraph repo clone, LegalBench/NLGraph downloads,
# and per-benchmark scoring all work before committing to a full multi-hour run.

set -euo pipefail
module load conda
conda activate search

export SCRATCH="${SCRATCH:-/scratch/gautschi/$USER}"
export HF_HOME="$SCRATCH/model_cache"
# Use cached datasets only — all HF datasets have been pre-downloaded by
# setup_cache_datasets.py into $HF_HOME; any cache miss should fail loudly
# rather than silently re-downloading (and possibly hitting rate limits).
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
mkdir -p slurm "$SCRATCH/triton_cache" "$SCRATCH/model_cache"
export PYTHONUNBUFFERED=1

# Propagate HF auth so streaming datasets (Lichess/chess-puzzles, gated FOLIO)
# don't hit anonymous-rate limits / auth errors.
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

cd /home/huan2073/nl-fine-tuning/nl/downstream

MODELS="base"

# All 14 paper benchmarks. Each adapter now returns BOTH log-lik and gen
# accuracy in a single forward pass (where labels are single-token compatible).
BENCHMARKS=(
    proofwriter proofwriter_cwa
    prontoqa_ood
    clutrr clutrr_fs
    stepgame
    # stepgame_gen dropped; redundant with fused stepgame
    folio
    logiqa logiqa_fs
    ruletaker ruletaker_fs
    logicbench_bqa logicbench_mcqa
    multilogieval multilogieval_fs
    nlgraph_gen
    # grapharena_gen — disabled; needs rdkit + dataset generation via build_dataset.py
    legal
    # chess_mate — dropped; 0% across all models at 0.6-1.7B scale (no signal)
    zebra_mc zebra_mc_fs
)

OUTPUT="results/smoke_${SLURM_JOB_ID}_qwen06b_base_n20.json"
HF_CACHE="$SCRATCH/model_cache"
# Hardcode PROMPTS_DIR — SLURM stages the script under /var/spool/slurm so
# `dirname $0` resolves to a read-only temp dir (breaks LegalBench template
# downloads + ProofWriter prompt fallbacks).
PROMPTS_DIR="/home/huan2073/nl-fine-tuning/nl/downstream/prompts"
CHECKPOINTS_ROOT="$SCRATCH/nl_output/search"
DATA_DIR="$SCRATCH/nl_eval"

echo "Models:           $MODELS"
echo "Benchmarks:       ${BENCHMARKS[*]}"
echo "n (global):       20"
echo "Output:           $OUTPUT"
echo "data_dir:         $DATA_DIR"

python eval_downstream.py \
    --models $MODELS \
    --benchmarks "${BENCHMARKS[@]}" \
    --n 20 \
    --debug-samples 2 \
    --output "$OUTPUT" \
    --hf-cache "$HF_CACHE" \
    --prompts-dir "$PROMPTS_DIR" \
    --checkpoints-root "$CHECKPOINTS_ROOT" \
    --data-dir "$DATA_DIR"

echo "DONE $(date)"

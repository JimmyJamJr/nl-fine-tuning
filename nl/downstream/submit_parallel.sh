#!/bin/bash
# Submit one SLURM job per model on Gilbreth. Each job runs the requested
# benchmark suite for a single model on a single A30 GPU. With 4 models + 4 GPUs
# they run fully in parallel, max wall-clock = max(per-model times), not sum.
#
# Usage (from Gilbreth):
#   cd /scratch/gilbreth/huan2073/nl_eval
#   bash submit_parallel.sh                                            # default: all 22 benchmarks
#   BENCHMARKS="legal proofwriter zebra_mc bbh standard" bash submit_parallel.sh
#   MODELS_LIST="base 6pct_L4=/path/..." bash submit_parallel.sh
#   DEBUG_SAMPLES=3 BENCHMARKS="legal proofwriter" bash submit_parallel.sh   # quick QC
#
# Walltime caveat: standby QoS has a 4h limit. If a single model can't finish
# all requested benchmarks in 4h, split into smaller benchmark sets and submit
# multiple times, or switch the sbatch script's QoS to one with longer walltime.

set -euo pipefail

SCRATCH="/scratch/gilbreth/$USER"
SCRIPT_DIR="$SCRATCH/nl_eval"

# Default fleet: base + instruct + 6% sweep (low to high L)
DEFAULT_MODELS=(
    "base"
    "instruct_only=$SCRATCH/models/instruct_only"
    "6pct_L16=$SCRATCH/models/6pct_L16"
    "6pct_L75=$SCRATCH/models/6pct_L75"
)

# Allow override via MODELS_LIST env var (space-separated)
if [ -n "${MODELS_LIST:-}" ]; then
    read -r -a MODELS_ARR <<< "$MODELS_LIST"
else
    MODELS_ARR=("${DEFAULT_MODELS[@]}")
fi

# Default to ALL 22 benchmarks. Override via env: BENCHMARKS="legal proofwriter zebra_mc"
BENCHMARKS="${BENCHMARKS:-legal legal_gen proofwriter proofwriter_cwa proofwriter_gen proofwriter_cwa_gen zebra_mc zebra_mc_gen stepgame stepgame_gen blocksworld blocksworld_logprob blocksworld_first mystery_blocksworld mystery_blocksworld_logprob mystery_blocksworld_first logistics logistics_logprob logistics_first chess_mate chess_mate_logprob chess_mate_first standard bbh bbh_cot}"

# Per-benchmark sample caps for slow generation-heavy benchmarks. Override via env.
N_OVERRIDES="${N_OVERRIDES:-blocksworld=50 blocksworld_first=50 mystery_blocksworld=50 mystery_blocksworld_first=50 logistics=50 logistics_first=50 chess_mate=50 chess_mate_first=50 stepgame_gen=100 proofwriter_gen=200 proofwriter_cwa_gen=200}"

# Debug samples per (benchmark, sub-task) — set >0 to print sample model outputs
DEBUG_SAMPLES="${DEBUG_SAMPLES:-0}"

mkdir -p "$SCRIPT_DIR/results" "$SCRIPT_DIR/slurm_logs"

echo "Submitting ${#MODELS_ARR[@]} parallel jobs (one per model)"
echo "  benchmarks:    $BENCHMARKS"
echo "  n_overrides:   $N_OVERRIDES"
echo "  debug_samples: $DEBUG_SAMPLES"
echo ""

TS=$(date +%Y%m%d_%H%M%S)_$$
for SPEC in "${MODELS_ARR[@]}"; do
    # Derive short name: everything before '=' (or the whole spec if no '=')
    NAME="${SPEC%%=*}"
    NAME="${NAME##*/}"  # if it's a bare path, use basename
    OUTPUT="$SCRIPT_DIR/results/${NAME}_${TS}.json"
    echo "  [$NAME] -> $OUTPUT"
    MODELS="$SPEC" \
    BENCHMARKS="$BENCHMARKS" \
    N_OVERRIDES="$N_OVERRIDES" \
    DEBUG_SAMPLES="$DEBUG_SAMPLES" \
    OUTPUT="$OUTPUT" \
        sbatch --job-name="eval_$NAME" "$SCRIPT_DIR/eval_downstream_gilbreth.sh"
done

echo ""
echo "Check status: squeue -u \$USER"
echo "Log dir:      $SCRIPT_DIR/slurm_logs/"
echo "Result dir:   $SCRIPT_DIR/results/"

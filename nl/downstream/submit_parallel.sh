#!/bin/bash
# Submit one SLURM job per model on Gilbreth. Each job runs the requested
# benchmark suite for a single model on a single A30 GPU. With N models + N GPUs
# they run fully in parallel, max wall-clock = max(per-model times), not sum.
#
# Usage (from Gilbreth):
#   cd /scratch/gilbreth/huan2073/nl_eval
#   bash submit_parallel.sh                                               # default: all 19 benchmarks
#   BENCHMARKS="proofwriter prontoqa_ood folio" bash submit_parallel.sh   # subset
#   MODELS_LIST="base 6pct_L32=/scratch/gilbreth/.../stage_32_*" bash submit_parallel.sh
#   DEBUG_SAMPLES=3 BENCHMARKS="proofwriter" bash submit_parallel.sh       # quick QC
#
# Walltime caveat: standby QoS has a 4h limit. If a single model can't finish
# all requested benchmarks in 4h on A30, split into smaller benchmark sets or
# switch eval_downstream_gilbreth.sh to a higher-walltime QoS.

set -euo pipefail

SCRATCH="/scratch/gilbreth/$USER"
SCRIPT_DIR="$SCRATCH/nl_eval"

# Default fleet: base + instruct + curriculum depth sweep. Use local paths for
# curriculum checkpoints since Gilbreth doesn't share scratch with Gautschi.
DEFAULT_MODELS=(
    "base"
    "instruct_only=$SCRATCH/models/instruct_only"
    "6pct_L8=$SCRATCH/models/6pct_L8"
    "6pct_L16=$SCRATCH/models/6pct_L16"
    "6pct_L32=$SCRATCH/models/6pct_L32"
    "6pct_L48=$SCRATCH/models/6pct_L48"
    "6pct_L64=$SCRATCH/models/6pct_L64"
    "6pct_L75=$SCRATCH/models/6pct_L75"
)

# Allow override via MODELS_LIST env var (space-separated)
if [ -n "${MODELS_LIST:-}" ]; then
    read -r -a MODELS_ARR <<< "$MODELS_LIST"
else
    MODELS_ARR=("${DEFAULT_MODELS[@]}")
fi

# Default: all 14 primary + 5 few-shot variants (19 total).
BENCHMARKS="${BENCHMARKS:-proofwriter proofwriter_cwa prontoqa_ood clutrr clutrr_fs stepgame folio logiqa logiqa_fs ruletaker ruletaker_fs logicbench_bqa logicbench_mcqa multilogieval multilogieval_fs nlgraph_gen legal zebra_mc zebra_mc_fs}"

N="${N:-1000}"
DEBUG_SAMPLES="${DEBUG_SAMPLES:-1}"

mkdir -p "$SCRIPT_DIR/results" "$SCRIPT_DIR/slurm_logs"

echo "Submitting ${#MODELS_ARR[@]} parallel jobs (one per model)"
echo "  benchmarks:    $BENCHMARKS"
echo "  n:             $N"
echo "  debug_samples: $DEBUG_SAMPLES"
echo ""

TS=$(date +%Y%m%d_%H%M%S)_$$
for SPEC in "${MODELS_ARR[@]}"; do
    NAME="${SPEC%%=*}"
    NAME="${NAME##*/}"
    OUTPUT="$SCRIPT_DIR/results/${NAME}_${TS}.json"
    echo "  [$NAME] -> $OUTPUT"
    MODELS="$SPEC" \
    BENCHMARKS="$BENCHMARKS" \
    N="$N" \
    DEBUG_SAMPLES="$DEBUG_SAMPLES" \
    OUTPUT="$OUTPUT" \
        sbatch --job-name="eval_$NAME" "$SCRIPT_DIR/eval_downstream_gilbreth.sh"
done

echo ""
echo "Check status: squeue -u \$USER"
echo "Log dir:      $SCRIPT_DIR/slurm_logs/"
echo "Result dir:   $SCRIPT_DIR/results/"

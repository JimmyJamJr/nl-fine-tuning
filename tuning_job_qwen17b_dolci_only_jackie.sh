#!/bin/bash
# =============================================================================
# Qwen 1.7B 100% Dolci-Instruct (no graph search, no curriculum)
# Match-compute ablation against the 6%-mix L=32 run.
# Target hardware: Jackie's 6x A100 80GB pod
#
# Saves persistent (model-only) checkpoints at the cumulative PFLOPS budgets
# matching the 6%-mix run's L=8/16/24/32 endpoints, then stops at the L=32
# budget. Each saved checkpoint pairs with one curriculum-eval point.
#
# Run on Jackie's 6x A100 pod after env setup:
#   bash tuning_job_qwen17b_dolci_only_jackie.sh
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR/nl"

# ========== Conda env ==========
if [ -f /home/ray/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/ray/anaconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-curriculum}"
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-base}"
fi

# ========== Paths ==========
export SCRATCH="${SCRATCH:-/home/ray/scratch}"
mkdir -p "$SCRATCH/nl_output" "$SCRATCH/model_cache" "$SCRATCH/triton_cache"
export HF_HOME="$SCRATCH/model_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
# Default to offline so 6 ranks * N dataloader workers don't hammer HF and trip 429.
# Caller is responsible for pre-warming the cache (see DOLCI_LOCAL_PATH below).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

JOB_ID="${JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_qwen17b_dolci_only}"
export SLURM_JOB_ID="$JOB_ID"
MASTER_PORT="${MASTER_PORT:-$((10000 + ($(echo "$JOB_ID" | cksum | cut -d' ' -f1) % 50000)))}"

# ========== Hardware ==========
GPUS_PER_NODE="${GPUS_PER_NODE:-6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR="127.0.0.1"
ulimit -n 131072 || true

# ==========================================================
#                      CONFIGURATION
# Same hyperparameters as 6%-mix run EXCEPT mix_pretrain_ratio=1.0
# (no graph search) and n_stages=1 (no curriculum L-progression).
#
# eff_batch = 6 GPU × 32 × GA=4 = 768 (matches 6%-mix's 4×48×4=768)
# This requires batch_size=32 instead of 48 to fit 6 GPUs cleanly.
# Optimizer math is identical (eff_batch unchanged); only per-microbatch
# shape differs.
# ==========================================================
TASK="search"   # task name unchanged for code path; with ratio=1.0 it's pure pretrain
MODEL_NAME="Qwen/Qwen3-1.7B"

# bs=64, ga=2 (eff=768): kernel-launch + occupancy gain over bs=32 ga=4 with GC
# on (~5–10% throughput). Memory ~42 GiB peak per A100 80GB (was ~31 GiB at
# bs=32). Override via env if you need to revert (e.g. on smaller GPUs).
BATCH_SIZE="${BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE=5e-5
WARMUP_STEPS=100
SEED=1234

# Single stage (no curriculum L-progression). 100% Dolci every batch.
N_STAGES=1
BASE_ALPHA=0.1
MAX_ALPHA=1.0
ACCURACY_THRESHOLD=0.98       # never reached — pure pretrain has no accuracy gate
MIN_STEPS_PER_STAGE=200
CHECK_EVERY=25
ACCURACY_WINDOW=200

MAX_INPUT_SIZE=192            # match 6%-mix run exactly
MAX_LOOKAHEAD=32              # nominal; not exercised at ratio=1.0
MAX_FRONTIER_SIZE=12
MAX_BRANCH_SIZE=12
REQUESTED_BACKTRACK=3

# With mix_pretrain_ratio=1.0 (all Dolci, seq_len up to 2048) on 6x A100 80GB,
# bs=32 OOM'd at ~78 GB peak. Enabling GC trades ~25-30% throughput for the
# ~50% activation-memory cut. Override to false (and reduce BATCH_SIZE) only
# if you have GPUs with more headroom.
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
USE_LIGER=true
USE_CHUNKED_CE=true
CE_CHUNK_SIZE=4096

# *** PFLOPS milestones — cumulative compute at which 6%-mix run finished each L ***
# Saves a model-only checkpoint at each crossing into pflops_checkpoints/pflops_<int>/
# Stop at the L=32 budget (= end of 6%-mix run = matches existing L=32 downstream eval).
# Both env-overridable so the same script can be reused to extend an existing run
# (e.g. PFLOPS_MILESTONES="...,107633,127421" MAX_TOTAL_PFLOPS=127421 bash <script>).
PFLOPS_MILESTONES="${PFLOPS_MILESTONES:-5362,25273,48253,85266}"
MAX_TOTAL_PFLOPS="${MAX_TOTAL_PFLOPS:-85266}"

DO_BASELINE=false
DO_FINAL_EVAL=false
DO_REDACTED_EVAL=false
DO_SEEN_EVAL=false
DO_STAGE_EVAL=false

# ==========================================================
echo "================================================="
echo "Qwen 1.7B 100% Dolci-Instruct — match-compute ablation"
echo "JOB_ID:           $JOB_ID"
echo "Effective batch:  $((GPUS_PER_NODE * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))  ($GPUS_PER_NODE × $BATCH_SIZE × $GRADIENT_ACCUMULATION_STEPS)"
echo "PFLOPS milestones: $PFLOPS_MILESTONES"
echo "Max total PFLOPS:  $MAX_TOTAL_PFLOPS"
echo "================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Build C++ generator if needed
python -c "
try:
    import generator
    print('[OK] C++ generator present')
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, 'nl_generator.py'])
"

ARGS=(
    --task "$TASK"
    --model_name "$MODEL_NAME"
    --cache_dir "$HF_HOME"
    --output_dir "$SCRATCH/nl_output"
    --scratch_dir "$SCRATCH"
    --job_id "$JOB_ID"

    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --warmup_steps "$WARMUP_STEPS"
    --seed "$SEED"
    --num_shots 0
    --first_token_soft_weight 0.0

    --n_stages "$N_STAGES"
    --base_alpha "$BASE_ALPHA"
    --max_alpha "$MAX_ALPHA"
    --accuracy_threshold "$ACCURACY_THRESHOLD"
    --min_steps_per_stage "$MIN_STEPS_PER_STAGE"
    --check_every "$CHECK_EVERY"
    --accuracy_window "$ACCURACY_WINDOW"
    --eval_every_steps 0

    --max_input_size "$MAX_INPUT_SIZE"
    --max_lookahead "$MAX_LOOKAHEAD"
    --max_frontier_size "$MAX_FRONTIER_SIZE"
    --max_branch_size "$MAX_BRANCH_SIZE"
    --requested_backtrack "$REQUESTED_BACKTRACK"

    --eval_samples 500
    --print_eval_examples 0
    --save_total_limit 2

    --use_packing
    --linear_lookahead
    --base_lookahead 1
    --lookahead_step 1

    # *** Pure pretrain — 100% Dolci-Instruct in every batch ***
    # Point at the locally-cached snapshot dir so 6 ranks * 4 dataloader workers
    # don't all race to call HF dataset_info() at startup (=> 429s). The dir must
    # contain a `data/*.parquet` layout, which `snapshot_download` produces.
    --mix_pretrain_data "${DOLCI_LOCAL_PATH:-/home/ray/scratch/model_cache/hub/datasets--allenai--Dolci-Instruct-SFT/snapshots/bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221}"
    --mix_pretrain_ratio 1.0
    --use_chat_template

    # *** PFLOPS milestone checkpoints + auto-stop ***
    --save_pflops_milestones "$PFLOPS_MILESTONES"
    --max_total_pflops "$MAX_TOTAL_PFLOPS"
)

$GRADIENT_CHECKPOINTING && ARGS+=(--gradient_checkpointing)
$USE_LIGER             && ARGS+=(--use_liger)
$USE_CHUNKED_CE        && ARGS+=(--use_chunked_ce --ce_chunk_size "$CE_CHUNK_SIZE")

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"

exec torchrun --nproc_per_node="$GPUS_PER_NODE" \
              --master_port="$MASTER_PORT" \
              --max_restarts=0 \
              tuning_nl.py "${ARGS[@]}"

#!/bin/bash
# =============================================================================
# Qwen 0.6B no-curriculum at L=120 — RESUME from step 16500 (~104K PFLOPS)
# Target hardware: Jackie's 2x H100 80GB pod
#
# This is the no-curriculum (single-stage) sweep arm for RQ 1.1 Part 1.
# Original Gautschi job 9495340 reached 104K PFLOPS / loss=0.026 before being
# preempted. The L=112 nocurr run (job 9495339) finished at 148K PFLOPS / 97.4%
# greedy_first @ alpha=1.0. We extend L=120 to ~150K+ PFLOPS to determine
# whether nocurr can still clear the 98% gate at this depth, or plateaus below.
#
# Resumes from a FULL checkpoint (model + optimizer.pt + scheduler.pt + rng).
# No fresh-AdamW spike — clean continuation.
#
# Pulls the seed checkpoint from HuggingFace Hub (public, no auth required).
#
# Run on Jackie's 2x H100 pod after env setup:
#   bash tuning_job_qwen06b_nocurr120_RESUME_jackie.sh
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
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

JOB_ID="${JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_qwen06b_nocurr120}"
JOB_DIR="$SCRATCH/nl_output/search/job_${JOB_ID}"

# ========== Step 1: pull seed checkpoint from HF Hub ==========
HF_REPO="${HF_REPO:-JimmyJamJr/qwen06b-nocurr-L120-step16500}"
SEED_DIR="$SCRATCH/qwen06b_nocurr120_step16500_seed"

if [ ! -f "$SEED_DIR/model.safetensors" ]; then
    echo "=== Downloading seed checkpoint from $HF_REPO ==="
    huggingface-cli download "$HF_REPO" --local-dir "$SEED_DIR" --local-dir-use-symlinks=False
fi

# ========== Step 2: fabricate Trainer-style checkpoint dir for auto-resume ==========
RESUME_STEP=16500
mkdir -p "$JOB_DIR/checkpoint-${RESUME_STEP}"

# Symlink ALL trainer files (we have full optimizer state — clean resume)
for f in model.safetensors config.json generation_config.json tokenizer.json tokenizer_config.json \
         special_tokens_map.json added_tokens.json chat_template.jinja merges.txt vocab.json \
         optimizer.pt scheduler.pt trainer_state.json training_args.bin curriculum_state.json \
         rng_state_0.pth rng_state_1.pth rng_state_2.pth; do
    [ -f "$SEED_DIR/$f" ] && ln -sf "$SEED_DIR/$f" "$JOB_DIR/checkpoint-${RESUME_STEP}/$f"
done

# Copy job-root state files
cp -f "$SEED_DIR/curriculum_state.json" "$JOB_DIR/" 2>/dev/null || true
cp -f "$SEED_DIR/loss_history.jsonl" "$JOB_DIR/" 2>/dev/null || true
cp -f "$SEED_DIR/stage_eval_history.json" "$JOB_DIR/" 2>/dev/null || true

echo "=== Fabricated resume checkpoint ==="
ls -la "$JOB_DIR/checkpoint-${RESUME_STEP}/"

# ========== Hardware ==========
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR="127.0.0.1"
MASTER_PORT="${MASTER_PORT:-$((10000 + ($(echo "$JOB_ID" | cksum | cut -d' ' -f1) % 50000)))}"
ulimit -n 131072 || true

# ==========================================================
#                      CONFIGURATION
# ==========================================================
# Original Gautschi config: 3 GPU × 32 × GA=8 = eff_batch=768  (GC=on)
# 2× H100 reshape:          2 GPU × 96 × GA=4 = eff_batch=768  (GC=off, ~40 GB / 80 GB)
# All other dynamics preserved (lr, warmup, n_stages=1, max_input_size, etc.)
# Memory math: per-GPU microbatch = 96 × 720 = 69K tokens; well under the 147K-token
# point where the L=256 step-size sweeps OOMed at GC=off on 4× H100 80GB. Half the
# seq length (720 vs 1536) gives ~half the activation memory.
#
# *** OOM FALLBACK ***
# If memory is somehow tight, flip GC=on with the largest possible microbatch
# (fastest GC=on option — fewer GA loops + better GPU saturation; ~32 GB peak):
#   BATCH_SIZE=384 GRADIENT_ACCUMULATION_STEPS=1 GRADIENT_CHECKPOINTING=true \
#     bash tuning_job_qwen06b_nocurr120_RESUME_jackie.sh
# ==========================================================
TASK="search"
MODEL_NAME="Qwen/Qwen3-0.6B"

BATCH_SIZE="${BATCH_SIZE:-96}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE=5e-5
WARMUP_STEPS="${WARMUP_STEPS:-100}"
SEED=1234

# *** NO CURRICULUM ***: single stage at fixed L=120, alpha pinned
N_STAGES=1
BASE_ALPHA=0.1
MAX_ALPHA=1.0
ACCURACY_THRESHOLD=0.98
MIN_STEPS_PER_STAGE=200
CHECK_EVERY=25
ACCURACY_WINDOW="${ACCURACY_WINDOW:-1000}"  # post all-gather fix: window is TOTAL samples (~±0.9pp noise at p=0.98 with 1000)

MAX_INPUT_SIZE=720         # 6 × 120
MAX_LOOKAHEAD=120
BASE_LOOKAHEAD=120         # nocurr: base=target
LOOKAHEAD_STEP=1
MAX_FRONTIER_SIZE=12
MAX_BRANCH_SIZE=12
REQUESTED_BACKTRACK=3

GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"  # off for ~30% speedup; flip true on OOM
USE_LIGER=true
USE_CHUNKED_CE=true
CE_CHUNK_SIZE=4096

ARGS=(
    --task "$TASK"
    --model_name "$MODEL_NAME"
    --cache_dir "$HF_HOME"
    --output_dir "$SCRATCH/nl_output"
    --scratch_dir "$SCRATCH"
    --job_id "$JOB_ID"
    --resume_from_job "$JOB_ID"

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
    --base_lookahead "$BASE_LOOKAHEAD"
    --lookahead_step "$LOOKAHEAD_STEP"

    --do_final_eval
    --do_redacted_eval
    --do_seen_eval
    --do_stage_eval
)

$GRADIENT_CHECKPOINTING && ARGS+=(--gradient_checkpointing)
$USE_LIGER             && ARGS+=(--use_liger)
$USE_CHUNKED_CE        && ARGS+=(--use_chunked_ce --ce_chunk_size "$CE_CHUNK_SIZE")

echo "================================================="
echo "Qwen 0.6B nocurr L=120 — RESUME from step ${RESUME_STEP}"
echo "JOB_ID:              $JOB_ID"
echo "Resume from:         $JOB_DIR/checkpoint-${RESUME_STEP}"
echo "Hardware:            $GPUS_PER_NODE GPUs, batch=$BATCH_SIZE, GA=$GRADIENT_ACCUMULATION_STEPS  ->  eff_batch=$((GPUS_PER_NODE * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "GC:                  $GRADIENT_CHECKPOINTING"
echo "Target L (no curriculum): $MAX_LOOKAHEAD"
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

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"

exec torchrun --nproc_per_node="$GPUS_PER_NODE" \
              --master_port="$MASTER_PORT" \
              --max_restarts=0 \
              tuning_nl.py "${ARGS[@]}"

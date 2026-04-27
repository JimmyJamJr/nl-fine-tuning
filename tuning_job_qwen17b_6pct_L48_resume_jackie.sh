#!/bin/bash
# =============================================================================
# Qwen 1.7B 6%-Dolci-Instruct curriculum — RESUME from stage 32 → extend to L=48
# Target hardware: Jackie's 2x H100 80GB pod
#
# Resumes from a model-only snapshot (original optimizer.pt was lost when the
# original RunPod was terminated). AdamW state will start fresh — expect a
# brief loss spike (~50-200 steps) at restart, then normal training.
#
# Pulls the seed checkpoint from HuggingFace Hub (private repo).
#
# Run on Jackie's 2x H100 pod after env setup:
#   bash tuning_job_qwen17b_6pct_L48_resume_jackie.sh
# =============================================================================

set -euo pipefail

# ========== Resolve repo paths ==========
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR/nl"
mkdir -p "$REPO_DIR/slurm"

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

# ========== Job ID ==========
JOB_ID="${JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_qwen17b_6pct_L48}"
JOB_DIR="$SCRATCH/nl_output/search/job_${JOB_ID}"

# ========== Step 1: pull seed checkpoint from HF Hub ==========
HF_REPO="${HF_REPO:-JimmyJamJr/qwen17b-6pct-dolci-stage32}"
SEED_DIR="$SCRATCH/qwen17b_6pct_stage32_seed"

if [ ! -f "$SEED_DIR/model.safetensors" ]; then
    echo "=== Downloading seed checkpoint from $HF_REPO ==="
    huggingface-cli download "$HF_REPO" --local-dir "$SEED_DIR" --local-dir-use-symlinks=False
fi

# ========== Step 2: fabricate a Trainer-style checkpoint dir so resume_from works ==========
# tuning_nl.py's auto-resume looks for $JOB_DIR/checkpoint-XXXXX/ with at least:
#   model.safetensors, config.json, trainer_state.json
# It tolerates missing optimizer.pt / scheduler.pt (logs a warning, fresh AdamW).
mkdir -p "$JOB_DIR/checkpoint-26400"

# Symlink model + tokenizer into the fabricated checkpoint
for f in model.safetensors config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json chat_template.jinja merges.txt; do
    [ -f "$SEED_DIR/$f" ] && ln -sf "$SEED_DIR/$f" "$JOB_DIR/checkpoint-26400/$f"
done

# Minimal trainer_state.json — the trainer will read global_step from this.
cat > "$JOB_DIR/checkpoint-26400/trainer_state.json" <<'EOF'
{
  "global_step": 26400,
  "epoch": 0.0,
  "max_steps": 1000000000,
  "num_train_epochs": 1000,
  "log_history": [],
  "best_metric": null,
  "best_model_checkpoint": null,
  "is_local_process_zero": true,
  "is_world_process_zero": true,
  "is_hyper_param_search": false,
  "trial_name": null,
  "trial_params": null
}
EOF

# Copy curriculum_state and loss_history to the JOB DIR root (these are loaded
# by tuning_nl.py via _try_restore_curriculum_state).
cp -f "$SEED_DIR/curriculum_state.json" "$JOB_DIR/" 2>/dev/null || true
cp -f "$SEED_DIR/loss_history.jsonl"    "$JOB_DIR/" 2>/dev/null || true
cp -f "$SEED_DIR/loss_history.json"     "$JOB_DIR/" 2>/dev/null || true
cp -f "$SEED_DIR/stage_eval_history.json" "$JOB_DIR/" 2>/dev/null || true

echo "=== Fabricated resume checkpoint ==="
ls -la "$JOB_DIR/checkpoint-26400/"

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
# Mirrors original 6%-mix run exactly EXCEPT:
#   - 2 GPUs instead of 4 (compensated by GA=8 vs GA=4 → eff_batch=768 same)
#   - max_lookahead 48 instead of 32 (extending curriculum)
#   - n_stages 48 instead of 32 (16 new stages: L=33..48)
#   - max_input_size 288 (=6*48) instead of 192 (=6*32) for new bigger samples
#   - warmup_steps 500 instead of 100 (cushion the fresh-AdamW spike)
# ==========================================================
TASK="search"
MODEL_NAME="Qwen/Qwen3-1.7B"

# eff_batch = 2 GPU × 48 × GA=8 = 768  (matches original 4×48×4 = 768)
BATCH_SIZE=48
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=5e-5
WARMUP_STEPS=500       # longer than original (100) to soften fresh-Adam restart
SEED=1234

N_STAGES=48            # extend from original 32 to 48
BASE_ALPHA=0.1
MAX_ALPHA=1.0
ACCURACY_THRESHOLD=0.98
MIN_STEPS_PER_STAGE=200
CHECK_EVERY=25
ACCURACY_WINDOW=200

MAX_INPUT_SIZE=288     # 6 × 48 (was 192 = 6×32)
MAX_LOOKAHEAD=48
BASE_LOOKAHEAD=1
LOOKAHEAD_STEP=1
MAX_FRONTIER_SIZE=12
MAX_BRANCH_SIZE=12
REQUESTED_BACKTRACK=3

GRADIENT_CHECKPOINTING=true   # match original; H100 80GB has room without but consistency matters
USE_LIGER=true                # Qwen-supported
USE_CHUNKED_CE=true
CE_CHUNK_SIZE=4096

# ==========================================================
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
    --linear_lookahead
    --base_lookahead "$BASE_LOOKAHEAD"
    --lookahead_step "$LOOKAHEAD_STEP"

    # Same Dolci 6% mix as original
    --mix_pretrain_data allenai/Dolci-Instruct-SFT
    --mix_pretrain_ratio 0.06
    --use_chat_template
)

$GRADIENT_CHECKPOINTING && ARGS+=(--gradient_checkpointing)
$USE_LIGER             && ARGS+=(--use_liger)
$USE_CHUNKED_CE        && ARGS+=(--use_chunked_ce --ce_chunk_size "$CE_CHUNK_SIZE")

echo "================================================="
echo "Qwen 1.7B 6%-mix resume — extend curriculum to L=48"
echo "JOB_ID:              $JOB_ID"
echo "Effective batch:     $((GPUS_PER_NODE * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))  ($GPUS_PER_NODE × $BATCH_SIZE × $GRADIENT_ACCUMULATION_STEPS)"
echo "Resume from:         $JOB_DIR/checkpoint-26400"
echo "Curriculum target:   L=$MAX_LOOKAHEAD ($N_STAGES stages)"
echo "================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"

# Build C++ generator if needed
python -c "
try:
    import generator
    print('[OK] C++ generator present')
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, 'nl_generator.py'])
"

exec torchrun --nproc_per_node="$GPUS_PER_NODE" \
              --master_port="$MASTER_PORT" \
              --max_restarts=0 \
              tuning_nl.py "${ARGS[@]}"

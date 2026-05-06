#!/bin/bash
# qwen06b_curr96_RESUME_2xh100.sh
# Resume Qwen3-0.6B curriculum (s=8, max_L=96) from checkpoint-6500 on 2xH100.
# Originally run on Gilbreth 2xA100 at bs=96 GA=4 (eff=768).
# H100 has more VRAM, so we use bs=192 GA=2 (eff=768 preserved) -- ~2x faster than A100.

set -uo pipefail

############################################
# >>> EDIT THESE 3 PATHS / NAMES IF NEEDED <<<
WORKDIR="$HOME/qwen06b_resume"          # where everything lives
CONDA_ENV="nl_resume"                    # your conda env with FA3, transformers, liger
CODE_DIR="$WORKDIR/nl-fine-tuning/nl"    # where tuning_nl_fa3.py lives (clone the repo)
HF_REPO="JimmyJamJr/qwen06b-curr96-step8-ckpt6500"
SLURM_JOBID=""                           # set to your reservation's job id if running via srun --jobid; else leave blank
############################################

JOB_ID="qwen06b_curr96_step8_regular"
OUTPUT_DIR="$WORKDIR/nl_output"
SCRATCH_DIR="$WORKDIR"
RUN_DIR="$OUTPUT_DIR/search/job_${JOB_ID}"

# --- Activate env ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --- HF cache (avoid filling $HOME) ---
export HF_HOME="$WORKDIR/.hf_home"
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR="$WORKDIR/.triton_cache"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR"

# --- Training env ---
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=14
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export BENCH_NUM_WORKERS=8

# --- One-time: pull checkpoint + metadata if not already present ---
if [ ! -f "$RUN_DIR/checkpoint-6500/model.safetensors" ]; then
    echo "=== Downloading checkpoint from HF Hub ($HF_REPO) ==="
    mkdir -p "$RUN_DIR"
    huggingface-cli download "$HF_REPO" --local-dir "$RUN_DIR" --quiet
    echo "Download complete."
else
    echo "=== Checkpoint already present at $RUN_DIR/checkpoint-6500 -- skipping download ==="
fi

# --- Sanity check checkpoint ---
for f in trainer_state.json model.safetensors optimizer.pt scheduler.pt rng_state_0.pth rng_state_1.pth; do
    if [ ! -f "$RUN_DIR/checkpoint-6500/$f" ]; then
        echo "[ERROR] Missing: $RUN_DIR/checkpoint-6500/$f"; exit 1
    fi
done
echo "[CKPT] All required checkpoint files present."

cd "$CODE_DIR"

echo "=== Resume Qwen3-0.6B curr96 step=8 RESUME (2xH100, bs=192 GA=2 GC=on) ==="
echo "JOB_ID:    $JOB_ID"
echo "config:    bs=192 GA=2 (eff=768), lr=5e-5, GC=on, FA3, liger, packing"
echo "resume from: $RUN_DIR/checkpoint-6500"

# --- Decide launch mode (held-out SLURM srun vs. plain torchrun) ---
if [ -n "$SLURM_JOBID" ]; then
    LAUNCH=(srun --jobid="$SLURM_JOBID" --overlap -c 14 --gres=gpu:2 -N1 -n1 torchrun)
else
    LAUNCH=(torchrun)
fi

exec "${LAUNCH[@]}" --nproc_per_node=2 --master_port=29501 --max_restarts=0 \
    tuning_nl_fa3.py \
    --task search \
    --model_name Qwen/Qwen3-0.6B \
    --cache_dir "$HF_HOME" \
    --output_dir "$OUTPUT_DIR" \
    --scratch_dir "$SCRATCH_DIR" \
    --job_id "$JOB_ID" \
    --resume_from_job "$JOB_ID" \
    --batch_size 192 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --warmup_steps 100 \
    --seed 1234 \
    --num_shots 0 \
    --first_token_soft_weight 0.0 \
    --n_stages 12 \
    --base_alpha 0.1 \
    --max_alpha 1.0 \
    --accuracy_threshold 0.98 \
    --min_steps_per_stage 200 \
    --check_every 25 \
    --accuracy_window 800 \
    --eval_every_steps 0 \
    --max_input_size 576 \
    --max_lookahead 96 \
    --max_frontier_size 12 \
    --max_branch_size 12 \
    --requested_backtrack 3 \
    --eval_samples 500 \
    --print_eval_examples 0 \
    --save_total_limit 2 \
    --use_packing \
    --linear_lookahead \
    --base_lookahead 8 \
    --lookahead_step 8 \
    --gradient_checkpointing \
    --use_liger \
    --use_chunked_ce \
    --ce_chunk_size 4096 \
    --persist_every 0 \
    --max_total_pflops 60000 \
    --do_final_eval

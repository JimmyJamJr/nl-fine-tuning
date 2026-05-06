#!/bin/bash
# pythia28b_RESUME_4xh100.sh
# Resume Pythia-2.8B curriculum (s=1, max_L=96) from checkpoint-18000 on 4xH100.
# Original was 8x H200 at bs=192 GA=1 GC=on (eff=1536).
# 4x H100 80GB (less VRAM than H200 141GB):
#   bs=192 GA=2 won't fit on H100 80GB (vast H200 used 94GB at bs=192).
#   bs=96 GA=4 GC=on -> 4*96*4=1536 (eff preserved), ~67-69 GB/GPU (proven on h009 8gpu before).
# GC=on mandatory on H100 80GB for Pythia-2.8B at L>=64.

set -uo pipefail

############################################
# >>> EDIT THESE 4 VALUES IF NEEDED <<<
WORKDIR="$HOME/pythia28b_resume"        # where checkpoint + outputs live
CONDA_ENV="nl_resume"                    # your env with FA3, transformers, liger, chunked_ce
CODE_DIR="$WORKDIR/nl-fine-tuning/nl"    # where tuning_nl_fa3.py lives (clone Jimson's repo)
SLURM_JOBID=""                           # set to your reservation jobid for srun --jobid; else leave blank
############################################

HF_REPO="JimmyJamJr/pythia28b-eff1536-ckpt18000"
JOB_ID="runpod_pythia28b_eff1536_gcoff"
OUTPUT_DIR="$WORKDIR/nl_output"
SCRATCH_DIR="$WORKDIR"
RUN_DIR="$OUTPUT_DIR/search/job_${JOB_ID}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="$WORKDIR/.hf_home"
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR="$WORKDIR/.triton_cache"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=14
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export BENCH_NUM_WORKERS=8

# --- One-time download from HF Hub ---
if [ ! -f "$RUN_DIR/checkpoint-18000/model-00001-of-00002.safetensors" ]; then
    echo "=== Downloading Pythia-2.8B checkpoint from HF Hub ($HF_REPO) ~17 GB ==="
    mkdir -p "$RUN_DIR"
    huggingface-cli download "$HF_REPO" --local-dir "$RUN_DIR" --quiet
    echo "Download complete."
else
    echo "=== Checkpoint already present at $RUN_DIR/checkpoint-18000 ==="
fi

# --- Sanity check ---
for f in trainer_state.json model.safetensors.index.json model-00001-of-00002.safetensors \
         model-00002-of-00002.safetensors optimizer.pt scheduler.pt rng_state_0.pth rng_state_1.pth \
         rng_state_2.pth rng_state_3.pth; do
    if [ ! -f "$RUN_DIR/checkpoint-18000/$f" ]; then
        echo "[ERROR] Missing: $RUN_DIR/checkpoint-18000/$f"; exit 1
    fi
done
echo "[CKPT] All required checkpoint files present."

cd "$CODE_DIR"

echo "=== Resume Pythia-2.8B (4xH100, bs=96 GA=4 GC=on FA3) ==="
echo "JOB_ID:    $JOB_ID"
echo "config:    bs=96 GA=4 (eff=1536, 4x96x4), lr=2.7e-5, GC=on, FA3, liger, packing"
echo "resume from: $RUN_DIR/checkpoint-18000"

if [ -n "$SLURM_JOBID" ]; then
    LAUNCH=(srun --jobid="$SLURM_JOBID" --overlap -c 56 --gres=gpu:4 -N1 -n1 torchrun)
else
    LAUNCH=(torchrun)
fi

exec "${LAUNCH[@]}" --nproc_per_node=4 --master_port=29503 --max_restarts=0 \
    tuning_nl_fa3.py \
    --task search \
    --model_name EleutherAI/pythia-2.8b \
    --cache_dir "$HF_HOME" \
    --output_dir "$OUTPUT_DIR" \
    --scratch_dir "$SCRATCH_DIR" \
    --job_id "$JOB_ID" \
    --resume_from_job "$JOB_ID" \
    --batch_size 96 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2.7e-5 \
    --warmup_steps 200 \
    --seed 1234 \
    --num_shots 0 \
    --first_token_soft_weight 0.0 \
    --n_stages 96 \
    --base_alpha 0.1 \
    --max_alpha 1.0 \
    --accuracy_threshold 0.98 \
    --min_steps_per_stage 0 \
    --check_every 25 \
    --accuracy_window 1000 \
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
    --base_lookahead 1 \
    --lookahead_step 1 \
    --use_chunked_ce \
    --ce_chunk_size 4096 \
    --gradient_checkpointing \
    --persist_every 0 \
    --max_total_pflops 500000 \
    --do_stage_eval \
    --stage_eval_every 8

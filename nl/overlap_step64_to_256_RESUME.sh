#!/bin/bash
# Qwen 0.6B step=64 RESUME from step 24500 ckpt → 300K PFLOPs cap.
# Same hypers as before: lr=5e-5, eff_batch=768 (4 GPU x 48 x GA=4), max_input=1536, GC=on.
# Fires into reservation 9530824 via srun --overlap.
set -euo pipefail
module load conda
conda activate search

export SCRATCH=/scratch/gautschi/huan2073
export HF_HOME=$SCRATCH/model_cache
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR=$SCRATCH/triton_cache
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=14
mkdir -p $SCRATCH/triton_cache

cd /home/huan2073/nl-fine-tuning/nl

# Resume from same dir as earlier run
JOB_ID="local_20260428_215710_qwen06b_step64_L256"

echo "=== step=64 RESUME, target=300K PFLOPs (currently at 133K, L=192) ==="
echo "JOB_ID: $JOB_ID"

exec torchrun --nproc_per_node=4 --master_port=41364 tuning_nl.py \
    --task search \
    --model_name Qwen/Qwen3-0.6B \
    --cache_dir $HF_HOME \
    --output_dir $SCRATCH/nl_output \
    --scratch_dir $SCRATCH \
    --job_id $JOB_ID \
    --resume_from_job $JOB_ID \
    --gradient_checkpointing \
    --batch_size 48 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --warmup_steps 100 \
    --seed 1234 \
    --num_shots 0 \
    --first_token_soft_weight 0.0 \
    --n_stages 4 \
    --base_alpha 0.1 \
    --max_alpha 1.0 \
    --accuracy_threshold 0.98 \
    --min_steps_per_stage 200 \
    --check_every 25 \
    --accuracy_window 1000 \
    --eval_every_steps 0 \
    --max_input_size 1536 \
    --max_lookahead 256 \
    --max_frontier_size 12 \
    --max_branch_size 12 \
    --requested_backtrack 3 \
    --eval_samples 500 \
    --print_eval_examples 0 \
    --save_total_limit 2 \
    --use_packing \
    --linear_lookahead \
    --base_lookahead 64 \
    --lookahead_step 64 \
    --use_liger \
    --use_chunked_ce \
    --ce_chunk_size 4096 \
    --persist_every 0 \
    --max_total_pflops 300000 \
    --do_stage_eval

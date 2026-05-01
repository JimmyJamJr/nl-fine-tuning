#!/bin/bash
# Pythia 1.4B step=1 → L=96 RANDOM INIT on Gautschi h013 (reservation 9673546, 2×H100).
# Matches the main 1.4B step=1 run's hypers exactly + adds --reinit_weights for random initialization.
# eff_batch=192 preserved (2 GPUs × bs=48 × GA=2 = 192) to match the main 4-GPU bs=48 GA=1 run.
# Safety: --max_restarts=0 so torchrun doesn't auto-restart on crash. Holder sbatch (9673546)
# keeps the 2×H100 allocation regardless of inner training state.
#
# Fires into reservation 9673546 via srun --overlap.

set -uo pipefail
module load conda
conda activate search

export SCRATCH=/scratch/gautschi/huan2073
export HF_HOME=$SCRATCH/model_cache
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR=$SCRATCH/triton_cache
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=14
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
mkdir -p $SCRATCH/triton_cache

cd /home/huan2073/nl-fine-tuning/nl

JOB_ID="local_$(date +%Y%m%d_%H%M%S)_pythia14b_step1_REINIT_L96"

echo "=== Pythia 1.4B step=1 RANDOM INIT (reinit_weights) FRESH ==="
echo "JOB_ID:    $JOB_ID"
echo "eff_batch: 192 (2 × 48 × GA=2)  GC=off  lr=2.7e-5  RANDOM INIT"

exec torchrun --nproc_per_node=2 --master_port=41336 --max_restarts=0 tuning_nl.py \
    --task search \
    --model_name EleutherAI/pythia-1.4b \
    --cache_dir $HF_HOME \
    --output_dir $SCRATCH/nl_output \
    --scratch_dir $SCRATCH \
    --job_id $JOB_ID \
    --reinit_weights \
    --batch_size 48 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2.7e-5 \
    --warmup_steps 100 \
    --seed 1234 \
    --num_shots 0 \
    --first_token_soft_weight 0.0 \
    --n_stages 96 \
    --base_alpha 0.1 \
    --max_alpha 1.0 \
    --accuracy_threshold 0.98 \
    --min_steps_per_stage 200 \
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
    --persist_every 0 \
    --max_total_pflops 500000 \
    --do_stage_eval \
    --stage_eval_every 8

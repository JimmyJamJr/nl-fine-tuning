#!/bin/bash
# Pythia 1.4B step=1 RANDOM INIT lr=1e-4 eff=768 on Gautschi h009 (reservation 9281360, 4×H100).
# Combo A from random-init hyper search: 4× lr + 4× batch vs the main pretrained 1.4B baseline.
# bs=192 GA=1 GC=on (bs=192 GC=off won't fit on H100 80GB; GC=on cuts activation memory ~75%).
# Same per-microbatch memory as nocurr128 (which uses bs=192 GA=1 GC=on).
# Safety: --max_restarts=0 so torchrun won't auto-restart on crash. Holder sbatch (9281360)
# keeps the 4×H100 allocation regardless of inner training state.
#
# Fires into reservation 9281360 via srun --overlap.

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

JOB_ID="local_$(date +%Y%m%d_%H%M%S)_pythia14b_step1_REINIT_lr1e4_eff768"

echo "=== Pythia 1.4B step=1 RANDOM INIT lr=1e-4 eff=768 GC=on FRESH ==="
echo "JOB_ID:    $JOB_ID"
echo "Combo A: 4x lr + 4x batch vs main pretrained 1.4B baseline"

exec torchrun --nproc_per_node=4 --master_port=41337 --max_restarts=0 tuning_nl.py \
    --task search \
    --model_name EleutherAI/pythia-1.4b \
    --cache_dir $HF_HOME \
    --output_dir $SCRATCH/nl_output \
    --scratch_dir $SCRATCH \
    --job_id $JOB_ID \
    --reinit_weights \
    --batch_size 192 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
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
    --gradient_checkpointing \
    --use_chunked_ce \
    --ce_chunk_size 4096 \
    --persist_every 0 \
    --max_total_pflops 500000 \
    --do_stage_eval \
    --stage_eval_every 8

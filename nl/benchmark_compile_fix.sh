#!/bin/bash
#SBATCH -J nl_compile_fix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=ai
#SBATCH -A asaparov
#SBATCH -q preemptible
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out
#SBATCH --open-mode=append

set -euo pipefail
mkdir -p ./slurm

echo "========================================"
echo "COMPILE FIX -- torch.compile + model.forward() retest"
echo "$(date)"
echo "========================================"

# ========== Environment ==========
module load conda
conda activate search

export SCRATCH="/scratch/gautschi/$USER"
mkdir -p "$SCRATCH/nl_output" "$SCRATCH/model_cache"
export HF_HOME="$SCRATCH/model_cache"
if [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
fi
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ========== Hardware ==========
GPUS_PER_NODE=2
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
ulimit -n 131072 || true

# ========== CUDA / Torch ==========
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# ========== Distributed ==========
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"
BASE_PORT=$((10000 + (SLURM_JOB_ID % 50000)))

echo "GPUS=$GPUS_PER_NODE  OMP_THREADS=$OMP_NUM_THREADS"
nvidia-smi || true

# ========== Build C++ generator ==========
python -c "
try:
    import generator
    print('[OK] C++ generator present')
except:
    print('[INFO] Building C++ generator...')
    import subprocess, sys
    subprocess.check_call([sys.executable, 'nl_generator.py'])
"

# ========== Cache model ==========
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
m, c = 'Qwen/Qwen3-0.6B', os.environ['HF_HOME']
AutoTokenizer.from_pretrained(m, cache_dir=c, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(m, cache_dir=c, trust_remote_code=True)
print('[OK] Model cached')
"

# ========== Benchmark configs ==========
# Retest: torch.compile + model.forward() (failed in bench3 due to accelerate bug, now fixed)
# Format: "label compile model_forward pack_length base_lookahead lookahead_step num_workers"
CONFIGS=(
    # --- L=16: compile + model.forward ---
    "modelfwd_compile_pack32k    1 1 32768  16 8 4"
    "modelfwd_compile_pack40k    1 1 40960  16 8 4"
    "modelfwd_compile_pack48k    1 1 49152  16 8 4"
    "modelfwd_compile_pack64k    1 1 65536  16 8 4"

    # --- L=128: compile + model.forward ---
    "modelfwd_compile_pack32k_L128  1 1 32768 128 0 4"
    "modelfwd_compile_pack40k_L128  1 1 40960 128 0 4"
    "modelfwd_compile_pack48k_L128  1 1 49152 128 0 4"
    "modelfwd_compile_pack64k_L128  1 1 65536 128 0 4"
)

BENCH_STEPS=150
BENCH_DIR="$SCRATCH/nl_output/compile_fix_${SLURM_JOB_ID}"
mkdir -p "$BENCH_DIR"

PORT=$BASE_PORT

for cfg in "${CONFIGS[@]}"; do
    read -r LABEL COMPILE MODEL_FWD PACK_LEN BASE_L L_STEP NUM_WORKERS <<< "$cfg"

    echo ""
    echo "========================================"
    echo "CONFIG: $LABEL"
    echo "  compile=$COMPILE model_forward=$MODEL_FWD pack_length=$PACK_LEN base_L=$BASE_L step=$L_STEP workers=$NUM_WORKERS"
    echo "========================================"

    COMMON_ARGS=(
        --task search
        --model_name "Qwen/Qwen3-0.6B"
        --cache_dir "$HF_HOME"
        --scratch_dir "$SCRATCH"
        --batch_size 96
        --gradient_accumulation_steps 1
        --learning_rate 1e-4
        --warmup_steps 100
        --seed 1234
        --num_shots 0
        --first_token_soft_weight 0.0
        --n_stages 10
        --base_alpha 0.1
        --max_alpha 1.0
        --accuracy_threshold 0.98
        --min_steps_per_stage 200
        --check_every 25
        --accuracy_window 200
        --eval_every_steps 0
        --max_input_size 768
        --max_lookahead 128
        --max_frontier_size 12
        --max_branch_size 12
        --requested_backtrack 3
        --eval_samples 100
        --print_eval_examples 0
        --use_packing
        --linear_lookahead
        --use_lora --lora_rank 16 --lora_dropout 0.10
        --gradient_checkpointing
        --use_liger
        --use_chunked_ce --ce_chunk_size 4096
        --benchmark_steps "$BENCH_STEPS"
        --output_dir "$BENCH_DIR"
        --job_id "$LABEL"
        --base_lookahead "$BASE_L"
        --lookahead_step "$L_STEP"
    )

    if [ "$COMPILE" -eq 1 ]; then
        COMMON_ARGS+=(--torch_compile)
    fi

    if [ "$MODEL_FWD" -eq 1 ]; then
        COMMON_ARGS+=(--use_model_forward)
    fi

    PORT=$((PORT + 1))

    export BENCH_NUM_WORKERS=$NUM_WORKERS

    set +e
    torchrun --nproc_per_node=$GPUS_PER_NODE \
             --master_port=$PORT \
             --max_restarts=0 \
             qwen_tuning_nl_multi.py "${COMMON_ARGS[@]}" 2>&1
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[WARN] Config $LABEL failed with exit $EXIT_CODE"
    fi

    echo ""
done

echo ""
echo "COMPILE FIX BENCHMARK COMPLETE -- $(date)"

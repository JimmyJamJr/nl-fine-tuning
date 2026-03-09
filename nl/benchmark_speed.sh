#!/bin/bash
#SBATCH -J nl_speed_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=ai
#SBATCH -A asaparov
#SBATCH -q normal
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out
#SBATCH --open-mode=append

set -euo pipefail
mkdir -p ./slurm

echo "========================================"
echo "SPEED BENCHMARK — $(date)"
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
# Format: "label compile pack_length base_lookahead lookahead_step"
# We test: compile (on/off) x pack_length (8192/16384) x lookahead (L=16 curriculum / L=128 fixed)
CONFIGS=(
    "nocompile_pack8k_L16    0  8192  16 8"
    "compile_pack8k_L16      1  8192  16 8"
    "nocompile_pack16k_L16   0 16384  16 8"
    "compile_pack16k_L16     1 16384  16 8"
    "nocompile_pack8k_L128   0  8192 128 0"
    "compile_pack8k_L128     1  8192 128 0"
    "nocompile_pack16k_L128  0 16384 128 0"
    "compile_pack16k_L128    1 16384 128 0"
)

BENCH_STEPS=150
BENCH_DIR="$SCRATCH/nl_output/benchmark_${SLURM_JOB_ID}"
mkdir -p "$BENCH_DIR"

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
)

PORT=$BASE_PORT

for cfg in "${CONFIGS[@]}"; do
    read -r LABEL COMPILE PACK_LEN BASE_L L_STEP <<< "$cfg"

    echo ""
    echo "========================================"
    echo "CONFIG: $LABEL"
    echo "  compile=$COMPILE pack_length=$PACK_LEN base_L=$BASE_L step=$L_STEP"
    echo "========================================"

    RUN_DIR="$BENCH_DIR/$LABEL"
    mkdir -p "$RUN_DIR"

    RUN_ARGS=("${COMMON_ARGS[@]}"
        --output_dir "$BENCH_DIR"
        --job_id "$LABEL"
        --base_lookahead "$BASE_L"
        --lookahead_step "$L_STEP"
    )

    if [ "$COMPILE" -eq 1 ]; then
        RUN_ARGS+=(--torch_compile)
    fi

    PORT=$((PORT + 1))

    set +e
    torchrun --nproc_per_node=$GPUS_PER_NODE \
             --master_port=$PORT \
             --max_restarts=0 \
             qwen_tuning_nl_multi.py "${RUN_ARGS[@]}" 2>&1
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[WARN] Config $LABEL failed with exit $EXIT_CODE"
    fi

    echo ""
done

# ========== Summary ==========
echo ""
echo "========================================"
echo "BENCHMARK SUMMARY"
echo "========================================"

python3 -c "
import json, os, glob

bench_dir = '$BENCH_DIR'
results = []

for d in sorted(glob.glob(os.path.join(bench_dir, 'search', '*', 'benchmark_result.json'))):
    label = d.split('/')[-2]
    with open(d) as f:
        r = json.load(f)
    results.append((label, r))

if not results:
    print('No benchmark results found!')
    exit(0)

# Header
print(f'{\"Config\":<28} {\"Steps/s\":>8} {\"Tok/s\":>12} {\"Mtok/s\":>8} {\"DataWait\":>10} {\"Compute\":>10} {\"DW%\":>5} {\"Compile\":>8} {\"PackLen\":>8} {\"Base_L\":>7}')
print('-' * 120)

# Sort by tokens/sec descending
results.sort(key=lambda x: x[1]['tokens_sec'], reverse=True)
best = results[0][1]['tokens_sec']

for label, r in results:
    pct = r['tokens_sec'] / best * 100
    dw = r.get('data_wait_ms', 0)
    cm = r.get('compute_ms', 0)
    dp = r.get('data_wait_pct', 0)
    print(f'{label:<28} {r[\"steps_sec\"]:>8.3f} {r[\"tokens_sec\"]:>12,.0f} {r[\"tokens_sec\"]/1e6:>8.3f} {dw:>8.1f}ms {cm:>8.1f}ms {dp:>4.1f}% {str(r[\"compile\"]):>8} {r[\"pack_length\"]:>8} {r[\"base_lookahead\"]:>7}  ({pct:.0f}%)')

print()
print('Fastest config:', results[0][0])
"

echo ""
echo "BENCHMARK COMPLETE — $(date)"

#!/bin/bash
#SBATCH -J nl_nocurr17b_L32_cm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=2-00:00:00
#SBATCH --partition=ai
#SBATCH -A asaparov
#SBATCH -q preemptible
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail
cd /home/huan2073/nl-fine-tuning/nl

# ========== Preemption handling ==========
trap 'echo "[SIG] USR1 @ $(date): walltime near, stopping trainer and requeueing"; [ -n "${TPID:-}" ] && kill -TERM "$TPID" 2>/dev/null; sleep 60; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
trap 'echo "[SIG] TERM @ $(date)"; exit 0' TERM

mkdir -p ./slurm
echo "JOB START $(date)"

# ========== Environment ==========
module load conda
conda activate search

export SCRATCH="/scratch/gautschi/$USER"
mkdir -p "$SCRATCH/nl_output" "$SCRATCH/model_cache" "$SCRATCH/triton_cache"
export HF_HOME="$SCRATCH/model_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"   # liger/triton must never write to home quota
if [ -f /home/huan2073/nl-fine-tuning/nl/.env ]; then
    source /home/huan2073/nl-fine-tuning/nl/.env
fi
export HF_HUB_OFFLINE=0  # Need online for streaming Dolci-Instruct-SFT mix
export HF_HUB_ETAG_TIMEOUT=120         # Hub metadata calls: default 10 s times out under load
export HF_HUB_DOWNLOAD_TIMEOUT=120
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ========== Hardware ==========
GPUS_PER_NODE=$(echo $SLURM_JOB_GPUS | tr "," "\n" | wc -l)
[ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
if [ "$GPUS_PER_NODE" -ne 4 ]; then
    echo "[FATAL] Need exactly 4 GPUs for eff_batch=768 (48 x 4 GA x 4 GPUs); got $GPUS_PER_NODE — refusing to train an incomparable run"
    exit 1
fi

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
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "GPUS=$GPUS_PER_NODE  OMP_THREADS=$OMP_NUM_THREADS"
nvidia-smi || true

# ==========================================================
#   Qwen3-1.7B, no curriculum, fixed L=32, 6% Dolci mix, compute-matched
#   Fixed-L flags follow the 0.6B compute-matched runs (job_retrain_nocurr_L*_cm, from job_8577238:
#   --n_stages 1 --linear_lookahead --base_lookahead 96 --lookahead_step 0,
#   MAX_INPUT_SIZE = 6*L), here at L=32; Dolci mix + chat template
#   flags replicate the 1.7B curriculum runs. All training dynamics (batch/GA/LR/warmup/
#   seed/packing/liger/chunked-CE, eff_batch 768) identical to the originals.
#   Budget = the 1.7B s=1 curriculum's cumulative compute at its Table 3 checkpoint
#   stage_32_step_26400_L32 (runpod L<=32 run, n=192): 86298 PFLOPs (loss_history tokens x 6 x 1,720,574,976).
#   Training dynamics copied from the 1.7B curriculum runs: bs 48 x GA 4 x 4 GPUs = 768, lr 5e-5, warmup 100, seed 1234.
# ==========================================================

TASK="search"
MODEL_NAME="Qwen/Qwen3-1.7B"

# Training
BATCH_SIZE=48
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=5e-5
SEED=1234
FIRST_TOKEN_SOFT_WEIGHT=0.0

# No curriculum: single fixed stage at L=32
N_STAGES=1
BASE_ALPHA=0.1
MAX_ALPHA=1.0
ACCURACY_THRESHOLD=1.1   # gate disabled: stop on compute cap only
MIN_STEPS_PER_STAGE=200
CHECK_EVERY=25
ACCURACY_WINDOW=800   # effective window of the original curriculum (200/rank x 4)
EVAL_EVERY_STEPS=1000                    # nocurr precedent (8577238): periodic eval, no stage transitions

# Task parameters
MAX_INPUT_SIZE=192                       # replicates the run that produced the Table 3 checkpoint stage_32_step_26400_L32 (runpod L<=32 run, n=192)
MAX_LOOKAHEAD=32

# Memory optimizations
CE_CHUNK_SIZE=4096

# Evaluation
EVAL_SAMPLES=500
PRINT_EVAL_EXAMPLES=5

# Budget (see header)
MAX_TOTAL_PFLOPS=86298   # 1.7B curriculum cumulative compute at the Table 3 checkpoint stage_32_step_26400_L32 (runpod L<=32 run, n=192); trainer accounting = 6 x 1,720,574,976 params per token

# ==========================================================

JOB_ID_OVERRIDE="nocurr17b_L32_cm"
EFFECTIVE_JOB_ID="${JOB_ID_OVERRIDE:-$SLURM_JOB_ID}"
echo "Task: $TASK | Model: $MODEL_NAME | Max Input: $MAX_INPUT_SIZE"
echo "Output: $SCRATCH/nl_output/$TASK/job_${EFFECTIVE_JOB_ID}"
echo "eff_batch = $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS x $GPUS_PER_NODE = $(( BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * GPUS_PER_NODE ))  (must be 768)"

# ========== Build C++ generator if needed ==========
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
m, c = '$MODEL_NAME', os.environ['HF_HOME']
AutoTokenizer.from_pretrained(m, cache_dir=c, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(m, cache_dir=c, trust_remote_code=True)
print('[OK] Model cached')
"

# ========== Build arguments ==========
ARGS=(
    --task "$TASK"
    --model_name "$MODEL_NAME"
    --cache_dir "$HF_HOME"
    --output_dir "$SCRATCH/nl_output"
    --scratch_dir "$SCRATCH"
    --job_id "$EFFECTIVE_JOB_ID"

    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --seed "$SEED"
    --first_token_soft_weight "$FIRST_TOKEN_SOFT_WEIGHT"

    --n_stages "$N_STAGES"
    --base_alpha "$BASE_ALPHA"
    --max_alpha "$MAX_ALPHA"
    --accuracy_threshold "$ACCURACY_THRESHOLD"
    --min_steps_per_stage "$MIN_STEPS_PER_STAGE"
    --check_every "$CHECK_EVERY"
    --accuracy_window "$ACCURACY_WINDOW"
    --eval_every_steps "$EVAL_EVERY_STEPS"

    --max_input_size "$MAX_INPUT_SIZE"
    --max_lookahead "$MAX_LOOKAHEAD"

    --eval_samples "$EVAL_SAMPLES"
    --print_eval_examples "$PRINT_EVAL_EXAMPLES"

    --use_packing

    --linear_lookahead
    --base_lookahead 32
    --lookahead_step 0
    --mix_pretrain_data allenai/Dolci-Instruct-SFT
    --mix_pretrain_ratio 0.06
    --use_chat_template
    --stage_eval_every 8

    --gradient_checkpointing
    --use_liger
    --ce_chunk_size "$CE_CHUNK_SIZE"

    --do_baseline
    --do_final_eval
    --do_stage_eval

    --persist_every 500
    --save_steps 250
    --max_total_pflops "$MAX_TOTAL_PFLOPS"
)
# Fresh run (no --resume_from_job). Trains until
# --max_total_pflops = the 1.7B curriculum's cumulative compute at its
# Table 3 checkpoint (trainer-internal 6N accounting). Threshold 1.1 keeps
# the gate from firing. Requeue re-entry resumes from this run's own stable
# dir (get_last_checkpoint on the stable job dir).

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"

# ========== Training loop with OOM restart ==========
MAX_RETRIES=10
RETRY_COUNT=0
JOB_OUTPUT_DIR="$SCRATCH/nl_output/$TASK/job_${EFFECTIVE_JOB_ID}"
# prune a checkpoint left half-written by a preemption kill (trainer_state.json and curriculum_state.json are written last)
for d in "$JOB_OUTPUT_DIR"/checkpoint-*; do [ -d "$d" ] || continue; { [ -f "$d/trainer_state.json" ] && python -c "import json,sys; json.load(open(sys.argv[1])); json.load(open(sys.argv[2]))" "$d/trainer_state.json" "$d/curriculum_state.json" 2>/dev/null; } || { echo "[PRUNE] removing partial $d"; rm -rf "$d"; }; done
RESTART_FLAG="$JOB_OUTPUT_DIR/RESTART_FLAG"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "========== Attempt $((RETRY_COUNT+1))/$MAX_RETRIES | Port $MASTER_PORT =========="

    set +e
    torchrun --nproc_per_node=$GPUS_PER_NODE \
             --master_port=$MASTER_PORT \
             --max_restarts=0 \
             tuning_nl.py "${ARGS[@]}" &
    TPID=$!
    wait "$TPID"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS $(date)"
        rm -f "$RESTART_FLAG"
        exit 0
    elif [ -f "$RESTART_FLAG" ]; then
        echo "[OOM] Restarting with reduced batch size..."
        MASTER_PORT=$((MASTER_PORT + 1))
        RETRY_COUNT=$((RETRY_COUNT + 1))
        rm -f "$RESTART_FLAG"
        sleep 5
    elif tail -n 400 "$(ls -t ./slurm/${SLURM_JOB_ID}_* | head -1)" 2>/dev/null | grep -qE "HfHubHTTPError|ReadTimeout|Read timed out|429 Client Error|ConnectionError"; then
        echo "[HF] transient Hub error at startup; retrying in 180 s ($(date))"
        sleep 180
        MASTER_PORT=$((MASTER_PORT + 1))
        RETRY_COUNT=$((RETRY_COUNT + 1))

    else
        echo "FAILED with exit code $EXIT_CODE $(date)"
        exit $EXIT_CODE
    fi
done

echo "Max retries ($MAX_RETRIES) reached"
exit 1

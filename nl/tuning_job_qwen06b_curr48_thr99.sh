#!/bin/bash
#SBATCH -J nl_qwen06b_curr48_thr99
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=56
#SBATCH --time=48:00:00
#SBATCH --partition=ai
#SBATCH -A asaparov
#SBATCH -q normal
#SBATCH --mail-user=huan2073@purdue.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

set -euo pipefail

# ========== Preemption handling ==========
# The first attempt of this run (16501338) hit its walltime and was NOT requeued: the
# trap never fired, because torchrun ran in the FOREGROUND and bash defers traps until
# the foreground command returns. The fix is to background torchrun and `wait` on it, so
# USR1 interrupts the wait and the handler actually runs. TPID is set by the training
# loop below; stop the trainer first so it can flush, then requeue.
# NO_REQUEUE=1 suppresses the requeue handler. Set it when running this script INSIDE an
# allocation that belongs to a different job, e.g. `srun --jobid=<other> --overlap`. There
# $SLURM_JOB_ID is the host job, so the handler would requeue THAT job rather than this
# training run, killing an unrelated workload. With the guard set the run simply stops at
# walltime and the next ordinary sbatch of this script resumes from the last checkpoint.
if [ "${NO_REQUEUE:-0}" != "1" ]; then
trap 'set +e; echo "[SIG] USR1 @ $(date) — walltime near, stopping trainer and requeueing"; [ -n "${TPID:-}" ] && kill -TERM "$TPID" 2>/dev/null; sleep 120; scontrol requeue "$SLURM_JOB_ID" || { sleep 30; scontrol requeue "$SLURM_JOB_ID"; }; exit 0' USR1
else
echo "[INFO] NO_REQUEUE=1: USR1 requeue handler disabled (running inside a foreign allocation)"
fi
trap 'echo "[SIG] TERM @ $(date)"; exit 0' TERM
TPID=""

mkdir -p ./slurm
echo "JOB START $(date)"

# ========== Environment ==========
module load conda
conda activate search

export SCRATCH="/scratch/gautschi/$USER"
mkdir -p "$SCRATCH/nl_output" "$SCRATCH/model_cache" "$SCRATCH/triton_cache"
export HF_HOME="$SCRATCH/model_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"   # liger/triton must never write to home quota
if [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
fi
export HF_HUB_OFFLINE=0  # Need online for streaming Dolci-Instruct-SFT mix
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
# MASTER_PORT_OVERRIDE is needed when running inside another job's allocation: the default
# is derived from $SLURM_JOB_ID, so a second run under the same allocation would pick the
# port the first torchrun already holds, and a paused trainer still holds its socket.
export MASTER_PORT="${MASTER_PORT_OVERRIDE:-$((10000 + (SLURM_JOB_ID % 50000)))}"

echo "GPUS=$GPUS_PER_NODE  OMP_THREADS=$OMP_NUM_THREADS"
nvidia-smi || true

# ==========================================================
#   0.6B curriculum, s=1, 6% Dolci, ADVANCEMENT THRESHOLD 0.99. Target L=48.
#   Created 2026-09-22.
#
#   WHY. Every Qwen3-0.6B curriculum run in this project used the 0.98 gate. The
#   1.7B arm was switched to 0.99 and the 0.98 arms at that scale were retired, so
#   there is currently no 0.6B run at the reported threshold. (One exists,
#   job_9048131, but it is n_stages=16, max_input_size=96, and its checkpoints were
#   purged on 2026-06-30, so it has no weights and stops at L=16.)
#
#   This is job_retrain_curr96_s1_dolci6 with exactly one training parameter
#   changed: accuracy_threshold 0.98 -> 0.99. Everything that sets training dynamics
#   is identical to that run (eff_batch 768 = 48 x 4GA x 4 GPUs, lr 5e-5, warmup 100,
#   seed 1234, s=1, accuracy_window 200, min_steps_per_stage 200, check_every 25,
#   max_input_size 576, 6% Dolci-Instruct-SFT, chat template), so the two are
#   directly comparable at every L and any difference is the threshold alone.
#
#   COST. The 0.98 arm completed L=48 at 51,341 PFLOPs and ran 93,819 PFLOPs in
#   24.8 h on 4xH100, i.e. ~3,790 PFLOPs/h. At 1.7B the 0.99 gate cost 1.78x the
#   0.98 arm to reach the same L (71,812 vs 40,292 PFLOPs at L=32). If that ratio
#   carries over, L=48 lands near 91,000 PFLOPs, about 24 h on 4xH100. The cap is
#   left at the 0.98 arm's 300,000 PFLOPs, which is roughly L=64-96 territory.
#
#   The gate is a rolling exact-match rate over the last 200 search examples: 0.98
#   permits four errors in the window, 0.99 permits two.
#
#   Stage checkpoints are written at EVERY stage into stage_checkpoints/ and are
#   never rotated, so L=8/16/32/48 are all retained for downstream evaluation even
#   though the rolling checkpoint-* directories keep only the most recent few.
#
#   Walltime 48h with --requeue and a USR1 trap that actually fires. The output
#   directory is pinned to job_qwen06b_curr48_thr99 (EFFECTIVE_JOB_ID default
#   below), so every entry resumes from the last checkpoint in that dir.
# ==========================================================

TASK="search"
MODEL_NAME="Qwen/Qwen3-0.6B"

# Training  (run_meta: --batch_size 48 --gradient_accumulation_steps 4
#            --learning_rate 5e-5 --seed 1234; world_size 4 -> eff_batch 768)
BATCH_SIZE=48
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=5e-5
SEED=1234
FIRST_TOKEN_SOFT_WEIGHT=0.0

# Curriculum (run_meta: --n_stages 96 --base_alpha 0.1 --max_alpha 1.0
#             --accuracy_threshold 0.98 --min_steps_per_stage 200 --check_every 25
#             --accuracy_window 200 --eval_every_steps 0)
N_STAGES=96
BASE_ALPHA=0.1
MAX_ALPHA=1.0
ACCURACY_THRESHOLD=0.99
MIN_STEPS_PER_STAGE=200
CHECK_EVERY=25
ACCURACY_WINDOW=200
EVAL_EVERY_STEPS=0

# Task parameters, copied from job_retrain_curr96_s1_dolci6 unchanged. n=576 gives
# L_eff = min(((576-4)//3 - 1)//2, (576-4)//6) = 94, comfortably past the L=48 target.
# It is NOT raised to match that run's own ceiling exactly, because the point of this
# run is comparability with it, and under --linear_lookahead the graph at a given L is
# the same for any n large enough to hold it.
MAX_INPUT_SIZE=576
MAX_LOOKAHEAD=96

# Memory optimizations
CE_CHUNK_SIZE=4096

# Evaluation (run_meta: --eval_samples 500 --print_eval_examples 5 + all --do_* flags)
EVAL_SAMPLES=500
PRINT_EVAL_EXAMPLES=5

# Stop cap (see header)
MAX_TOTAL_PFLOPS=300000

# ==========================================================

# Pinned: this script only ever continues the one run in job_curr48_thr99. A bare
# `sbatch` without JOB_ID_OVERRIDE used to default to $SLURM_JOB_ID and start a fresh
# run in a new directory, silently abandoning the chain.
EFFECTIVE_JOB_ID="${JOB_ID_OVERRIDE:-qwen06b_curr48_thr99}"
echo "Task: $TASK | Model: $MODEL_NAME | Max Input: $MAX_INPUT_SIZE"
echo "Output: $SCRATCH/nl_output/$TASK/job_${EFFECTIVE_JOB_ID}"
echo "eff_batch = $BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS x $GPUS_PER_NODE = $(( BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * GPUS_PER_NODE ))  (must be 768)"

# Partial-checkpoint guard (added 2026-09-23 for the follow-on after 16544683): a trainer killed at a
# walltime can leave a half-written checkpoint-N, and the auto-resume takes the highest N. Any
# checkpoint missing a readable curriculum_state.json, model.safetensors, optimizer.pt or
# trainer_state.json is RENAMED (not deleted) to <name>.partial so the resume skips it.
OUT_DIR_GUARD="$SCRATCH/nl_output/$TASK/job_${EFFECTIVE_JOB_ID}"
for d in "$OUT_DIR_GUARD"/checkpoint-*; do
    [ -d "$d" ] || continue
    case "$d" in *.partial) continue;; esac
    if ! python -c "import json,sys; json.load(open(sys.argv[1]))" "$d/curriculum_state.json" 2>/dev/null \
       || [ ! -s "$d/model.safetensors" ] || [ ! -s "$d/optimizer.pt" ] || [ ! -s "$d/trainer_state.json" ]; then
        echo "[GUARD] incomplete checkpoint $d -> $d.partial"; mv "$d" "$d.partial"
    fi
done

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

    --linear_lookahead
    --base_lookahead 1
    --lookahead_step 1
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
    --max_total_pflops "$MAX_TOTAL_PFLOPS"
)
# NOTE: no --resume_from_job. Every entry, first submission or requeue, auto-resumes
# from the last complete checkpoint-* in the pinned job dir (get_last_checkpoint).

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"

# ========== Training loop with OOM restart ==========
MAX_RETRIES=10
RETRY_COUNT=0
JOB_OUTPUT_DIR="$SCRATCH/nl_output/$TASK/job_${EFFECTIVE_JOB_ID}"
RESTART_FLAG="$JOB_OUTPUT_DIR/RESTART_FLAG"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "========== Attempt $((RETRY_COUNT+1))/$MAX_RETRIES | Port $MASTER_PORT =========="

    set +e
    # Backgrounded so the USR1 trap can fire; see the trap comment at the top.
    torchrun --nproc_per_node=$GPUS_PER_NODE \
             --master_port=$MASTER_PORT \
             --max_restarts=0 \
             tuning_nl.py "${ARGS[@]}" &
    TPID=$!
    wait $TPID
    EXIT_CODE=$?
    TPID=""
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
    else
        echo "FAILED with exit code $EXIT_CODE $(date)"
        exit $EXIT_CODE
    fi
done

echo "Max retries ($MAX_RETRIES) reached"
exit 1

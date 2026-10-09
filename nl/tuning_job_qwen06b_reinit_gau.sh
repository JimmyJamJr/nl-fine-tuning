#!/bin/bash
#SBATCH -J nl_qwen06b_reinit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=7-00:00:00
#SBATCH --partition=ai
#SBATCH -A asaparov
#SBATCH -q normal
#SBATCH -o ./slurm/%j_%x.out
#SBATCH -e ./slurm/%j_%x.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#
# Qwen3-0.6B from RANDOM INITIALIZATION, curriculum s=1, continued on Gautschi.
#
# WHY. The randomly initialized trajectory has completed 61 stages and sits at only
# ~51% of its fitted ceiling (L_max ~ 120), so that ceiling is not identified: the
# residual block bootstrap gives [85, 3130] against [308, 345] for the pretrained
# model. Reaching L ~ 90 (about 75% of the fitted ceiling, matching where the
# pretrained trajectory sits) would pin it well enough to state whether random
# initialization has a genuinely lower asymptote or only a slower approach.
#
# CONTINUATION, NOT A NEW EXPERIMENT. This resumes gilbreth job 11695538, whose
# checkpoint-1217500 and loss_history were copied to
# $SCRATCH/nl_output/search/job_11695538_gilbreth. Every parameter below is taken
# from that job's run_meta.json so the trajectory stays a single comparable chain:
#   eff_batch 1536 = 96 x 4 GA x 4 GPUs (Table 7: random init, batch 1536)
#   lr 3e-4, warmup 2000, seed 1234, s=1, n_stages 256, n=1536, L cap 256
#   threshold 0.98, window 800, min_steps_per_stage 200, check_every 25
# The GPUs differ (4x H100 here, 4x A100-40GB on gilbreth) but eff_batch does not,
# so the compute-per-stage accounting is unaffected.
#
# --reinit_weights is deliberately ABSENT. The weights were randomised once at the
# start of the chain (job 11590956); re-randomising on resume would destroy it.
#
# The predecessor's CLI also passed --use_chunked_ce, a no-op flag removed from tuning_nl.py on
# 2026-10-09 (chunked CE is always on), so dropping it changes nothing.

set -uo pipefail
cd /home/huan2073/nl-fine-tuning/nl

# torchrun is backgrounded so the USR1 trap can actually fire: bash defers traps until
# a foreground command returns, which is why an earlier job hit its walltime without
# requeueing.
trap 'set +e; echo "[USR1] $(date): walltime near, stopping trainer and requeueing"; [ -n "${TPID:-}" ] && kill -TERM "$TPID" 2>/dev/null; sleep 120; scontrol requeue "$SLURM_JOB_ID" || { sleep 30; scontrol requeue "$SLURM_JOB_ID"; }; exit 0' USR1
TPID=""

module load conda
conda activate search

export SCRATCH="/scratch/gautschi/$USER"
export HF_HOME="$SCRATCH/model_cache"
export TORCH_HOME="$SCRATCH/model_cache"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NL_DDP_TIMEOUT_MIN=240
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$SCRATCH/triton_cache_reinit"
mkdir -p slurm "$TRITON_CACHE_DIR"

GPUS_PER_NODE=$(echo "$SLURM_JOB_GPUS" | tr "," "\n" | wc -l)
[ "$GPUS_PER_NODE" -eq 4 ] || { echo "FATAL: need 4 GPUs for eff_batch 1536 (96 x 4 x 4); got $GPUS_PER_NODE"; exit 1; }
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

JOB_ID="${JOB_ID_OVERRIDE:-qwen06b_reinit_gau}"
PREV_JOB="11695538_gilbreth"
PREV_DIR="$SCRATCH/nl_output/search/job_$PREV_JOB"
ls "$PREV_DIR"/checkpoint-*/model.safetensors >/dev/null 2>&1 || {
    echo "FATAL: no transferred checkpoint under $PREV_DIR"; exit 1; }
echo "predecessor: $(ls -d $PREV_DIR/checkpoint-* | tail -1)"

ARGS=(
    --task search
    --model_name Qwen/Qwen3-0.6B
    --cache_dir "$HF_HOME"
    --output_dir "$SCRATCH/nl_output"
    --scratch_dir "$SCRATCH"
    --job_id "$JOB_ID"
    --resume_from_job "$PREV_JOB"

    --batch_size 96
    --gradient_accumulation_steps 4
    --learning_rate 3e-4
    --seed 1234
    --first_token_soft_weight 0.0

    --n_stages 256
    --base_alpha 0.1
    --max_alpha 1.0
    --accuracy_threshold 0.98
    --min_steps_per_stage 200
    --check_every 25
    --accuracy_window 800
    --eval_every_steps 0

    --max_input_size 1536
    --max_lookahead 256
    --base_lookahead 1
    --lookahead_step 1

    --eval_samples 500
    --print_eval_examples 0
    --save_total_limit 2
    --ce_chunk_size 4096

    --linear_lookahead
    --use_liger
    --gradient_checkpointing
)

echo "JOB START $(date)  job_id=$JOB_ID  resume_from=$PREV_JOB"
echo "eff_batch = 96 x 4 x $GPUS_PER_NODE = $(( 96 * 4 * GPUS_PER_NODE ))  (must be 1536)"
echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE tuning_nl.py ${ARGS[*]}"
torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}" &
TPID=$!
wait $TPID
echo "JOB END $(date) rc=$?"

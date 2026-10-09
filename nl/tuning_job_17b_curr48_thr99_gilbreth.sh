#!/bin/bash
#SBATCH -J nl_17b_curr48_thr99_gil
#SBATCH -o slurm/%j_%x.out
#SBATCH -e slurm/%j_%x.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH -A asaparov
#SBATCH -p a100-80gb
#SBATCH -q standby
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#
# ==========================================================
#   GILBRETH continuation of job_curr48_thr99 (1.7B, s=1, 6% Dolci, gate 0.99,
#   ceiling L=128). Same chain as Gautschi's tuning_job_17b_curr48_thr99.sh; it
#   resumes from checkpoint-22000 (stage 33, L=32 completed), which was copied to
#   $SCRATCH/nl_output/search/job_curr48_thr99 on 2026-09-21 and verified
#   byte-exact. The Gautschi job (16544683) and this one are submitted together and
#   whichever trains first keeps the chain; the other is cancelled.
#
#   Training dynamics are unchanged: eff_batch 768 = 48 x GA x GPUs with GA
#   derived from the GPU count (a100-80gb nodes are 2-GPU, so GA=8). lr 5e-5,
#   warmup 100, seed 1234, threshold 0.99, window 200, min_steps 200, check_every
#   25, n=780, L cap 128, PFLOPs cap 2M.
#
#   Gilbreth-specific (all per gilbreth_pythia14b_REINIT3_resume.sh):
#   - The account owns zero a100-80gb GPUs, so this runs as 4h standby chunks.
#     --requeue + USR1 trap re-enter the same job id; every entry auto-resumes
#     from the latest intact checkpoint-* in the pinned run dir.
#   - --save_steps 150 / --save_total_limit 3 bound the loss per chunk to ~150
#     steps. Checkpoint cadence does not affect training dynamics.
#     stage_checkpoints/ (every stage) and persistent_checkpoints/ are unaffected.
#   - In-run evals are OFF (--eval_samples 0, no --do_* flags): rank-0 eval-set
#     generation ate whole 4h chunks on standby nodes for the Pythia chains. The
#     gate uses the training window, the L-vs-compute curve uses loss_history
#     stage transitions, and downstream evals load stage_checkpoints/, so nothing
#     the paper uses is lost. The Gautschi copy keeps its evals.
#   - HF_HUB_OFFLINE=0: the Dolci mix is loaded with streaming=True, which
#     refuses to start offline even with the dataset cached. A pre-flight probe
#     fails fast and loudly if the compute node cannot reach the Hub.
#   - NCCL P2P/IB disabled: 2-GPU NCCL hung at the first collective on the
#     a100-80gb k-nodes.
#   - Runs from nl_code_thr99/ (Gautschi's tuning_nl.py + nl_generator.py) so the
#     Pythia chains' nl_code/ is untouched; the C++ generator is built on first use.
#   - Pre-flight quarantines a truncated latest checkpoint (the Gautschi chain was
#     killed mid-save once and the next start died on the half-written file).
# ==========================================================

source /etc/profile
set -euo pipefail

TPID=""
trap 'set +e; echo "[USR1] $(date) walltime near: stopping trainer, requeueing"; if [ -n "${TPID:-}" ]; then kill -TERM "$TPID" 2>/dev/null; for i in $(seq 1 24); do kill -0 "$TPID" 2>/dev/null || break; sleep 5; done; fi; scontrol requeue "$SLURM_JOB_ID" || { sleep 20; scontrol requeue "$SLURM_JOB_ID"; }; exit 0' USR1
trap 'echo "[TERM] $(date)"; exit 0' TERM

module load conda/2025.09 cuda/12.6.0
conda activate "${NL_CONDA_ENV:-/scratch/gilbreth/$USER/conda_envs/search}"

export SCRATCH="/scratch/gilbreth/$USER"
export HF_HOME="$SCRATCH/model_cache"
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR="$SCRATCH/triton_cache_thr99"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NL_DDP_TIMEOUT_MIN=40
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p slurm "$TRITON_CACHE_DIR"

cd "$SCRATCH/nl_code_thr99"
echo "JOB START $(date)  host=$(hostname)  restarts=${SLURM_RESTART_COUNT:-0}"

# ========== Pre-flight: Hub reachability (Dolci streaming needs it) ==========
code=$(curl -sS -m 20 -o /dev/null -w "%{http_code}" https://huggingface.co/api/datasets/allenai/Dolci-Instruct-SFT || echo "000")
if [ "$code" != "200" ]; then
    echo "[PREFLIGHT] FATAL: huggingface.co unreachable from $(hostname) (HTTP $code); the Dolci mix cannot stream. Not requeueing."
    exit 1
fi
echo "[PREFLIGHT] Hub reachable (HTTP $code)"

# ========== Pre-flight: quarantine a truncated latest checkpoint ==========
TASK="search"
EFFECTIVE_JOB_ID="${JOB_ID_OVERRIDE:-curr48_thr99}"
RUN_DIR="$SCRATCH/nl_output/$TASK/job_${EFFECTIVE_JOB_ID}"
python - "$RUN_DIR" <<'PY'
import os, sys, json, struct, glob, shutil
d = sys.argv[1]
q = os.path.join(os.path.dirname(d), "_corrupt")
NEED = ["model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json", "curriculum_state.json"]
def intact(c):
    if any(not os.path.isfile(os.path.join(c, f)) for f in NEED):
        return False
    p = os.path.join(c, "model.safetensors")
    try:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        end = max(v["data_offsets"][1] for k, v in h.items() if k != "__metadata__")
        return 8 + n + end == os.path.getsize(p)
    except Exception:
        return False
cks = sorted(glob.glob(os.path.join(d, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
while cks and not intact(cks[-1]):
    bad = cks.pop()
    os.makedirs(q, exist_ok=True)
    dst = os.path.join(q, os.path.basename(d) + "_" + os.path.basename(bad))
    shutil.move(bad, dst)
    print(f"[PREFLIGHT] quarantined truncated {bad} -> {dst}")
if not cks:
    print(f"[PREFLIGHT] FATAL: no intact checkpoint under {d}")
    sys.exit(1)
print(f"[PREFLIGHT] resume checkpoint: {cks[-1]}")
PY

# ========== Hardware / eff_batch ==========
GPUS_PER_NODE=$(echo "$SLURM_JOB_GPUS" | tr "," "\n" | wc -l)
[ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
BATCH_SIZE=48
case "$GPUS_PER_NODE" in
    2) GRADIENT_ACCUMULATION_STEPS=8 ;;
    4) GRADIENT_ACCUMULATION_STEPS=4 ;;
    *) echo "[FATAL] need 2 or 4 GPUs for eff_batch 768 at bs=48; got $GPUS_PER_NODE"; exit 1 ;;
esac
EFF=$(( BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * GPUS_PER_NODE ))
[ "$EFF" -eq 768 ] || { echo "[FATAL] eff_batch=$EFF != 768"; exit 1; }
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "GPUS=$GPUS_PER_NODE  bs=$BATCH_SIZE  GA=$GRADIENT_ACCUMULATION_STEPS  eff_batch=$EFF  OMP=$OMP_NUM_THREADS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ========== Build C++ generator if needed ==========
python -c "
try:
    import generator
    print('[OK] C++ generator present')
except Exception:
    print('[INFO] Building C++ generator...')
    import subprocess, sys
    subprocess.check_call([sys.executable, 'nl_generator.py'])
"

# ========== Arguments (training dynamics identical to the Gautschi script) ==========
ARGS=(
    --task "$TASK"
    --model_name "Qwen/Qwen3-1.7B"
    --cache_dir "$HF_HOME"
    --output_dir "$SCRATCH/nl_output"
    --scratch_dir "$SCRATCH"
    --job_id "$EFFECTIVE_JOB_ID"

    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate 5e-5
    --warmup_steps 100
    --seed 1234
    --num_shots 0
    --first_token_soft_weight 0.0

    --n_stages 128
    --base_alpha 0.1
    --max_alpha 1.0
    --accuracy_threshold 0.99
    --min_steps_per_stage 200
    --check_every 25
    --accuracy_window 200
    --eval_every_steps 0

    --max_input_size 780
    --max_lookahead 128
    --max_frontier_size 12
    --max_branch_size 12
    --requested_backtrack 3

    --eval_samples 0
    --print_eval_examples 0

    --use_packing
    --linear_lookahead
    --base_lookahead 1
    --lookahead_step 1
    --mix_pretrain_data allenai/Dolci-Instruct-SFT
    --mix_pretrain_ratio 0.06
    --use_chat_template
    --stage_eval_every 8

    --gradient_checkpointing
    --use_liger
    --ce_chunk_size 4096

    --save_steps 150
    --save_total_limit 3
    --persist_every 500
    --max_total_pflops 2000000
)

echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"
set +e
torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" --max_restarts=0 \
         tuning_nl.py "${ARGS[@]}" &
TPID=$!
wait "$TPID"
EXIT=$?
TPID=""
set -e
echo "torchrun exited $EXIT at $(date)"
exit $EXIT

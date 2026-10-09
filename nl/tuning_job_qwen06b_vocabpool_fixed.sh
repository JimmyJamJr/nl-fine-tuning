#!/bin/bash
# GAUTSCHI (preemptible, 4 H100s, auto-requeue): entity-vocabulary ablation, Qwen3-0.6B, s=8.
#   Arm "fixed": vertex-ID -> name dictionary with the ID range pinned at the context maximum: 255 names (IDs 1..255) throughout.
#   Both arms replicate the Section 4.1 joint run (jobs 8533255 -> 8555128) exactly: bs 48 x GA 4 x 4 GPUs
#   = 768, lr 5e-5, warmup 100, seed 1234, n = 768, s = 8 to L = 128 (16 stages), gate 98% on W = 800,
#   packing / liger / chunked CE / grad checkpointing, no Dolci mix, no chat template. Budget = the joint
#   run's compute at completing L = 128 under the paper's concatenated-chain accounting (87,483 PFLOPs at step 22,775 of
#   job 8555128; Figure 4 / Table 10); the trainer stops there or when
#   all 16 stages are complete, whichever comes first.
#SBATCH -J nl_qwen06b_vocabpool_fixed
#SBATCH -o /home/huan2073/nl-fine-tuning/nl/slurm/%j_%x.out
#SBATCH -e /home/huan2073/nl-fine-tuning/nl/slurm/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=2-00:00:00
#SBATCH -A asaparov
#SBATCH -p ai
#SBATCH -q preemptible
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300

set -euo pipefail
cd /home/huan2073/nl-fine-tuning/nl
module load conda
conda activate search
trap 'set +e; echo "[USR1] $(date): walltime near, stopping trainer and requeueing"; [ -n "${TPID:-}" ] && kill -TERM "$TPID" 2>/dev/null; sleep 60; scontrol requeue "$SLURM_JOB_ID" || { sleep 30; scontrol requeue "$SLURM_JOB_ID"; }; exit 0' USR1

export SCRATCH="/scratch/gautschi/$USER"
export HF_HOME="$SCRATCH/model_cache"
export TORCH_HOME="$SCRATCH/model_cache"
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR="$SCRATCH/triton_cache_vocabpool_fixed"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NL_DDP_TIMEOUT_MIN=240
export PYTHONUNBUFFERED=1
# The fast tokenizer otherwise spawns its own Rust thread pool inside every forked dataloader
# worker, on top of the OpenMP threads they inherit. The trainer pins workers to one thread via
# worker_init_fn; this closes the same hole for the tokenizer. The nocurr17b scripts already set
# this and these did not, which was an oversight rather than a deliberate difference.
export TOKENIZERS_PARALLELISM=false

# Selective gradient checkpointing: recompute every 2nd layer instead of every layer while the
# micro-batch is small enough to hold the extra activations, and fall back to every layer above
# that. Gradients are identical either way, so training dynamics are unchanged; this is purely a
# memory/compute trade. Measured on this exact configuration (Qwen3-0.6B, batch 48 x grad-accum 4
# x 4 ranks, 4xH100 80GB): 310,358 tok/s against 218,664 at L=88, a 1.42x gain, with peak memory
# 69,949 MiB of 81,559. The gain disappears above ~88k tokens per micro-batch, where every-2nd
# OOMs (80,837 MiB at L=104), so the gate hands those stages back to every-layer, which needs only
# 74,127 MiB even at L=128. This run climbs to L=128, so it crosses the boundary and relies on the
# fallback. The thresholds are specific to this model and batch geometry; see tuning_nl.py.
export NL_CKPT_EVERY_N_LAYERS=2
export NL_CKPT_RELAX_MAX_TOKENS=88000
mkdir -p slurm "$TRITON_CACHE_DIR"

GPUS_PER_NODE=$(echo "$SLURM_JOB_GPUS" | tr "," "\n" | wc -l)
[ "$GPUS_PER_NODE" -eq 4 ] || { echo "FATAL: need 4 GPUs for eff_batch 768"; exit 1; }
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

JOB_ID="qwen06b_vocabpool_fixed"
OUT_DIR="$SCRATCH/nl_output/search/job_$JOB_ID"
# prune a checkpoint left half-written by a preemption kill (curriculum_state.json is written last)
for d in "$OUT_DIR"/checkpoint-*; do [ -d "$d" ] || continue; python -c "import json,sys; json.load(open(sys.argv[1]))" "$d/curriculum_state.json" 2>/dev/null || { echo "[PRUNE] removing partial $d"; rm -rf "$d"; }; done

ARGS=(
    --task search
    --model_name Qwen/Qwen3-0.6B
    --cache_dir "$HF_HOME"
    --output_dir "$SCRATCH/nl_output"
    --scratch_dir "$SCRATCH"
    --job_id "$JOB_ID"
    --batch_size 48
    --gradient_accumulation_steps 4
    --learning_rate 5e-5
    --seed 1234
    --first_token_soft_weight 0.0
    --n_stages 16
    --base_alpha 0.1
    --max_alpha 1.0
    --accuracy_threshold 0.98
    --min_steps_per_stage 200
    --check_every 25
    --accuracy_window 800
    --eval_every_steps 0
    --max_input_size 768
    --max_lookahead 128
    --linear_lookahead
    --base_lookahead 8
    --lookahead_step 8
    --vocab_pool fixed
    --eval_samples 500
    --print_eval_examples 0
    --ce_chunk_size 4096
    --use_liger
    --gradient_checkpointing
    --do_stage_eval
    --do_final_eval
    --persist_every 1000
    --save_steps 250
    --max_total_pflops 87483
)
echo "eff_batch = 48 x 4 x $GPUS_PER_NODE = $(( 48 * 4 * GPUS_PER_NODE ))  (must be 768) | arm fixed | output $OUT_DIR | chunk $(date)"
echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT tuning_nl.py ${ARGS[*]}"
torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}" &
TPID=$!
wait "$TPID"

#!/bin/bash
# Breadth ablation at the main-text step size: Qwen3-0.6B, s=8, gate window 1000, breadth B (3 or 4).
# Created 2026-09-25 at the user's request ("one with b=3 and one with b=4, using step size = 8 and W = 1000").
#
# CONFIG. Replicates the paper's s=8 joint run (jobs 8533255 -> 8555128): batch 48 x GA 4 x 4 GPUs = 768,
# lr 5e-5, warmup 100, seed 1234, no Dolci mix, no chat template, packing / liger / grad checkpointing,
# gate 98% with min 200 steps per stage checked every 25. Changed on purpose: accuracy_window 1000 (joint run
# used 200) and --breadth B (joint run was the default 2). Target L = 256 as in the paper, n_stages 32.
# max_input_size 3080 is the smallest n with (n-5)//3 = (n-4)//3 >= 4*256+1, so b=4 can reach L=256
# (n = 2 mod 3 avoids the (n-4)//3 vs (n-5)//3 mismatch). Under --linear_lookahead graph size per stage does
# not depend on n. Budget 1,500,000 PFLOPs so compute never ends a run before its gate does.
#
# KNOWN BEHAVIOUR, KEPT DELIBERATELY (user decision 2026-09-25). Training batches pass the run's global
# max_lookahead (256) to the generator, not the stage's L, so the generator caps an instance's lookahead at
# ~(B*L+1)/2. At B>2 a stage labelled L therefore trains on lookaheads 0..min(B*L/2, 256): up to 1.5L at B=3,
# 2L at B=4. This matches Max's s=1 breadth runs. For B=2 it is exactly 0..L.
#
# RUN IT INSIDE A FROZEN ALLOCATION (no USR1 requeue handler here on purpose; `scontrol requeue` inside a
# borrowed job would restart that job's own script):
#   B=4 on h015 (job 16580669, frozen shell 3173573):
#   setsid nohup srun --jobid=16580669 --overlap env SLURM_JOB_GPUS=0,1,2,3 SLURM_CPUS_PER_TASK=56 B=4 \
#       bash tuning_job_qwen06b_s8_breadth_w1000.sh > slurm/qwen06b_s8_b4_w1000.out 2>&1 < /dev/null & disown
#   B=3 on h011 (job 16455237, frozen shell 394789): same with --jobid=16455237 B=3 and the b3 log name.
#   (B=3 started on h019 / job 16455236 but was moved at step ~1,200 on 2026-09-25 01:55: h019 GPU 2 sits at
#   86-88 C with SW thermal slowdown, 405-735 MHz, which halved the whole DDP run. It resumed from checkpoint-1000.)
# Re-running the same command resumes from the latest checkpoint-N in the run directory.

set -uo pipefail
cd /home/huan2073/nl-fine-tuning/nl
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
# Selective gradient checkpointing (memory/compute trade only; gradients identical).
export NL_CKPT_EVERY_N_LAYERS=2
export NL_CKPT_RELAX_MAX_TOKENS=88000

: "${B:?B must be set to 3 or 4}"
case "$B" in 3|4) ;; *) echo "FATAL: B must be 3 or 4 (got $B)"; exit 1 ;; esac
S_STEP=8; MAXL=256; NSTAGES=32; NIN=3080; PFLOP_CAP=1500000; WINDOW=1000
[ $(( MAXL / S_STEP )) -eq "$NSTAGES" ] || { echo "FATAL: n_stages $NSTAGES != $MAXL/$S_STEP"; exit 1; }
[ $(( (NIN - 5) / 3 )) -ge $(( B * MAXL + 1 )) ] || { echo "FATAL: n=$NIN cannot reach L=$MAXL at breadth $B"; exit 1; }
export TRITON_CACHE_DIR="$SCRATCH/triton_cache_s8_b${B}_w1000"
mkdir -p slurm "$TRITON_CACHE_DIR"

GPUS_PER_NODE=$(echo "${SLURM_JOB_GPUS:-}" | tr "," "\n" | grep -c .)
[ "$GPUS_PER_NODE" -eq 4 ] || { echo "FATAL: need 4 GPUs for eff_batch 768 (SLURM_JOB_GPUS='${SLURM_JOB_GPUS:-}')"; exit 1; }
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-56} / GPUS_PER_NODE ))
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

JOB_ID="qwen06b_s8_b${B}_w1000"
OUT_DIR="$SCRATCH/nl_output/search/job_$JOB_ID"
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
    --n_stages "$NSTAGES"
    --base_alpha 0.1
    --max_alpha 1.0
    --accuracy_threshold 0.98
    --min_steps_per_stage 200
    --check_every 25
    --accuracy_window "$WINDOW"
    --eval_every_steps 0
    --max_input_size "$NIN"
    --max_lookahead "$MAXL"
    --linear_lookahead
    --base_lookahead "$S_STEP"
    --lookahead_step "$S_STEP"
    --breadth "$B"
    --eval_samples 500
    --print_eval_examples 0
    --ce_chunk_size 4096
    --use_packing
    --use_liger
    --gradient_checkpointing
    --do_stage_eval
    --do_final_eval
    --persist_every 2000
    --save_steps 500
    --save_total_limit 2
    --max_total_pflops "$PFLOP_CAP"
)
echo "JOB START $(date)  s=$S_STEP  breadth=$B  window=$WINDOW  n=$NIN  cap=$MAXL  budget=$PFLOP_CAP  (inside SLURM job ${SLURM_JOB_ID:-?})"
echo "Command: torchrun --nproc_per_node=$GPUS_PER_NODE tuning_nl.py ${ARGS[*]}"
exec torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" tuning_nl.py "${ARGS[@]}"

#!/bin/bash
# Context-length test (reviewer point: are the Pythia ceilings confounded by the 2,048-token
# pretraining context?), 2026-09-23. See context_length_eval.py for the two designs.
# Runs inside eval_worker job 16604009 on g005 with priority, like run_frontier_eval.sh: the
# depth-benchmark eval processes are SIGSTOPped for the duration and resumed on exit.
cd /home/huan2073/nl-fine-tuning/nl
export HF_HOME=/scratch/gautschi/huan2073/model_cache TORCH_HOME=/scratch/gautschi/huan2073/model_cache
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=$HOME/.conda/envs/search/bin/python
S=/scratch/gautschi/huan2073/nl_output/search
mkdir -p eval_context

PIDS=$(pgrep -u "$USER" -f "eval_downstream.py" | tr '\n' ' ')
echo "[context] $(date +%H:%M:%S) pausing depth-benchmark eval pids: ${PIDS:-none}"
[ -n "$PIDS" ] && kill -STOP $PIDS
trap '[ -n "$PIDS" ] && kill -CONT $PIDS; echo "[context] $(date +%H:%M:%S) resumed: ${PIDS:-none}"' EXIT

echo "######## pythia28b_L133 $(date +%H:%M:%S)"
$PY context_length_eval.py --ckpt $S/job_11647896/stage_checkpoints/stage_133_step_60025_L133 \
    --model_name EleutherAI/pythia-2.8b --n 1536 --max_lookahead 256 --stage_L 133 \
    --la_lo 30 --la_hi 70 --n_natural 600 --la 40,60 --n_base 150 --targets 1500,1900,2300,2700,3100,3500 \
    --out eval_context/pythia28b_L133.json || echo "FAILED pythia28b"
# 1.4B trained only up to ~2,100 tokens (L=75), so lengths past that are also past its training
# lengths; keep the sweep short and read anything above ~2,100 with that in mind.
echo "######## pythia14b_L75 $(date +%H:%M:%S)"
$PY context_length_eval.py --ckpt $S/job_15232695/stage_checkpoints/stage_75_step_2375625_L75 \
    --model_name EleutherAI/pythia-1.4b --n 576 --max_lookahead 96 --stage_L 75 \
    --la_lo 30 --la_hi 70 --n_natural 600 --la 40 --n_base 150 --targets 1500,1900,2300,2700 \
    --out eval_context/pythia14b_L75.json || echo "FAILED pythia14b"
echo "######## ALL DONE $(date +%H:%M:%S)"

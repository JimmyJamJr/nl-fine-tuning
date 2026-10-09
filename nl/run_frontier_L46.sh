#!/bin/bash
# Threshold 0.99 vs 0.98 at the deepest clean point for Qwen3-1.7B (2026-09-23). The 0.98 run used
# n=288, whose generator cap is lookahead 46 ((288-5)//3-1)//2), so its "L=47/48" stages passed their
# gates on graphs no deeper than 46 and lookahead-48 items cannot be generated under its config. L=46 is
# therefore the deepest like-for-like pair (same edge budget: alpha x (n-5)//3 = 93 edges in both runs).
# The 0.99 run's L=48 checkpoint (the downstream curriculum arm) is scored alone.
# Short prompts (~1K tokens): runs alongside the depth-benchmark evals, no pausing.
cd /home/huan2073/nl-fine-tuning/nl
export HF_HOME=/scratch/gautschi/huan2073/model_cache TORCH_HOME=/scratch/gautschi/huan2073/model_cache
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=$HOME/.conda/envs/search/bin/python
S=/scratch/gautschi/huan2073/nl_output/search
both() { tag=$1; shift
  echo "######## $tag $(date +%H:%M:%S)"
  $PY frontier_eval.py "$@" --out eval_frontier/$tag.json || echo "FAILED $tag"
  $PY frontier_eval.py "$@" --exact_only 300 --out eval_frontier/exact_$tag.json || echo "FAILED exact_$tag"; }
Q17=Qwen/Qwen3-1.7B
both qwen17b_thr99_L46 --ckpt $S/job_curr48_thr99/stage_checkpoints/stage_46_step_30550_L46 --model_name $Q17 --L 46 --n 780 --max_lookahead 128 --chat
both qwen17b_thr98_L46 --ckpt $S/job_retrain_17b_curr48_s1_dolci6/stage_checkpoints/stage_46_step_17775_L46 --model_name $Q17 --L 46 --n 288 --max_lookahead 48 --chat
both qwen17b_thr99_L48 --ckpt $S/job_curr48_thr99/stage_checkpoints/stage_48_step_31825_L48 --model_name $Q17 --L 48 --n 780 --max_lookahead 128 --chat
echo "######## ALL DONE $(date +%H:%M:%S)"

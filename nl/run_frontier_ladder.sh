#!/bin/bash
# Threshold 0.99 vs 0.98 on the standard Qwen3-1.7B ladder L = 8, 16, 32, 48 (2026-09-23).
# L=32 was run in run_frontier_gates.sh. L=48: the 0.98 run's own config (n=288) caps generated
# lookahead at 46, so BOTH arms are scored on the 0.99 run's L=48 distribution (n=780, same seed, same
# items); the 0.98 checkpoint never trained on depth 47-48, which this measures. L=8/16 use each run's own
# config (identical items: same seed and edge budget). 0.98 L=48 first: g005 ends 05:16.
cd /home/huan2073/nl-fine-tuning/nl
export HF_HOME=/scratch/gautschi/huan2073/model_cache TORCH_HOME=/scratch/gautschi/huan2073/model_cache
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=$HOME/.conda/envs/search/bin/python
S=/scratch/gautschi/huan2073/nl_output/search
both() { tag=$1; shift
  echo "######## $tag $(date +%H:%M:%S)"
  $PY frontier_eval.py "$@" --exact_only 300 --out eval_frontier/exact_$tag.json || echo "FAILED exact_$tag"
  $PY frontier_eval.py "$@" --out eval_frontier/$tag.json || echo "FAILED $tag"; }
Q17=Qwen/Qwen3-1.7B; T99=$S/job_curr48_thr99/stage_checkpoints; T98=$S/job_retrain_17b_curr48_s1_dolci6/stage_checkpoints
both qwen17b_thr98_L48_on780 --ckpt $T98/stage_48_step_18325_L48 --model_name $Q17 --L 48 --n 780 --max_lookahead 128 --chat
both qwen17b_thr99_L8  --ckpt $T99/stage_8_step_3350_L8   --model_name $Q17 --L 8  --n 780 --max_lookahead 128 --chat
both qwen17b_thr98_L8  --ckpt $T98/stage_8_step_2300_L8   --model_name $Q17 --L 8  --n 288 --max_lookahead 48  --chat
both qwen17b_thr99_L16 --ckpt $T99/stage_16_step_9600_L16 --model_name $Q17 --L 16 --n 780 --max_lookahead 128 --chat
both qwen17b_thr98_L16 --ckpt $T98/stage_16_step_5025_L16 --model_name $Q17 --L 16 --n 288 --max_lookahead 48  --chat
echo "######## ALL DONE $(date +%H:%M:%S)"

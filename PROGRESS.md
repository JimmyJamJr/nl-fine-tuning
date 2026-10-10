# Experiment 2: Pythia-1.4B Intermediate Pretraining Checkpoints (Step Size 1)

**Base commit:** `da7b19c` (`main`, plus `--revision` support in `nl/tuning_nl.py`)
**Branch:** `jackierwzhang/exp2-pythia14b-revisions`
**VM Instance:** `nl-exp2-pythia14b` (`a2-highgpu-8g`, 8x NVIDIA A100-SXM4-40GB, `asia-southeast1-c`)
**Last updated:** 2026-10-10 01:47 UTC

## Shared Configuration

| Parameter | Value |
| :--- | :--- |
| Model | `EleutherAI/pythia-1.4b` (`--revision step1000`, `step10000`, `step100000`) |
| Task | `search` |
| Seed | `1234` |
| Topology | 4x NVIDIA A100-SXM4-40GB per run (`NL_DDP_GRAD_AVERAGE` unset, `NL_ATTN_KERNEL` unset) |
| Batch size / Grad accum | `24` per GPU, `gradient_accumulation_steps=2` |
| Learning rate | `2.7e-5` (constant, 0 warmup) |
| Curriculum | `base_lookahead=1`, `lookahead_step=1`, `n_stages=96` (`L=1..96`), `--linear_lookahead` |
| Gate | `accuracy_threshold=0.98`, `accuracy_window=800`, `check_every=25`, `min_steps_per_stage=200` |
| Context & Alpha | `max_input_size=576`, `max_lookahead=96`, `base_alpha=0.1`, `max_alpha=1.0` |
| Loss & Memory | `first_token_soft_weight=0.0`, `ce_chunk_size=4096`, `--gradient_checkpointing` |
| Evals & Checkpoints | `--do_stage_eval`, `stage_eval_every=8`, `eval_samples=500`, `eval_every_steps=0`, `save_steps=500`, `save_total_limit=2`, `persist_every=0` |
| Compute Budget | `max_total_pflops=500000` (0.5M PFLOPs per run) |

## Run Status

| Arm | Revision | Pretraining Tokens | Job ID | GPUs | Status | Step | Stage / $L$ | Rolling Full Acc | Rolling First Acc | Recent Loss | PFLOPs / 500k | TFLOP/s | Wall Time |
| :--- | :--- | :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `step1000` | `step1000` | ~2.1B | `exp2_pythia14b_step1000` | GPUs 0,1,2,3 | Running | 7150 | 4/96 (`L=4`) | 87.88% | 88.38% | 0.1018 | 1,233.6 | 403.7 | 0.88h |
| `step10000` | `step10000` | ~21.0B | `exp2_pythia14b_step10000` | GPUs 4,5,6,7 | Running | 6060 | 6/96 (`L=6`) | 98.75% | 99.12% | 0.0210 | 1,436.4 | 452.2 | 0.88h |
| `step100000` | `step100000` | ~209.7B | `exp2_pythia14b_step100000` | Wave 2 (4 GPUs) | Queued | 0 | - | - | - | - | 0.0 | - | 0.00h |

## Stage Evaluation Summary (`stage_eval_history.json`, every 8 stages)

_No stage evaluations recorded yet (first stage evaluation runs upon completing Stage 8, $L=8$)._

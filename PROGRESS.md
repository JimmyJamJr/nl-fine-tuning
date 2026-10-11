# Experiment 2: Pythia-1.4B Intermediate Pretraining Checkpoints (Step Size 1)

**Base commit:** `da7b19c` (`main`, plus `--revision` support in `nl/tuning_nl.py`)
**Branch:** `jackierwzhang/exp2-pythia14b-revisions`
**VM Instances:** `nl-exp2-pythia14b` (`a2-highgpu-8g`, 8x A100-SXM4-40GB, `asia-southeast1-c`) + `nl-exp2-spot-step100000` (`a3-highgpu-4g` Spot, 4x H100-80GB)
**Last updated:** 2026-10-11 02:59 UTC

## Shared Configuration

| Parameter | Value |
| :--- | :--- |
| Model | `EleutherAI/pythia-1.4b` (`--revision step1000`, `step10000`, `step100000`) |
| Task | `search` |
| Seed | `1234` |
| Topology | 4 GPUs per run (`step1000` & `step10000` on 4x A100-40GB; `step100000` on 4x H100-80GB Spot with `NL_ATTN_KERNEL=fa2`; `NL_DDP_GRAD_AVERAGE` unset) |
| Batch size / Grad accum | `24` per GPU, `gradient_accumulation_steps=2` |
| Learning rate | `2.7e-5` (constant, 0 warmup) |
| Curriculum | `base_lookahead=1`, `lookahead_step=1`, `n_stages=96` (`L=1..96`), `--linear_lookahead` |
| Gate | `accuracy_threshold=0.98`, `accuracy_window=800`, `check_every=25`, `min_steps_per_stage=200` |
| Context & Alpha | `max_input_size=576`, `max_lookahead=96`, `base_alpha=0.1`, `max_alpha=1.0` |
| Loss & Memory | `first_token_soft_weight=0.0`, `ce_chunk_size=4096`, `--gradient_checkpointing` |
| Evals & Checkpoints | `--do_stage_eval`, `stage_eval_every=8`, `eval_samples=500`, `eval_every_steps=0`, `save_steps=500`, `save_total_limit=2`, `persist_every=0` |
| Compute Budget | `max_total_pflops=500000` (0.5M PFLOPs per run) |

## Run Status

| Arm | Revision | Pretraining Tokens | Job ID | GPUs | Status | Step | Stage / $L$ | Rolling Full Acc | Rolling First Acc | Recent Loss | PFLOPs / 500k | TFLOP/s | Wall Time | Est. Remaining |
| :--- | :--- | :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `step1000` | `step1000` | ~2.1B | `exp2_pythia14b_step1000` | GPUs 0,1,2,3 | Running | 121100 | 14/96 (`L=14`) | 96.62% | 98.00% | 0.0300 | 49,596.7 | 518.2 | 26.05h | ~153.0h (stg rate, 14.6% L) / 224.0h (500k cap @ 2,011 PF/h) |
| `step10000` | `step10000` | ~21.0B | `exp2_pythia14b_step10000` | GPUs 4,5,6,7 | Running | 101570 | 17/96 (`L=17`) | 96.50% | 99.00% | 0.0231 | 51,276.6 | 528.7 | 26.05h | ~121.4h (stg rate, 17.7% L) / 218.9h (500k cap @ 2,050 PF/h) |
| `step100000` | `step100000` | ~209.7B | `exp2_pythia14b_step100000` | Spot 4x H100 (us-east4-b) | Running | 112960 | 39/96 (`L=39`) | 96.38% | 98.62% | 0.0245 | 105,808.0 | 1287.2 | 22.84h | ~33.4h (stg rate, 40.6% L) / 78.9h (500k cap @ 4,996 PF/h) |

## Stage Evaluation Summary (`stage_eval_history.json`, every 8 stages)

### Arm: `step1000` (`exp2_pythia14b_step1000`)

| Stage | Step | $L$ | Stage $\alpha$ | TF Loss ($\alpha=1.0$) | Greedy First ($\alpha=1.0$) | Greedy Full ($\alpha=1.0$) | Greedy First (Stage $\alpha$) | Greedy Full (Stage $\alpha$) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 36500 | 8 | 0.0895 | 0.7263 | 34.80% | 34.00% | 99.00% | 98.80% |

### Arm: `step10000` (`exp2_pythia14b_step10000`)

| Stage | Step | $L$ | Stage $\alpha$ | TF Loss ($\alpha=1.0$) | Greedy First ($\alpha=1.0$) | Greedy Full ($\alpha=1.0$) | Greedy First (Stage $\alpha$) | Greedy Full (Stage $\alpha$) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 13025 | 8 | 0.0895 | 0.6055 | 43.40% | 41.60% | 99.00% | 99.00% |
| 16 | 88650 | 16 | 0.1737 | 0.5499 | 41.00% | 39.80% | 98.20% | 98.00% |

### Arm: `step100000` (`exp2_pythia14b_step100000`)

| Stage | Step | $L$ | Stage $\alpha$ | TF Loss ($\alpha=1.0$) | Greedy First ($\alpha=1.0$) | Greedy Full ($\alpha=1.0$) | Greedy First (Stage $\alpha$) | Greedy Full (Stage $\alpha$) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 3725 | 8 | 0.0895 | 0.8706 | 36.80% | 35.60% | 98.60% | 98.40% |
| 16 | 16175 | 16 | 0.1737 | 0.5736 | 43.40% | 42.00% | 98.80% | 98.60% |
| 24 | 43500 | 24 | 0.2579 | 0.5343 | 45.20% | 44.00% | 98.80% | 98.60% |
| 32 | 81725 | 32 | 0.3421 | 0.3851 | 55.20% | 54.80% | 98.40% | 98.20% |


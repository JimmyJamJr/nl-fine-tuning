# Experiment 2: Pythia-1.4B Intermediate Pretraining Checkpoints (Step Size 1)

**Base commit:** `da7b19c` (`main`, plus `--revision` support in `nl/tuning_nl.py`)
**Branch:** `jackierwzhang/exp2-pythia14b-revisions`
**VM Instances:** `nl-exp2-spot-step1000`, `nl-exp2-spot-step10000`, `nl-exp2-spot-step100000` (3x `a3-highgpu-4g` Spot, 4x H100-80GB each)
**Last updated:** 2026-10-13 09:14 UTC

## Shared Configuration

| Parameter | Value |
| :--- | :--- |
| Model | `EleutherAI/pythia-1.4b` (`--revision step1000`, `step10000`, `step100000`) |
| Task | `search` |
| Seed | `1234` |
| Topology | 4 GPUs per run (`a3-highgpu-4g` Spot 4x H100-80GB per arm with `NL_ATTN_KERNEL=fa2`; `NL_DDP_GRAD_AVERAGE` unset) |
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
| `step1000` | `step1000` | ~2.1B | `exp2_pythia14b_step1000` | Spot 4x H100 (us-east4-a) | Running | 453180 | 26/96 (`L=26`) | 95.00% | 96.88% | 0.0265 | 293,037.6 | 1219.5 | 78.52h | ~212.0h (stg rate, 27.1% L) / 43.8h (500k cap @ 4,730 PF/h) |
| `step10000` | `step10000` | ~21.0B | `exp2_pythia14b_step10000` | Spot 4x H100 (us-east4-a) | Running | 412690 | 26/96 (`L=26`) | 95.62% | 97.75% | 0.0294 | 305,503.4 | 1246.1 | 79.54h | ~214.7h (stg rate, 27.1% L) / 40.2h (500k cap @ 4,834 PF/h) |
| `step100000` | `step100000` | ~209.7B | `exp2_pythia14b_step100000` | Spot 4x H100 (us-east4-b) | Running | 281000 | 55/96 (`L=55`) | 97.00% | 98.12% | 0.0285 | 376,577.0 | 1288.9 | 77.10h | ~57.5h (stg rate, 57.3% L) / 24.7h (500k cap @ 5,002 PF/h) |

## Stage Evaluation Summary (`stage_eval_history.json`, every 8 stages)

### Arm: `step1000` (`exp2_pythia14b_step1000`)

| Stage | Step | $L$ | Stage $\alpha$ | TF Loss ($\alpha=1.0$) | Greedy First ($\alpha=1.0$) | Greedy Full ($\alpha=1.0$) | Greedy First (Stage $\alpha$) | Greedy Full (Stage $\alpha$) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 36500 | 8 | 0.0895 | 0.7263 | 34.80% | 34.00% | 99.00% | 98.80% |
| 16 | 179250 | 16 | 0.1737 | 0.5908 | 42.00% | 41.00% | 97.80% | 97.80% |
| 24 | 415075 | 24 | 0.2579 | 0.5144 | 43.80% | 42.80% | 97.80% | 97.60% |

### Arm: `step10000` (`exp2_pythia14b_step10000`)

| Stage | Step | $L$ | Stage $\alpha$ | TF Loss ($\alpha=1.0$) | Greedy First ($\alpha=1.0$) | Greedy Full ($\alpha=1.0$) | Greedy First (Stage $\alpha$) | Greedy Full (Stage $\alpha$) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 13025 | 8 | 0.0895 | 0.6055 | 43.40% | 41.60% | 99.00% | 99.00% |
| 16 | 88650 | 16 | 0.1737 | 0.5499 | 41.00% | 39.80% | 98.20% | 98.00% |
| 24 | 297975 | 24 | 0.2579 | 0.4048 | 43.60% | 42.40% | 97.00% | 96.60% |

### Arm: `step100000` (`exp2_pythia14b_step100000`)

| Stage | Step | $L$ | Stage $\alpha$ | TF Loss ($\alpha=1.0$) | Greedy First ($\alpha=1.0$) | Greedy Full ($\alpha=1.0$) | Greedy First (Stage $\alpha$) | Greedy Full (Stage $\alpha$) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 3725 | 8 | 0.0895 | 0.8706 | 36.80% | 35.60% | 98.60% | 98.40% |
| 16 | 16175 | 16 | 0.1737 | 0.5736 | 43.40% | 42.00% | 98.80% | 98.60% |
| 24 | 43500 | 24 | 0.2579 | 0.5343 | 45.20% | 44.00% | 98.80% | 98.60% |
| 32 | 81725 | 32 | 0.3421 | 0.3851 | 55.20% | 54.80% | 98.40% | 98.20% |
| 40 | 125475 | 40 | 0.4263 | 0.3266 | 59.20% | 58.20% | 98.00% | 98.00% |
| 48 | 199625 | 48 | 0.5105 | 0.2943 | 65.80% | 64.60% | 98.00% | 98.00% |


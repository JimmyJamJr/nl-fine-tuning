# Experiment 2: Pythia-1.4B Intermediate Pretraining Checkpoints (Step Size 1)

**Base commit:** `da7b19c` (`main`, plus `--revision` support in `nl/tuning_nl.py`)
**Branch:** `jackierwzhang/exp2-pythia14b-revisions`
**VM Instances:** `nl-exp2-pythia14b` (`a2-highgpu-8g`, 8x A100-SXM4-40GB, `asia-southeast1-c`) + `nl-exp2-spot-step100000` (`a3-highgpu-4g` Spot, 4x H100-80GB)
**Last updated:** 2026-10-10 09:33 UTC

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
| `step1000` | `step1000` | ~2.1B | `exp2_pythia14b_step1000` | GPUs 0,1,2,3 | Running | 49890 | 10/96 (`L=10`) | 95.00% | 97.00% | 0.0381 | 15,192.7 | 500.3 | 8.63h | ~74.6h (stg rate, 10.4% L) / 249.9h (500k cap @ 1,940 PF/h) |
| `step10000` | `step10000` | ~21.0B | `exp2_pythia14b_step10000` | GPUs 4,5,6,7 | Running | 41330 | 12/96 (`L=12`) | 97.62% | 98.88% | 0.0229 | 16,234.8 | 507.3 | 8.65h | ~60.7h (stg rate, 12.5% L) / 245.8h (500k cap @ 1,968 PF/h) |
| `step100000` | `step100000` | ~209.7B | `exp2_pythia14b_step100000` | Spot 4x H100 (us-east4-b) | Running | 34420 | 22/96 (`L=22`) | 96.25% | 97.00% | 0.0225 | 20,117.0 | 1241.7 | 5.42h | ~18.3h (stg rate, 22.9% L) / 99.6h (500k cap @ 4,819 PF/h) |

## Stage Evaluation Summary (`stage_eval_history.json`, every 8 stages)

### Arm: `step1000` (`exp2_pythia14b_step1000`)

| Stage | Step | $L$ | Stage $\alpha$ | TF Loss ($\alpha=1.0$) | Greedy First ($\alpha=1.0$) | Greedy Full ($\alpha=1.0$) | Greedy First (Stage $\alpha$) | Greedy Full (Stage $\alpha$) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 36500 | 8 | 0.0895 | 0.7263 | 34.80% | 34.00% | 99.00% | 98.80% |

### Arm: `step10000` (`exp2_pythia14b_step10000`)

| Stage | Step | $L$ | Stage $\alpha$ | TF Loss ($\alpha=1.0$) | Greedy First ($\alpha=1.0$) | Greedy Full ($\alpha=1.0$) | Greedy First (Stage $\alpha$) | Greedy Full (Stage $\alpha$) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 13025 | 8 | 0.0895 | 0.6055 | 43.40% | 41.60% | 99.00% | 99.00% |

### Arm: `step100000` (`exp2_pythia14b_step100000`)

| Stage | Step | $L$ | Stage $\alpha$ | TF Loss ($\alpha=1.0$) | Greedy First ($\alpha=1.0$) | Greedy Full ($\alpha=1.0$) | Greedy First (Stage $\alpha$) | Greedy Full (Stage $\alpha$) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 3725 | 8 | 0.0895 | 0.8706 | 36.80% | 35.60% | 98.60% | 98.40% |
| 16 | 16175 | 16 | 0.1737 | 0.5736 | 43.40% | 42.00% | 98.80% | 98.60% |


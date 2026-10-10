# Experiment 1: Qwen3-0.6B LoRA vs Full Fine-Tuning (Step Size 8)

**Base commit:** `da7b19c` (`main`)
**Branch:** `jackierwzhang/exp1-qwen06b-lora-vs-fullft`
**VMs:** `nl-exp1-qwen06b` (`a2-highgpu-8g`, On-Demand A100-40GB, `asia-southeast1-c`), `nl-exp1-spot-r8` (`a3-highgpu-4g`, 4x H100-80GB Spot, `us-east4-a`), `nl-exp1-spot-r64` (`a3-highgpu-4g`, 4x H100-80GB Spot, `us-east4-a`), `nl-exp1-spot-r256` (`a3-highgpu-4g`, 4x H100-80GB Spot, `us-east4-b`)
**Last updated:** 2026-10-10 11:35 UTC

## Shared Configuration

| Parameter | Value |
| :--- | :--- |
| Model | `Qwen/Qwen3-0.6B` |
| Task | `search` |
| Seed | `1234` |
| Topology | 4 GPUs per run (`NL_DDP_GRAD_AVERAGE` unset): 1x 4x A100-SXM4-40GB (`Full-FT`) + 3x 4x H100-80GB Spot (`LoRA R=8` migrating at `checkpoint-3000`, `LoRA R=64`, `LoRA R=256`) |
| Batch size / Grad accum | `48` per GPU, `gradient_accumulation_steps=4` |
| Learning rate | `5e-5` (constant, 0 warmup) |
| Curriculum | `base_lookahead=8`, `lookahead_step=8`, `n_stages=16` (`L=8..128`), `--linear_lookahead` |
| Gate | `accuracy_threshold=0.98`, `accuracy_window=800`, `check_every=25`, `min_steps_per_stage=200` |
| Context & Alpha | `max_input_size=768`, `max_lookahead=128`, `base_alpha=0.1`, `max_alpha=1.0` |
| Loss & Kernels | `first_token_soft_weight=0.0`, `ce_chunk_size=4096`, `NL_ATTN_KERNEL=fa2`, `--use_liger`, `--gradient_checkpointing` |
| Evals & Checkpoints | `--do_stage_eval`, `eval_samples=500`, `eval_every_steps=0`, `save_steps=500`, `save_total_limit=2`, `persist_every=0` |

## Run Status

| Arm | Job ID | Mode | GPUs | Status | Step | Stage / $L$ | Rolling Full Acc | Rolling First Acc | Loss | PFLOPs | TFLOP/s | Wall Time | Est. Remaining |
| :--- | :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Full-FT | `exp1_qwen06b_fullft` | Full FT | 0,1,2,3 (On-Demand 4x A100: nl-exp1-qwen06b, asia-southeast1-c) | Running | 6210 | 6 (`L=48`) | 96.38% | 98.75% | 0.0256 | 10967.10 | 212.0 | 11.08h | ~62.3h (stage-wt) / ~71.8h (82k PFLOPs ref) |
| LoRA `R=8` | `exp1_qwen06b_lora_r8` | `--use_lora --lora_rank 8` (`alpha=16, dropout=0.1`) | 0,1,2,3 (Spot 4x H100: nl-exp1-spot-r8, us-east4-a) | Running | 7360 | 4 (`L=32`) | 95.88% | 97.38% | 0.0265 | 9591.82 | 246.4 | 11.05h | ~104.5h (stage-wt) / ~60.5h (82k PFLOPs ref) |
| LoRA `R=64` | `exp1_qwen06b_lora_r64` | `--use_lora --lora_rank 64` (`alpha=128, dropout=0.1`) | 0,1,2,3 (Spot 4x H100: nl-exp1-spot-r64, us-east4-a) | Running | 4950 | 5 (`L=40`) | 95.50% | 97.88% | 0.0248 | 7401.78 | 226.0 | 7.18h | ~59.7h (stage-wt) / ~72.4h (82k PFLOPs ref) |
| LoRA `R=256` | `exp1_qwen06b_lora_r256` | `--use_lora --lora_rank 256` (`alpha=512, dropout=0.1`) | 0,1,2,3 (Spot 4x H100: nl-exp1-spot-r256, us-east4-b) | Running | 4250 | 4 (`L=32`) | 96.25% | 97.75% | 0.0247 | 5868.89 | 160.7 | 7.91h | ~103.2h (stage-wt) / ~102.6h (82k PFLOPs ref) |

> **ETA Methodology & Overall Experiment 1 Completion (All 4 Arms Running Concurrently on 4x A100 + 12x Spot H100):** `stage-wt` scales current wall time by remaining lookahead units ($\sum_{k=1}^{16} k = 136$); `82k PFLOPs ref` divides remaining compute against the collaborator's `Qwen3-0.6B` `step=8` cold-start reference (`82,000 PFLOPs` to `L=128`) by the arm's measured `PFLOPs/h` rate. Fastest arm completion is estimated in **~59.7h (stage-wt) / ~60.5h (82k PFLOPs ref)**; full 4-arm concurrent completion is bounded by the slowest arm at **~104.5h (stage-wt) to ~102.6h (82k PFLOPs ref)**.

## Stage Evaluation Summary (`stage_eval_history.json`)

| Arm | Stage | Step | $L$ | Stage $\alpha$ | TF Loss ($\alpha=1.0$) | Greedy First ($\alpha=1.0$) | Greedy Full ($\alpha=1.0$) | Greedy First (Stage $\alpha$) | Greedy Full (Stage $\alpha$) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-FT | 1 | 250 | 8 | 0.067 | 0.5895 | 38.60% | 38.40% | 98.80% | 98.20% |
| Full-FT | 2 | 1650 | 16 | 0.130 | 0.5030 | 42.60% | 42.00% | 99.20% | 99.00% |
| Full-FT | 3 | 2100 | 24 | 0.193 | 0.3774 | 47.20% | 46.60% | 97.00% | 96.80% |
| Full-FT | 4 | 3250 | 32 | 0.256 | 0.3644 | 50.80% | 50.40% | 97.60% | 97.20% |
| Full-FT | 5 | 5925 | 40 | 0.319 | 0.2980 | 60.00% | 59.60% | 98.60% | 98.40% |
| LoRA `R=8` | 1 | 475 | 8 | 0.067 | 0.4921 | 40.00% | 39.40% | 98.80% | 98.80% |
| LoRA `R=8` | 2 | 2700 | 16 | 0.130 | 0.3356 | 43.20% | 42.40% | 99.60% | 99.60% |
| LoRA `R=8` | 3 | 6000 | 24 | 0.193 | 0.3237 | 48.20% | 47.60% | 97.40% | 97.40% |
| LoRA `R=64` | 1 | 450 | 8 | 0.067 | 0.5967 | 39.40% | 39.40% | 98.80% | 98.80% |
| LoRA `R=64` | 2 | 1300 | 16 | 0.130 | 0.3640 | 45.80% | 45.20% | 98.80% | 98.80% |
| LoRA `R=64` | 3 | 2675 | 24 | 0.193 | 0.3256 | 51.20% | 51.00% | 99.20% | 99.20% |
| LoRA `R=64` | 4 | 4375 | 32 | 0.256 | 0.3216 | 51.80% | 51.20% | 98.80% | 98.80% |
| LoRA `R=256` | 1 | 300 | 8 | 0.067 | 0.5629 | 40.40% | 40.00% | 99.40% | 99.40% |
| LoRA `R=256` | 2 | 1025 | 16 | 0.130 | 0.4768 | 43.00% | 42.80% | 98.80% | 98.80% |
| LoRA `R=256` | 3 | 3175 | 24 | 0.193 | 0.3778 | 51.60% | 50.60% | 97.80% | 97.80% |

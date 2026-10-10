# Experiment 1: Qwen3-0.6B LoRA vs Full Fine-Tuning (Step Size 8)

**Base commit:** `da7b19c` (`main`)
**Branch:** `jackierwzhang/exp1-qwen06b-lora-vs-fullft`
**VMs:** `nl-exp1-qwen06b` (`a2-highgpu-8g`, 8x A100-40GB On-Demand, `asia-southeast1-c`), `nl-exp1-spot-r64` (`a3-highgpu-4g`, 4x H100-80GB Spot, `us-east4-a`), `nl-exp1-spot-r256` (`a3-highgpu-4g`, 4x H100-80GB Spot, `us-east4-b`)
**Last updated:** 2026-10-10 04:27 UTC

## Shared Configuration

| Parameter | Value |
| :--- | :--- |
| Model | `Qwen/Qwen3-0.6B` |
| Task | `search` |
| Seed | `1234` |
| Topology | 4 GPUs per run (`NL_DDP_GRAD_AVERAGE` unset): 2x 4x A100-SXM4-40GB (`Full-FT`, `LoRA R=8`) + 2x 4x H100-80GB Spot (`LoRA R=64`, `LoRA R=256`) |
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
| Full-FT | `exp1_qwen06b_fullft` | Full FT | 0,1,2,3 (On-Demand 4x A100: nl-exp1-qwen06b, asia-southeast1-c) | Running | 2980 | 4 (`L=32`) | 96.75% | 98.12% | 0.0294 | 3755.72 | 213.0 | 3.95h | ~51.5h (stage-wt) / ~82.3h (82k PFLOPs ref) |
| LoRA `R=8` | `exp1_qwen06b_lora_r8` | `--use_lora --lora_rank 8` (`alpha=16, dropout=0.1`) | 4,5,6,7 (On-Demand 4x A100: nl-exp1-qwen06b, asia-southeast1-c) | Running | 2410 | 2 (`L=16`) | 97.75% | 98.88% | 0.0177 | 2183.87 | 120.9 | 3.95h | ~185.0h (stage-wt) / ~144.2h (82k PFLOPs ref) |
| LoRA `R=64` | `exp1_qwen06b_lora_r64` | `--use_lora --lora_rank 64` (`alpha=128, dropout=0.1`) | 0,1,2,3 (Spot 4x H100: nl-exp1-spot-r64, us-east4-a) | Running | 980 | 2 (`L=16`) | 97.50% | 98.88% | 0.0238 | 779.31 | 231.9 | 0.83h | ~39.1h (stage-wt) / ~86.8h (82k PFLOPs ref) |
| LoRA `R=256` | `exp1_qwen06b_lora_r256` | `--use_lora --lora_rank 256` (`alpha=512, dropout=0.1`) | 0,1,2,3 (Spot 4x H100: nl-exp1-spot-r256, us-east4-b) | Running | 660 | 2 (`L=16`) | 94.25% | 96.75% | 0.0272 | 525.28 | 204.4 | 0.79h | ~37.2h (stage-wt) / ~123.1h (82k PFLOPs ref) |

> **ETA Methodology & Overall Experiment 1 Completion (All 4 Arms Running Concurrently on 8x A100 + 8x Spot H100):** `stage-wt` scales current wall time by remaining lookahead units ($\sum_{k=1}^{16} k = 136$); `82k PFLOPs ref` divides remaining compute against the collaborator's `Qwen3-0.6B` `step=8` cold-start reference (`82,000 PFLOPs` to `L=128`) by the arm's measured `PFLOPs/h` rate. Fastest arm completion is estimated in **~37.2h (stage-wt) / ~82.3h (82k PFLOPs ref)**; full 4-arm concurrent completion is bounded by the slowest arm at **~185.0h (stage-wt) to ~144.2h (82k PFLOPs ref)**.

## Stage Evaluation Summary (`stage_eval_history.json`)

| Arm | Stage | Step | $L$ | Stage $\alpha$ | TF Loss ($\alpha=1.0$) | Greedy First ($\alpha=1.0$) | Greedy Full ($\alpha=1.0$) | Greedy First (Stage $\alpha$) | Greedy Full (Stage $\alpha$) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-FT | 1 | 250 | 8 | 0.067 | 0.5895 | 38.60% | 38.40% | 98.80% | 98.20% |
| Full-FT | 2 | 1650 | 16 | 0.130 | 0.5030 | 42.60% | 42.00% | 99.20% | 99.00% |
| Full-FT | 3 | 2100 | 24 | 0.193 | 0.3774 | 47.20% | 46.60% | 97.00% | 96.80% |
| LoRA `R=8` | 1 | 475 | 8 | 0.067 | 0.4921 | 40.00% | 39.40% | 98.80% | 98.80% |
| LoRA `R=64` | 1 | 450 | 8 | 0.067 | 0.5967 | 39.40% | 39.40% | 98.80% | 98.80% |
| LoRA `R=256` | 1 | 300 | 8 | 0.067 | 0.5629 | 40.40% | 40.00% | 99.40% | 99.40% |

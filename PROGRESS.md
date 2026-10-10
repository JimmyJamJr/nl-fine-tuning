# Experiment 1: Qwen3-0.6B LoRA vs Full Fine-Tuning (Step Size 8)

**Base commit:** `da7b19c` (`main`)
**Branch:** `jackierwzhang/exp1-qwen06b-lora-vs-fullft`
**VMs:** `nl-exp1-qwen06b` (`a2-highgpu-8g`, 8x A100-40GB On-Demand, `asia-southeast1-c`), `nl-exp1-spot-r64` (`a3-highgpu-4g`, 4x H100-80GB Spot, `us-east4-a`), `nl-exp1-spot-r256` (`a3-highgpu-4g`, 4x H100-80GB Spot, `us-east4-b`)
**Last updated:** 2026-10-10 03:42 UTC

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
| Full-FT | `exp1_qwen06b_fullft` | Full FT | 0,1,2,3 (On-Demand 4x A100: nl-exp1-qwen06b, asia-southeast1-c) | Running | 2550 | 4 (`L=32`) | 95.25% | 97.12% | 0.0309 | 2973.63 | 212.6 | 3.19h | ~41.7h (stage-wt) / ~84.9h (82k PFLOPs ref) |
| LoRA `R=8` | `exp1_qwen06b_lora_r8` | `--use_lora --lora_rank 8` (`alpha=16, dropout=0.1`) | 4,5,6,7 (On-Demand 4x A100: nl-exp1-qwen06b, asia-southeast1-c) | Running | 1970 | 2 (`L=16`) | 97.25% | 98.62% | 0.0210 | 1749.26 | 120.9 | 3.20h | ~149.8h (stage-wt) / ~146.6h (82k PFLOPs ref) |
| LoRA `R=64` | `exp1_qwen06b_lora_r64` | `--use_lora --lora_rank 64` (`alpha=128, dropout=0.1`) | 0,1,2,3 (Spot 4x H100: nl-exp1-spot-r64, us-east4-a) | Running | 130 | 1 (`L=8`) | 92.50% | 94.12% | 0.0453 | 74.41 | 218.4 | 0.08h | ~28.1h (stage-wt) / ~84.7h (82k PFLOPs ref) |
| LoRA `R=256` | `exp1_qwen06b_lora_r256` | `--use_lora --lora_rank 256` (`alpha=512, dropout=0.1`) | 0,1,2,3 (Spot 4x H100: nl-exp1-spot-r256, us-east4-b) | Running | 50 | 1 (`L=8`) | 80.25% | 81.25% | 0.1141 | 28.58 | 195.2 | 0.04h | ~39.3h (stage-wt) / ~118.5h (82k PFLOPs ref) |

> **ETA Methodology & Overall Experiment 1 Completion (All 4 Arms Running Concurrently on 8x A100 + 8x Spot H100):** `stage-wt` scales current wall time by remaining lookahead units ($\sum_{k=1}^{16} k = 136$); `82k PFLOPs ref` divides remaining compute against the collaborator's `Qwen3-0.6B` `step=8` cold-start reference (`82,000 PFLOPs` to `L=128`) by the arm's measured `PFLOPs/h` rate. Fastest arm completion is estimated in **~28.1h (stage-wt) / ~84.7h (82k PFLOPs ref)**; full 4-arm concurrent completion is bounded by the slowest arm at **~149.8h (stage-wt) to ~146.6h (82k PFLOPs ref)**.

## Stage Evaluation Summary (`stage_eval_history.json`)

| Arm | Stage | Step | $L$ | Stage $\alpha$ | TF Loss ($\alpha=1.0$) | Greedy First ($\alpha=1.0$) | Greedy Full ($\alpha=1.0$) | Greedy First (Stage $\alpha$) | Greedy Full (Stage $\alpha$) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-FT | 1 | 250 | 8 | 0.067 | 0.5895 | 38.60% | 38.40% | 98.80% | 98.20% |
| Full-FT | 2 | 1650 | 16 | 0.130 | 0.5030 | 42.60% | 42.00% | 99.20% | 99.00% |
| Full-FT | 3 | 2100 | 24 | 0.193 | 0.3774 | 47.20% | 46.60% | 97.00% | 96.80% |
| LoRA `R=8` | 1 | 475 | 8 | 0.067 | 0.4921 | 40.00% | 39.40% | 98.80% | 98.80% |

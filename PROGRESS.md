# Experiment 1: Qwen3-0.6B LoRA vs Full Fine-Tuning (Step Size 8)

**Base commit:** `da7b19c` (`main`)
**Branch:** `jackierwzhang/exp1-qwen06b-lora-vs-fullft`
**VM:** `nl-exp1-qwen06b` (`a2-highgpu-8g`, 8x NVIDIA A100-SXM4-40GB, `asia-southeast1-c`)
**Last updated:** 2026-10-10 00:48 UTC

## Shared Configuration

| Parameter | Value |
| :--- | :--- |
| Model | `Qwen/Qwen3-0.6B` |
| Task | `search` |
| Seed | `1234` |
| Topology | 4x NVIDIA A100-SXM4-40GB per run (`NL_DDP_GRAD_AVERAGE` unset) |
| Batch size / Grad accum | `48` per GPU, `gradient_accumulation_steps=4` |
| Learning rate | `5e-5` (constant, 0 warmup) |
| Curriculum | `base_lookahead=8`, `lookahead_step=8`, `n_stages=16` (`L=8..128`), `--linear_lookahead` |
| Gate | `accuracy_threshold=0.98`, `accuracy_window=800`, `check_every=25`, `min_steps_per_stage=200` |
| Context & Alpha | `max_input_size=768`, `max_lookahead=128`, `base_alpha=0.1`, `max_alpha=1.0` |
| Loss & Kernels | `first_token_soft_weight=0.0`, `ce_chunk_size=4096`, `NL_ATTN_KERNEL=fa2`, `--use_liger`, `--gradient_checkpointing` |
| Evals & Checkpoints | `--do_stage_eval`, `eval_samples=500`, `eval_every_steps=0`, `save_steps=500`, `save_total_limit=2`, `persist_every=0` |

## Run Status

| Arm | Job ID | Mode | GPUs | Status | Step | Stage / $L$ | Rolling Full Acc | Rolling First Acc | Loss | PFLOPs | TFLOP/s | Wall Time |
| :--- | :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-FT | `exp1_qwen06b_fullft` | Full FT | 0,1,2,3 (Wave 1) | Running | 330 | 2 (`L=16`) | 90.50% | 92.62% | 0.0627 | 220.36 | 204.7 | 0.31h |
| LoRA `R=8` | `exp1_qwen06b_lora_r8` | `--use_lora --lora_rank 8` (`alpha=16, dropout=0.1`) | 4,5,6,7 (Wave 1) | Running | 290 | 1 (`L=8`) | 94.25% | 95.75% | 0.0399 | 165.92 | 113.8 | 0.31h |
| LoRA `R=64` | `exp1_qwen06b_lora_r64` | `--use_lora --lora_rank 64` (`alpha=128, dropout=0.1`) | Queued (Wave 2) | Queued | 0 | - | - | - | - | 0.00 | - | 0.00h |
| LoRA `R=256` | `exp1_qwen06b_lora_r256` | `--use_lora --lora_rank 256` (`alpha=512, dropout=0.1`) | Queued (Wave 2) | Queued | 0 | - | - | - | - | 0.00 | - | 0.00h |

## Stage Evaluation Summary (`stage_eval_history.json`)

| Arm | Stage | Step | $L$ | Stage $\alpha$ | TF Loss ($\alpha=1.0$) | Greedy First ($\alpha=1.0$) | Greedy Full ($\alpha=1.0$) | Greedy First (Stage $\alpha$) | Greedy Full (Stage $\alpha$) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-FT | 1 | 250 | 8 | 0.067 | 0.5895 | 38.60% | 38.40% | 98.80% | 98.20% |

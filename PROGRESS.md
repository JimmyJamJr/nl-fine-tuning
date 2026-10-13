# Experiment 1: Qwen3-0.6B LoRA vs Full Fine-Tuning (Step Size 8)

**Base commit:** `da7b19c` (`main`)
**Branch:** `jackierwzhang/exp1-qwen06b-lora-vs-fullft`
**VMs:** `nl-exp1-spot-fullft` (`europe-west1-c`, deleted after completion), `nl-exp1-spot-r8` (`a3-highgpu-4g`, 4x H100-80GB Spot, `us-east4-a`), `nl-exp1-spot-r64` (`a3-highgpu-4g`, 4x H100-80GB Spot, `europe-west1-c`), `nl-exp1-spot-r256` (`a3-highgpu-4g`, 4x H100-80GB Spot, `us-east4-b`)
**Last updated:** 2026-10-13 19:36 UTC

## Shared Configuration

| Parameter | Value |
| :--- | :--- |
| Model | `Qwen/Qwen3-0.6B` |
| Task | `search` |
| Seed | `1234` |
| Topology | 4 GPUs per run (`NL_DDP_GRAD_AVERAGE` unset): 4x 4x H100-80GB Spot (`Full-FT` migrated at `checkpoint-11000`, `LoRA R=8` migrated at `checkpoint-3000`, `LoRA R=64`, `LoRA R=256`) |
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
| Full-FT | `exp1_qwen06b_fullft` | Full FT | 0,1,2,3 (Spot 4x H100: nl-exp1-spot-fullft, europe-west1-c) | Completed | 27570 | 16 (`L=128`) | 97.62% | 98.88% | 0.0163 | 116023.73 | 405.3 | 71.26h | 0.0h (Done) |
| LoRA `R=8` | `exp1_qwen06b_lora_r8` | `--use_lora --lora_rank 8` (`alpha=16, dropout=0.1`) | 0,1,2,3 (Spot 4x H100: nl-exp1-spot-r8, us-east4-a) | Running | 33270 | 13 (`L=104`) | 96.12% | 98.12% | 0.0202 | 101994.15 | 240.2 | 89.49h | ~45.8h (stage-wt) / ~0.0h (82k PFLOPs ref) |
| LoRA `R=64` | `exp1_qwen06b_lora_r64` | `--use_lora --lora_rank 64` (`alpha=128, dropout=0.1`) | 0,1,2,3 (Spot 4x H100: nl-exp1-spot-r64, europe-west1-c) | Running | 24260 | 14 (`L=112`) | 96.88% | 98.62% | 0.0212 | 88794.66 | 220.4 | 82.48h | ~25.5h (stage-wt) / ~0.0h (82k PFLOPs ref) |
| LoRA `R=256` | `exp1_qwen06b_lora_r256` | `--use_lora --lora_rank 256` (`alpha=512, dropout=0.1`) | 0,1,2,3 (Spot 4x H100: nl-exp1-spot-r256, us-east4-b) | Running | 25090 | 9 (`L=72`) | 96.12% | 97.88% | 0.0214 | 68177.60 | 160.3 | 87.89h | ~182.0h (stage-wt) / ~17.8h (82k PFLOPs ref) |

> **ETA Methodology & Overall Experiment 1 Completion (3 Active Arms on 12x Spot H100; Full-FT Completed):** `stage-wt` scales current wall time by remaining lookahead units ($\sum_{k=1}^{16} k = 136$); `82k PFLOPs ref` divides remaining compute against the collaborator's `Qwen3-0.6B` `step=8` cold-start reference (`82,000 PFLOPs` to `L=128`) by the arm's measured `PFLOPs/h` rate. Next arm completion is estimated in **~25.5h (stage-wt) / ~0.0h (82k PFLOPs ref)**; full 4-arm concurrent completion is bounded by the slowest arm at **~182.0h (stage-wt) to ~17.8h (82k PFLOPs ref)**.

## Stage Evaluation Summary (`stage_eval_history.json`)

| Arm | Stage | Step | $L$ | Stage $\alpha$ | TF Loss ($\alpha=1.0$) | Greedy First ($\alpha=1.0$) | Greedy Full ($\alpha=1.0$) | Greedy First (Stage $\alpha$) | Greedy Full (Stage $\alpha$) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-FT | 1 | 250 | 8 | 0.067 | 0.5895 | 38.60% | 38.40% | 98.80% | 98.20% |
| Full-FT | 2 | 1650 | 16 | 0.130 | 0.5030 | 42.60% | 42.00% | 99.20% | 99.00% |
| Full-FT | 3 | 2100 | 24 | 0.193 | 0.3774 | 47.20% | 46.60% | 97.00% | 96.80% |
| Full-FT | 4 | 3250 | 32 | 0.256 | 0.3644 | 50.80% | 50.40% | 97.60% | 97.20% |
| Full-FT | 5 | 5925 | 40 | 0.319 | 0.2980 | 60.00% | 59.60% | 98.60% | 98.40% |
| Full-FT | 6 | 7075 | 48 | 0.382 | 0.2386 | 64.20% | 63.80% | 98.40% | 98.40% |
| Full-FT | 7 | 7700 | 56 | 0.445 | 0.1977 | 67.40% | 67.00% | 98.00% | 97.80% |
| Full-FT | 8 | 10700 | 64 | 0.508 | 0.1549 | 76.60% | 76.60% | 97.80% | 97.60% |
| Full-FT | 9 | 12300 | 72 | 0.571 | 0.1657 | 80.20% | 79.80% | 98.20% | 97.80% |
| Full-FT | 10 | 14975 | 80 | 0.634 | 0.1040 | 87.20% | 87.20% | 98.40% | 98.40% |
| Full-FT | 11 | 15300 | 88 | 0.697 | 0.0947 | 88.00% | 88.00% | 98.00% | 97.80% |
| Full-FT | 12 | 19550 | 96 | 0.760 | 0.0466 | 94.20% | 94.20% | 97.80% | 97.60% |
| Full-FT | 13 | 21825 | 104 | 0.823 | 0.0409 | 92.80% | 92.80% | 97.40% | 97.40% |
| Full-FT | 14 | 23275 | 112 | 0.886 | 0.0347 | 97.60% | 97.60% | 97.60% | 97.40% |
| Full-FT | 15 | 24350 | 120 | 0.949 | 0.0214 | 97.00% | 97.00% | 98.00% | 98.00% |
| Full-FT | 16 | 27575 | 128 | 1.000 | 0.0222 | 98.60% | 98.60% | - | - |
| LoRA `R=8` | 1 | 475 | 8 | 0.067 | 0.4921 | 40.00% | 39.40% | 98.80% | 98.80% |
| LoRA `R=8` | 2 | 2700 | 16 | 0.130 | 0.3356 | 43.20% | 42.40% | 99.60% | 99.60% |
| LoRA `R=8` | 3 | 6000 | 24 | 0.193 | 0.3237 | 48.20% | 47.60% | 97.40% | 97.40% |
| LoRA `R=8` | 4 | 8575 | 32 | 0.256 | 0.2849 | 51.80% | 51.20% | 98.00% | 98.00% |
| LoRA `R=8` | 5 | 12525 | 40 | 0.319 | 0.2457 | 57.40% | 56.80% | 99.20% | 99.20% |
| LoRA `R=8` | 6 | 15775 | 48 | 0.382 | 0.2110 | 66.00% | 65.60% | 97.60% | 97.60% |
| LoRA `R=8` | 7 | 17175 | 56 | 0.445 | 0.1816 | 66.40% | 65.80% | 97.60% | 97.60% |
| LoRA `R=8` | 8 | 22175 | 64 | 0.508 | 0.1326 | 76.60% | 76.40% | 99.00% | 99.00% |
| LoRA `R=8` | 9 | 24100 | 72 | 0.571 | 0.1077 | 83.00% | 82.80% | 97.80% | 97.60% |
| LoRA `R=8` | 10 | 27900 | 80 | 0.634 | 0.0853 | 86.60% | 86.40% | 99.00% | 98.80% |
| LoRA `R=8` | 11 | 30225 | 88 | 0.697 | 0.0637 | 89.40% | 89.20% | 98.40% | 98.00% |
| LoRA `R=8` | 12 | 31000 | 96 | 0.760 | 0.0679 | 91.60% | 91.40% | 97.80% | 97.80% |
| LoRA `R=64` | 1 | 450 | 8 | 0.067 | 0.5967 | 39.40% | 39.40% | 98.80% | 98.80% |
| LoRA `R=64` | 2 | 1300 | 16 | 0.130 | 0.3640 | 45.80% | 45.20% | 98.80% | 98.80% |
| LoRA `R=64` | 3 | 2675 | 24 | 0.193 | 0.3256 | 51.20% | 51.00% | 99.20% | 99.20% |
| LoRA `R=64` | 4 | 4375 | 32 | 0.256 | 0.3216 | 51.80% | 51.20% | 98.80% | 98.80% |
| LoRA `R=64` | 5 | 6900 | 40 | 0.319 | 0.2766 | 60.40% | 60.20% | 99.20% | 99.20% |
| LoRA `R=64` | 6 | 8125 | 48 | 0.382 | 0.2613 | 60.80% | 60.40% | 98.80% | 98.80% |
| LoRA `R=64` | 7 | 9250 | 56 | 0.445 | 0.2050 | 67.80% | 67.20% | 98.20% | 98.20% |
| LoRA `R=64` | 8 | 12600 | 64 | 0.508 | 0.1526 | 74.40% | 74.40% | 98.40% | 98.40% |
| LoRA `R=64` | 9 | 13725 | 72 | 0.571 | 0.1169 | 82.40% | 82.20% | 97.80% | 97.80% |
| LoRA `R=64` | 10 | 15450 | 80 | 0.634 | 0.1018 | 86.40% | 86.20% | 98.20% | 98.00% |
| LoRA `R=64` | 11 | 17525 | 88 | 0.697 | 0.0748 | 89.20% | 89.20% | 98.00% | 97.80% |
| LoRA `R=64` | 12 | 19475 | 96 | 0.760 | 0.0578 | 94.40% | 94.40% | 98.60% | 98.60% |
| LoRA `R=64` | 13 | 22125 | 104 | 0.823 | 0.0375 | 95.40% | 95.40% | 97.80% | 97.80% |
| LoRA `R=256` | 1 | 300 | 8 | 0.067 | 0.5629 | 40.40% | 40.00% | 99.40% | 99.40% |
| LoRA `R=256` | 2 | 1025 | 16 | 0.130 | 0.4768 | 43.00% | 42.80% | 98.80% | 98.80% |
| LoRA `R=256` | 3 | 3175 | 24 | 0.193 | 0.3778 | 51.60% | 50.60% | 97.80% | 97.80% |
| LoRA `R=256` | 4 | 4900 | 32 | 0.256 | 0.3145 | 54.20% | 54.00% | 97.80% | 97.40% |
| LoRA `R=256` | 5 | 10500 | 40 | 0.319 | 0.2944 | 58.60% | 58.40% | 98.60% | 98.60% |
| LoRA `R=256` | 6 | 13675 | 48 | 0.382 | 0.2203 | 67.80% | 67.60% | 98.60% | 98.60% |
| LoRA `R=256` | 7 | 18125 | 56 | 0.445 | 0.2113 | 71.80% | 71.00% | 99.00% | 99.00% |
| LoRA `R=256` | 8 | 19225 | 64 | 0.508 | 0.1456 | 77.20% | 76.80% | 98.40% | 98.40% |

# Qwen 0.6B nocurr L=120 resume handoff — Jackie's 2× H100 pod

Single run, one launcher script. Continues an in-progress no-curriculum sweep arm for RQ 1.1 Part 1.

| Run | Hardware | Wall time est. | Goal |
|---|---|---|---|
| **Qwen 0.6B no-curr L=120 RESUME** | 2× H100 80GB | ~5-10h | Determine whether nocurr can clear 98% gate at L=120 (or plateaus) |

Saves to `/home/ray/scratch/nl_output/search/job_<id>/`. Auto-sync to `gs://jackierwzhang-purdue-research-curriculum/` if your existing sync hook is active.

---

## Background

The no-curriculum sweep arm of RQ 1.1 Part 1 establishes that without curriculum learning, transformers cannot reach high reasoning depth. Existing endpoints:

| L target | PFLOPS spent | Final greedy_first @ α=1.0 | Outcome |
|---|---|---|---|
| 96 | 95,524 | **98.8%** | Cleared 98% gate ✓ |
| 104 | 106,627 | 97.8% | Just under, called success |
| 112 | 148,663 | 97.4% | Just under — borderline |
| **120** | **104K (preempted)** | (no eval yet) | **In progress — needs resume** |
| 128 | 154K (preempted) | (no eval yet) | Loss=0.044 (vs 0.022 at L=112) — likely struggling |

L=120 is the **critical boundary point**. Whether it clears the 98% gate or plateaus determines exactly where curriculum becomes necessary.

The original Gautschi run (job 9495340) was preempted at step 16,500 / loss 0.026. We extend it on Jackie's 2× H100 pod by another ~50K PFLOPS (~5-10h wall) to reach final convergence or confirm plateau.

---

## Step 1: prerequisites on Jackie's pod

If you already set up the env for the 1.7B handoffs or the step-size sweep handoffs, **skip — same env works**.

```bash
conda create -n curriculum python=3.11 -y
conda activate curriculum
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install \
  numpy==1.26.4 transformers==4.57.5 tokenizers==0.22.2 huggingface-hub==0.34.4 \
  peft==0.18.1 accelerate==1.12.0 datasets==4.0.0 fsspec==2024.10.0 safetensors==0.6.2 \
  pybind11 faker scipy matplotlib pandas liger-kernel
pip install flash-attn==2.8.3 --no-build-isolation
pip install nvidia-nvshmem-cu12==3.3.20

cd nl-fine-tuning/nl
python nl_generator.py
cd ..
```

The seed checkpoint is **public on HuggingFace, no auth required**: [`JimmyJamJr/qwen06b-nocurr-L120-step16500`](https://huggingface.co/JimmyJamJr/qwen06b-nocurr-L120-step16500) (~3.4 GB).

---

## Step 2: launch

```bash
tmux new -s train_nocurr120
bash tuning_job_qwen06b_nocurr120_RESUME_jackie.sh 2>&1 | tee /home/ray/scratch/train_nocurr120.log
# Ctrl-b d to detach
```

The launcher:
1. Downloads seed checkpoint from HF (~3.4 GB, one-time, cached)
2. Fabricates `checkpoint-16500/` in the new job dir with full optimizer state
3. Restores `curriculum_state.json` and `loss_history.jsonl` so cumulative PFLOPS continues from ~104K
4. Launches torchrun with `--resume_from_job <new_job_id>` (auto-resumes from the fabricated checkpoint)
5. Continues training until either the 98% gate is cleared (`[FINISHED]`) or the model plateaus

---

## Config summary

| Param | Value | Note |
|---|---|---|
| Model | Qwen/Qwen3-0.6B | unchanged |
| **eff_batch** | **768** | unchanged (was 3 GPU × 32 × GA=8 on Gautschi; now 2 GPU × 96 × GA=4) |
| LR | 5e-5 | unchanged |
| Warmup | 100 | unchanged |
| **No curriculum** | n_stages=1, base_lookahead=120, lookahead_step=1 | fixed L=120 throughout |
| max_input_size | 720 | =6×120 |
| **GC** | **off** | ~30% speedup vs original; ~40 GB / 80 GB peak per GPU (half max_input_size of step sweeps that OOMed) |
| Liger | on | unchanged |

---

## Expected behavior at restart

```
[CKPT] Auto-resuming from local run: .../checkpoint-16500
[CURRICULUM] Restored stage=1, ...
[Stage 1/1 L=120] step 16510 ... loss=0.025+ ...   ← clean continuation
... eventually one of two outcomes:
 - Loss drops below ~0.02 + rolling accuracy hits 98% → [FINISHED] Curriculum complete
 - Loss plateaus at ~0.025 across many thousands of steps → manual stop after enough evidence
```

Either outcome is a clean result for the paper.

---

## OOM fallback

If you see OOM, flip GC=on with the biggest microbatch (~32 GB peak, same eff_batch=768):

```bash
BATCH_SIZE=384 GRADIENT_ACCUMULATION_STEPS=1 GRADIENT_CHECKPOINTING=true \
  bash tuning_job_qwen06b_nocurr120_RESUME_jackie.sh
```

---

## Sync results back to Gautschi

Either:
- Auto-sync via your existing GCS hook to `gs://jackierwzhang-purdue-research-curriculum/`
- Or `gsutil -m rsync -r /home/ray/scratch/nl_output/search/job_<id> gs://jackierwzhang-purdue-research-curriculum/qwen06b-nocurr-L120/job_<id>` (manual)

Then on Gautschi side: `gsutil -m rsync -r gs://...` to pull.

---

## Estimated wall-clock

- 2× H100 at eff_batch=768 with Qwen 0.6B at max_input_size=720, GC=on: roughly **8-10K PFLOPS/h aggregate**
- Need another ~50-100K PFLOPS to reach 154K (matching where 128 plateaued) → **~5-10 hours wall**
- If model converges and clears the 98% gate before 50K additional PFLOPS, training will stop earlier via `[FINISHED]`

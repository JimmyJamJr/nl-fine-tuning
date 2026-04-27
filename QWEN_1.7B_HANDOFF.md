# Qwen 1.7B handoff — two parallel runs for Jackie's pods

Two independent runs, can launch in parallel.

| Job | Hardware | Wall time est. | Goal |
|---|---|---|---|
| **(A) 6%-mix resume → L=48** | 2× H100 80GB | ~24-36h | Clear L=32 + extend curriculum to L=48 |
| **(B) 100% Dolci-Instruct ablation** | 6× A100 80GB | ~52h | Match-compute baseline at 5,362 / 25,273 / 48,253 / 85,266 PFLOPS |

Both runs save to `/home/ray/scratch/nl_output/search/job_<id>/`. Auto-sync to `gs://jackierwzhang-purdue-research-curriculum/` if Jackie's existing sync setup is active. Otherwise rsync back periodically.

---

## (A) 6%-mix resume → L=48 on 2× H100

### Background

Original run (`job_runpod_qwen17b_L32_20260418_085434`) was Qwen3-1.7B with 6% Dolci-Instruct mix and step=1 curriculum to L=32. Pod was killed mid-stage-32 — model never cleared the 98% accuracy gate. We have:

- **Model weights** at end-of-run (~step 26,400)
- **Curriculum state** (stage=32, L=32)
- **Loss history** (full 26,400 entries)
- **No optimizer.pt** (lost when pod was terminated)

All four are bundled in the public HF repo: **[`JimmyJamJr/qwen17b-6pct-dolci-stage32`](https://huggingface.co/JimmyJamJr/qwen17b-6pct-dolci-stage32)** (public, no auth required).

So this is a **model-only resume with fresh AdamW**. Brief loss spike expected (50-200 steps); recovers to normal training thereafter. Downstream eval impact at L=32 should be <1pp vs an idealized continuous resume — within noise floor.

### Step 1: prerequisites on Jackie's pod

```bash
# Conda env named 'curriculum' (or edit script)
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

# Get repo onto pod (use whatever method — git clone, scp, etc.)
# The two launcher scripts and this README go in the repo root.
cd nl-fine-tuning/nl
python nl_generator.py   # build C++ generator
cd ..
```

**No HF login needed** — the seed checkpoint repo is public (`JimmyJamJr/qwen17b-6pct-dolci-stage32`, no auth required).

### Step 2: launch

```bash
# Recommended in tmux so it survives SSH drops
tmux new -s train_6pct
bash tuning_job_qwen17b_6pct_L48_resume_jackie.sh 2>&1 | tee /home/ray/scratch/train_6pct.log
# Ctrl-b d to detach
```

The launcher will:
1. Download seed model from `JimmyJamJr/qwen17b-6pct-dolci-stage32` on HF (~3.4 GB, one-time, cached)
2. Fabricate a `checkpoint-26400/` dir in the new job dir with model + tokenizer + minimal trainer_state.json
3. Restore `curriculum_state.json` (stage=32) and `loss_history.jsonl` (so cumulative PFLOPS continues from ~85K)
4. Launch torchrun with `--resume_from_job <new_job_id>` (auto-resumes from the fabricated checkpoint)
5. Train through stages 32-48, advancing as the model masters each L

Config summary (matches original 6%-mix exactly except where noted):

| Param | Value | Note |
|---|---|---|
| Model | Qwen/Qwen3-1.7B | unchanged |
| eff_batch | 768 | unchanged (2 GPU × 48 × GA=8) |
| LR | 5e-5 | unchanged |
| Warmup | **500** | bumped from 100 to cushion fresh-Adam restart |
| max_lookahead | **48** | extended from 32 |
| n_stages | **48** | extended from 32 |
| max_input_size | **288** | extended from 192 (=6×48) |
| Mix ratio | 0.06 Dolci | unchanged |
| GC | on | unchanged (matches original) |
| Liger | on | unchanged |

### Expected behavior at restart

```
[CKPT] Auto-resuming from local run: .../checkpoint-26400
[CKPT][WARN] optimizer.pt not found — starting with fresh AdamW   ← expected
[CURRICULUM] Restored stage=32, ...                                 ← expected
[Stage 32/48 L=32] step 26410 ... loss=0.5+ ... (loss spike)        ← expected, ~50-200 steps
[Stage 32/48 L=32] step 26900 ... loss=0.05 ... (recovered)         ← back to normal
... eventually:
 -> Advanced to stage 33 | alpha=... | L=33/48
... continues through L=48
[FINISHED] Curriculum complete (reached max_lookahead)              ← stops here
```

---

## (B) 100% Dolci-Instruct ablation on 6× A100

### Background

Match-compute baseline: train Qwen 1.7B on 100% Dolci-Instruct (no graph search, no curriculum) to the same total compute the 6%-mix run had spent at each curriculum L. Save persistent checkpoints at:

| Milestone | PFLOPS | Description |
|---|---|---|
| L=8 match | 5,362 | Compute when 6%-mix mastered L=8 |
| L=16 match | 25,273 | Compute when 6%-mix mastered L=16 |
| L=24 match | 48,253 | Compute when 6%-mix mastered L=24 |
| L=32 match | 85,266 | Compute at end of 6%-mix run (mid-stage-32) |

Run **stops automatically** at 85,266 PFLOPS. Each saved checkpoint gives an instruct-only model trained to the same compute as a curriculum-eval point — clean apples-to-apples for downstream eval.

### Step 1: prerequisites

Same as (A). If you already set up the env for (A), skip.

### Step 2: launch

```bash
tmux new -s train_dolci
bash tuning_job_qwen17b_dolci_only_jackie.sh 2>&1 | tee /home/ray/scratch/train_dolci.log
# Ctrl-b d to detach
```

Config summary:

| Param | Value | Note |
|---|---|---|
| Model | Qwen/Qwen3-1.7B (fresh, no curriculum start) | |
| eff_batch | 768 | 6 GPU × 32 × GA=4 |
| LR / warmup | 5e-5 / 100 | matches 6%-mix |
| Mix ratio | **1.0** Dolci | 100% Dolci-Instruct, no graph search |
| n_stages | 1 | no curriculum |
| GC | off | A100 80GB has room; ~25-30% speedup |
| save_pflops_milestones | 5362,25273,48253,85266 | model-only checkpoints saved at each |
| max_total_pflops | 85266 | auto-stop |

### Expected behavior

```
[Stage 1/1 L=1] step 10 loss=2.5 ...   (initial high loss — pretrain on Dolci)
[Stage 1/1 L=1] step 1000 loss=0.8 ... (decreasing as model learns Dolci)
[PFLOPS] Reached 5362 PFLOPS — saving checkpoint at milestone 5362  ← first save
[PFLOPS] Reached 25273 PFLOPS — saving checkpoint at milestone 25273
[PFLOPS] Reached 48253 PFLOPS — saving checkpoint at milestone 48253
[PFLOPS] Reached 85266 PFLOPS — saving checkpoint at milestone 85266
[PFLOPS] Reached 85266 PFLOPS ≥ max 85266 — stopping training      ← auto-stop
```

After stop: 4 checkpoints in `pflops_checkpoints/pflops_5362/`, `pflops_25273/`, `pflops_48253/`, `pflops_85266/` — each is the Dolci-only model at the matching compute budget. Use these as the "instruct-only" arm in your downstream eval at L=8/16/24/32.

### If 6% mix later finishes L=32 cleanly + extends to L=48

The compute budgets for "L=32 cleared", "L=40 cleared", "L=48 cleared" will be NEW values higher than 85,266. To extend the Dolci-only run to match those:

```bash
# Same JOB_ID = continues from the saved checkpoint at 85,266 PFLOPS
JOB_ID=<original-dolci-job-id> \
  PFLOPS_MILESTONES="5362,25273,48253,85266,<new-L32>,<new-L40>,<new-L48>" \
  MAX_TOTAL_PFLOPS=<new-L48> \
  bash tuning_job_qwen17b_dolci_only_jackie.sh
```

The milestone tracker auto-skips the first 4 (already saved) and only saves the new ones. Trainer state restored from the full HF checkpoint at 85,266 → continues with full Adam state continuity (this run's checkpoints DO have optimizer.pt, since training is bit-continuous within this run).

---

## Sync results back to Gautschi

Either:
- Auto-sync via Jackie's existing GCS hook to `gs://jackierwzhang-purdue-research-curriculum/`
- Or `gsutil -m rsync -r /home/ray/scratch/nl_output/search/job_<id> gs://jackierwzhang-purdue-research-curriculum/<subdir>/job_<id>` (manual)

Then on Gautschi side: `gsutil -m rsync -r gs://...` to pull.

---

## Estimated wall-clock

- (A) 6%-mix resume on 2× H100: ~24-36 hours (stage 32 clearance + 16 new stages 33-48)
- (B) Dolci-only on 6× A100: ~52 hours (85K PFLOPS at ~0.45 PFLOPS/s aggregate)

Both can run simultaneously since they're on different pods.

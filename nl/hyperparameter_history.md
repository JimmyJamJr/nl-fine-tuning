# NL Fine-Tuning Hyperparameter History

All experiments use curriculum learning with linear lookahead on the search task.
"Eff Batch" = GPUs x Batch Size x Grad Accumulation Steps.

---

## Pythia 160M

### LoRA (rank=16)

| Jobs | LR | BS | GA | GPUs | Eff Batch | Notes |
|------|-----|-----|-----|------|-----------|-------|
| 8292431, 8292501, 8292519 | 1e-4 | 96 | 1 | 2 | 192 | 3 attempts, same config |
| 8292534 | 5e-4 | 96 | 1 | 2 | 192 | |
| 8306676 | 1e-4 | 96 | 2 | 2 | 384 | |
| 8313431 | 5e-4 | 192 | 2 | 2 | 768 | |

### LoRA (rank=64)

| Jobs | LR | BS | GA | GPUs | Eff Batch | Notes |
|------|-----|-----|-----|------|-----------|-------|
| 8292555 | 5e-4 | 96 | 1 | 2 | 192 | |
| 8292568 | 1e-4 | 96 | 1 | 2 | 192 | |

### Full Fine-Tuning

| Jobs | LR | BS | GA | GPUs | Eff Batch | Notes |
|------|-----|-----|-----|------|-----------|-------|
| 8305248 | 1e-5 | 48 | 1 | 2 | 96 | |
| 8305249 | 1e-5 | 96 | 1 | 2 | 192 | |
| 8305250 | 5e-5 | 48 | 1 | 2 | 96 | |
| 8305216 | 5e-5 | 96 | 1 | 2 | 192 | Same config as Pythia 410M best |
| 8304850, 8305418 | 5e-5 | 96 | 1 | 4 | 384 | |
| 8305246 | 1e-4 | 48 | 1 | 2 | 96 | |
| 8305247 | 1e-4 | 96 | 1 | 2 | 192 | |
| 8305417, 8315046 | 1e-4 | 96 | 1 | 4 | 384 | |
| 8305544 | 2e-4 | 96 | 1 | 4 | 384 | |
| **8319879 → 8324386 → 8378391** | **1e-4** | **96** | **2** | **2** | **384** | **Vanilla baseline** (resumed from 8315046). Stage 2 (L=16), plateaued at loss=0.050, full_acc=91% |
| 8387575 | 1e-4 | 96 | 2 | 2 | 384 | **Plateau LR spike (2x)** — resumes from 8324386 Stage 2 checkpoint. `plateau_action=lr_spike, lr_spike_factor=2.0`. Flat at loss=0.053, full_acc=92%. No improvement over vanilla |
| ~~8387560~~ | 2e-4 | 96 | 2 | 2 | 384 | **SGDR cosine restarts** — fresh from Stage 1 weights. `cosine_t0=10000, t_mult=2, eta_min_ratio=0.5`. **Cancelled** — much worse than vanilla (loss=0.22, acc=50% vs vanilla's 0.05/91% at same steps-in-stage). Cosine restarts hurt curriculum learning |
| ~~earlier spike~~ | 1e-4 | 96 | 2 | 2 | 384 | **Plateau LR spike (5x)** — collapsed accuracy from 94% to 70% after repeated spikes. Too aggressive |

| 8388117 | 1e-4 | 48 | 1 | 4 | 192 | 4-GPU variant, lower eff_batch. Had loss history gap (missing steps 1-204K). 748K steps, Stage 2 |

All Pythia 160M runs: 12 stages, max_L=96, max_input=576. Cleared Stage 1 (L=8), stuck on Stage 2 (L=16).

**160M findings:**
- Full FT with lr=1e-4, eff_batch=384 is the best config (cleared Stage 1, reached 91% on Stage 2)
- LR spike (5x) harmful, LR spike (2x) neutral, SGDR cosine restarts harmful
- Constant LR is optimal for curriculum learning (consistent with Luo et al. 2025)
- Model appears capacity-limited at Stage 2 complexity

---

## Pythia 410M

### LoRA (rank=16)

| Jobs | LR | BS | GA | GPUs | Eff Batch | Notes |
|------|-----|-----|-----|------|-----------|-------|
| 8305896 | 1e-4 | 96 | 2 | 2 | 384 | |

### Full Fine-Tuning

| Jobs | LR | BS | GA | GPUs | Eff Batch | Notes |
|------|-----|-----|-----|------|-----------|-------|
| 8304849 → 8319876 | 5e-5 | 96 | 1 | 2 | 192 | Reached Stage 6 (L=48) |
| **8324385 → 8415213 → 8500174** | **5e-5** | **48** | **1** | **4** | **192** | **Current run** (restarted from 8324385 checkpoint-449000 due to loss history gap). Reached **Stage 8 (L=64)** at 361K PFLOPs / 587 GPU-hrs. Massive wall at L=48-56 (381K+266K steps). Still grinding on L=64 at 540K PFLOPs / 1.07M steps. Per-stage PFLOPs: L=8@496, L=16@2,328, L=24@3,572, L=32@7,254, L=40@28,352, **L=48@177,301**, **L=56@141,519**, L=64@136,513* |

| ~~8424517~~ | 2e-5 | 48 | 1 | 4 | 192 | **Cancelled** — reached L=16 at 1,140 PFLOPs (baseline: 496). At 2,821 PFLOPs still on L=16 while baseline had cleared L=24 at 2,824. 2.3x worse |
| ~~8424518~~ | 1e-4 | 48 | 1 | 4 | 192 | **Cancelled** — still on L=8 at 1,963 PFLOPs with 58% acc. Baseline cleared L=16 at 496 PFLOPs. Same failure pattern as 1B lr=1e-4 |
| ~~8424519~~ | 5e-5 | 48 | 2 | 4 | 384 | **Cancelled** — 2.4-4x worse PFLOPs than baseline at every stage (L=16@2,009 vs 496, L=24@7,316 vs 2,824, L=32@15,488 vs 6,396). Larger batch hurts 410M |

### Full Fine-Tuning (step=4 curriculum)

| Jobs | LR | BS | GA | GPUs | Eff Batch | Notes |
|------|-----|-----|-----|------|-----------|-------|
| **8446252 → 8478416 → 8566793** | **5e-5** | **64** | **1** | **3** | **192** | **Step=4 curriculum** (L=4,8,...,96, 24 stages). Same eff_batch as baseline. **~2x worse at early L but crossover at L=56**: L=16@1,302 vs 496 (2.6x), L=24@5,744 vs 2,824 (2.0x), L=32@13,663 vs 6,396 (2.1x), L=40@30,705 vs 13,650 (2.2x), L=48@63,226 vs 42,002 (1.5x), **L=56@110,817 vs 219,303 (step=4 wins 2.0x!)**, **L=64@222,308 vs 360,821 (step=4 wins 1.6x!)**. Finer curriculum avoids massive walls. Currently at L=64, 258K PFLOPs, grinding |

### Full Fine-Tuning (Reinit — random weights)

| Jobs | LR | BS | GA | GPUs | Eff Batch | Schedule | Notes |
|------|-----|-----|-----|------|-----------|----------|-------|
| ~~8446090 → 8483032~~ | 1e-4 | 48 | 1 | 4 | 192 | constant | **Cancelled** — plateaued at ~58% acc on L=8 after 55K PFLOPs (loss=0.18). Baseline cleared L=48 at same compute |
| **8527729** | **1e-4** | **64** | **4** | **4** | **1024** | **constant** | **Best reinit config** (Max's batch size). Cleared L=4 at 14.3K PFLOPs. On L=8 at 30K PFLOPs, loss=0.065, acc=85%. Still improving (loss was 0.118→0.070 before preemption). First reinit to advance a stage |
| ~~8445190~~ | 1e-4 | 48 | 2 | 4 | 384 | constant | **Cancelled** — same plateau as b=192, both stuck at ~50% acc on L=8. Converges to same loss ~0.21 |
| ~~8424851~~ | 5e-5 | 48 | 1 | 4 | 192 | constant | **Cancelled** — plateaued at loss=0.50 after 18K PFLOPs. Too low LR for random init |
| ~~8442036~~ | 5e-4 | 48 | 1 | 4 | 192 | constant | **Cancelled** — loss barely dropping (3.87→3.51 over 7.5K steps). Too high |
| ~~8444382~~ | 3e-4 | 48 | 1 | 4 | 192 | constant | **Cancelled** — loss=2.47 at 2.5K PFLOPs. lr=1e-4 was ~0.48 at same point. Too high |
| ~~8444383~~ | 3e-4 | 48 | 2 | 4 | 384 | constant | **Cancelled** — lr=3e-4 b=192 already worse |
| ~~8445497~~ | 1e-3 | 48 | 4 | 4 | 768 | constant | **Cancelled** — loss=3.63→3.43 over 17K steps. Too aggressive |
| ~~8446092~~ | 1e-3 | 48 | 4 | 4 | 768 | HF cosine | **Cancelled** — cosine decay bug: HF scheduler needs max_steps, which is unknown in curriculum. LR never actually decayed |
| ~~8459553 → 8478417~~ | 3e-4 | 48 | 1 | 4 | 192 | cosine_restart | **Cancelled** — LR oscillated (3e-4→8.8e-5→1.4e-4) due to restart cycling. Loss=0.88 at 8K PFLOPs while lr=1e-4 constant was 0.21 at same point. 15x less efficient. Cosine restart wrong approach — should have used cosine (no restart) |

All Pythia 410M runs: 12 stages, max_L=96, max_input=576. Reached Stage 8 (L=64) with baseline config.

**410M findings:**
- lr=5e-5 is the sweet spot for fine-tuning — both lr=2e-5 and lr=1e-4 are worse
- Higher LR (1e-4) consistently bad for fine-tuning across both 410M and 1B Pythia models
- Larger batch (384) hurts 410M fine-tuning (2.4-4x worse PFLOPs) — opposite of 1B where it helps
- **Step=4 curriculum: ~2x worse early but wins at L=56**: extra intermediate stages cost compute at L=16-48 (2.0-2.6x ratio), but avoid the massive wall baseline hits at L=48-56. **Step=4 reaches L=56 at 111K PFLOPs vs baseline's 219K — 2x faster**. Crossover between L=48 and L=56
- **Reinit (random weights) is massively slower**: pretrained baseline clears L=8 at 496 PFLOPs. Best reinit (lr=1e-4, b=192) plateaued at 58% on L=8 after 55K PFLOPs. **Reinit with b=1024 (Max's config) cleared L=4 at 14.3K PFLOPs** and reached 85% on L=8 — first reinit to advance a stage. Larger batch is critical for random init
- For reinit, lr=1e-4 is the best LR. All other LRs worse. LR schedules don't help. **Batch size matters more than LR schedule for reinit**: b=1024 succeeds where b=192 plateaus
- **410M hits a massive wall at L=48-56**: 381K + 266K steps vs <73K for all earlier stages. The 1B doesn't have this wall (15K steps each for L=48 and L=56), suggesting it's a capacity limit

---

## Pythia 1B

### Full Fine-Tuning

| Jobs | LR | BS | GA | GPUs | Eff Batch | Notes |
|------|-----|-----|-----|------|-----------|-------|
| **8386517** | **5e-5** | **48** | **1** | **4** | **192** | **Baseline** (fresh start). Stage 7 (L=56), 313K PFLOPs. Cancelled to free GPUs — data preserved |
| ~~8421101~~ | 1e-4 | 48 | 1 | 4 | 192 | **Cancelled** — 3.4x slower to reach L=16 (1.7B tokens vs 0.5B baseline). Too aggressive for 1B |
| **8421102 → 8454846 → 8521540** | **5e-5** | **48** | **2** | **4** | **384** | **Best 1B config**. Reached **L=64** at 337K PFLOPs. Advantage over b=192 baseline at each L: 0%→5%→22%→30%→18%→23%. Per-stage PFLOPs: L=8@2,904, L=16@11,886, L=24@13,235, L=32@24,206, L=40@38,418, L=48@34,646, **L=56@212,070**, L=64@160,317*. **1B clears L=48 in 15K steps vs 410M's 381K — no wall at L=48** but hits wall at L=56-64 |
| 8544599 | 5e-5 | 48 | 4 | 4 | 768 | **New run** — same hypers as Qwen FT for direct comparison. Fresh start. Normal QOS |
| ~~8421103~~ | 2e-5 | 48 | 1 | 4 | 192 | **Cancelled** — reached L=16 fast (1.9K PFLOPs vs 2.9K baseline) but then stalled. At 16.2K PFLOPs still on L=16 while baseline already cleared L=24 at 15.5K. Too slow in tail convergence |
| ~~8415468~~ | 5e-5 | 48 | 1 | 4 | 192 | **No-curriculum** (n_stages=1, L=96). Reached 35% acc at 27K steps, plateauing. Cancelled |

**1B findings:**
- lr=1e-4 too aggressive (3.4x slower), lr=2e-5 fast at first but stalls in tail convergence
- **batch=384 is the best 1B config**: 23% fewer PFLOPs to reach L=56 (125K vs 163K). Advantage grows with stage depth (0% at L=16, 30% at L=40, 23% at L=56)
- No-curriculum plateaus at ~35% on L=96
- Optimal batch differs by model size: 192 for 410M, 384 for 1B

All Pythia 1B runs: 12 stages, max_L=96, max_input=576.

---

## Qwen3 0.6B

All Qwen 0.6B runs use LoRA rank=16 unless noted.

### Early experiments (8 GPUs, max_L=256, max_input=1536)

| Jobs | LR | BS | GA | GPUs | Eff Batch | Notes |
|------|-----|-----|-----|------|-----------|-------|
| 6308044 + resumes | 3e-5 | 16 | 1 | 8 | 128 | No linear lookahead initially |
| 6308222 + resumes | 3e-5 | 16 | 1 | 8 | 128 | Linear lookahead added |
| 7531208, 7539481 | 3e-4 | 16 | 1 | 8 | 128 | No grad checkpoint |
| 7541246 + many resumes | 1e-4 | 16 | 1 | 8 | 128 | No grad checkpoint initially, then with |
| 7547486, 7569649 | 2e-4 | 16 | 1 | 8 | 128 | |
| 7618364 → ... → 8022825 | 1e-4 | 16 | 1 | 8 | 128 | Long chain of resumes |

### max_L=96 experiments (8 GPUs, max_input=1536)

| Jobs | LR | BS | GA | GPUs | Eff Batch | Notes |
|------|-----|-----|-----|------|-----------|-------|
| 8184167 → 8215015 | 1e-4 | 16 | 1 | 8 | 128 | max_input=1536 (too large for L=96) |
| 8194590 → 8242867 | 1e-4 | 16 | 1 | 8 | 128 | Same issue: alpha maxed at 0.378 |

### max_L=128 experiments (8 GPUs, max_input=768)

| Jobs | LR | BS | GA | GPUs | Eff Batch | Notes |
|------|-----|-----|-----|------|-----------|-------|
| 6308280, 6312603 + resumes | 3e-5 | 16 | 1 | 8 | 128 | |
| 6336183, 6464277, 6464372 | 3e-5 | 16 | 1 | 8 | 128 | |
| 6905898 + resumes | 3e-5 | 32 | 1 | 8 | 256 | |
| 6905901 + resumes | 3e-5 | 32 | 1 | 8 | 256 | |
| 8245533, 8245683, 8248186, 8249556 | 1e-4 | 16 | 1 | 8 | 128 | |
| 8254806 → 8260161 | 1e-4 | 16 | 1 | 8 | 128 | |
| 8254807 → 8263556 | 1e-4 | 16 | 1 | 8 | 128 | |
| 8265549 → 8293261 | 1e-4 | 16 | 1 | 8 | 128 | |
| 8291685, 8291686 | 1e-4 | 96 | 1 | 8 | 768 | Larger batch |
| **8305304 → 8319839 → 8353090** | **1e-4** | **96** | **2** | **4** | **768** | **Current run** — Stage 15/15 (L=128), loss=0.020, full_acc=97%, greedy=97%. Near completion. Eval on L=96 at each stage: 48%@L=16 → 67%@L=40 → 85%@L=64 → 97%@L=80 → **99%@L=96** |
| 8313552 | 1e-4 | 96 | 2 | 4 | 768 | No-curriculum variant (L=128) |
| **8421087** | **1e-4** | **96** | **2** | **4** | **768** | **No-curriculum L=96 — FINISHED**. Step 21.9K, greedy=97.6%, loss=0.023, 112K PFLOPs, 261 GPU-hrs. Curriculum reached 98.6% at 94K PFLOPs — **curriculum 1.2x less compute, 1% higher acc**. At same compute (~67K PF): curr=97.8% vs nocurr=96.4%. Gap closes late but curriculum consistently ahead |
| ~~8469854~~ | 1e-4 | 96 | 2 | 4 | 768 | **No-curriculum L=112 (LoRA)** — cancelled. Step 9K, 53K PFLOPs, 65% acc. LoRA discontinued in favor of full FT |

### Full Fine-Tuning (curriculum, max_L=96, max_input=576)

| Jobs | LR | BS | GA | GPUs | Eff Batch | Notes |
|------|-----|-----|-----|------|-----------|-------|
| **8533255** | **5e-5** | **48** | **4** | **4** | **768** | **COMPLETED L=96** at 98.4% greedy, 47.9K PFLOPs, 55 GPU-hrs. **2x faster than LoRA** at every stage. Extended to L=128 via job 8555128 |
| **8555128** | **5e-5** | **48** | **4** | **4** | **768** | **COMPLETED L=128** (resumed from 8533255). 98.6% greedy, 83.3K PFLOPs, 95 GPU-hrs total. Per-stage: L=104@67.7K, L=112@69.6K, L=120@75.7K, L=128@83.3K. **2.4x faster than LoRA** (LoRA: 202K PFLOPs for L=128). No walls — Qwen handles all L smoothly |
| ~~8544599~~ | 5e-5 | 48 | 4 | 4 | 768 | Pythia 1B with same hypers — cancelled |

### Full Fine-Tuning No-Curriculum (various max_L, max_input=6*L)

| Jobs | LR | BS | GA | GPUs | Eff Batch | Target L | Notes |
|------|-----|-----|-----|------|-----------|----------|-------|
| **8555129 → 8577238** | **5e-5** | **48** | **4** | **4** | **768** | **96** | **COMPLETED**. Step 18.9K, 96.5K PFLOPs, 98.8% greedy. Curriculum reached same at 47.9K PFLOPs — **curriculum 2.0x faster** |
| **8555130 → 8577239** | **5e-5** | **48** | **4** | **4** | **768** | **104** | **COMPLETED**. Step 19.5K, 107.7K PFLOPs, 97.6% greedy. Curriculum reached L=104 at 67.7K PFLOPs — **curriculum 1.6x faster** |
| **8555131 → 8577240** | **5e-5** | **48** | **4** | **4** | **768** | **112** | Running. Step 6K, 35.6K PFLOPs, 57.2% acc |
| **8555132 → 8585533** | **5e-5** | **48** | **4** | **4** | **768** | **120** | Running. Step 5.5K, 35K PFLOPs, 53% acc |
| **8555133 → 8585534** | **5e-5** | **48** | **4** | **4** | **768** | **128** | Running. Step 4K, 27.1K PFLOPs, 44.8% acc |

### Full FT + Reinit (max_L=64, max_input=384)

| Jobs | LR | BS | GA | GPUs | Eff Batch | Notes |
|------|-----|-----|-----|------|-----------|-------|
| 8247102, 8248165 | 1e-4 | 16 | 1 | 8 | 128 | Reinit weights, no LoRA |
| 8254808, 8260176 | 1e-4 | 16 | 1 | 8 | 128 | Reinit weights, no LoRA |

### No-curriculum (1 stage, max_L=128, max_input=768)

| Jobs | LR | BS | GA | GPUs | Eff Batch | Notes |
|------|-----|-----|-----|------|-----------|-------|
| 8263630, 8263665, 8263706, 8263763 | 1e-4 | 16 | 1 | 1 | 16 | Single GPU |
| 8264187 | 1e-4 | 16 | 1 | 2 | 32 | |
| 8264462, 8264547 | 1e-4 | 16 | 1 | 1 | 16 | |

### Small batch experiments (2 GPUs, max_L=128, max_input=768)

| Jobs | LR | BS | GA | GPUs | Eff Batch | Notes |
|------|-----|-----|-----|------|-----------|-------|
| many (no job IDs) | 1e-4 | 16 | 1 | 2 | 32 | ~50+ runs with same config |
| (no job IDs) | 1e-4 | 192 | 1 | 2 | 384 | |
| (no job IDs) | 1e-4 | 288 | 1 | 2 | 576 | |
| (no job IDs) | 1e-4 | 384 | 1 | 2 | 768 | |
| (no job IDs) | 1e-4 | 480 | 1 | 2 | 960 | |

---

## Qwen3 1.7B

All Qwen 1.7B runs use LoRA rank=16.

### Early experiments (no packing, no linear lookahead)

| Jobs | LR | BS | GA | GPUs | Eff Batch | max_L | max_input | Notes |
|------|-----|-----|-----|------|-----------|-------|-----------|-------|
| 5216111 | 2e-5 | 16 | 1 | 4 | 64 | 128 | 768 | |
| 5216438 + resumes | 2e-5 | 16 | 1 | 8 | 128 | 128 | 768 | |
| 5342184 + resumes | 2e-5 | 16 | 1 | 8 | 128 | 128 | 768 | Added Liger |
| 5343739 + resumes | 2e-5 | 16 | 1 | 8 | 128 | 128 | 768 | 2 stages only |
| 5344221 | 2e-5 | 16 | 1 | 8 | 128 | 128 | 768 | No curriculum |
| 5345387, 5345396, 5347694 | 2e-5 | 16 | 1 | 8 | 128 | 128 | 768 | 10 stages |
| 5350905 + resumes | 2e-5 | 16 | 1 | 8 | 128 | 512 | 3072 | |
| 5474538 + resumes | 2e-5 | 16 | 1 | 8 | 128 | 256 | 1536 | |
| 5474563 | 2e-5 | 16 | 1 | 8 | 128 | 384 | 2304 | |
| 5572855, 5572858 | 2e-5 | 16 | 1 | 8 | 128 | 128 | 768 | |
| 5651166 → 6267216 | 2e-5 | 16 | 1 | 8 | 128 | 192 | 1152 | |

### With packing + linear lookahead

| Jobs | LR | BS | GA | GPUs | Eff Batch | max_L | max_input | Notes |
|------|-----|-----|-----|------|-----------|-------|-----------|-------|
| 6278757, 6278784, 6281139 | 2e-5 | 16 | 1 | 4 | 64 | 256 | 1536 | 5 stages |
| 6281157, 6283040, 6284094 | 2e-5 | 16 | 1 | 4 | 64 | 256 | 1536 | No curriculum |
| 6284102 + many | 2e-5 | 16 | 1 | 8 | 128 | 256 | 1536 | No curriculum |
| 6305989 + resumes | 2e-5 | 16 | 1 | 8 | 128 | 256 | 1536 | 10 stages |
| 6306108 + many | 2e-5 | 16 | 1 | 8 | 128 | 256 | 1536 | No curriculum, packing |
| 6307036 | 2e-5 | 16 | 1 | 8 | 128 | 256 | 1536 | 10 stages |
| 6307413 + many | 3e-5 | 16 | 1 | 8 | 128 | 256 | 1536 | No curriculum |
| 6314287 → 6585900 | 3e-5 | 16 | 1 | 8 | 128 | 128 | 768 | Long chain of resumes |

---

## Qwen3 4B

All Qwen 4B runs use LoRA rank=16, 8 GPUs.

| Jobs | LR | BS | GA | GPUs | Eff Batch | max_L | max_input | Notes |
|------|-----|-----|-----|------|-----------|-------|-----------|-------|
| 5345417, 5345497, 5345688 | 2e-5 | 16 | 1 | 8 | 128 | 512 | 3072 | No curriculum |
| 5346403, 5347682 | 2e-5 | 16 | 1 | 8 | 128 | 512 | 3072 | 10 stages |
| 5651164 | 2e-5 | 16 | 1 | 8 | 128 | 128 | 768 | |
| 5685704 → 6272089 | 2e-5 | 16 | 1 | 8 | 128 | 256 | 1536 | Chain of resumes |
| 6905911 | 3e-5 | 32 | 1 | 8 | 256 | 128 | 768 | Linear lookahead |

---

## Symbolic Transformer (6 layers, 1 head, 16 hidden dim)

All symbolic runs use SophiaG optimizer, max_input_size=256, max_lookahead=40, batch_size=512.

| Jobs | LR | Optimizer | AMP | Compile | base_L | step_L | Threshold | Notes |
|------|-----|-----------|-----|---------|--------|--------|-----------|-------|
| various | 1e-5 | SophiaG | no | no | 8 | 8 | 0.99 | Baseline — stuck on Stage 3 (L=24) |
| various | 1e-5 | AdamW | yes | yes | 8 | 8 | 0.99 | Same wall at Stage 3 |
| ~~8377756~~ | 1e-5 | SophiaG+hessian | yes | yes | 8 | 8 | 0.99 | **Cancelled** — 14K epochs on Stage 3, acc declined from 80% to 71%. Capacity wall confirmed |
| ~~8388056~~ | 1e-5 | SophiaG+hessian | yes | yes | 2 | 2 | 0.98 | **Old code** (post-update accuracy). Reached Stage 8 (L=16). Train=95% but fresh same-α eval only 86-91% — inflated accuracy. Cancelled |
| **8420638** | **1e-5** | **SophiaG+hessian** | **yes** | **yes** | **2** | **2** | **0.98** | **New code** (pre-update accuracy, eval on advancement, constant LR). Cleared stages 1-6, stuck on Stage 7 (L=14) for 13.4K epochs. Train acc peaked at 95.2% (epoch ~10-11K) then **destabilized** to 88% (epoch 16.5K). Loss was still slowly decreasing (0.175→0.133) right before collapse. Stage eval confirms train≈stage_test (within 0.5%). test_loss(α=1) improved stages 1-5 then reversed at stage 6. **Cancelled** — L=14 is the capacity ceiling |
| **8453744** | **1e-5** | **SophiaG+hessian** | **yes** | **yes** | **2** | **2** | **0.98** | **Per-stage cosine decay** (peak=1e-5, min=1e-7, T_max=20000). Added periodic test eval every 50 epochs. Cleared stages 1-6 in identical epochs as constant LR run. Stage 7: peaked at 95.2% (same as constant), then **same destabilization** pattern — loss rose at epoch ~12K. LR decayed from 1e-5 to ~4.7e-6 (47% of peak) but collapse still occurred at same point. **Cosine decay does not prevent destabilization** — confirms L=14 is a hard capacity wall, not an LR issue. Data deleted but log preserved. |

| ~~8488181~~ | **3e-5** | **SophiaG+hessian** | **yes** | **yes** | **2** | **2** | **0.98** | **Per-stage cosine, higher peak LR (3e-5)**. Cleared stages 1-4 faster than 1e-5 (109 vs 342 epochs for stage 4), but **stuck on Stage 5 (L=10) for 15K+ epochs** at 73% acc (1e-5 cleared Stage 5 in 120 epochs). Higher LR overshoots at moderate difficulty. **Cancelled** — worse than baseline |
| ~~8480401~~ | **1e-4** | **SophiaG+hessian** | **yes** | **yes** | **2** | **2** | **0.98** | **Per-stage cosine, peak=1e-4**. Loss went **NaN at epoch 3** — AMP overflow. 1e-4 is too high for this architecture with mixed precision. **Cancelled** |

**Symbolic findings:**
- All optimizer/AMP/compile variants hit same Stage 3 wall with step=8 curriculum
- With step=8: model reaches ~80% on Stage 3 (L=24, alpha=0.58) then declines over 14K epochs
- With step=2 (new code): cleared through L=12 but stuck on L=14. Train acc peaked at 95.2% (epoch 10-11K) then **destabilized** — loss was still slowly decreasing right before collapse
- **Cosine LR decay does NOT prevent destabilization**: per-stage cosine (1e-5→4.7e-6) produced identical trajectory — same peak (95.2%), same collapse point (~12K epochs), same loss floor (0.1287). Confirms capacity wall, not LR issue
- **Higher peak LR (3e-5) is worse**: clears early stages faster but gets stuck on Stage 5 (L=10) — baseline cleared it in 120 epochs. LR=1e-4 causes NaN with AMP
- Old code had inflated accuracy: post-update measurement gave 98% but fresh same-α eval was 86-91%. New pre-update rolling window gives honest measurement (train≈stage_test within 0.5%)
- test_loss(α=1.0) improved stages 1-5 then reversed at stage 6 — overfitting pattern
- torch.compile bug: model gets re-wrapped in OptimizedModule on each preemption resume (fixed)
- New code ~2.5x faster per epoch (7.7s vs 18.3s) by removing per-epoch eval/save
- **Capacity ceiling is L=12-14** for this architecture (6 layers, 360 hidden dim, ~4.6M params)
- New symbolic run submitted (8480401) with properly working cosine decay (peak=1e-4, min=1e-6) — higher peak LR to test if more aggressive training can break through L=14

---

## Summary of Unique Configs Tried

| Model | Training | LR Range | Eff Batch Range | Unique Configs | Best Stage |
|-------|----------|----------|-----------------|----------------|------------|
| Pythia 160M | LoRA r16 | 1e-4 – 5e-4 | 192 – 768 | 4 | Stage 1 |
| Pythia 160M | LoRA r64 | 1e-4 – 5e-4 | 192 | 2 | Stage 1 |
| Pythia 160M | Full FT | 1e-5 – 2e-4 | 96 – 384 | 8+3 sched | Stage 2 (L=16), plateaued |
| Pythia 410M | LoRA r16 | 1e-4 | 384 | 1 | Stage 1 |
| Pythia 410M | Full FT | 2e-5 – 1e-4 | 192 – 384 | 5 | **Stage 8 (L=64)**. lr=5e-5 b=192 optimal. batch=384 2.4-4x worse. **step=4 wins at L=56** (2x fewer PFLOPs) despite being 2x worse at L=16-48 |
| Pythia 410M | Full FT Reinit | 5e-5 – 1e-3 | 192 – 1024 | 10 | b=192: stuck at 58% on L=8. **b=1024: cleared L=4, 85% on L=8** — larger batch critical for reinit |
| Pythia 1B | Full FT | 2e-5 – 1e-4 | 192 – 768 | 5 | **Stage 8 (L=64)**. batch=384 best (23% fewer PFLOPs). No wall at L=48 unlike 410M. b=768 run in progress |
| Qwen 0.6B | LoRA r16 | 1e-4 – 5e-4 | 16 – 960 | ~15 | Stage 15 (L=128). Nocurr L=96 finished (97.6%). Nocurr L=112 running |
| Qwen 0.6B | **Full FT** | **5e-5** | **768** | **1** | **L=48 in 10.8 GPU-hrs** — 3-20x faster than LoRA, faster than all Pythia models |
| Qwen 0.6B | Full FT+Reinit | 1e-4 | 128 | 1 | — |
| Qwen 1.7B | LoRA r16 | 2e-5 – 3e-5 | 64 – 128 | ~10 | — |
| Qwen 4B | LoRA r16 | 2e-5 – 3e-5 | 128 – 256 | ~5 | — |
| Symbolic 4.6M | SophiaG | 1e-5 | 512 | 4+1 sched | Capacity ceiling L=12-14 (step=2). L=24 wall (step=8) |

---

## Key Findings

1. **LR schedules hurt curriculum learning**: Constant LR is optimal. SGDR cosine restarts significantly degraded performance (4x higher loss). LR spikes (5x) collapsed accuracy; 2x spikes were neutral. Consistent with Luo et al. (2025).

2. **Model size determines curriculum ceiling**: 160M plateaus at Stage 2 (L=16), 410M at Stage 7 (L=56), while 1B and Qwen 0.6B continue advancing. Larger models can handle deeper graph reasoning.

3. **Full FT > LoRA**: For Pythia models, full fine-tuning consistently outperforms LoRA on this task.

4. **Gradual curriculum matters for small models**: Symbolic transformer hit a hard wall at Stage 3 with step=8 (L jumps of 8). Restarted with step=2 for more gradual progression.

5. **lr=5e-5 is robust for 410M and 1B**: Both higher (1e-4) and lower (2e-5) LR are worse for these model sizes. Higher LR consistently bad across both. Lower LR reaches early stages faster but stalls on later stages.

6. **Larger batch (384) helps 1B but hurts 410M**: 1B reaches L=56 with 23% fewer PFLOPs at batch=384. 410M is 2.4-4x worse at batch=384. Optimal batch scales with model size.

7. **Pretraining provides massive compute advantage**: Pretrained 410M clears L=8 at 496 PFLOPs. Reinit with b=192 plateaus at 58% on L=8 after 55K PFLOPs. **Reinit with b=1024 cleared L=4** (first reinit to advance) and reached 85% on L=8 — larger batch is critical for random init. 10 reinit configs tested.

8. **410M is most compute-efficient up to L=48, then 1B takes over**: 410M reaches each L with 2-6x fewer PFLOPs than 1B for L=8 through L=48. But 410M hits a massive wall at L=48-56 (177K+142K PFLOPs per stage) that the 1B doesn't have (35K+212K). Both hit walls at L=64 (~136K-160K PFLOPs per stage, still grinding).

9. **Curriculum > no-curriculum** (Qwen 0.6B LoRA, L=96 target): Curriculum reached 98.6% at 94K PFLOPs. Nocurr finished at 97.6% / 112K PFLOPs — curriculum 1.2x less compute, 1% higher acc. At same compute curriculum is consistently 2-24% ahead. Nocurr L=112 in progress.

10. **Symbolic capacity ceiling is L=14**: Confirmed with both constant LR and per-stage cosine decay — identical trajectories, same peak (95.2%), same destabilization point (~12K epochs). Not an LR issue — hard capacity wall for this architecture.

11. **Step=4 curriculum: worse early, better late**: 2.0-2.6x worse PFLOPs at L=16-48, but **2x BETTER at L=56** (111K vs 219K PFLOPs). Finer curriculum avoids the massive wall baseline hits. Crossover between L=48 and L=56.

12. **Qwen Full FT >> LoRA >> Pythia**: Qwen 0.6B full FT reaches L=48 at 8.4K PFLOPs — **3.2x faster than LoRA** (27K) and **5x faster than Pythia 410M** (42K). Architecture matters more than model size: Qwen 0.6B outperforms Pythia 1B despite being smaller. Possible causes: better pretraining, higher rope_theta (1M vs 10K), 100% rotary coverage, GQA, more layers (28 vs 16/24).

13. **Per-stage compute shows architectural differences**: Qwen cost per stage grows gradually (3K→7K→18K→43K PFLOPs). Pythia has sudden 10-20x jumps (410M: 7K→28K→177K at L=32→40→48). Pythia hits walls; Qwen doesn't. Likely due to positional encoding differences (25% rotary coverage in Pythia vs 100% in Qwen).

---

## Performance Optimization Benchmarks

### Flash Attention 3 vs FA2 (Qwen 0.6B, H100)
- FA3 is 1.5-1.7x faster for the attention kernel alone at L=32+
- But attention is <2% of step time, so end-to-end speedup is <1%
- Not worth the code change

### Fused AdamW
- No speedup on any model (Pythia 410M, 1B, Qwen 0.6B) at any L
- Tested with isolated cold-start jobs

### Liger Kernels (Qwen only, Pythia not supported)
- 29-30% speedup for Qwen 0.6B at all L values (cold-start verified)
- Benchmarks (ms/step): L=8: 279→197, L=32: 790→549, L=64: 1504→1048, L=96: 2283→1612

### torch.compile
- Works with HF standard forward (24% speedup) but NOT with our custom PackedSequenceTrainer varlen forward
- With native HF varlen forward: no additional speedup over native alone

### Native HF varlen forward (FlashAttentionKwargs)
- HF supports passing cu_seq_lens through model forward natively
- Tested but SLOWER than our custom forward at L=32+ (549ms vs 385ms at L=32, 1067ms vs 769ms at L=64)
- Custom forward is faster because it handles gradient checkpointing more efficiently (per-layer vs whole-model)
- Only faster at L=8 (191ms vs 197ms) which is negligible

### Best config per model:
- **Qwen 0.6B**: Liger + regular AdamW + custom forward (our current config)
- **Pythia 410M/1B**: No Liger (not supported) + regular AdamW + custom forward (our current config)
- No changes needed — current config is already optimal

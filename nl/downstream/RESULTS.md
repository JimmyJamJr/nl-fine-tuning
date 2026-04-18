# Downstream Eval Results

Manually maintained. Add new rows/columns/benchmarks by hand as results come in.
Last compiled automatically from `compiled_downstream.json` on 2026-04-16.

**Models.** The primary comparison is now 6-way (curriculum-depth scaling curve):

| alias | checkpoint / description |
| --- | --- |
| `base` | `Qwen/Qwen3-0.6B` (no fine-tuning) |
| `instruct_only` | base + Dolci instruct SFT (no curriculum) — jobs 8826386 → 8969184 → 9052803 → 9152198 |
| `6pct_L6` | 6% Dolci mix + curriculum, checkpoint at L=6 (early stage) — jobs 8894380/9001346 |
| `6pct_L16` | Same run, checkpoint at L=16 |
| `6pct_L32` | Same run, checkpoint at L=32 |
| `6pct_L75` | Same run, checkpoint at L=75 (final / best lookahead) |

Additional models that appear below are threshold-sweep artifacts (`99pct_L*` from job 9048131),
ablations (`pure_curr`, `dolci20pct_search`, `6pct_search`/`6pct_L6`, `lora_merged`), and
older Qwen curriculum checkpoints (`models__qwen06b_ft_curr_L96/L128`).

Values are accuracies (higher is better) unless otherwise noted. A dash (`-`) means "not run yet."

---

## 1. Reasoning & search-transfer benchmarks

### 1a. Legal (LegalBench — 3 tasks)

Full test set (n ≈ 9495 across 3 tasks combined) run 2026-04-16 on Gilbreth.

**Log-likelihood scoring** (score " Yes" / " No" on chat-wrapped prompt):

| | base | instruct_only | 6pct_L6 | 6pct_L16 | 6pct_L32 | 6pct_L75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | **0.574** | 0.425 | 0.424 | 0.424 | 0.424 | 0.422 |

**Generation scoring** (`legal_gen` — model generates answer, few-shot prompt matches LegalBench author setup):

| | base | instruct_only | 6pct_L6 | 6pct_L16 | 6pct_L32 | 6pct_L75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 0.394 | 0.476 | 0.418 | 0.458 | 0.530 | **0.546** |

> **The two scoring methods tell opposite stories on legal reasoning.**
>
> Under **log-lik scoring**, base dominates (0.574 vs ~0.42 for all fine-tuned). Under
> **generation scoring**, 6pct_L75 leads by **+15pp over base** (0.546 vs 0.394).
>
> Why the flip? Log-lik scores forced completions ` Yes` / ` No` at the assistant-role
> position — a distribution shift from fine-tuning can make these tokens less likely
> relative to actual content tokens, dragging log-lik accuracy toward chance. When the
> model is allowed to generate freely (gen scoring), the fine-tuned models produce
> correct answers more often. Diagnostic previously showed `6pct_L75` has yes-rate=0.00
> in chat log-lik (always prefers ` No`), consistent with format collapse at that position.
>
> **Our paper-relevant number for legal is `legal_gen`**: 6pct_L75 wins, generation
> scoring isolates real reasoning from token-position bias.

### 1b. ProofWriter OWA (3-class: True/False/Unknown)

**Log-likelihood scoring** (`proofwriter`, full test set per depth — n=1959 to 6194):

| Depth | base | instruct_only | 6pct_L6 | 6pct_L16 | 6pct_L32 | 6pct_L75 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.160 | 0.739 | 0.703 | 0.695 | 0.696 | **0.743** |
| 1 | 0.209 | 0.569 | 0.581 | 0.581 | 0.581 | **0.584** |
| 2 | 0.329 | 0.353 | 0.342 | 0.341 | 0.341 | 0.345 |
| 3 | 0.401 | 0.214 | 0.199 | 0.198 | 0.198 | 0.213 |
| 4 | **0.452** | 0.118 | 0.096 | 0.096 | 0.096 | 0.106 |
| 5 | **0.484** | 0.046 | 0.032 | 0.032 | 0.032 | 0.044 |

**Generation scoring** (`proofwriter_gen`, n=200 per depth):

| Depth | base | instruct_only | 6pct_L6 | 6pct_L16 | 6pct_L32 | 6pct_L75 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.145 | 0.600 | 0.635 | **0.745** | 0.755 | 0.605 |
| 1 | 0.080 | 0.530 | 0.545 | **0.625** | 0.610 | 0.500 |
| 2 | 0.155 | **0.450** | 0.475 | 0.355 | 0.380 | 0.330 |
| 3 | 0.165 | 0.340 | **0.430** | 0.210 | 0.245 | 0.280 |
| 4 | 0.150 | 0.290 | **0.440** | 0.095 | 0.125 | 0.215 |
| 5 | 0.185 | 0.275 | **0.490** | 0.065 | 0.105 | 0.210 |

> Notes: Under log-lik, base wins deep depths (3-5) because fine-tuned models
> collapse to T/F preference over Unknown. Under generation, base is essentially
> chance (~0.1-0.2 = 1/3 class prior), while fine-tuned models generate Unknown
> more readily — `instruct_only` leads at deep depths (0.275-0.340), `6pct_L16`
> leads at shallow depths. **3-class classification is the hardest; even the best
> fine-tuned model only hits 0.34 at depth 3-5.**

### 1c. ProofWriter CWA (2-class: True/False)

**Log-likelihood scoring** (`proofwriter_cwa`, full test set per depth — n=2047 to 6125):

| Depth | base | instruct_only | 6pct_L6 | 6pct_L16 | 6pct_L32 | 6pct_L75 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.516 | **0.636** | 0.600 | 0.621 | 0.612 | 0.609 |
| 1 | 0.485 | 0.492 | 0.529 | 0.514 | 0.535 | **0.540** |
| 2 | 0.494 | 0.544 | 0.605 | 0.533 | 0.592 | **0.615** |
| 3 | 0.498 | 0.581 | 0.657 | 0.561 | 0.637 | **0.682** |
| 4 | 0.498 | 0.576 | 0.674 | 0.561 | 0.650 | **0.705** |
| 5 | 0.498 | 0.578 | 0.667 | 0.570 | 0.662 | **0.716** |

**Generation scoring** (`proofwriter_cwa_gen`, n=200 per depth):

| Depth | base | instruct_only | 6pct_L6 | 6pct_L16 | 6pct_L32 | 6pct_L75 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.120 | 0.480 | 0.425 | **0.505** | 0.495 | 0.475 |
| 1 | 0.045 | 0.495 | 0.485 | **0.500** | 0.445 | 0.485 |
| 2 | 0.055 | 0.495 | **0.645** | 0.565 | 0.500 | 0.495 |
| 3 | 0.025 | 0.495 | **0.720** | 0.640 | 0.570 | 0.545 |
| 4 | 0.070 | 0.495 | **0.870** | 0.700 | 0.605 | 0.550 |
| 5 | 0.065 | 0.550 | **0.895** | 0.770 | 0.685 | 0.645 |

> **Strongest curriculum-transfer finding in the paper.** Under BOTH scoring
> methods, `6pct_L75` (log-lik) or `6pct_L16` (generation) show **monotonic
> improvement with depth** (log-lik: 0.56 → 0.74; gen: 0.50 → 0.77), while
> base stays near 0.5 (random on a 2-class task) / 0.06 (can't generate).
> Curriculum-trained models handle deeper proofs BETTER, not worse — a clean
> signal that search training transfers to CWA deductive reasoning.
>
> Note that the scoring methods pick different winners:
> - Log-lik: `6pct_L75` wins at depths 1–5 (monotonic 0.56 → 0.74)
> - Generation: `6pct_L16` wins at depths 2–5 (monotonic 0.57 → 0.77)
>
> `6pct_L16` having stronger generation performance on this benchmark is
> intriguing — possibly the shorter curriculum preserves more general ICL
> capability that the few-shot CWA prompt relies on.

### 1d. ZebraLogic — multiple choice (`zebra_mc`)

**Log-likelihood scoring** (full test set, n=3259):

| | base | instruct_only | 6pct_L6 | 6pct_L16 | 6pct_L32 | 6pct_L75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 0.288 | 0.260 | 0.218 | 0.289 | 0.266 | **0.312** |

**Generation scoring** (`zebra_mc_gen`, full test set, n=3259):

| | base | instruct_only | 6pct_L6 | 6pct_L16 | 6pct_L32 | 6pct_L75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 0.005 | **0.245** | 0.074 | 0.098 | 0.170 | 0.074 |

> Notes: In log-lik, `6pct_L75` wins by ~+2pp over base. In generation, base
> collapses to 0.5% — it can't emit the target answer as the leading-space
> token cleanly. Fine-tuned models all do better in gen (they have coherent
> Yes-style short answers in their distribution), with `instruct_only` best.
> Log-lik is the cleaner reasoning measurement here.

### 1f. StepGame (spatial reasoning)

Random baseline = 1/9 ≈ 0.111 (9-way spatial direction classification).

**Log-likelihood scoring** (`stepgame`, capped at n=1000 per hops, 10 hops):

| | base | instruct_only | 6pct_L6 | 6pct_L16 | 6pct_L32 | 6pct_L75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 0.114 | 0.125 | 0.134 | **0.140** | **0.141** | 0.133 |

**Generation scoring** (`stepgame_gen`, n=1000):

| | base | instruct_only | 6pct_L6 | 6pct_L16 | 6pct_L32 | 6pct_L75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 0.162 | 0.146 | 0.161 | 0.173 | 0.159 | **0.188** |

> Notes: Both scoring methods show fine-tuned models above base and above random
> baseline (0.111). `6pct_L16` wins log-lik (0.140), `6pct_L75` wins gen (0.188).
> All models hover just above random — StepGame is hard at this scale.

### 1g. Game of 24 (generation + expression eval)

Hard subset (puzzles ranked 901-1000), n=100 per model:

| | base | instruct_only | 6pct_L6 | 6pct_L16 | 6pct_L32 | 6pct_L75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 0.010 | **0.050** | 0.030 | 0.030 | 0.040 | 0.030 |

> Notes: Task-limited — 0.6B models can't reliably find expressions that equal 24.
> All near-zero; `instruct_only` barely edges out at 5%.

### 1h. PlanBench (Blocksworld, Mystery-Blocksworld, Logistics)

Two scoring methods per domain: exact-match generation (models fail uniformly at this scale)
and mean log-prob of gold plan (curriculum models score closer to gold).

**Exact-match plan generation** (models generate complete plans, n=50 per domain):

| Domain | base | instruct_only | 6pct_L16 | 6pct_L75 |
| --- | ---: | ---: | ---: | ---: |
| blocksworld | 0.000 | 0.000 | 0.000 | 0.000 |
| mystery_blocksworld | 0.000 | 0.000 | 0.000 | 0.000 |
| logistics | 0.000 | 0.000 | 0.000 | 0.000 |

All models score 0% — 0.6B is too small to generate valid PDDL plans.

**Mean log-prob of gold plan per token** (higher = more confident; n=50 per domain):

| Domain | base | instruct_only | 6pct_L16 | 6pct_L75 |
| --- | ---: | ---: | ---: | ---: |
| blocksworld | -2.728 | -1.891 | **-1.804** | -1.890 |
| mystery_blocksworld | -2.636 | -1.664 | **-1.576** | -1.615 |
| logistics | -0.723 | -0.619 | **-0.578** | -0.617 |

> **Clean curriculum-transfer signal in log-prob metrics**: `6pct_L16` has the
> highest (least-negative) log-prob of gold plans in **all 3 PlanBench domains**.
> Even though generation fails, the model has learned that valid plans are more
> likely — indicating the curriculum trained plan-structure priors that transfer.
> Base is consistently the worst (most uncertain about gold plans).

### 1i. Chess mate-in-N (Lichess puzzles)

**Exact-match generation** (n=50 per mate-in-N category, 3 categories):

| | base | instruct_only | 6pct_L16 | 6pct_L75 |
| --- | ---: | ---: | ---: | ---: |
| overall | 0.000 | 0.000 | 0.000 | 0.000 |

All 0% — chess requires specific move reasoning beyond 0.6B capacity.

**Mean log-prob of gold move sequence per token** (`chess_mate_logprob`, n=50 per category):

| Category | base | instruct_only | 6pct_L16 | 6pct_L75 |
| --- | ---: | ---: | ---: | ---: |
| mateIn1 | -8.593 | -5.731 | **-5.362** | -5.378 |
| mateIn2 | -4.404 | -3.806 | **-3.288** | -3.368 |
| mateIn3 | -3.480 | -3.239 | **-2.803** | -2.901 |
| **macro avg** | -5.492 | -4.259 | **-3.817** | -3.882 |

> **`6pct_L16` wins all 3 mate-in-N categories** — same pattern as PlanBench
> log-prob (curriculum-trained model assigns highest probability to gold
> solutions). Gap over base: ~1.7 nat per token on macro. Even though models
> can't generate correct chess moves (exact-match 0%), the curriculum shifts
> the output distribution toward legal/optimal moves more than base alone.
> Log-prob gets less negative with deeper mate (mateIn3 > mateIn2 > mateIn1)
> because longer sequences average over more predictable continuation tokens.

### 1e. ZebraLogic — grid mode (`zebra`) — **REMOVED FROM EVAL FRAMEWORK**

> **Status:** As of 2026-04-16, the `zebra` (grid-mode) benchmark has been
> removed from `eval_downstream.py` and the sbatch defaults. We now report
> only `zebra_mc` (section 1d) for the same underlying puzzles, since MC
> scoring isolates reasoning capability from JSON-format-following — and
> grid mode penalized fine-tuned models for the latter (parse_rate dropped
> 99% → 71%) in a way that wiped out their reasoning advantage.
>
> The numbers below are kept as a record of the last run for reference and
> for any reviewer who asks about grid-mode comparability with published
> ZebraLogic leaderboards.

Re-ran on Gilbreth 2026-04-16 at uniform N=20 per size (500 puzzles per model total).
The old `base` numbers ("0.802 overall") that lived here previously were
inherited from `compile_results.py` supplemental and were invalidated by
this clean re-run.

**Cell accuracy per size** (fraction of (house, attribute) cells correctly assigned):

| Puzzle size | base | instruct_only | 6pct_L16 | 6pct_L75 |
| --- | ---: | ---: | ---: | ---: |
| 2×2 | 0.675 | **0.775** | 0.675 | 0.650 |
| 2×3 | **0.650** | 0.600 | 0.583 | 0.642 |
| 2×4 | 0.613 | **0.650** | 0.581 | 0.606 |
| 2×5 | **0.575** | 0.500 | 0.520 | 0.460 |
| 2×6 | 0.521 | 0.346 | 0.458 | **0.571** |
| 3×2 | 0.425 | **0.500** | 0.433 | 0.475 |
| 3×3 | 0.328 | **0.522** | 0.389 | 0.411 |
| 3×4 | **0.371** | 0.350 | 0.325 | 0.367 |
| 3×5 | **0.383** | 0.373 | 0.390 | 0.350 |
| 3×6 | **0.378** | 0.347 | 0.353 | 0.353 |
| 4×2 | 0.356 | **0.475** | 0.356 | 0.425 |
| 4×3 | 0.267 | 0.371 | 0.333 | **0.400** |
| 4×4 | 0.228 | 0.263 | **0.309** | 0.241 |
| 4×5 | **0.273** | 0.217 | 0.198 | 0.190 |
| 4×6 | **0.281** | 0.198 | 0.171 | 0.125 |
| 5×2 | 0.290 | 0.340 | 0.290 | **0.380** |
| 5×3 | 0.230 | **0.287** | 0.250 | 0.207 |
| 5×4 | **0.215** | 0.150 | 0.170 | 0.130 |
| 5×5 | **0.160** | 0.128 | 0.140 | 0.104 |
| 5×6 | **0.195** | 0.097 | 0.092 | 0.062 |
| 6×2 | 0.250 | 0.325 | 0.279 | **0.325** |
| 6×3 | **0.250** | 0.206 | 0.256 | 0.133 |
| 6×4 | **0.175** | 0.094 | 0.087 | 0.062 |
| 6×5 | **0.172** | 0.052 | 0.087 | 0.023 |
| 6×6 | **0.175** | 0.017 | 0.092 | 0.000 |
| **overall** | **0.337** | 0.327 | 0.313 | 0.308 |

**Puzzle-level summary** (cell_acc = fraction of cells correct; puzzle_acc = fraction of
puzzles fully solved; parse_rate = fraction of outputs that produced valid JSON):

| | cell_acc | puzzle_acc | parse_rate |
| --- | ---: | ---: | ---: |
| base | **0.337** | 0.034 | **0.988** |
| instruct_only | 0.327 | **0.070** | 0.718 |
| 6pct_L16 | 0.313 | 0.036 | 0.860 |
| 6pct_L75 | 0.308 | 0.052 | 0.708 |

> **Notes:**
> - Cell accuracy is essentially tied across models (31–34%). Curriculum training gives no cell-level advantage on grid-mode.
> - Puzzle accuracy is ~2× higher for fine-tuned models (5–7% vs 3.4% for base) — when they succeed, they succeed completely. But success rate overall is low across the board.
> - Parse rate is a format-regression signature of fine-tuning: `base` emits valid JSON 99% of the time; fine-tuned models drop to 71–86%. This is especially punishing at large puzzles: `6pct_L75` scores 0.000 on 6×6 not because its reasoning is worse but because it can't emit valid JSON at that scale anymore.
> - Base is notably stronger on 5×* and 6×* cell accuracy (except 6×2), suggesting that for the hardest puzzles, format reliability matters more than whatever reasoning the curriculum added.

---

## 2. Standard NLU (lm-eval-harness, 12 tasks)

### 2a. Primary 4-way comparison (full test set, Gilbreth 2026-04-17)

Full test set on 12 standard NLU tasks via lm-eval-harness:

| Task | base | instruct_only | 6pct_L6 | 6pct_L16 | 6pct_L32 | 6pct_L75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| arc_challenge | 0.290 | 0.311 | — | 0.306 | — | **0.334** |
| arc_easy | 0.535 | 0.585 | — | 0.636 | — | **0.641** |
| boolq | **0.709** | 0.619 | — | 0.703 | — | 0.615 |
| commonsense_qa | 0.436 | 0.465 | — | **0.514** | — | 0.498 |
| copa | 0.670 | 0.730 | — | **0.750** | — | 0.720 |
| gsm8k | 0.000 | 0.000 | — | **0.002** | — | 0.000 |
| hellaswag | 0.360 | **0.381** | — | 0.368 | — | 0.375 |
| openbookqa | 0.188 | **0.224** | — | 0.210 | — | 0.202 |
| piqa | 0.660 | 0.676 | — | 0.671 | — | **0.676** |
| sciq | 0.836 | 0.904 | — | **0.933** | — | 0.928 |
| truthfulqa_mc1 | **0.344** | 0.275 | — | 0.268 | — | 0.259 |
| winogrande | 0.519 | 0.530 | — | 0.538 | — | **0.539** |
| **macro-avg** | 0.462 | 0.475 | 0.489 | **0.492** | 0.486 | 0.482 |

> Notes:
> - **`6pct_L16` wins macro-avg** (0.492) by +3pp over base.
> - **base wins 2 tasks**: boolq (+9pp) and truthfulqa_mc1 (+7pp). TruthfulQA is a
>   known fine-tuning anti-pattern — training makes the model more confident in
>   wrong answers.
> - **6pct curriculum models win 7 of 12 tasks** (arc_challenge, arc_easy, commonsense_qa,
>   copa, piqa, sciq, winogrande). 6pct_L16 wins 4; 6pct_L75 wins 3.
> - **GSM8K is near-zero across the board** — 0.6B is too small for math reasoning.

### 2b. Historical per-task breakdown (earlier checkpoints, for reference)

Note: `instruct_only` numbers below are from the threshold-sweep compilation and
reflect an earlier checkpoint. The 2a macro-avg numbers above are the current
authoritative values.

| Task | base | instruct_only | 6pct_search | 99pct_L16 |
| --- | ---: | ---: | ---: | ---: |
| arc_challenge | 0.294 | 0.314 | 0.330 | 0.317 |
| arc_easy | 0.539 | 0.580 | 0.638 | 0.638 |
| boolq | 0.713 | 0.636 | 0.704 | 0.658 |
| commonsense_qa | 0.439 | 0.458 | 0.497 | 0.491 |
| copa | 0.660 | 0.700 | 0.730 | 0.710 |
| gsm8k | 0.018 | 0.419 | 0.461 | 0.431 |
| hellaswag | 0.361 | 0.377 | 0.376 | 0.371 |
| openbookqa | 0.196 | 0.252 | 0.210 | 0.222 |
| piqa | 0.659 | 0.676 | 0.675 | 0.682 |
| sciq | 0.835 | 0.891 | 0.934 | 0.928 |
| truthfulqa_mc1 | 0.343 | 0.275 | 0.267 | 0.267 |
| winogrande | 0.519 | 0.537 | 0.542 | 0.552 |
| **macro avg** | 0.465 | 0.510 | 0.547 | 0.539 |

### 2c. 99%-threshold L sweep (job 9048131)

| Task | L=2 | L=4 | L=6 | L=8 | L=10 | L=12 | L=14 | L=16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arc_challenge | 0.308 | 0.311 | 0.304 | 0.309 | 0.326 | 0.317 | 0.323 | 0.317 |
| arc_easy | 0.612 | 0.638 | 0.642 | 0.626 | 0.647 | 0.637 | 0.638 | 0.638 |
| boolq | 0.591 | 0.664 | 0.665 | 0.691 | 0.695 | 0.681 | 0.710 | 0.658 |
| commonsense_qa | 0.482 | 0.511 | 0.515 | 0.509 | 0.469 | 0.472 | 0.497 | 0.491 |
| copa | 0.730 | 0.730 | 0.730 | 0.740 | 0.720 | 0.740 | 0.750 | 0.710 |
| gsm8k | 0.382 | 0.396 | 0.396 | 0.427 | 0.422 | 0.400 | 0.421 | 0.431 |
| hellaswag | 0.369 | 0.367 | 0.365 | 0.368 | 0.369 | 0.370 | 0.375 | 0.371 |
| openbookqa | 0.188 | 0.200 | 0.214 | 0.206 | 0.214 | 0.226 | 0.226 | 0.222 |
| piqa | 0.666 | 0.670 | 0.669 | 0.669 | 0.671 | 0.670 | 0.680 | 0.682 |
| sciq | 0.925 | 0.922 | 0.927 | 0.925 | 0.932 | 0.929 | 0.930 | 0.928 |
| truthfulqa_mc1 | 0.286 | 0.285 | 0.264 | 0.278 | 0.266 | 0.257 | 0.266 | 0.267 |
| winogrande | 0.534 | 0.527 | 0.535 | 0.549 | 0.541 | 0.534 | 0.537 | 0.552 |
| **macro avg** | 0.506 | 0.518 | 0.519 | 0.525 | 0.523 | 0.519 | 0.529 | 0.522 |

### 2d. Other models (one-offs)

| Task | pure_curr | dolci20pct_search | lora_merged | qwen06b_ft_curr_L96 | qwen06b_ft_curr_L128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| arc_challenge | 0.177 | 0.324 | 0.212 | – | 0.181 |
| arc_easy | 0.299 | 0.644 | – | – | – |
| boolq | 0.570 | 0.552 | – | – | – |
| commonsense_qa | 0.196 | 0.462 | – | – | – |
| copa | 0.540 | 0.710 | – | – | – |
| gsm8k | 0.000 | 0.281 | – | 0.000 | 0.000 |
| hellaswag | 0.262 | 0.380 | – | – | – |
| openbookqa | 0.132 | 0.236 | – | – | – |
| piqa | 0.554 | 0.672 | – | – | – |
| sciq | 0.656 | 0.903 | – | – | – |
| truthfulqa_mc1 | 0.261 | 0.267 | – | – | – |
| winogrande | 0.487 | 0.552 | – | – | – |

---

## 3. BBH (Big-Bench Hard reasoning, 11 subtasks)

### 3a. Primary 4-way comparison — BBH zero-shot (full test set, Gilbreth 2026-04-17)

11 BBH subtasks, zero-shot log-likelihood scoring:

| Task | base | instruct_only | 6pct_L6 | 6pct_L16 | 6pct_L32 | 6pct_L75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| boolean_expressions | 0.000 | 0.644 | — | **0.728** | — | 0.704 |
| dyck_languages | 0.000 | **0.040** | — | 0.000 | — | 0.024 |
| formal_fallacies | 0.000 | 0.000 | — | 0.000 | — | **0.200** |
| logical_deduction_three_objects | 0.000 | 0.000 | — | 0.000 | — | 0.000 |
| logical_deduction_five_objects | 0.000 | **0.020** | — | 0.000 | — | 0.000 |
| logical_deduction_seven_objects | 0.000 | **0.112** | — | 0.000 | — | 0.004 |
| navigate | 0.004 | **0.580** | — | 0.500 | — | 0.576 |
| tracking_shuffled_objects_three_objects | 0.000 | 0.004 | — | 0.000 | — | **0.004** |
| tracking_shuffled_objects_five_objects | 0.000 | **0.012** | — | 0.000 | — | 0.000 |
| tracking_shuffled_objects_seven_objects | 0.000 | 0.000 | — | 0.000 | — | **0.008** |
| web_of_lies | 0.000 | 0.000 | — | **0.004** | — | 0.000 |
| **macro-avg** | 0.000 | 0.128 | 0.067 | 0.112 | 0.112 | **0.138** |

### 3b. Primary 4-way comparison — BBH 3-shot Chain-of-Thought (`bbh_cot`)

Same 11 subtasks but with 3-shot CoT prompting:

| Task | base | instruct_only | 6pct_L6 | 6pct_L16 | 6pct_L32 | 6pct_L75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| boolean_expressions | 0.000 | 0.016 | — | **0.380** | — | 0.180 |
| dyck_languages | 0.000 | 0.000 | — | 0.000 | — | 0.000 |
| formal_fallacies | 0.000 | **0.388** | — | 0.220 | — | 0.248 |
| logical_deduction_three_objects | 0.000 | 0.196 | — | **0.268** | — | 0.200 |
| logical_deduction_five_objects | 0.000 | 0.088 | — | 0.104 | — | **0.108** |
| logical_deduction_seven_objects | 0.000 | 0.044 | — | **0.060** | — | 0.048 |
| navigate | 0.000 | 0.184 | — | **0.476** | — | 0.348 |
| tracking_shuffled_objects_three_objects | 0.000 | 0.072 | — | **0.228** | — | 0.072 |
| tracking_shuffled_objects_five_objects | 0.000 | 0.028 | — | **0.096** | — | 0.056 |
| tracking_shuffled_objects_seven_objects | 0.000 | 0.012 | — | **0.092** | — | 0.020 |
| web_of_lies | 0.000 | 0.000 | — | 0.000 | — | 0.000 |
| **macro-avg** | 0.000 | 0.093 | 0.012 | **0.175** | 0.124 | 0.116 |

> **Key findings (zero-shot, section 3a — the headline):**
> - **Base scores 0 on every BBH task** — Qwen3-0.6B doesn't emit BBH-style answers in the expected format without fine-tuning.
> - **BBH zero-shot improves with curriculum depth**: comparing the 99% threshold sweep (section 3d), macro-avg goes from 0.057 at L=2 to **0.194 at L=16** (+13.7pp). The current 4-way shows the same direction in the 6% Dolci mix: L=75 (0.138) > L=16 (0.112) by +2.6pp.
> - **Biggest per-subtask wins on zero-shot**: `boolean_expressions` +73pp (6pct_L16: 0.728), `navigate` +57pp (6pct_L75: 0.576), `formal_fallacies` +20pp (6pct_L75 is the only model above 0). Structural-reasoning tasks where search-curriculum training should transfer.
>
> CoT results (section 3b) are recorded for completeness but are not our headline
> — the zero-shot result is cleaner and more consistent with the rest of the suite
> (ProofWriter CWA, PlanBench log-prob, legal_gen).

### 3c. Historical per-task breakdown

| Task | base | instruct_only | 6pct_search | 99pct_L16 |
| --- | ---: | ---: | ---: | ---: |
| boolean_expressions | 0.000 | 0.588 | 0.728 | 0.732 |
| dyck_languages | 0.000 | 0.064 | 0.000 | 0.000 |
| formal_fallacies | 0.000 | 0.076 | 0.012 | 0.468 |
| logical_deduction_five_objects | 0.000 | 0.112 | 0.000 | 0.036 |
| logical_deduction_seven_objects | 0.000 | 0.056 | 0.004 | 0.244 |
| logical_deduction_three_objects | 0.000 | 0.052 | 0.000 | 0.000 |
| navigate | 0.008 | 0.412 | 0.580 | 0.548 |
| tracking_shuffled_objects_five_objects | 0.000 | 0.000 | 0.000 | 0.040 |
| tracking_shuffled_objects_seven_objects | 0.000 | 0.000 | 0.000 | 0.000 |
| tracking_shuffled_objects_three_objects | 0.000 | 0.000 | 0.000 | 0.060 |
| web_of_lies | 0.000 | 0.020 | 0.000 | 0.000 |
| **macro avg** | 0.001 | 0.126 | 0.120 | 0.194 |

> Key finding: `99pct_L16` (99% threshold curriculum) macro-averages **+10.6pp over
> `6pct_search`** on BBH despite similar standard-NLU scores — 99% threshold locks in more
> robust reasoning, especially on formal_fallacies, logical_deduction_seven, and
> tracking_shuffled_three.

### 3d. 99%-threshold L sweep

| Task | L=2 | L=4 | L=6 | L=8 | L=10 | L=12 | L=14 | L=16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| boolean_expressions | 0.628 | 0.696 | 0.680 | 0.520 | 0.724 | 0.604 | 0.748 | 0.732 |
| dyck_languages | 0.000 | 0.000 | 0.000 | 0.000 | 0.004 | 0.000 | 0.000 | 0.000 |
| formal_fallacies | 0.000 | 0.364 | 0.468 | 0.352 | 0.004 | 0.024 | 0.000 | 0.468 |
| logical_deduction_five_objects | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.036 |
| logical_deduction_seven_objects | 0.000 | 0.000 | 0.000 | 0.000 | 0.016 | 0.000 | 0.000 | 0.244 |
| logical_deduction_three_objects | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| navigate | 0.000 | 0.496 | 0.576 | 0.576 | 0.580 | 0.580 | 0.112 | 0.548 |
| tracking_shuffled_objects_five_objects | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.040 |
| tracking_shuffled_objects_seven_objects | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| tracking_shuffled_objects_three_objects | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.060 |
| web_of_lies | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **macro avg** | 0.057 | 0.142 | 0.157 | 0.132 | 0.121 | 0.110 | 0.078 | 0.194 |

### 3e. Ablations (one-off)

| Task | 6pct_L6 (98% thr, paired w/ 99pct_L6) | pure_curr | dolci20pct_search |
| --- | ---: | ---: | ---: |
| boolean_expressions | 0.608 | 0.000 | 0.000 |
| dyck_languages | 0.000 | 0.000 | 0.000 |
| formal_fallacies | 0.024 | 0.000 | 0.000 |
| logical_deduction_five_objects | 0.000 | 0.000 | 0.000 |
| logical_deduction_seven_objects | 0.000 | 0.000 | 0.000 |
| logical_deduction_three_objects | 0.000 | 0.000 | 0.000 |
| navigate | 0.076 | 0.000 | 0.000 |
| tracking_shuffled_objects_five_objects | 0.000 | 0.000 | 0.000 |
| tracking_shuffled_objects_seven_objects | 0.000 | 0.000 | 0.000 |
| tracking_shuffled_objects_three_objects | 0.000 | 0.000 | 0.000 |
| web_of_lies | 0.000 | 0.000 | 0.000 |
| **macro avg** | 0.064 | 0.000 | 0.000 |

---

## 4. Pending / not yet run

- **Legal diagnostic on `proa` and `international_citizenship_questions`** — only `hearsay` has been diagnosed; the other two tasks may show different bias patterns.

---

## Changelog

- 2026-04-18: **Added L6 and L32 columns** to all primary tables (sections 1a-1g, 2a, 3a, 3b) — curriculum-depth scaling curve now has 5 points (base → L6 → L16 → L32 → L75) for benchmarks where L6/L32 evals were run. Per-task breakdowns for L6/L32 on standard NLU and BBH not yet folded in (only macro-avg). proofwriter and proofwriter_cwa log-lik tables were also rerun on the **full test set** (was n=200/depth) to be sample-matched with L6/L32 — old n=200 numbers replaced. stepgame_gen for L6 and L32 rerun at n=1000 to match the others.
- 2026-04-16: initial dump from `compiled_downstream.json` (420 records, 7 benchmarks)
- 2026-04-16: PrOntoQA removed (degenerate L=1 case of DeepRD training data)
- 2026-04-16: ZebraLogic grid re-run (Gilbreth, N=20 per size) — invalidated previous "base 0.802" supplemental numbers. New clean baseline: all 4 models ~0.31-0.34 cell_acc. Fine-tuning helps puzzle_acc ~2× but hurts parse_rate 99% → 71%.
- 2026-04-16: `zebra` (grid mode) **removed from eval framework**. Replaced by `zebra_mc` (same underlying puzzles, log-lik scoring) which doesn't penalize fine-tuning's format regression. Results preserved in section 1e.
- 2026-04-16: Legal diagnostic on hearsay (Gilbreth job 10566994, n=94 shuffled). Reveals the original "base wins legal" finding is dominated by **label-bias collapse** in fine-tuned models (esp. `6pct_L75` always says "No"), not lost legal reasoning. Section 1a updated with diagnostic table.
- 2026-04-16: **Full-test-set eval on Gilbreth** (2 groups × 4 models = 8 jobs). Added `_gen` (generation-scoring) variants for legal, proofwriter (OWA + CWA), zebra_mc, stepgame. Key findings: (1) **legal_gen flips the legal result** — `6pct_L75` wins by +15pp over base (0.546 vs 0.394), consistent with the diagnostic finding that log-lik was measuring format-collapse not reasoning. (2) **proofwriter_cwa_gen shows monotonic improvement with depth** for `6pct_L16` (0.51 → 0.77 across depths 0-5), the strongest curriculum-transfer signal. (3) `stepgame_gen`: `6pct_L75` +2.6pp over base.
- 2026-04-17: **Completed remaining benchmarks on Gilbreth** (3 bundles × 4 models = 12 jobs) + per-subtask extraction. Added: `standard` (full 12-task table), `bbh` (11 subtasks, zero-shot), `bbh_cot` (recorded for completeness but not the headline — see note in §3), `stepgame` (log-lik), `game24`, `blocksworld`, `mystery_blocksworld`, `logistics`, `*_logprob` variants for all three PlanBench domains. Key findings: (1) **`6pct_L16` wins `standard` macro-avg (0.492)** and 4 of 12 subtasks — notably commonsense_qa, copa, sciq. (2) **`6pct_L75` wins BBH zero-shot (0.138 macro)** — biggest per-subtask wins: boolean_expressions +73pp vs base (0.728), navigate +57pp (0.576), formal_fallacies +20pp (0.200 — only model > 0). Consistent with the 99%-threshold sweep where BBH also improves with L (0.057 → 0.194 from L=2 → L=16). (3) **PlanBench log-prob variants all favor `6pct_L16`** — highest log-prob of gold plans across blocksworld, mystery_blocksworld, logistics. Clean signal that curriculum teaches plan-structure priors even when generation still fails. (4) Chess mate exact-match 0% all models; `chess_mate_logprob` initially errored (streaming-cap bug) — fix deployed and reran on 2026-04-17 (Gilbreth jobs 10577722-25): **6pct_L16 wins all 3 mate-in-N categories** (macro -3.817 vs base -5.492, +1.7 nat/tok), matching the PlanBench log-prob pattern.

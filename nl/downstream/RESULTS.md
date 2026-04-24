# Downstream Eval Results

Hand-compiled summary of `results/eval_n1000_*.json` outputs. Last major
refactor: 2026-04-24 (calibration, fused scorer, consolidated 14 benchmarks,
added 5 few-shot variants).

**This file will be regenerated after the current calibrated evals finish.**
The n=1000 results from 2026-04-23/24 — 5 Qwen 0.6B models (base, instruct_only,
6pct_L{8,16,32,48,64,75}) and 4 Qwen 1.7B models (base, 6pct_L{8,16,32}) —
are in the `results/eval_n1000_*.json` files but predate the calibration
implementation. Calibrated re-runs are in progress (jobs 9522434–9522441);
1.7B calibrated will follow when queue opens.

## Models evaluated

| alias | checkpoint |
|---|---|
| `base` | `Qwen/Qwen3-0.6B` (no fine-tuning) |
| `instruct_only` | `job_9152198/checkpoint-87000` — 100% Dolci-Instruct-SFT, no curriculum |
| `6pct_L{8,16,32,48}` | `job_8894380/stage_checkpoints/stage_L_*` — 6% Dolci mix + linear curriculum (L=1→96, step=1) |
| `6pct_L{64,75}` | `job_9001346/stage_checkpoints/stage_L_*` — continuation of same curriculum |
| `base_1.7b` | `Qwen/Qwen3-1.7B` |
| `6pct_1.7b_L{8,16,32}` | `job_runpod_qwen17b_L32_*/stage_checkpoints/stage_L_*` — same curriculum protocol at 1.7B |

## Benchmark methods summary

See `README.md` for full per-benchmark protocol table. Three metrics reported per fused benchmark:
- **log-lik**: `log p(c | prompt)` argmax over candidate labels (lm-eval-harness convention)
- **calibrated**: `log p(c | prompt) − log p(c | content-free)` — Zhao et al. ICML 2021
- **gen**: constrained greedy (argmax over full vocab, mapped to candidates)

Gen-canonical benchmarks (`nlgraph_gen`, `legal`) use their authors' own scorers.

## Placeholder — tables will go here

### Qwen 0.6B scaling (base → instruct_only → 6pct_L8 → ... → L75)

TBD after calibrated evals land.

### Qwen 1.7B scaling (base → 6pct_L8 → L16 → L32)

TBD after calibrated 1.7B evals land.

### 0-shot vs few-shot comparison (on the 5 `_fs` variants)

TBD.

### Per-class breakdowns (where class collapse is revealed by calibration)

Key benchmarks: ProofWriter OWA (Unknown collapse), ProofWriter CWA (False collapse),
LogicBench MCQA (A-bias), LogiQA (A-bias), Multi-LogiEval (Yes-majority).

TBD.

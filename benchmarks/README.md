# Benchmark Workspace

This workspace compares:

- `Qwen/Qwen3-0.6B`
- `JimmyJamJr/qwen06b-search-curr-L128`

Phase-1 benchmark bundle:

- `IFEval`
- `IFBench`
- `BigBenchHard`
- `AGI Eval English`

## Layout

- `common/`: shared configs, orchestration, and result aggregation
- `ifeval/`: benchmark-specific env spec, docs, and wrapper
- `ifbench/`: benchmark-specific env spec, docs, and wrapper
- `bbh/`: benchmark-specific env spec, docs, and wrapper
- `agieval_en/`: benchmark-specific env spec, docs, and wrapper

## Quick Start

Create the benchmark env you want to run:

```bash
conda env create -f benchmarks/ifeval/environment.yml
conda activate qwen-ifeval
```

Run a smoke test on one benchmark:

```bash
python benchmarks/ifeval/run_compare.py --gpus 0,1 --limit 5
```

Run the full phase-1 bundle:

```bash
python benchmarks/common/run_all.py --gpus 0,1,2,3,4,5,6,7
```

Per-benchmark outputs land under `benchmarks/<benchmark>/results/`. A cross-benchmark summary is written to `benchmarks/results/summary.md`.

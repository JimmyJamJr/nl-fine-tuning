# BigBenchHard

This benchmark uses `lm-evaluation-harness` with the `vllm` backend and the `bbh` task group.

## Environment

```bash
conda env create -f benchmarks/bbh/environment.yml
conda activate qwen-bbh
```

## Smoke Test

```bash
python benchmarks/bbh/run_compare.py --gpus 0,1 --limit 5
```

## Full Run

```bash
python benchmarks/bbh/run_compare.py --gpus 0,1
```

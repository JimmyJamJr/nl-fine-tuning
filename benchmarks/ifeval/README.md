# IFEval

This benchmark uses `lm-evaluation-harness` with the `vllm` backend.

## Environment

```bash
conda env create -f benchmarks/ifeval/environment.yml
conda activate qwen-ifeval
```

## Smoke Test

```bash
python benchmarks/ifeval/run_compare.py --gpus 0,1 --limit 5
```

## Full Run

```bash
python benchmarks/ifeval/run_compare.py --gpus 0,1
```

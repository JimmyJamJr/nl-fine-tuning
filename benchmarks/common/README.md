# Shared Benchmark Utilities

Key entrypoints:

- `benchmark_config.py`: model ids and benchmark shortlist
- `compare_benchmark.py`: run one benchmark for both models and aggregate the result
- `run_all.py`: launch all four benchmark comparisons across available GPU pairs
- `lm_eval_runner.py`: shared `lm-evaluation-harness` + `vllm` runner for `IFEval`, `BigBenchHard`, and `AGI Eval English`
- `ifbench_runner.py`: `vllm` generation plus official `IFBench` scoring

Base env specs:

- `base-lm-eval-environment.yml`
- `base-ifbench-environment.yml`

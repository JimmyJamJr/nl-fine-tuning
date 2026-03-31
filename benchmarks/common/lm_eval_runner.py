from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.benchmark_config import BENCHMARKS_ROOT, get_benchmark, get_model
from benchmarks.common.io_utils import (
    ensure_dir,
    find_first_json_with_key,
    flatten_numeric_metrics,
    pick_primary_metric,
    read_json,
    utc_timestamp,
    write_json,
)


def _pythonpath_env() -> str:
    current = os.environ.get("PYTHONPATH", "")
    repo = str(BENCHMARKS_ROOT.parent)
    if not current:
        return repo
    if repo in current.split(os.pathsep):
        return current
    return os.pathsep.join([repo, current])


def _ld_library_path() -> str:
    entries = [str(Path(sys.prefix) / "lib"), "/usr/local/nvidia/lib64"]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if current:
        entries.append(current)
    return os.pathsep.join(entries)


def build_model_args(model_id: str, dtype: str, max_model_len: int, max_num_seqs: int) -> str:
    args = {
        "pretrained": model_id,
        "dtype": dtype,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "trust_remote_code": True,
    }
    return ",".join(f"{key}={value}" for key, value in args.items())


def build_lm_eval_command(
    benchmark_slug: str,
    model_alias: str,
    output_dir: Path,
    limit: int | None,
) -> list[str]:
    benchmark = get_benchmark(benchmark_slug)
    model = get_model(model_alias)
    runner = shutil.which("lm_eval")

    command = [runner] if runner else [sys.executable, "-m", "lm_eval"]
    command.extend(
        [
            "--model",
            "vllm",
            "--tasks",
            benchmark.task_name,
            "--model_args",
            build_model_args(
                model_id=model.model_id,
                dtype=model.dtype,
                max_model_len=model.max_model_len,
                max_num_seqs=model.max_num_seqs,
            ),
            "--batch_size",
            "auto",
            "--output_path",
            str(output_dir),
            "--log_samples",
            "--gen_kwargs",
            f"temperature=0,top_p=1,max_gen_toks={benchmark.max_new_tokens}",
        ]
    )
    if model.apply_chat_template:
        command.append("--apply_chat_template")
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def collect_metrics(benchmark_slug: str, output_dir: Path) -> dict:
    benchmark = get_benchmark(benchmark_slug)
    raw_results_path = find_first_json_with_key(output_dir, "results")
    if raw_results_path is None:
        raise FileNotFoundError(f"No lm-eval results JSON found in {output_dir}")

    payload = read_json(raw_results_path)
    results = payload.get("results", {})

    task_metrics = results.get(benchmark.task_name)
    if task_metrics is None and results:
        task_name, task_metrics = next(iter(results.items()))
    else:
        task_name = benchmark.task_name

    numeric_metrics = flatten_numeric_metrics(task_metrics)
    primary_metric_name, primary_metric_value = pick_primary_metric(
        numeric_metrics, benchmark.metric_preferences
    )

    summary = {
        "benchmark": benchmark.slug,
        "display_name": benchmark.display_name,
        "task_name": task_name,
        "raw_results_path": str(raw_results_path),
        "metrics": numeric_metrics,
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": primary_metric_value,
    }
    write_json(output_dir / "metrics.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lm-eval + vLLM for one benchmark/model pair.")
    parser.add_argument("--benchmark", required=True, choices=["ifeval", "bbh", "agieval_en"])
    parser.add_argument("--model-alias", required=True, choices=["base", "finetuned"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-id", required=True, help="Single GPU id to expose to this run.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    metadata = {
        "benchmark": args.benchmark,
        "model_alias": args.model_alias,
        "model_id": get_model(args.model_alias).model_id,
        "gpu_id": args.gpu_id,
        "started_at_utc": utc_timestamp(),
        "kind": "lm_eval",
    }
    write_json(output_dir / "run_metadata.json", metadata)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    env["PYTHONPATH"] = _pythonpath_env()
    env["PATH"] = os.pathsep.join(["/usr/local/nvidia/bin", env.get("PATH", "")])
    env["LD_LIBRARY_PATH"] = _ld_library_path()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    command = build_lm_eval_command(args.benchmark, args.model_alias, output_dir, args.limit)
    log_path = output_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("Command:\n")
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        subprocess.run(command, check=True, env=env, stdout=handle, stderr=subprocess.STDOUT)

    summary = collect_metrics(args.benchmark, output_dir)
    metadata["completed_at_utc"] = utc_timestamp()
    metadata["primary_metric_name"] = summary["primary_metric_name"]
    metadata["primary_metric_value"] = summary["primary_metric_value"]
    write_json(output_dir / "run_metadata.json", metadata)


if __name__ == "__main__":
    main()

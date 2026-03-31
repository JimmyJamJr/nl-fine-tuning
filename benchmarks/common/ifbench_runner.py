from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.benchmark_config import BENCHMARKS_ROOT, get_benchmark, get_model
from benchmarks.common.io_utils import ensure_dir, read_json, utc_timestamp, write_json


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


def ensure_ifbench_repo(repo_dir: Path) -> None:
    if repo_dir.exists():
        return
    ensure_dir(repo_dir.parent)
    subprocess.run(
        ["git", "clone", "https://github.com/allenai/IFBench.git", str(repo_dir)],
        check=True,
    )


def load_examples(limit: int | None) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("allenai/IFBench_test", split="train")
    examples = [dict(row) for row in dataset]
    if limit is not None:
        examples = examples[:limit]
    return examples


def write_examples(path: Path, examples: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(row) + "\n")


def generate_responses(
    model_alias: str,
    output_dir: Path,
    gpu_id: str,
    limit: int | None,
) -> tuple[Path, Path]:
    from vllm import LLM, SamplingParams

    benchmark = get_benchmark("ifbench")
    model = get_model(model_alias)
    examples = load_examples(limit)

    dataset_path = output_dir / "ifbench_test.jsonl"
    responses_path = output_dir / f"{model_alias}-responses.jsonl"
    write_examples(dataset_path, examples)

    llm = LLM(
        model=model.model_id,
        trust_remote_code=True,
        dtype=model.dtype,
        tensor_parallel_size=1,
        max_model_len=model.max_model_len,
        max_num_seqs=model.max_num_seqs,
        gpu_memory_utilization=model.gpu_memory_utilization,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=benchmark.max_new_tokens,
    )
    prompts = [row["prompt"] for row in examples]
    generations = llm.generate(prompts, sampling)

    with responses_path.open("w", encoding="utf-8") as handle:
        for example, generation in zip(examples, generations, strict=True):
            text = generation.outputs[0].text if generation.outputs else ""
            handle.write(
                json.dumps({"prompt": example["prompt"], "response": text}, ensure_ascii=True) + "\n"
            )

    metadata = {
        "benchmark": benchmark.slug,
        "model_alias": model_alias,
        "model_id": model.model_id,
        "gpu_id": gpu_id,
        "completed_at_utc": utc_timestamp(),
        "example_count": len(examples),
    }
    write_json(output_dir / "generation_metadata.json", metadata)
    return dataset_path, responses_path


def score_responses(repo_dir: Path, dataset_path: Path, responses_path: Path, output_dir: Path) -> dict:
    ensure_ifbench_repo(repo_dir)
    eval_output_dir = ensure_dir(output_dir / "eval")

    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath_env()
    env["PATH"] = os.pathsep.join(["/usr/local/nvidia/bin", env.get("PATH", "")])
    env["LD_LIBRARY_PATH"] = _ld_library_path()

    command = [
        sys.executable,
        "-m",
        "run_eval",
        f"--input_data={dataset_path}",
        f"--input_response_data={responses_path}",
        f"--output_dir={eval_output_dir}",
    ]
    with (output_dir / "score.log").open("w", encoding="utf-8") as handle:
        handle.write("Command:\n")
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        subprocess.run(
            command,
            check=True,
            cwd=repo_dir,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )

    input_basename = responses_path.name
    if "-responses.jsonl" in input_basename:
        model_name = input_basename.replace("-responses.jsonl", "")
    else:
        model_name = responses_path.stem

    loose_path = eval_output_dir / f"{model_name}-eval_results_loose.jsonl"
    strict_path = eval_output_dir / f"{model_name}-eval_results_strict.jsonl"
    return summarize_scores(loose_path, strict_path, output_dir)


def summarize_scores(loose_path: Path, strict_path: Path, output_dir: Path) -> dict:
    def read_rows(path: Path) -> list[dict]:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))
        return rows

    loose_rows = read_rows(loose_path)
    strict_rows = read_rows(strict_path)

    prompt_loose = sum(1 for row in loose_rows if row["follow_all_instructions"]) / len(loose_rows)
    prompt_strict = sum(1 for row in strict_rows if row["follow_all_instructions"]) / len(strict_rows)

    loose_hits = sum(sum(1 for item in row["follow_instruction_list"] if item) for row in loose_rows)
    strict_hits = sum(sum(1 for item in row["follow_instruction_list"] if item) for row in strict_rows)
    instruction_total = sum(len(row["follow_instruction_list"]) for row in loose_rows)

    summary = {
        "benchmark": "ifbench",
        "display_name": get_benchmark("ifbench").display_name,
        "metrics": {
            "prompt_level_loose_accuracy": prompt_loose,
            "instruction_level_loose_accuracy": loose_hits / instruction_total,
            "prompt_level_strict_accuracy": prompt_strict,
            "instruction_level_strict_accuracy": strict_hits / instruction_total,
        },
        "primary_metric_name": "prompt_level_loose_accuracy",
        "primary_metric_value": prompt_loose,
        "raw_loose_results_path": str(loose_path),
        "raw_strict_results_path": str(strict_path),
    }
    write_json(output_dir / "metrics.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IFBench generation and scoring for one model.")
    parser.add_argument("--model-alias", required=True, choices=["base", "finetuned"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--ifbench-repo",
        default=str(BENCHMARKS_ROOT / "ifbench" / "vendor" / "IFBench"),
    )
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    write_json(
        output_dir / "run_metadata.json",
        {
            "benchmark": "ifbench",
            "kind": "ifbench",
            "model_alias": args.model_alias,
            "model_id": get_model(args.model_alias).model_id,
            "gpu_id": args.gpu_id,
            "started_at_utc": utc_timestamp(),
        },
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["PYTHONPATH"] = _pythonpath_env()
    os.environ["PATH"] = os.pathsep.join(["/usr/local/nvidia/bin", os.environ.get("PATH", "")])
    os.environ["LD_LIBRARY_PATH"] = _ld_library_path()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dataset_path, responses_path = generate_responses(
        model_alias=args.model_alias,
        output_dir=output_dir,
        gpu_id=args.gpu_id,
        limit=args.limit,
    )
    summary = score_responses(Path(args.ifbench_repo), dataset_path, responses_path, output_dir)

    metadata = read_json(output_dir / "run_metadata.json")
    metadata["completed_at_utc"] = utc_timestamp()
    metadata["primary_metric_name"] = summary["primary_metric_name"]
    metadata["primary_metric_value"] = summary["primary_metric_value"]
    write_json(output_dir / "run_metadata.json", metadata)


if __name__ == "__main__":
    main()

"""One-shot script to pre-cache all downstream eval datasets into $SCRATCH.

Triggers HF download of every dataset eval_downstream.py uses so subsequent
SLURM runs hit the local cache (no streaming, no rate limits). For
Lichess/chess-puzzles (streaming-only by default, 4M+ rows), filters to
mate-in-1/2/3 puzzles and saves the filtered subset as a local Parquet file
under $DATA_DIR/chess_puzzles/mate_puzzles.parquet. eval_chess_mate will
auto-load this file if present (see eval_downstream.py).

Usage:
    export HF_HOME=/scratch/gautschi/$USER/model_cache
    export DATA_DIR=/scratch/gautschi/$USER/nl_eval
    python setup_cache_datasets.py
"""
import os, sys, json, time

HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
DATA_DIR = os.environ.get("DATA_DIR", "/scratch/gautschi/huan2073/nl_eval")
os.environ["HF_HOME"] = HF_HOME
print(f"HF_HOME:  {HF_HOME}")
print(f"DATA_DIR: {DATA_DIR}")
os.makedirs(DATA_DIR, exist_ok=True)

# Propagate HF token if cached
token_path = os.path.expanduser("~/.cache/huggingface/token")
if os.path.isfile(token_path) and "HF_TOKEN" not in os.environ:
    with open(token_path) as f:
        os.environ["HF_TOKEN"] = f.read().strip()
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    print("Propagated HF_TOKEN from ~/.cache/huggingface/token")

from datasets import load_dataset

# (repo_id, config, split) — all non-streaming HF datasets we use
HF_DATASETS = [
    ("yale-nlp/FOLIO",                None,          "validation"),
    ("yale-nlp/FOLIO",                None,          "train"),
    ("hails/agieval-logiqa-en",       None,          "test"),
    ("kendrivp/CLUTRR_v1_extracted",  None,          "test"),
    ("ZhengyanShi/StepGame",          None,          "test"),
    ("ZhengyanShi/StepGame",          None,          "train"),
    ("WildEval/ZebraLogic",           "mc_mode",     "test"),
    ("tasksource/nlgraph",            None,          "test"),
    ("hitachi-nlp/ruletaker",         None,          "test"),
    ("Rowan/hellaswag",               None,          "validation"),
]

def fetch(repo, cfg, split):
    key = f"{repo}" + (f"/{cfg}" if cfg else "") + f":{split}"
    print(f"[fetch] {key} ...", flush=True)
    t0 = time.time()
    try:
        if cfg:
            ds = load_dataset(repo, cfg, split=split)
        else:
            ds = load_dataset(repo, split=split)
        print(f"  ok: {len(ds)} rows ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")

for repo, cfg, split in HF_DATASETS:
    fetch(repo, cfg, split)

# LegalBench has per-task configs; warm just the 8 we actually use.
print("\n[legalbench per-task]")
LEGAL_TASKS = ["diversity_1", "diversity_2", "diversity_3", "diversity_4",
               "diversity_5", "diversity_6", "sara_entailment", "sara_numeric"]
for task in LEGAL_TASKS:
    fetch("nguha/legalbench", task, "test")

# Lichess chess-puzzles — streaming-only mirror. Filter to mate-in-1/2/3
# puzzles and cache locally as Parquet.
chess_dir = os.path.join(DATA_DIR, "chess_puzzles")
chess_path = os.path.join(chess_dir, "mate_puzzles.parquet")
if os.path.isfile(chess_path):
    print(f"\n[lichess] already cached at {chess_path}; skipping")
else:
    print(f"\n[lichess] streaming + filtering to mate-in-1/2/3 puzzles ...")
    os.makedirs(chess_dir, exist_ok=True)
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    targets = {"mateIn1": [], "mateIn2": [], "mateIn3": []}
    want = 2000  # 2k of each category -> 6k total, covers n=1000 per-category easily
    import ast as _ast
    def _themes(t):
        if isinstance(t, list): return t
        if isinstance(t, str):
            try:
                v = _ast.literal_eval(t)
                if isinstance(v, list): return v
            except Exception: pass
            return t.split()
        return []
    scanned = 0
    for ex in ds:
        scanned += 1
        if scanned % 20000 == 0:
            print(f"  scanned {scanned}; have " +
                  ", ".join(f"{k}={len(v)}" for k, v in targets.items()),
                  flush=True)
        th = _themes(ex.get("Themes", []))
        for key in list(targets.keys()):
            if key in th and len(targets[key]) < want:
                # Keep only fields we use
                targets[key].append({
                    "PuzzleId": ex.get("PuzzleId"),
                    "FEN": ex.get("FEN"),
                    "Moves": ex.get("Moves"),
                    "Rating": ex.get("Rating"),
                    "Themes": th,
                    "category": key,
                })
        if all(len(v) >= want for v in targets.values()):
            break
    rows = []
    for lst in targets.values():
        rows.extend(lst)
    print(f"  collected {len(rows)} puzzles ({scanned} scanned)")
    import pandas as pd
    pd.DataFrame(rows).to_parquet(chess_path, index=False)
    print(f"  wrote {chess_path}")

print("\n[done]")

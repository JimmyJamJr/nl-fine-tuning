#!/usr/bin/env python3
"""Unified downstream eval: select models + benchmarks via CLI.

Usage:
  python eval_downstream.py \
      --models base instruct_only 6pct_L16 6pct_L75 \
      --benchmarks zebra_mc legal \
      --n 100 \
      --output results/zebra_mc_legal.json

Registries below map short names → checkpoint paths and benchmark runners.
Edit MODEL_REGISTRY to add models; register a new benchmark with @register.
"""

import argparse
import glob
import json
import os
import sys
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# All external paths are configured from CLI args (see main() --hf-cache,
# --prompts-dir, --checkpoints-root, --data-dir) and stashed into these globals.
# HF_HOME env var still works for cache_dir (transformers reads it directly).
cache_dir = os.environ.get("HF_HOME")  # may be None; set by --hf-cache in main()
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
CHECKPOINTS_ROOT = None  # required iff any MODEL_REGISTRY alias is used
DATA_DIR = None  # required iff game24 / proofwriter benchmarks are requested


def _load_prompt(rel_path: str) -> str:
    """Load a prompt template from PROMPTS_DIR (--prompts-dir)."""
    p = os.path.join(PROMPTS_DIR, rel_path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Prompt '{rel_path}' not found under {PROMPTS_DIR}")
    with open(p) as f:
        return f.read()


# Set USE_TRAIN_ICES=1 in the environment to use K in-context examples drawn
# from the dataset's train split instead of the hand-crafted prompt files for
# ProofWriter / StepGame. Defaults to 0 (use the static prompt files).
_USE_TRAIN_ICES = os.environ.get("USE_TRAIN_ICES", "0") not in ("0", "", "false", "False")
_TRAIN_ICE_K = int(os.environ.get("TRAIN_ICE_K", "5"))
_train_ice_cache = {}  # benchmark_name -> cached few-shot prompt string


# ==========================================================
# Model registry
# ==========================================================

def _latest_ckpt(job_id):
    if CHECKPOINTS_ROOT is None:
        raise RuntimeError(
            "--checkpoints-root is required to resolve MODEL_REGISTRY aliases "
            "(e.g. 'instruct_only', '6pct_L16'). Pass a raw path or set it."
        )
    paths = sorted(
        glob.glob(os.path.join(CHECKPOINTS_ROOT, f"job_{job_id}", "checkpoint-*")),
        key=lambda x: int(x.rsplit("-", 1)[-1]),
    )
    return paths[-1] if paths else None


def _stage_ckpt(job_id, L):
    if CHECKPOINTS_ROOT is None:
        raise RuntimeError(
            "--checkpoints-root is required to resolve MODEL_REGISTRY aliases "
            "(e.g. 'instruct_only', '6pct_L16'). Pass a raw path or set it."
        )
    matches = glob.glob(
        os.path.join(CHECKPOINTS_ROOT, f"job_{job_id}", "stage_checkpoints", f"stage_{L}_*_L{L}")
    )
    return matches[0] if matches else None


MODEL_REGISTRY = {
    "base": "Qwen/Qwen3-0.6B",
    "instruct_only": lambda: _latest_ckpt("9152198") or _latest_ckpt("9052803") or _latest_ckpt("8969184"),
    "6pct_L4":  lambda: _stage_ckpt("8894380", 4),
    "6pct_L8":  lambda: _stage_ckpt("8894380", 8),
    "6pct_L16": lambda: _stage_ckpt("8894380", 16),
    "6pct_L32": lambda: _stage_ckpt("8894380", 32),
    "6pct_L48": lambda: _stage_ckpt("8894380", 48),
    "6pct_L64": lambda: _stage_ckpt("9001346", 64),
    "6pct_L75": lambda: _stage_ckpt("9001346", 75),
}


def resolve_model(spec):
    """Resolve a model spec to (display_name, path).

    Accepts:
      - registry key:                     '6pct_L16'
      - explicit alias+path:              'my6pct=/scratch/.../stage_64_step_104850_L64'
      - bare path (absolute or HF repo):  '/scratch/.../checkpoint-87000'  or  'Qwen/Qwen3-0.6B'
    """
    if "=" in spec and not spec.startswith("/"):
        name, path = spec.split("=", 1)
        return name, path
    if spec in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[spec]
        path = entry() if callable(entry) else entry
        if path is None:
            raise ValueError(f"Could not resolve checkpoint for model '{spec}'")
        return spec, path
    # Bare path (local dir) or HF repo (contains /)
    if "/" in spec:
        name = spec.rstrip("/").rsplit("/", 1)[-1]
        return name, spec
    raise ValueError(
        f"Unknown model spec '{spec}'. Expected a registry key "
        f"({sorted(MODEL_REGISTRY.keys())}), a 'name=path' alias, or a path/HF repo."
    )


# ==========================================================
# Model / generation helpers
# ==========================================================

def load_model(path):
    """Load tokenizer + model. Auto-detects whether the model has a chat template.

    Returns (tokenizer, model, use_chat_template_flag).
    - Qwen checkpoints: tokenizer has `chat_template`, we use it with `enable_thinking=False`.
    - Pythia / base LMs: no chat template, use plain completion.
    """
    tok = AutoTokenizer.from_pretrained(path, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    use_chat = getattr(tok, "chat_template", None) is not None
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"  [load] {path}  chat_template={'yes' if use_chat else 'no'}  "
          f"tokenizer={tok.__class__.__name__}")
    return tok, model, use_chat


def _chat_format(tokenizer, user_content):
    """Apply chat template with graceful fallback for non-Qwen3 chat models."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )


def generate(model, tokenizer, user_content, use_chat, max_new_tokens=30):
    """Unified generate. Uses chat template iff the tokenizer has one."""
    if use_chat:
        prompt = _chat_format(tokenizer, user_content)
    else:
        prompt = user_content
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    ids = enc["input_ids"]
    in_len = ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()


# Back-compat alias so existing benchmark functions don't all need rewriting.
def generate_chat(model, tokenizer, user_content, max_new_tokens=30, use_chat=True):
    return generate(model, tokenizer, user_content, use_chat, max_new_tokens)


@torch.no_grad()
def _loglik_completion(model, tokenizer, prefix: str, completion: str) -> float:
    """Compute log p(completion | prefix), handling BPE boundary merges.

    Walks forward the first position where `tokenize(prefix)` and `tokenize(prefix+completion)`
    diverge, and sums log-probs of the suffix tokens from that position onward. This is the
    same trick lm-eval-harness uses for its `loglikelihood` request type.
    """
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(prefix + completion, add_special_tokens=False)

    # Find the longest common prefix of token-id lists
    split = 0
    while split < len(prefix_ids) and split < len(full_ids) and prefix_ids[split] == full_ids[split]:
        split += 1
    if split == len(full_ids):
        return 0.0  # completion absorbed into last prefix token (shouldn't happen in practice)

    inp = torch.tensor([full_ids], device=model.device)
    logits = model(inp).logits[0]  # [seq_len, vocab]
    log_probs = torch.log_softmax(logits, dim=-1)

    # log p(token[t] | tokens[:t]) = log_probs[t-1, token[t]]
    # Sum over t in [split, len(full_ids))
    total = 0.0
    for t in range(split, len(full_ids)):
        total += log_probs[t - 1, full_ids[t]].item()
    return total


# ==========================================================
# Debug sample printing
# ==========================================================
# Set via --debug-samples N. When > 0, the first N examples per (benchmark, sub-task)
# print their prompt excerpt + gold + pred + per-class log-likelihoods (or generation
# output for gen benchmarks). Each scoring helper accepts _dbg_label and _dbg_gold;
# generation benchmarks call _dbg_log_gen() after generate().

DEBUG_SAMPLES = 0
_dbg_counters = {}


def _dbg_should_print(label):
    """Return True iff debug is on and this label hasn't hit its sample quota."""
    if DEBUG_SAMPLES <= 0 or label is None:
        return False
    return _dbg_counters.get(label, 0) < DEBUG_SAMPLES


def _dbg_bump(label):
    _dbg_counters[label] = _dbg_counters.get(label, 0) + 1


def _dbg_print_header(label, prompt):
    n = _dbg_counters[label]
    bench, _, sub = label.partition(":")
    excerpt = (prompt[-220:]).replace("\n", "\\n")
    print(f"\n  [DBG {label} #{n}]  prompt …{excerpt!r}")


def _dbg_log_loglik(label, prompt, gold, pred, scores: dict):
    """Print log-likelihood scoring debug for one example. `scores` is {label: log-lik}."""
    if not _dbg_should_print(label):
        return
    _dbg_bump(label)
    _dbg_print_header(label, prompt)
    score_str = "  ".join(f"{k}={v:.3f}" for k, v in scores.items())
    correct = "✓" if (gold is not None and str(gold).lower().strip() == str(pred).lower().strip()) else "✗"
    print(f"    gold={gold!r}  pred={pred!r}  {correct}   {score_str}")


def _dbg_log_gen(label, prompt, gold, gen_text, pred=None):
    """Print generation-based debug for one example. `pred` is the parsed answer if any."""
    if not _dbg_should_print(label):
        return
    _dbg_bump(label)
    _dbg_print_header(label, prompt)
    gen_excerpt = gen_text[:120].replace("\n", "\\n")
    line = f"    gold={gold!r}  gen={gen_excerpt!r}"
    if pred is not None:
        correct = "✓" if (gold is not None and str(gold).lower().strip() == str(pred).lower().strip()) else "✗"
        line += f"  →  pred={pred!r}  {correct}"
    print(line)


def score_yesno(model, tokenizer, prompt: str, use_chat: bool, _dbg_label=None, _dbg_gold=None) -> str:
    """Return 'yes' or 'no' by comparing log-likelihood of the two completions.

    When use_chat is True, wraps the prompt in the chat template (with enable_thinking=False
    for Qwen3) so the scoring reflects the distribution the chat-tuned model actually produces.
    Pass _dbg_label="bench:sub" and _dbg_gold for debug output (controlled by DEBUG_SAMPLES).
    """
    prefix = _chat_format(tokenizer, prompt) if use_chat else prompt
    # Score " Yes" and " No" — leading space is lm-eval convention and ensures
    # BPE tokenizers pick up the standalone token rather than a merged prefix.
    ll_yes = _loglik_completion(model, tokenizer, prefix, " Yes")
    ll_no = _loglik_completion(model, tokenizer, prefix, " No")
    pred = "yes" if ll_yes > ll_no else "no"
    _dbg_log_loglik(_dbg_label, prompt, _dbg_gold, pred,
                    {" Yes": ll_yes, " No": ll_no, "gap": ll_yes - ll_no})
    return pred


def score_truefalse(model, tokenizer, prompt: str, use_chat: bool, _dbg_label=None, _dbg_gold=None) -> str:
    """Return 'true' or 'false' by log-likelihood comparison."""
    prefix = _chat_format(tokenizer, prompt) if use_chat else prompt
    ll_t = _loglik_completion(model, tokenizer, prefix, " True")
    ll_f = _loglik_completion(model, tokenizer, prefix, " False")
    pred = "true" if ll_t > ll_f else "false"
    _dbg_log_loglik(_dbg_label, prompt, _dbg_gold, pred,
                    {" True": ll_t, " False": ll_f, "gap": ll_t - ll_f})
    return pred


def score_tfu(model, tokenizer, prompt: str, use_chat: bool, _dbg_label=None, _dbg_gold=None) -> str:
    """Return 'true', 'false', or 'unknown' by 3-way log-likelihood comparison.

    This is the correct scoring for ProofWriter under the open-world assumption:
    Unknown is a first-class label meaning the query is neither provable nor
    disprovable from the theory. Filtering it out would bias toward easier examples
    and change the task semantics.
    """
    prefix = _chat_format(tokenizer, prompt) if use_chat else prompt
    scores = {
        "true":    _loglik_completion(model, tokenizer, prefix, " True"),
        "false":   _loglik_completion(model, tokenizer, prefix, " False"),
        "unknown": _loglik_completion(model, tokenizer, prefix, " Unknown"),
    }
    pred = max(scores, key=scores.get)
    _dbg_log_loglik(_dbg_label, prompt, _dbg_gold, pred, scores)
    return pred


def _yesno(pred):
    s = pred.lower().strip()
    if "yes" in s[:10] or "true" in s[:10]:
        return "yes"
    if "no" in s[:10] or "false" in s[:10]:
        return "no"
    return ""


# ==========================================================
# Benchmark registry
# ==========================================================

BENCHMARKS = {}


def register(name):
    def _wrap(fn):
        BENCHMARKS[name] = fn
        return fn
    return _wrap


# ---------------- ZebraLogic ----------------

# NOTE: The grid-mode `zebra` benchmark was REMOVED on 2026-04-16. ZebraLogic-grid
# bundles reasoning capability with JSON-format-following, and fine-tuned models
# pay a "format tax" (parse_rate drops from 99% → 71%) that wipes out their
# reasoning advantage. The MC version (`zebra_mc` below) tests the same puzzles
# via log-likelihood scoring, which isolates reasoning from format-following.
# Old grid results are preserved in RESULTS.md but no new grid-mode runs.


@register("zebra_mc")
def eval_zebra_mc(model, tokenizer, use_chat, n=100):
    """ZebraLogic multiple-choice mode via log-likelihood scoring.

    Uses WildEval/ZebraLogic/mc_mode (3260 questions across 25 puzzle sizes 2*2..6*6).
    Each example: (puzzle, question, choices[list], answer[str]).
    Scoring: log-likelihood of each choice as a completion; argmax -> predicted choice.

    This is a non-paper-standard variant that's more reliable for small models —
    it measures whether the model can pick the correct attribute from a closed set,
    rather than whether it can emit valid JSON for a full solution table.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("WildEval/ZebraLogic", "mc_mode", split="test")
    except Exception as e:
        return {"error": f"load_dataset failed: {e}"}
    print(f"  [zebra_mc] loaded WildEval/ZebraLogic/mc_mode: {len(ds)} questions")

    def extract_size(ex_id):
        try:
            return ex_id.split("-")[2].replace("x", "*")
        except Exception:
            return "?"

    by_size = {}
    for ex in ds:
        by_size.setdefault(extract_size(ex["id"]), []).append(ex)

    def _size_key(s):
        try:
            return tuple(int(x) for x in s.split("*"))
        except Exception:
            return (99, 99)

    results = {}
    for size in sorted(by_size.keys(), key=_size_key):
        group = by_size[size][:n]
        correct = 0
        total = 0
        for ex in group:
            prompt = f"{ex['puzzle']}\n\nQuestion: {ex['question']}\nAnswer:"
            prefix = _chat_format(tokenizer, prompt) if use_chat else prompt
            scores = [
                _loglik_completion(model, tokenizer, prefix, " " + str(c))
                for c in ex["choices"]
            ]
            pred = ex["choices"][max(range(len(scores)), key=lambda i: scores[i])]
            _dbg_log_loglik(f"zebra_mc:{size}", prompt, ex["answer"], pred,
                            {f"choice[{i}]={c!r}": scores[i] for i, c in enumerate(ex["choices"])})
            if pred == ex["answer"]:
                correct += 1
            total += 1
        results[size] = {
            "correct": correct, "total": total,
            "accuracy": correct / total if total else 0.0,
        }
        print(f"  [zebra_mc] size={size:>5} n={total:>3}  acc={results[size]['accuracy']:.1%}")

    sizes = [v for v in results.values() if v["total"] > 0]
    all_correct = sum(v["correct"] for v in sizes)
    all_total = sum(v["total"] for v in sizes)
    results["overall"] = {
        "correct": all_correct,
        "total": all_total,
        "accuracy": all_correct / all_total if all_total else 0.0,
    }
    return results


# ---------------- LegalBench ----------------

# LegalBench task adapters. Template paths are resolved lazily by `_legal_template`
# so PROMPTS_DIR can still be overridden via --prompts-dir (set in main()).
LEGAL_TASKS = {
    "hearsay": {
        "input_col": "text", "answer_col": "answer",
        "template_path": "legal/hearsay.txt",
    },
    "international_citizenship_questions": {
        "input_col": "question", "answer_col": "answer",
        "template_path": "legal/international_citizenship_questions.txt",
    },
    "proa": {
        "input_col": "text", "answer_col": "answer",
        "template_path": "legal/proa.txt",
    },
}


def _legal_template(spec):
    """Lazy-load the prompt template — `_load_prompt` reads PROMPTS_DIR which
    isn't finalized until main() parses --prompts-dir."""
    return _load_prompt(spec["template_path"])


@register("legal")
def eval_legal(model, tokenizer, use_chat, n=100):
    """LegalBench: 3 binary-classification subtasks using the official few-shot templates."""
    from datasets import load_dataset

    results = {}
    for task, spec in LEGAL_TASKS.items():
        try:
            ds = load_dataset("nguha/legalbench", task, split="test")
        except Exception as e:
            print(f"  [legal:{task}] load failed: {e}")
            results[task] = {"error": str(e)}
            continue
        if len(ds) == 0:
            results[task] = {"error": "empty test split"}
            continue

        # Fallback to 'label' if 'answer' is missing
        ans_col = spec["answer_col"]
        if ans_col not in ds.column_names:
            ans_col = "label" if "label" in ds.column_names else ds.column_names[-1]

        correct = 0
        total = 0
        for ex in list(ds)[:n]:
            text = ex.get(spec["input_col"])
            gold = ex.get(ans_col)
            if text is None or gold is None:
                continue
            prompt = _legal_template(spec).format(input=text)
            pred_yn = score_yesno(model, tokenizer, prompt, use_chat,
                                  _dbg_label=f"legal:{task}", _dbg_gold=gold)
            gold_yn = _yesno(str(gold))
            if pred_yn == gold_yn and pred_yn:
                correct += 1
            total += 1
        acc = correct / total if total else 0.0
        results[task] = {"correct": correct, "total": total, "accuracy": acc}
        print(f"  [legal:{task}]: {correct}/{total} = {acc:.1%}")

    # LegalBench convention: macro-average across tasks (each task weighted
    # equally). Micro would be dominated by international_citizenship_questions
    # which has ~9k samples vs ~95 for hearsay/proa. Per LegalBench paper
    # (Guha et al., NeurIPS 2023) and HELM leaderboard.
    per_task_accs = [v["accuracy"] for v in results.values() if "accuracy" in v]
    macro_avg = sum(per_task_accs) / len(per_task_accs) if per_task_accs else 0.0
    all_correct = sum(v.get("correct", 0) for v in results.values())
    all_total = sum(v.get("total", 0) for v in results.values())
    results["overall"] = {
        "correct": all_correct,
        "total": all_total,
        "accuracy": macro_avg,           # macro-avg is the headline metric
        "micro_accuracy": all_correct / all_total if all_total else 0.0,
    }
    print(f"  [legal] macro-avg: {macro_avg:.3f}  (micro: "
          f"{all_correct/max(all_total,1):.3f} over {all_total} samples)")
    return results


# ---------------- Game of 24 ----------------

# Unregistered: game24 is consistently <5% across all 0.6B models (instruct=0.05,
# all others 0.01–0.04). Too hard for 0.6B; revive for Qwen 1.7B+ if needed.
# @register("game24")
def eval_game24(model, tokenizer, use_chat, n=100):
    """Game of 24: given 4 numbers, combine with +-*/() to make 24.

    Uses the 100-puzzle Hard subset from Yao et al. (Tree-of-Thoughts), puzzles
    at indices 900..999 (medium-hard difficulty, "solved rate" ~25-55%).

    Scoring: generation-based. Model outputs an expression. We parse it,
    verify it uses exactly the 4 input numbers (each once), and check that
    it evaluates to 24. Random baseline is ~0% (tiny expression space of valid).
    """
    if DATA_DIR is None:
        return {"error": "Game of 24 requires --data-dir (looks for game24_data/24.csv inside it)"}
    csv_path = os.path.join(DATA_DIR, "game24_data", "24.csv")
    if not os.path.isfile(csv_path):
        return {"error": f"Game of 24 data not found at {csv_path}"}

    import csv as _csv
    puzzles = []
    with open(csv_path) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            p = row.get("Puzzles", "").strip()
            if p:
                puzzles.append(p)

    # Hard subset as in ToT paper: rank 901-1000 (indices 900-999)
    if len(puzzles) >= 1000:
        test_set = puzzles[900:1000]
    else:
        test_set = puzzles[-min(len(puzzles), 100):]
    test_set = test_set[:n] if n else test_set

    import re as _re
    correct = 0
    total = 0
    for nums_str in test_set:
        nums = [int(x) for x in nums_str.split()]
        prompt = (
            f"Use each of these 4 numbers exactly once with +, -, *, /, "
            f"and parentheses to make an arithmetic expression that equals 24: "
            f"{nums_str}\n"
            "Answer with only the expression, no explanation."
        )
        pred = generate_chat(model, tokenizer, prompt, max_new_tokens=40, use_chat=use_chat)
        _dbg_log_gen("game24:overall", prompt, f"=24 from {nums_str}", pred)

        # Extract the first plausible expression from the output
        # Look for a line/fragment with digits and operators
        candidate = pred.strip().split("\n")[0]
        # Keep only characters that can appear in a valid expression
        clean = "".join(c for c in candidate if c in "0123456789+-*/()= .")
        # Drop anything after an '=' if present
        clean = clean.split("=")[0].strip()
        if not clean:
            total += 1
            continue

        # Verify the digits used
        digits_in_expr = [int(d) for d in _re.findall(r"\d+", clean)]
        if sorted(digits_in_expr) != sorted(nums):
            total += 1
            continue

        # Evaluate safely
        try:
            val = eval(clean, {"__builtins__": {}}, {})
            if abs(val - 24) < 1e-6:
                correct += 1
        except Exception:
            pass
        total += 1

    acc = correct / total if total else 0.0
    print(f"  [game24] hard-subset: {correct}/{total} = {acc:.1%}")
    return {
        "hard_subset": {"correct": correct, "total": total, "accuracy": acc},
        "overall": {"correct": correct, "total": total, "accuracy": acc},
    }


# ---------------- StepGame ----------------

# Canonical label set from StepGame benchmark (Shi et al., AAAI 2022)
STEPGAME_LABELS = [
    "left", "right", "above", "below",
    "upper-left", "upper-right", "lower-left", "lower-right",
    "overlap",
]


@register("stepgame")
def eval_stepgame(model, tokenizer, use_chat, n=100):
    """StepGame: chain spatial relations to deduce the position of one agent
    relative to another. Config `qaK` has K reasoning hops.

    Scoring: log-likelihood over 9 direction candidates. Random = 11.1%.

    Stratified by `config` (qa1..qa9) to show depth-scaling behavior, same
    flavor as ProofWriter.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("tasksource/stepgame", split="test")
    except Exception as e:
        return {"error": f"tasksource/stepgame load failed: {e}"}
    print(f"  [stepgame] loaded: {len(ds)} examples")

    by_hops = {}
    for ex in ds:
        cfg = ex.get("config", "qa?")
        by_hops.setdefault(cfg, []).append(ex)

    def _hops(c):
        try:
            return int(c.replace("qa", ""))
        except Exception:
            return 99

    results = {}
    for cfg in sorted(by_hops.keys(), key=_hops):
        group = by_hops[cfg][:n]
        correct = 0
        total = 0
        for ex in group:
            prompt = (
                f"{ex['story']}\n\n"
                f"Question: {ex['question']}\n"
                f"Answer:"
            )
            prefix = _chat_format(tokenizer, prompt) if use_chat else prompt
            scores = {
                lbl: _loglik_completion(model, tokenizer, prefix, " " + lbl)
                for lbl in STEPGAME_LABELS
            }
            pred = max(scores, key=scores.get)
            gold = ex.get("label", "").strip().lower()
            _dbg_log_loglik(f"stepgame:{cfg}", prompt, gold, pred, scores)
            if pred == gold:
                correct += 1
            total += 1
        acc = correct / total if total else 0.0
        results[cfg] = {"correct": correct, "total": total, "accuracy": acc}
        print(f"  [stepgame] {cfg}: {correct}/{total} = {acc:.1%}")

    all_correct = sum(v["correct"] for v in results.values())
    all_total = sum(v["total"] for v in results.values())
    results["overall"] = {
        "correct": all_correct, "total": all_total,
        "accuracy": all_correct / all_total if all_total else 0.0,
    }
    return results


# ---------------- PlanBench Blocksworld ----------------

def _eval_planbench_domain(model, tokenizer, use_chat, domain, n):
    """Shared loader for any PlanBench task_1_plan_generation domain."""
    from datasets import load_dataset
    import re as _re
    try:
        ds = load_dataset("tasksource/planbench", "task_1_plan_generation", split="train")
    except Exception as e:
        return {"error": f"planbench load failed: {e}"}
    matched = [ex for ex in ds if ex.get("domain") == domain][:n]
    print(f"  [{domain}] {len(matched)} instances")

    def _normalize_plan(text):
        actions = []
        for m in _re.finditer(r"\(([^)]+)\)", text):
            tokens = m.group(1).lower().strip().split()
            if tokens:
                actions.append(tuple(tokens))
        return actions

    correct = 0
    total = 0
    n_actions_correct = 0
    n_actions_total = 0
    first_action_correct = 0
    for ex in matched:
        gold_actions = _normalize_plan(ex["ground_truth_plan"])
        if not gold_actions:
            continue
        prompt = ex["query"]
        pred_text = generate_chat(model, tokenizer, prompt, max_new_tokens=200, use_chat=use_chat)
        pred_actions = _normalize_plan(pred_text)
        _dbg_log_gen(f"{domain}:overall", prompt,
                     " | ".join(" ".join(a) for a in gold_actions[:3]) + (" …" if len(gold_actions) > 3 else ""),
                     pred_text,
                     " | ".join(" ".join(a) for a in pred_actions[:3]) + (" …" if len(pred_actions) > 3 else ""))
        if pred_actions == gold_actions:
            correct += 1
        if pred_actions and pred_actions[0] == gold_actions[0]:
            first_action_correct += 1
        total += 1
        k = 0
        while k < min(len(pred_actions), len(gold_actions)) and pred_actions[k] == gold_actions[k]:
            k += 1
        n_actions_correct += k
        n_actions_total += len(gold_actions)

    plan_acc = correct / total if total else 0.0
    prefix_acc = n_actions_correct / n_actions_total if n_actions_total else 0.0
    first_acc = first_action_correct / total if total else 0.0
    print(f"  [{domain}] plan_exact={correct}/{total}={plan_acc:.1%}  "
          f"first_action={first_action_correct}/{total}={first_acc:.1%}  "
          f"prefix_match={prefix_acc:.1%}")
    return {
        domain: {
            "correct": correct, "total": total,
            "accuracy": plan_acc,
            "first_action_accuracy": first_acc,
            "prefix_match_rate": prefix_acc,
        },
        "overall": {
            "correct": correct, "total": total,
            "accuracy": plan_acc,
            "first_action_accuracy": first_acc,
        },
    }


@register("blocksworld")
def eval_blocksworld(model, tokenizer, use_chat, n=50):
    """PlanBench Blocksworld: plan generation task.

    Uses tasksource/planbench task_1_plan_generation filtered to domain=blocksworld.
    Model generates a plan as a sequence of (action arg1 arg2) tuples. We compare
    against the gold plan via exact-match on normalized action sequences.

    Scoring: exact-match plan accuracy. This is stricter than "did the plan
    reach the goal" (which requires a simulator) — a model that produces a
    different valid plan will score 0. Small models will mostly fail exact
    match, but the differential between instruct_only and 6pct+search is
    still informative.
    """
    from datasets import load_dataset
    import re as _re

    try:
        ds = load_dataset("tasksource/planbench", "task_1_plan_generation", split="train")
    except Exception as e:
        return {"error": f"planbench load failed: {e}"}

    bw = [ex for ex in ds if ex.get("domain") == "blocksworld"][:n]
    print(f"  [blocksworld] {len(bw)} instances")

    def _normalize_plan(text):
        """Extract list of (action, *args) tuples from text."""
        actions = []
        for m in _re.finditer(r"\(([^)]+)\)", text):
            tokens = m.group(1).lower().strip().split()
            if tokens:
                actions.append(tuple(tokens))
        return actions

    return _eval_planbench_domain(model, tokenizer, use_chat, "blocksworld", n)


@register("mystery_blocksworld")
def eval_mystery_blocksworld(model, tokenizer, use_chat, n=50):
    """PlanBench Mystery-Blocksworld: same underlying planning structure as
    blocksworld but with all predicate and action names obfuscated to
    nonsense words (e.g., 'Paltry o1 o2 o3' instead of 'pick-up block1').

    This is a crucial ablation: it tests whether the model has learned the
    abstract planning structure vs. just memorized the blocksworld vocabulary.
    A model that does well on blocksworld but poorly on mystery-blocksworld
    is reciting surface patterns; one that does well on both has internalized
    the state-space-search structure.

    Scoring: same as blocksworld — exact-match on action sequence + prefix
    match rate as partial credit.
    """
    return _eval_planbench_domain(model, tokenizer, use_chat, "mystery_blocksworld", n)


@register("logistics")
def eval_logistics(model, tokenizer, use_chat, n=50):
    """PlanBench Logistics: classic AI planning benchmark for package delivery
    across multiple cities using trucks and airplanes. Larger state space and
    longer plans than blocksworld (~10-30 actions).
    """
    return _eval_planbench_domain(model, tokenizer, use_chat, "logistics", n)


# ---------------- First-action variants (easier scoring) ----------------
# Wrap the existing PlanBench evals and surface first_action_accuracy as the
# headline `accuracy`. These give signal even when full-plan exact-match is 0%.

def _planbench_first_action_wrapper(domain):
    def _runner(model, tokenizer, use_chat, n=50):
        result = _eval_planbench_domain(model, tokenizer, use_chat, domain, n)
        if "error" in result:
            return result
        d = result.get(domain, {})
        first = d.get("first_action_accuracy", 0.0)
        return {
            domain: {
                "correct": int(first * d.get("total", 0)),
                "total": d.get("total", 0),
                "accuracy": first,
                "exact_match_accuracy": d.get("accuracy", 0.0),
            },
            "overall": {
                "correct": int(first * d.get("total", 0)),
                "total": d.get("total", 0),
                "accuracy": first,
            },
        }
    return _runner

@register("blocksworld_first")
def eval_blocksworld_first(model, tokenizer, use_chat, n=50):
    """First-action accuracy for blocksworld. Easier than full plan match —
    scores 1 if the model's first emitted action matches the gold first action."""
    return _planbench_first_action_wrapper("blocksworld")(model, tokenizer, use_chat, n)

@register("mystery_blocksworld_first")
def eval_mystery_blocksworld_first(model, tokenizer, use_chat, n=50):
    """First-action accuracy for mystery_blocksworld."""
    return _planbench_first_action_wrapper("mystery_blocksworld")(model, tokenizer, use_chat, n)

@register("logistics_first")
def eval_logistics_first(model, tokenizer, use_chat, n=50):
    """First-action accuracy for logistics."""
    return _planbench_first_action_wrapper("logistics")(model, tokenizer, use_chat, n)


# ==========================================================
# Log-probability variants for sequence-generation tasks
# These score teacher-forced log p(gold_output | prompt), normalized by token
# count. Higher (less negative) = model assigns more probability mass to the
# correct answer. Not an accuracy metric, but the natural log-lik analog when
# the output space is a sequence rather than a closed set of labels.
# ==========================================================


def _score_gold_logprob(model, tokenizer, prompt, gold, use_chat):
    """Per-token mean log-probability of `gold` as completion of `prompt`."""
    prefix = _chat_format(tokenizer, prompt) if use_chat else prompt
    ll = _loglik_completion(model, tokenizer, prefix, gold)
    n_toks = max(1, len(tokenizer.encode(gold, add_special_tokens=False)))
    return ll / n_toks


def _eval_planbench_logprob(model, tokenizer, use_chat, domain, n):
    from datasets import load_dataset
    try:
        ds = load_dataset("tasksource/planbench", "task_1_plan_generation", split="train")
    except Exception as e:
        return {"error": f"load failed: {e}"}
    matched = [ex for ex in ds if ex.get("domain") == domain][:n]
    total_lp = 0.0
    count = 0
    for ex in matched:
        lp = _score_gold_logprob(model, tokenizer, ex["query"], ex["ground_truth_plan"], use_chat)
        _dbg_log_loglik(f"{domain}_logprob:overall", ex["query"], ex["ground_truth_plan"][:60],
                        f"lp/tok={lp:.4f}", {"mean_logprob_per_token": lp})
        total_lp += lp
        count += 1
    mean_lp = total_lp / count if count else 0.0
    print(f"  [{domain}_logprob] n={count}  mean log-prob/token = {mean_lp:.4f}")
    return {domain: {"n": count, "mean_logprob_per_token": mean_lp},
            "overall": {"n": count, "mean_logprob_per_token": mean_lp}}


@register("blocksworld_logprob")
def eval_blocksworld_logprob(model, tokenizer, use_chat, n=50):
    """Blocksworld: mean per-token log-prob of gold plan. Higher = better."""
    return _eval_planbench_logprob(model, tokenizer, use_chat, "blocksworld", n)


@register("mystery_blocksworld_logprob")
def eval_mystery_blocksworld_logprob(model, tokenizer, use_chat, n=50):
    return _eval_planbench_logprob(model, tokenizer, use_chat, "mystery_blocksworld", n)


@register("logistics_logprob")
def eval_logistics_logprob(model, tokenizer, use_chat, n=50):
    return _eval_planbench_logprob(model, tokenizer, use_chat, "logistics", n)


@register("chess_mate_logprob")
def eval_chess_mate_logprob(model, tokenizer, use_chat, n=50):
    """Chess mate-in-N: mean per-token log-prob of gold move sequence."""
    # Streaming dataset — cap required. If main() passed n=None (full test set
    # default), fall back to signature default so we have a termination condition.
    n = n or 50
    from datasets import load_dataset
    try:
        ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    except Exception as e:
        return {"error": f"load failed: {e}"}

    targets = {"mateIn1": [], "mateIn2": [], "mateIn3": []}

    def _themes_list(t):
        if isinstance(t, list): return t
        if isinstance(t, str):
            import ast
            try:
                v = ast.literal_eval(t)
                if isinstance(v, list): return v
            except Exception: pass
            return t.split()
        return []

    for ex in ds:
        themes = _themes_list(ex.get("Themes", []))
        for key in list(targets.keys()):
            if key in themes and len(targets[key]) < n:
                targets[key].append(ex)
        if all(len(v) >= n for v in targets.values()):
            break

    results = {}
    for cat in ["mateIn1", "mateIn2", "mateIn3"]:
        group = targets[cat][:n]
        total_lp = 0.0; count = 0
        for ex in group:
            moves = ex["Moves"].split()
            if len(moves) < 2: continue
            setup, gold_seq = moves[0], " ".join(moves[1:])
            prompt = (f"Chess puzzle in UCI notation.\n"
                      f"Position (FEN): {ex['FEN']}\n"
                      f"The opponent just played {setup}.\n"
                      f"The forced mate sequence in UCI format is:")
            lp = _score_gold_logprob(model, tokenizer, prompt, " " + gold_seq, use_chat)
            _dbg_log_loglik(f"chess_mate_logprob:{cat}", prompt, gold_seq[:60],
                            f"lp/tok={lp:.4f}", {"mean_logprob_per_token": lp})
            total_lp += lp
            count += 1
        mean_lp = total_lp / count if count else 0.0
        results[cat] = {"n": count, "mean_logprob_per_token": mean_lp}
        print(f"  [chess_mate_logprob] {cat}: n={count} mean_lp={mean_lp:.4f}")
    all_lp = sum(v["mean_logprob_per_token"] * v["n"] for v in results.values())
    all_n = sum(v["n"] for v in results.values())
    results["overall"] = {"n": all_n,
                          "mean_logprob_per_token": all_lp / all_n if all_n else 0.0}
    return results


# Note: game24 has no canonical gold expression in the dataset, so a log-prob
# variant would require a brute-force solver to pick a representative solution
# per puzzle. Given the arbitrariness of which solution to choose, we omit
# game24_logprob and keep only the exact-eval generation-based game24.


# ==========================================================
# Paper-matching variants: generation-based scoring with few-shot
# These are alternatives to the log-lik variants above, scoring how closely
# the authors' original evaluation method would rate the model.
# Each emits a short generation and applies exact-match over expected labels.
# ==========================================================


def _extract_first_token_label(pred: str, candidates):
    """Return the first candidate label that appears in pred (case-insensitive),
    matching the paper convention of picking the first word after "A:"."""
    pred_low = pred.strip().lower()
    # Limit to first ~20 chars to avoid matching labels that appear later in prose
    head = pred_low[:32]
    # Sort by length desc so "upper-left" beats "left" when both appear
    for cand in sorted(candidates, key=lambda s: -len(s)):
        if cand.lower() in head:
            return cand
    return None


@register("legal_gen")
def eval_legal_gen(model, tokenizer, use_chat, n=100):
    """LegalBench with paper-matching generation-based scoring (Guha 2023).
    Same few-shot base_prompt templates; greedy generation; first-word Yes/No match."""
    from datasets import load_dataset
    results = {}
    for task, spec in LEGAL_TASKS.items():
        try:
            ds = load_dataset("nguha/legalbench", task, split="test")
        except Exception as e:
            results[task] = {"error": str(e)}; continue
        ans_col = spec["answer_col"]
        if ans_col not in ds.column_names:
            ans_col = "label" if "label" in ds.column_names else ds.column_names[-1]
        correct = 0; total = 0
        for ex in list(ds)[:n]:
            text = ex.get(spec["input_col"])
            gold = ex.get(ans_col)
            if text is None or gold is None: continue
            prompt = _legal_template(spec).format(input=text)
            pred = generate_chat(model, tokenizer, prompt, max_new_tokens=5, use_chat=use_chat)
            pred_yn = _yesno(pred)
            gold_yn = _yesno(str(gold))
            _dbg_log_gen(f"legal_gen:{task}", prompt, gold, pred, pred_yn or "?")
            if pred_yn == gold_yn and pred_yn: correct += 1
            total += 1
        acc = correct / total if total else 0.0
        results[task] = {"correct": correct, "total": total, "accuracy": acc}
        print(f"  [legal_gen:{task}]: {correct}/{total} = {acc:.1%}")
    all_c = sum(v.get("correct", 0) for v in results.values())
    all_t = sum(v.get("total", 0) for v in results.values())
    # Macro-average across tasks (LegalBench convention)
    per_task_accs = [v["accuracy"] for v in results.values() if "accuracy" in v]
    macro_avg = sum(per_task_accs) / len(per_task_accs) if per_task_accs else 0.0
    results["overall"] = {
        "correct": all_c, "total": all_t,
        "accuracy": macro_avg,
        "micro_accuracy": all_c/all_t if all_t else 0.0,
    }
    print(f"  [legal_gen] macro-avg: {macro_avg:.3f}  (micro: "
          f"{all_c/max(all_t,1):.3f} over {all_t} samples)")
    return results


@register("zebra_mc_gen")
def eval_zebra_mc_gen(model, tokenizer, use_chat, n=100):
    """ZebraLogic MC with generation-based scoring. Model generates an answer;
    we match it against the enumerated choices via substring presence."""
    from datasets import load_dataset
    try:
        ds = load_dataset("WildEval/ZebraLogic", "mc_mode", split="test")
    except Exception as e:
        return {"error": f"load failed: {e}"}
    print(f"  [zebra_mc_gen] loaded {len(ds)} questions")

    def extract_size(ex_id):
        try: return ex_id.split("-")[2].replace("x", "*")
        except: return "?"

    by_size = {}
    for ex in ds: by_size.setdefault(extract_size(ex["id"]), []).append(ex)

    results = {}
    for size in sorted(by_size.keys(), key=lambda s: tuple(int(x) for x in s.split("*")) if "*" in s else (99,99)):
        group = by_size[size][:n]
        correct = 0; total = 0
        for ex in group:
            prompt = (f"{ex['puzzle']}\n\nQuestion: {ex['question']}\n"
                      f"Choices: {', '.join(str(c) for c in ex['choices'])}\n"
                      f"Answer with only the chosen value.")
            pred = generate_chat(model, tokenizer, prompt, max_new_tokens=15, use_chat=use_chat)
            pred_norm = pred.strip().lower().strip('."\'')
            gold = str(ex["answer"]).strip().lower()
            _dbg_log_gen(f"zebra_mc_gen:{size}", prompt, gold, pred, pred_norm[:32])
            # Match if gold appears in first 32 chars of the prediction
            if gold in pred_norm[:32]:
                correct += 1
            total += 1
        results[size] = {"correct": correct, "total": total,
                         "accuracy": correct/total if total else 0.0}
        print(f"  [zebra_mc_gen] size={size:>5} n={total:>3} acc={results[size]['accuracy']:.1%}")
    all_c = sum(v["correct"] for v in results.values() if "correct" in v)
    all_t = sum(v["total"] for v in results.values() if "total" in v)
    results["overall"] = {"correct": all_c, "total": all_t, "accuracy": all_c/all_t if all_t else 0.0}
    return results


def _proofwriter_few_shot_block(mode):
    """Produce ProofWriter few-shot block. Either loaded from prompts/ or built
    from the train split via _load_proofwriter_raw if USE_TRAIN_ICES=1.

    mode is 'OWA' or 'CWA'.
    """
    cache_key = f"proofwriter_{mode}"
    if cache_key in _train_ice_cache:
        return _train_ice_cache[cache_key]
    if not _USE_TRAIN_ICES:
        path = f"proofwriter_{mode.lower()}_fewshot.txt"
        block = _load_prompt(path)
    else:
        # Load first K from train split — call with depth-3 (medium) for diversity
        root = _find_proofwriter_raw()
        if root is None:
            return _load_prompt(f"proofwriter_{mode.lower()}_fewshot.txt")
        p = os.path.join(root, mode, "depth-3", "meta-train.jsonl")
        if not os.path.isfile(p):
            return _load_prompt(f"proofwriter_{mode.lower()}_fewshot.txt")
        examples = []
        label_counts = {"True": 0, "False": 0, "Unknown": 0}
        want_per_label = _TRAIN_ICE_K // 3 + 1
        with open(p) as f:
            for line in f:
                if all(c >= want_per_label for c in label_counts.values()):
                    break
                rec = json.loads(line)
                theory = rec.get("theory", "")
                for q in rec.get("questions", {}).values():
                    ans = _normalize_pw_answer(q.get("answer"))
                    if ans is None:
                        continue
                    if mode == "CWA" and ans not in ("True", "False"):
                        continue
                    if label_counts[ans] >= want_per_label:
                        continue
                    label_counts[ans] += 1
                    examples.append((theory, q.get("question", ""), ans))
                    if len(examples) >= _TRAIN_ICE_K:
                        break
        closed = " (closed-world assumption)" if mode == "CWA" else ""
        tf_u = "true or false" if mode == "CWA" else "True, False, or Unknown"
        blocks = []
        for theory, qtext, ans in examples[:_TRAIN_ICE_K]:
            blocks.append(
                f"Facts and rules:\n{theory}\n"
                f"Question: Based only on the facts and rules above{closed}, "
                f"is the following statement {tf_u}?\n"
                f"Statement: {qtext}\nAnswer: {ans}"
            )
        block = "\n\n".join(blocks) + "\n\n"
    _train_ice_cache[cache_key] = block
    return block


@register("proofwriter_gen")
def eval_proofwriter_gen(model, tokenizer, use_chat, n=200):
    """ProofWriter OWA with 3-shot generation (approximates paper setup for LLMs)."""
    examples = _load_proofwriter_raw("OWA", "depth-5")
    if examples is None:
        return {"error": "proofwriter raw dataset not found"}
    by_depth = {}
    for ex in examples:
        ans = _normalize_pw_answer(ex["answer"])
        if ans not in ("True", "False", "Unknown"): continue
        ex["answer"] = ans
        by_depth.setdefault(ex["QDep"], []).append(ex)
    results = {}
    for depth in sorted(by_depth.keys()):
        if depth is None or depth > 5: continue
        correct = 0; total = 0
        for ex in by_depth[depth][:n]:
            prompt = (_proofwriter_few_shot_block("OWA") +
                      f"Facts and rules:\n{ex['theory']}\n"
                      f"Question: Based only on the facts and rules above, is the following "
                      f"statement True, False, or Unknown?\n"
                      f"Statement: {ex['question']}\n"
                      f"Answer:")
            pred = generate_chat(model, tokenizer, prompt, max_new_tokens=5, use_chat=use_chat)
            pl = pred.strip().lower()[:15]
            pa = "Unknown" if "unknown" in pl else ("True" if "true" in pl else ("False" if "false" in pl else ""))
            _dbg_log_gen(f"proofwriter_gen:depth_{depth}", prompt, ex["answer"], pred, pa or "?")
            if pa == ex["answer"]: correct += 1
            total += 1
        acc = correct/total if total else 0.0
        results[depth] = {"correct": correct, "total": total, "accuracy": acc}
        print(f"  [proofwriter_gen] depth={depth}: {correct}/{total} = {acc:.1%}")
    return results


@register("proofwriter_cwa_gen")
def eval_proofwriter_cwa_gen(model, tokenizer, use_chat, n=200):
    """ProofWriter CWA with 3-shot generation."""
    examples = _load_proofwriter_raw("CWA", "depth-5")
    if examples is None:
        return {"error": "proofwriter raw dataset not found"}
    by_depth = {}
    for ex in examples:
        ans = _normalize_pw_answer(ex["answer"])
        if ans not in ("True", "False"): continue
        ex["answer"] = ans
        by_depth.setdefault(ex["QDep"], []).append(ex)
    results = {}
    for depth in sorted(by_depth.keys()):
        if depth is None or depth > 5: continue
        correct = 0; total = 0
        for ex in by_depth[depth][:n]:
            prompt = (_proofwriter_few_shot_block("CWA") +
                      f"Facts and rules:\n{ex['theory']}\n"
                      f"Question: Based only on the facts and rules above (closed-world "
                      f"assumption), is the following statement true or false?\n"
                      f"Statement: {ex['question']}\n"
                      f"Answer:")
            pred = generate_chat(model, tokenizer, prompt, max_new_tokens=5, use_chat=use_chat)
            pl = pred.strip().lower()[:15]
            pa = "True" if "true" in pl else ("False" if "false" in pl else "")
            _dbg_log_gen(f"proofwriter_cwa_gen:depth_{depth}", prompt, ex["answer"], pred, pa or "?")
            if pa == ex["answer"]: correct += 1
            total += 1
        acc = correct/total if total else 0.0
        results[depth] = {"correct": correct, "total": total, "accuracy": acc}
        print(f"  [proofwriter_cwa_gen] depth={depth}: {correct}/{total} = {acc:.1%}")
    return results


def _stepgame_few_shot_block():
    """StepGame few-shot: loaded from prompts/ by default, or built from train
    split if USE_TRAIN_ICES=1."""
    if "stepgame" in _train_ice_cache:
        return _train_ice_cache["stepgame"]
    if not _USE_TRAIN_ICES:
        block = _load_prompt("stepgame_fewshot.txt")
    else:
        from datasets import load_dataset
        try:
            ds = load_dataset("tasksource/stepgame", split="train")
        except Exception:
            block = _load_prompt("stepgame_fewshot.txt")
            _train_ice_cache["stepgame"] = block
            return block
        examples = []
        seen_labels = set()
        for ex in ds:
            lbl = ex.get("label", "").strip().lower()
            if lbl in seen_labels or lbl not in STEPGAME_LABELS:
                continue
            seen_labels.add(lbl)
            examples.append(ex)
            if len(examples) >= _TRAIN_ICE_K:
                break
        blocks = []
        for ex in examples:
            blocks.append(
                f"Story: {ex['story']}\n"
                f"Question: {ex['question']}\n"
                f"Answer: {ex['label']}"
            )
        block = "\n\n".join(blocks) + "\n\n"
    _train_ice_cache["stepgame"] = block
    return block


@register("stepgame_gen")
def eval_stepgame_gen(model, tokenizer, use_chat, n=100):
    """StepGame with 5-shot generation (approximates Li et al. 2024)."""
    from datasets import load_dataset
    try:
        ds = load_dataset("tasksource/stepgame", split="test")
    except Exception as e:
        return {"error": f"load failed: {e}"}
    by_hops = {}
    for ex in ds: by_hops.setdefault(ex.get("config", "qa?"), []).append(ex)
    def _hops(c):
        try: return int(c.replace("qa",""))
        except: return 99
    results = {}
    for cfg in sorted(by_hops.keys(), key=_hops):
        group = by_hops[cfg][:n]
        correct = 0; total = 0
        for ex in group:
            prompt = (_stepgame_few_shot_block() +
                      f"Story: {ex['story']}\n"
                      f"Question: {ex['question']}\n"
                      f"Answer:")
            pred = generate_chat(model, tokenizer, prompt, max_new_tokens=6, use_chat=use_chat)
            pred_lbl = _extract_first_token_label(pred, STEPGAME_LABELS)
            gold = ex.get("label", "").strip().lower()
            _dbg_log_gen(f"stepgame_gen:{cfg}", prompt, gold, pred, pred_lbl or "?")
            if pred_lbl == gold: correct += 1
            total += 1
        acc = correct/total if total else 0.0
        results[cfg] = {"correct": correct, "total": total, "accuracy": acc}
        print(f"  [stepgame_gen] {cfg}: {correct}/{total} = {acc:.1%}")
    all_c = sum(v["correct"] for v in results.values())
    all_t = sum(v["total"] for v in results.values())
    results["overall"] = {"correct": all_c, "total": all_t,
                          "accuracy": all_c/all_t if all_t else 0.0}
    return results


# ---------------- Chess mate-in-N ----------------

@register("chess_mate")
def eval_chess_mate(model, tokenizer, use_chat, n=50):
    """Chess mate-in-N puzzles from Lichess.

    Streams the Lichess/chess-puzzles dataset, filters to mateIn1/2/3 puzzles.
    For each: give the model the FEN and the opponent's setup move, ask for
    the mating sequence in UCI format.

    Scoring: exact-match on the solver's move sequence (UCI notation). Stratified
    by mate depth. Random baseline is ~0% — chess has ~30 legal moves per position
    so picking the right sequence by chance is vanishingly unlikely.

    This is hard for a 0.6B model. Expect low numbers in absolute terms; the
    differential between instruct_only and 6pct+search is the informative signal.
    """
    # Streaming dataset — cap required. If main() passed n=None, fall back to default.
    n = n or 50
    from datasets import load_dataset

    try:
        ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    except Exception as e:
        return {"error": f"Lichess puzzles load failed: {e}"}

    targets = {"mateIn1": [], "mateIn2": [], "mateIn3": []}
    per_cat = max(n, 20)

    def _themes_list(t):
        if isinstance(t, list):
            return t
        if isinstance(t, str):
            # Try ast.literal_eval, then fall back to whitespace split
            import ast
            try:
                v = ast.literal_eval(t)
                if isinstance(v, list):
                    return v
            except Exception:
                pass
            return t.split()
        return []

    for ex in ds:
        themes = _themes_list(ex.get("Themes", []))
        for key in list(targets.keys()):
            if key in themes and len(targets[key]) < per_cat:
                targets[key].append(ex)
        if all(len(v) >= per_cat for v in targets.values()):
            break

    # If we couldn't find enough of some category, just use what we have
    total_puzzles = sum(len(v) for v in targets.values())
    print(f"  [chess_mate] collected: " +
          ", ".join(f"{k}={len(v)}" for k, v in targets.items()) +
          f"  total={total_puzzles}")

    results = {}
    for cat in ["mateIn1", "mateIn2", "mateIn3"]:
        group = targets[cat][:n]
        if not group:
            continue
        correct = 0
        first_move_correct = 0
        total = 0
        for ex in group:
            fen = ex["FEN"]
            moves = ex["Moves"].split()
            if len(moves) < 2:
                continue
            setup = moves[0]
            gold_seq = moves[1:]  # solver's moves + forced replies

            prompt = (
                f"Chess puzzle in UCI notation.\n"
                f"Position (FEN): {fen}\n"
                f"The opponent just played {setup}.\n"
                f"Find {cat.replace('mateIn','the forced mate in ')} moves. "
                f"Output only the sequence of moves in UCI format, "
                f"space-separated (e.g., 'e2e4 e7e5 g1f3'). Include the "
                f"opponent's forced replies. No explanation."
            )
            pred = generate_chat(model, tokenizer, prompt, max_new_tokens=40, use_chat=use_chat)

            # Extract UCI-like tokens from the output (4-5 char alphanumeric)
            import re as _re
            pred_moves = _re.findall(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", pred.lower())
            _dbg_log_gen(f"chess_mate:{cat}", prompt, " ".join(gold_seq), pred, " ".join(pred_moves))

            if pred_moves == gold_seq:
                correct += 1
            if pred_moves and pred_moves[0] == gold_seq[0]:
                first_move_correct += 1
            total += 1
        exact_acc = correct / total if total else 0.0
        first_acc = first_move_correct / total if total else 0.0
        results[cat] = {
            "correct": correct, "total": total,
            "accuracy": exact_acc,
            "first_move_accuracy": first_acc,
        }
        print(f"  [chess_mate] {cat}: exact={correct}/{total}={exact_acc:.1%}  "
              f"first_move={first_move_correct}/{total}={first_acc:.1%}")

    all_correct = sum(v["correct"] for v in results.values())
    all_total = sum(v["total"] for v in results.values())
    all_first = sum(v.get("first_move_accuracy", 0) * v["total"] for v in results.values())
    results["overall"] = {
        "correct": all_correct, "total": all_total,
        "accuracy": all_correct / all_total if all_total else 0.0,
        "first_move_accuracy": all_first / all_total if all_total else 0.0,
    }
    return results


@register("chess_mate_first")
def eval_chess_mate_first(model, tokenizer, use_chat, n=50):
    """First-move accuracy for chess mate-in-N. Easier than full mating sequence —
    scores 1 if the model's first emitted move matches the gold first move.
    Aligned with how Lichess scores their puzzles for smaller models."""
    raw = eval_chess_mate(model, tokenizer, use_chat, n=n)
    if "error" in raw:
        return raw
    out = {}
    for cat in ["mateIn1", "mateIn2", "mateIn3"]:
        if cat in raw:
            d = raw[cat]
            first = d.get("first_move_accuracy", 0.0)
            out[cat] = {
                "correct": int(first * d.get("total", 0)),
                "total": d.get("total", 0),
                "accuracy": first,
                "exact_match_accuracy": d.get("accuracy", 0.0),
            }
    if "overall" in raw:
        d = raw["overall"]
        first = d.get("first_move_accuracy", 0.0)
        out["overall"] = {
            "correct": int(first * d.get("total", 0)),
            "total": d.get("total", 0),
            "accuracy": first,
        }
    return out


# ---------------- lm-eval-harness adapter ----------------

# 12 standard NLU tasks used in the paper
STANDARD_TASKS = [
    "hellaswag", "winogrande", "piqa", "arc_easy", "arc_challenge",
    "boolq", "openbookqa", "sciq", "copa", "commonsense_qa",
    "truthfulqa_mc1", "gsm8k",
]

# 11 BBH zero-shot reasoning subtasks used in the paper (same set as eval_bbh.sh)
BBH_TASKS = [
    "bbh_zeroshot_boolean_expressions",
    "bbh_zeroshot_dyck_languages",
    "bbh_zeroshot_formal_fallacies",
    "bbh_zeroshot_logical_deduction_three_objects",
    "bbh_zeroshot_logical_deduction_five_objects",
    "bbh_zeroshot_logical_deduction_seven_objects",
    "bbh_zeroshot_navigate",
    "bbh_zeroshot_tracking_shuffled_objects_three_objects",
    "bbh_zeroshot_tracking_shuffled_objects_five_objects",
    "bbh_zeroshot_tracking_shuffled_objects_seven_objects",
    "bbh_zeroshot_web_of_lies",
]


def _run_lm_eval(model, tokenizer, use_chat, task_list, n=None):
    """Run a list of lm-eval-harness tasks on an already-loaded model.

    For Qwen3 models, monkey-patches tokenizer.apply_chat_template so every
    chat-wrapping call forces enable_thinking=False — matching how the paper's
    existing eval scripts invoked lm-eval.
    """
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    if use_chat:
        _orig = tokenizer.apply_chat_template
        def _patched(*args, **kwargs):
            kwargs.setdefault("enable_thinking", False)
            return _orig(*args, **kwargs)
        tokenizer.apply_chat_template = _patched

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=16)
    raw = simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=0,
        limit=n,
        apply_chat_template=use_chat,
    )

    out = {}
    for task_name, task_result in raw.get("results", {}).items():
        # Prefer common accuracy keys, fall back to exact_match / first numeric
        acc = (
            task_result.get("acc,none")
            or task_result.get("exact_match,none")
            or task_result.get("acc_norm,none")
        )
        if acc is None:
            for k, v in task_result.items():
                if isinstance(v, (int, float)) and not k.endswith("_stderr,none"):
                    acc = v
                    break
        out[task_name] = {"accuracy": acc, "total": task_result.get("samples", None)}
        print(f"  [lm_eval:{task_name}]: {acc*100:.1f}%" if acc is not None else f"  [lm_eval:{task_name}]: ?")

    # Macro-average across tasks
    accs = [v["accuracy"] for v in out.values() if v.get("accuracy") is not None]
    if accs:
        out["overall"] = {"accuracy": sum(accs) / len(accs), "total": len(accs)}
    return out


@register("standard")
def eval_standard(model, tokenizer, use_chat, n=None):
    """12 standard NLU benchmarks via lm-eval-harness (same tasks as eval_benchmark.sh)."""
    return _run_lm_eval(model, tokenizer, use_chat, STANDARD_TASKS, n=n)


@register("bbh")
def eval_bbh(model, tokenizer, use_chat, n=None):
    """11 BBH zero-shot reasoning subtasks via lm-eval-harness.

    NOTE: ignore the `use_chat` arg — chat template wraps the prompt in Qwen's
    assistant format, which makes the model emit markdown-wrapped answers
    ("**No**", "- Yes.") that fail lm-eval-harness's exact_match filter.
    Published BBH numbers are from the raw Suzgun prompt (no chat wrap)."""
    return _run_lm_eval(model, tokenizer, use_chat=False, task_list=BBH_TASKS, n=n)


# Paper-matching variant: 3-shot CoT, as in Suzgun et al. 2023's headline eval.
BBH_COT_TASKS = [t.replace("bbh_zeroshot_", "bbh_cot_fewshot_") for t in BBH_TASKS]


@register("bbh_cot")
def eval_bbh_cot(model, tokenizer, use_chat, n=None):
    """11 BBH subtasks with 3-shot Chain-of-Thought prompting (paper-matching).

    Uses lm-eval-harness's `bbh_cot_fewshot_*` configs, which replicate
    Suzgun et al. 2023's evaluation: 3 in-context examples with reasoning
    traces, then the test question, with exact-match on the final answer
    after the CoT.

    NOTE: ignores `use_chat` — see eval_bbh. The 3-shot CoT exemplars condition
    the model to emit "So the answer is X." format; chat-template wrapping
    breaks this."""
    return _run_lm_eval(model, tokenizer, use_chat=False, task_list=BBH_COT_TASKS, n=n)


# ---------------- ProofWriter ----------------

def _find_proofwriter_raw():
    """Locate ProofWriter raw dataset: $DATA_DIR/proofwriter_raw/proofwriter-dataset-V2020.12.3."""
    if DATA_DIR is None:
        return None
    p = os.path.join(DATA_DIR, "proofwriter_raw", "proofwriter-dataset-V2020.12.3")
    if os.path.isdir(os.path.join(p, "CWA")) and os.path.isdir(os.path.join(p, "OWA")):
        return p
    return None


def _load_proofwriter_raw(mode: str, depth_dir: str = "depth-5"):
    """Load AllenAI ProofWriter meta-test.jsonl for a given assumption mode.

    Flattens each record's nested questions into (theory, question, answer, QDep)
    tuples. Uses the depth-5 directory by default so we get questions at all
    QDep values (0 through 5) from theories that require up to 5-hop reasoning —
    the most comprehensive single file available.
    """
    root = _find_proofwriter_raw()
    if root is None:
        return None
    path = os.path.join(root, mode, depth_dir, "meta-test.jsonl")
    if not os.path.isfile(path):
        return None

    examples = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            theory = rec.get("theory", "")
            for qid, q in rec.get("questions", {}).items():
                examples.append({
                    "theory": theory,
                    "question": q.get("question", ""),
                    "answer": q.get("answer"),
                    "QDep": q.get("QDep"),
                })
    return examples


def _normalize_pw_answer(ans):
    """ProofWriter answers are JSON booleans for True/False and the string
    'Unknown' for the open-world case. Normalize to canonical strings."""
    if ans is True:
        return "True"
    if ans is False:
        return "False"
    if isinstance(ans, str):
        return ans
    return None


@register("proofwriter")
def eval_proofwriter(model, tokenizer, use_chat, n=200):
    """ProofWriter OWA 3-class classification (True / False / Unknown).

    Loads AllenAI's raw ProofWriter zip (proofwriter-dataset-V2020.12.3/OWA/depth-5/
    meta-test.jsonl). Groups questions by their own QDep level, so we get proper
    per-depth breakdowns.

    Each query has three possible answers: True, False, or Unknown (neither
    provable nor disprovable from the theory under open-world semantics). Scored
    via log-likelihood over all three candidates. Random-guess baseline = 33.3%.
    """
    examples = _load_proofwriter_raw("OWA", "depth-5")
    if examples is None:
        return {"error": "proofwriter raw dataset not found; run setup to extract"}
    print(f"  [proofwriter] loaded OWA/depth-5: {len(examples)} questions")

    by_depth = {}
    for ex in examples:
        ans = _normalize_pw_answer(ex["answer"])
        if ans not in ("True", "False", "Unknown"):
            continue
        ex["answer"] = ans
        by_depth.setdefault(ex["QDep"], []).append(ex)

    class_label = {"true": "True", "false": "False", "unknown": "Unknown"}
    results = {}
    for depth in sorted(by_depth.keys()):
        if depth is None or depth > 5:
            continue
        correct = 0
        total = 0
        by_class = {"True": [0, 0], "False": [0, 0], "Unknown": [0, 0]}
        for ex in by_depth[depth][:n]:
            answer = ex["answer"]
            user_content = (
                f"Facts and rules:\n{ex['theory']}\n\n"
                f"Based only on the facts and rules above, is the following "
                f"statement True, False, or Unknown (meaning it can be neither "
                f"proven nor disproven)?\n"
                f"{ex['question']}\n"
                "Answer with only True, False, or Unknown."
            )
            pred = class_label[score_tfu(model, tokenizer, user_content, use_chat,
                                         _dbg_label=f"proofwriter:depth_{depth}",
                                         _dbg_gold=answer)]
            by_class[answer][1] += 1
            if pred == answer:
                correct += 1
                by_class[answer][0] += 1
            total += 1
        acc = correct / total if total else 0.0
        results[depth] = {
            "correct": correct, "total": total, "accuracy": acc,
            "per_class": {
                cls: {"correct": c[0], "total": c[1],
                      "accuracy": c[0] / c[1] if c[1] else 0.0}
                for cls, c in by_class.items()
            },
        }
        per_class_str = "  ".join(f"{cls[:1]}={c[0]}/{c[1]}" for cls, c in by_class.items())
        print(f"  [proofwriter] depth={depth}: {correct}/{total} = {acc:.1%}  ({per_class_str})")
    return results


@register("proofwriter_cwa")
def eval_proofwriter_cwa(model, tokenizer, use_chat, n=200):
    """ProofWriter CWA binary classification (True / False).

    Uses the closed-world variant where the theory is assumed complete:
    any statement that can't be proven true is False (negation-as-failure).
    Each query has exactly two answers: True or False. Scored via log-likelihood
    over both candidates. Random-guess baseline = 50%.
    """
    examples = _load_proofwriter_raw("CWA", "depth-5")
    if examples is None:
        return {"error": "proofwriter raw dataset not found; run setup to extract"}
    print(f"  [proofwriter_cwa] loaded CWA/depth-5: {len(examples)} questions")

    by_depth = {}
    for ex in examples:
        ans = _normalize_pw_answer(ex["answer"])
        if ans not in ("True", "False"):
            continue
        ex["answer"] = ans
        by_depth.setdefault(ex["QDep"], []).append(ex)

    results = {}
    for depth in sorted(by_depth.keys()):
        if depth is None or depth > 5:
            continue
        correct = 0
        total = 0
        by_class = {"True": [0, 0], "False": [0, 0]}
        for ex in by_depth[depth][:n]:
            answer = ex["answer"]
            user_content = (
                f"Facts and rules:\n{ex['theory']}\n\n"
                f"Based only on the facts and rules above, is the following "
                f"statement true or false? Assume anything not provable is "
                f"False.\n"
                f"{ex['question']}\n"
                "Answer with only True or False."
            )
            pred_tf = score_truefalse(model, tokenizer, user_content, use_chat,
                                      _dbg_label=f"proofwriter_cwa:depth_{depth}",
                                      _dbg_gold=answer)
            pa = "True" if pred_tf == "true" else "False"
            by_class[answer][1] += 1
            if pa == answer:
                correct += 1
                by_class[answer][0] += 1
            total += 1
        acc = correct / total if total else 0.0
        results[depth] = {
            "correct": correct, "total": total, "accuracy": acc,
            "per_class": {
                cls: {"correct": c[0], "total": c[1],
                      "accuracy": c[0] / c[1] if c[1] else 0.0}
                for cls, c in by_class.items()
            },
        }
        per_class_str = "  ".join(f"{cls[:1]}={c[0]}/{c[1]}" for cls, c in by_class.items())
        print(f"  [proofwriter_cwa] depth={depth}: {correct}/{total} = {acc:.1%}  ({per_class_str})")
    return results


# ==========================================================
# Main
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="Registry key, 'name=path' alias, or bare path/HF repo. "
                             f"Registry keys: {sorted(MODEL_REGISTRY.keys())}")
    parser.add_argument("--benchmarks", nargs="+", required=True,
                        choices=list(BENCHMARKS.keys()))
    parser.add_argument("--n", type=int, default=None,
                        help="Default examples per sub-task. None (default) = full test set. "
                             "Use --n-per-benchmark to cap only the slow ones.")
    parser.add_argument("--n-per-benchmark", nargs="*", default=[],
                        help="Per-benchmark caps as 'bench=N' pairs, e.g. "
                             "'--n-per-benchmark game24=100 blocksworld=50'. "
                             "Unlisted benchmarks fall back to --n (full test set by default).")
    parser.add_argument("--chat-template", choices=["auto", "always", "never"], default="auto",
                        help="auto: use chat template iff tokenizer has one (Qwen yes, Pythia no). "
                             "Override with always/never for ablations.")
    parser.add_argument("--debug-samples", type=int, default=0,
                        help="If >0, print the first N examples per (benchmark, sub-task) "
                             "with prompt excerpt + gold + pred + per-class log-likelihoods "
                             "(or generation output for gen benchmarks). Useful for spotting "
                             "label-bias collapse or format issues. Reset between models.")
    parser.add_argument("--output", required=True)

    # --- Path configuration (no more hardcoded /scratch paths) ---
    parser.add_argument("--hf-cache", default=None,
                        help="HuggingFace cache dir. Falls back to $HF_HOME.")
    parser.add_argument("--prompts-dir", default=None,
                        help="Directory containing prompt template files. "
                             "Default: 'prompts/' next to this script.")
    parser.add_argument("--checkpoints-root", default=None,
                        help="Root dir containing job_<id>/ subdirs. Required iff "
                             "any MODEL_REGISTRY alias is used (instruct_only, 6pct_L*).")
    parser.add_argument("--data-dir", default=None,
                        help="Dir containing benchmark data subdirs "
                             "(game24_data/, proofwriter_raw/). Required iff "
                             "game24 or proofwriter benchmarks are requested.")

    args = parser.parse_args()

    # Stash paths into module globals so benchmark helpers can read them
    global cache_dir, PROMPTS_DIR, CHECKPOINTS_ROOT, DATA_DIR, DEBUG_SAMPLES
    if args.hf_cache:
        cache_dir = args.hf_cache
    if args.prompts_dir:
        PROMPTS_DIR = args.prompts_dir
    CHECKPOINTS_ROOT = args.checkpoints_root
    DATA_DIR = args.data_dir
    DEBUG_SAMPLES = args.debug_samples

    print(f"Paths:")
    print(f"  hf_cache:         {cache_dir}")
    print(f"  prompts_dir:      {PROMPTS_DIR}")
    print(f"  checkpoints_root: {CHECKPOINTS_ROOT}")
    print(f"  data_dir:         {DATA_DIR}")

    # Parse --n-per-benchmark into {bench: n}
    n_overrides = {}
    for spec in args.n_per_benchmark:
        if "=" not in spec:
            raise ValueError(f"--n-per-benchmark entry '{spec}' must be bench=N")
        b, v = spec.split("=", 1)
        if b not in BENCHMARKS:
            raise ValueError(f"--n-per-benchmark: unknown benchmark '{b}'")
        n_overrides[b] = int(v)

    def n_for(bname):
        return n_overrides.get(bname, args.n)

    print(f"Benchmarks: {args.benchmarks}")
    print(f"n default: {args.n if args.n is not None else 'full test set'}")
    if n_overrides:
        print(f"n overrides: {n_overrides}")

    # Resolve all model specs to (name, path) up-front so we fail fast
    resolved = [resolve_model(spec) for spec in args.models]
    print("Models:")
    for name, path in resolved:
        print(f"  {name} -> {path}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # JSON-safe keys
    def fix(obj):
        if isinstance(obj, dict):
            return {str(k): fix(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [fix(x) for x in obj]
        return obj

    def save(all_results):
        """Atomic write — dump to .tmp then rename — so we never corrupt partial results."""
        tmp = args.output + ".tmp"
        with open(tmp, "w") as f:
            json.dump(fix(all_results), f, indent=2)
        os.replace(tmp, args.output)

    all_results = {}
    for mname, path in resolved:
        print(f"\n{'='*60}\n{mname} ({path})\n{'='*60}")
        # Reset debug counters per model so each model gets its own N samples per benchmark
        _dbg_counters.clear()
        tok, model, use_chat = load_model(path)
        if args.chat_template == "always":
            use_chat = True
        elif args.chat_template == "never":
            use_chat = False
        all_results[mname] = {"_path": path, "_chat_template": use_chat}
        save(all_results)  # persist that we started this model
        for bname in args.benchmarks:
            bn = n_for(bname)
            print(f"\n--- {bname} (chat={use_chat}, n={bn}) ---")
            try:
                all_results[mname][bname] = BENCHMARKS[bname](model, tok, use_chat, n=bn)
            except Exception as e:
                traceback.print_exc()
                all_results[mname][bname] = {"error": str(e)}
            save(all_results)  # persist after each benchmark so preemption doesn't lose the fast ones
        del model
        torch.cuda.empty_cache()

    print(f"\nResults saved to {args.output}")

    # Summary — iterate over RESOLVED short names (not raw specs), since
    # all_results is keyed by resolved name (e.g. "instruct_only", not
    # "instruct_only=/scratch/.../instruct_only").
    resolved_names = [name for name, _ in resolved]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for bname in args.benchmarks:
        print(f"\n{bname}:")
        header = f"  {'sub-task':>20}"
        for mname in resolved_names:
            header += f"  {mname:>18}"
        print(header)

        # Collect all sub-task keys across models
        all_keys = set()
        for mname in resolved_names:
            res = all_results[mname].get(bname, {})
            if isinstance(res, dict):
                all_keys.update(k for k, v in res.items() if isinstance(v, dict) and "accuracy" in v)
        for key in sorted(all_keys, key=lambda k: (str(k).isdigit(), str(k))):
            row = f"  {str(key):>20}"
            for mname in resolved_names:
                acc = all_results[mname].get(bname, {}).get(key, {}).get("accuracy", None)
                row += f"  {acc*100:>17.1f}%" if acc is not None else f"  {'—':>18}"
            print(row)


if __name__ == "__main__":
    main()

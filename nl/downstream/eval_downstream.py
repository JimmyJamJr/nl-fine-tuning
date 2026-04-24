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


# Default: draw K in-context examples from each dataset's train split (matches
# the canonical FOLIO-style protocol used when the original paper's few-shot
# uses CoT we can't use). Set USE_TRAIN_ICES=0 to fall back to the legacy
# hand-written prompt files in prompts/ for ProofWriter / StepGame.
_USE_TRAIN_ICES = os.environ.get("USE_TRAIN_ICES", "1") not in ("0", "", "false", "False")
_TRAIN_ICE_K = int(os.environ.get("TRAIN_ICE_K", "8"))
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
    "6pct_L6":  lambda: _stage_ckpt("8894380", 6),
    "6pct_L8":  lambda: _stage_ckpt("8894380", 8),
    "6pct_L16": lambda: _stage_ckpt("8894380", 16),
    "6pct_L32": lambda: _stage_ckpt("8894380", 32),
    "6pct_L48": lambda: _stage_ckpt("8894380", 48),
    "6pct_L64": lambda: _stage_ckpt("9001346", 64),
    "6pct_L75": lambda: _stage_ckpt("9001346", 75),
    # Qwen 1.7B
    "base_1.7b":    "Qwen/Qwen3-1.7B",
    "6pct_1.7b_L8":  lambda: _stage_ckpt("runpod_qwen17b_L32_20260418_085434", 8),
    "6pct_1.7b_L16": lambda: _stage_ckpt("runpod_qwen17b_L32_20260418_085434", 16),
    "6pct_1.7b_L32": lambda: _stage_ckpt("runpod_qwen17b_L32_20260418_085434", 32),
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


def _score_choices_fused(model, tokenizer, prompt: str, choices, use_chat: bool,
                         baseline_logprobs=None):
    """Run the model ONCE on `prompt` and return both:
       - per-choice log-prob (for log-lik scoring)
       - greedy next-token argmax (for "gen" scoring)
       - calibrated prediction (if `baseline_logprobs` is provided)

    For multi-token choices, falls back to per-choice teacher-forced log-lik
    (still single forward pass per choice).

    Contextual Calibration (Zhao et al. ICML 2021): pass `baseline_logprobs`
    = the same model's logprobs over the same choices on a content-free prompt
    (e.g., replace task input with "N/A"). We subtract those from the per-example
    logprobs before argmax, which cancels out the model's prompt-independent
    class prior.

    Returns dict:
      'logprobs': {choice: float}            log p(" choice" | prompt)
      'loglik_pred': str                     argmax over choices by logprob
      'gen_pred': str                        greedy argmax of next token, mapped to a choice
      'calibrated_pred': str or None         argmax over (logprobs - baseline), if provided
    """
    prefix = _chat_format(tokenizer, prompt) if use_chat else prompt
    choice_token_ids = {}
    for c in choices:
        ids = tokenizer.encode(" " + c, add_special_tokens=False)
        choice_token_ids[c] = ids

    all_single = all(len(ids) == 1 for ids in choice_token_ids.values())

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    inp = torch.tensor([prefix_ids], device=model.device)
    with torch.no_grad():
        logits = model(inp).logits[0, -1, :]  # next-token logits
    log_probs = torch.log_softmax(logits, dim=-1)

    if all_single:
        scores = {c: log_probs[ids[0]].item() for c, ids in choice_token_ids.items()}
    else:
        scores = {c: _loglik_completion(model, tokenizer, prefix, " " + c)
                  for c in choices}

    loglik_pred = max(scores, key=scores.get)

    greedy_id = int(torch.argmax(logits).item())
    greedy_token = tokenizer.decode([greedy_id])
    gen_pred = _extract_label_from_anchored(greedy_token, choices) or loglik_pred

    calibrated_pred = None
    if baseline_logprobs is not None:
        calibrated = {c: scores[c] - baseline_logprobs.get(c, 0.0) for c in choices}
        calibrated_pred = max(calibrated, key=calibrated.get)

    return {"logprobs": scores, "loglik_pred": loglik_pred, "gen_pred": gen_pred,
            "calibrated_pred": calibrated_pred}


def _content_free_baseline(model, tokenizer, cf_prompt: str, choices, use_chat: bool):
    """Compute the model's content-free class prior by running the fused scorer
    on a neutral prompt (same format as real prompts, task-specific content
    replaced by 'N/A' or similar). Used for Contextual Calibration."""
    out = _score_choices_fused(model, tokenizer, cf_prompt, choices, use_chat)
    return out["logprobs"]


def _subsample(items, cap, seed=42):
    """Return up to `cap` items from `items`, sampled randomly with a fixed seed.

    Avoids bias from order-preserving slicing (LogicBench files alternate yes/no,
    ProofWriter groups by question type, etc. — taking first N can give
    class-imbalanced samples). Uses Python's stdlib `random.sample` for
    reproducibility across runs.
    """
    if cap <= 0: return []
    if len(items) <= cap: return list(items)
    import random as _random
    return _random.Random(seed).sample(list(items), cap)


def _distribute_n(n, num_subsets, sizes=None):
    """Distribute a per-benchmark `n` cap across `num_subsets` subsets.

    If `sizes` is provided (list of per-subset available counts), the per-subset
    cap is min(n // num_subsets, size). Any leftover from small subsets is
    redistributed across the larger ones up to their size.

    Returns a list of per-subset caps (length = num_subsets).
    """
    if num_subsets <= 0: return []
    if sizes is None:
        base = n // num_subsets
        rem = n - base * num_subsets
        # Distribute remainder one-per-subset to first `rem` subsets.
        return [base + (1 if i < rem else 0) for i in range(num_subsets)]
    # Adaptive: if a subset has fewer than its share, redistribute leftover.
    caps = [min(sizes[i], n // num_subsets) for i in range(num_subsets)]
    leftover = n - sum(caps)
    # Round-robin add leftover to subsets with headroom
    while leftover > 0:
        progress = False
        for i in range(num_subsets):
            if caps[i] < sizes[i]:
                caps[i] += 1
                leftover -= 1
                progress = True
                if leftover == 0: break
        if not progress: break
    return caps


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


def _yesno(pred):
    s = pred.lower().strip()
    if "yes" in s[:10] or "true" in s[:10]:
        return "yes"
    if "no" in s[:10] or "false" in s[:10]:
        return "no"
    return ""


# ==========================================================
# Shared constants and helpers (added 2026-04-24 overhaul)
# ==========================================================

DEFAULT_MAX_N = 1000  # cap for benchmarks with large test sets

# CLUTRR's 21 kinship labels (Sinha 2019)
CLUTRR_LABELS = [
    "aunt", "uncle", "father", "mother", "brother", "sister",
    "son", "daughter", "grandmother", "grandfather",
    "granddaughter", "grandson", "niece", "nephew",
    "husband", "wife", "father-in-law", "mother-in-law",
    "son-in-law", "daughter-in-law", "sister-in-law",
]
FOLIO_LABELS = ["True", "False", "Uncertain"]
LOGIQA_LABELS = ["A", "B", "C", "D"]
RULETAKER_LABELS = ["True", "False"]


def _clamp_n(n, max_n=DEFAULT_MAX_N):
    """Cap n at DEFAULT_MAX_N=1000 unless explicitly set lower."""
    if n is None:
        return max_n
    return min(n, max_n)


def _extract_label_from_anchored(text, candidates, anchors=None):
    """Extract label after a matching anchor phrase, falling back to last-occurrence
    or first-occurrence search. Used for generation+regex scoring across multiple
    benchmarks (FOLIO, LogiQA, ProofWriter-gen, etc.).

    For single-character candidates (e.g. ['A','B','C','D']) we require a word
    boundary to avoid matching arbitrary letters inside words like "analyze"
    or "answer".

    Args:
        text: model's generated text
        candidates: list of valid label strings (e.g. ["True","False","Uncertain"])
        anchors: regex-like anchor phrases to look for first. If found, take the
            text immediately after.
    Returns the matched candidate label (preserving case from candidates list) or None.
    """
    import re as _re
    if anchors is None:
        anchors = [r"answer\s*(?:is)?\s*:?\s*", r"final answer\s*:?\s*",
                   r"the correct option is\s*:?\s*", r"therefore,?\s+"]
    text_low = text.strip().lower()
    cands_sorted = sorted([c.lower() for c in candidates], key=lambda s: -len(s))
    cand_orig = {c.lower(): c for c in candidates}
    single_char = all(len(c) == 1 for c in cands_sorted)

    def _find(cand, hay):
        """Substring search with word-boundary enforcement for single chars."""
        if single_char:
            m = _re.search(r"(?<![a-z0-9])" + _re.escape(cand) + r"(?![a-z0-9])", hay)
            return m.start() if m else -1
        return hay.find(cand)

    def _rfind(cand, hay):
        if single_char:
            matches = list(_re.finditer(
                r"(?<![a-z0-9])" + _re.escape(cand) + r"(?![a-z0-9])", hay))
            return matches[-1].start() if matches else -1
        return hay.rfind(cand)

    # 1. Anchored match
    for anchor in anchors:
        m = _re.search(anchor, text_low)
        if m:
            tail = text_low[m.end():m.end() + 64]
            best_pos, best_cand = -1, None
            for cand in cands_sorted:
                p = _find(cand, tail)
                if p >= 0 and (best_pos < 0 or p < best_pos):
                    best_pos, best_cand = p, cand
            if best_cand is not None:
                return cand_orig[best_cand]
    # 2. Last-occurrence match (scan from right)
    last_pos, last_label = -1, None
    for cand in cands_sorted:
        pos = _rfind(cand, text_low)
        if pos > last_pos:
            last_pos, last_label = pos, cand
    if last_label is not None:
        return cand_orig[last_label]
    # 3. First-occurrence in head (legacy fallback)
    head = text_low[:64]
    best_pos, best_cand = -1, None
    for cand in cands_sorted:
        p = _find(cand, head)
        if p >= 0 and (best_pos < 0 or p < best_pos):
            best_pos, best_cand = p, cand
    return cand_orig[best_cand] if best_cand else None


def _format_few_shot_block(examples, format_fn):
    """Build a few-shot prompt block from a list of (input, output) examples.

    examples: list of dicts with whatever fields format_fn expects.
    format_fn(ex) -> str returning the formatted Q+A for that example.
    Returns a single string with all examples concatenated by blank lines.
    """
    blocks = [format_fn(ex) for ex in examples]
    return "\n\n".join(blocks) + "\n\n"


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
def eval_zebra_mc(model, tokenizer, use_chat, n=None):
    """ZebraLogic mc_mode — single forward pass returns BOTH log-lik and
    constrained-greedy gen accuracy. Choices are puzzle-specific (5-6 names
    per puzzle, multi-token). n is total cap distributed across 25 puzzle sizes.
    """
    from datasets import load_dataset
    n = _clamp_n(n)
    try:
        ds = load_dataset("WildEval/ZebraLogic", "mc_mode", split="test")
    except Exception as e:
        return {"error": f"load_dataset failed: {e}"}
    print(f"  [zebra_mc] loaded WildEval/ZebraLogic/mc_mode: {len(ds)} questions")

    def extract_size(ex_id):
        try: return ex_id.split("-")[2].replace("x", "*")
        except Exception: return "?"

    by_size = {}
    for ex in ds:
        by_size.setdefault(extract_size(ex["id"]), []).append(ex)
    sizes = sorted(by_size.keys(), key=lambda s: tuple(int(x) for x in s.split("*"))
                                                 if "*" in s else (99,99))
    size_counts = [len(by_size[s]) for s in sizes]
    caps = _distribute_n(n, len(sizes), size_counts)

    # Content-free baseline per puzzle size (choice set varies per example,
    # so we use each puzzle's own choices with a content-free prompt).
    results = {}
    for size, cap in zip(sizes, caps):
        ll_c = gen_c = cal_c = total = 0
        for ex in _subsample(by_size[size], cap):
            prompt = f"{ex['puzzle']}\n\nQuestion: {ex['question']}\nAnswer:"
            choices = [str(c) for c in ex["choices"]]
            # Per-example baseline since choice set varies across puzzles
            cf_prompt = f"N/A\n\nQuestion: N/A\nAnswer:"
            baseline_lp = _content_free_baseline(model, tokenizer, cf_prompt, choices, use_chat)
            out = _score_choices_fused(model, tokenizer, prompt, choices, use_chat,
                                       baseline_logprobs=baseline_lp)
            _dbg_log_loglik(f"zebra_mc:{size}", prompt, ex["answer"],
                            out["loglik_pred"], out["logprobs"])
            if out["loglik_pred"] == ex["answer"]: ll_c += 1
            if out["gen_pred"] == ex["answer"]: gen_c += 1
            if out["calibrated_pred"] == ex["answer"]: cal_c += 1
            total += 1
        results[size] = {"loglik_accuracy":     ll_c/total if total else 0.0,
                          "gen_accuracy":        gen_c/total if total else 0.0,
                          "calibrated_accuracy": cal_c/total if total else 0.0,
                          "total": total,
                          "accuracy": ll_c/total if total else 0.0}
        print(f"  [zebra_mc] size={size:>5} n={total:>3}  ll={ll_c/max(total,1):.1%}  "
              f"gen={gen_c/max(total,1):.1%}  cal={cal_c/max(total,1):.1%}")
    all_ll = sum(v["loglik_accuracy"] * v["total"] for v in results.values())
    all_gen = sum(v["gen_accuracy"] * v["total"] for v in results.values())
    all_cal = sum(v["calibrated_accuracy"] * v["total"] for v in results.values())
    all_t = sum(v["total"] for v in results.values())
    results["overall"] = {"loglik_accuracy":     all_ll/all_t if all_t else 0.0,
                           "gen_accuracy":        all_gen/all_t if all_t else 0.0,
                           "calibrated_accuracy": all_cal/all_t if all_t else 0.0,
                           "total": all_t,
                           "accuracy": all_ll/all_t if all_t else 0.0}
    return results


# ---------------- LegalBench ----------------

# LegalBench task adapters. Updated 2026-04-24 — switched from
# {hearsay, international_citizenship_questions, proa} (1-hop, dominated by
# majority-class baseline) to {diversity_1..6 + sara_entailment + sara_numeric}
# which test multi-condition reasoning (diversity) and statutory interpretation
# (sara). Templates are downloaded from HazyResearch/legalbench's base_prompt.txt
# on first use and cached locally under PROMPTS_DIR/legal/.
#
# Per-task scoring:
#   diversity_1..6   -> binary Yes/No (exact-match after normalization)
#   sara_entailment  -> 2-class {Entailment, Contradiction}
#   sara_numeric     -> first integer extracted, ±10% tolerance vs gold
LEGAL_TASKS = {
    "diversity_1": {"input_col": "text", "answer_col": "answer",
                    "template_path": "legal/diversity_1.txt", "score_type": "yesno"},
    "diversity_2": {"input_col": "text", "answer_col": "answer",
                    "template_path": "legal/diversity_2.txt", "score_type": "yesno"},
    "diversity_3": {"input_col": "text", "answer_col": "answer",
                    "template_path": "legal/diversity_3.txt", "score_type": "yesno"},
    "diversity_4": {"input_col": "text", "answer_col": "answer",
                    "template_path": "legal/diversity_4.txt", "score_type": "yesno"},
    "diversity_5": {"input_col": "text", "answer_col": "answer",
                    "template_path": "legal/diversity_5.txt", "score_type": "yesno"},
    "diversity_6": {"input_col": "text", "answer_col": "answer",
                    "template_path": "legal/diversity_6.txt", "score_type": "yesno"},
    "sara_entailment": {"input_col": "text", "answer_col": "answer",
                        "template_path": "legal/sara_entailment.txt",
                        "score_type": "entailment"},
    "sara_numeric": {"input_col": "text", "answer_col": "answer",
                     "template_path": "legal/sara_numeric.txt",
                     "score_type": "numeric"},
}

LEGAL_ENTAILMENT_LABELS = ["Entailment", "Contradiction"]


def _ensure_legal_template(spec):
    """Download the base_prompt.txt for a LegalBench task from GitHub if not local."""
    import os as _os, urllib.request as _ureq
    rel = spec["template_path"]
    full = _os.path.join(PROMPTS_DIR, rel)
    if _os.path.isfile(full):
        return
    task_name = _os.path.splitext(_os.path.basename(rel))[0]
    url = f"https://raw.githubusercontent.com/HazyResearch/legalbench/main/tasks/{task_name}/base_prompt.txt"
    _os.makedirs(_os.path.dirname(full), exist_ok=True)
    try:
        _ureq.urlretrieve(url, full)
        print(f"  [legal] Downloaded template: {rel}")
    except Exception as e:
        print(f"  [legal] WARNING: failed to download {url}: {e}")


def _legal_normalize(text, stem=False):
    """Port of HazyResearch/legalbench/evaluation.py:normalize() — strip
    punctuation, lowercase, optionally Porter-stem. Matches their canonical
    exact-match-balanced-accuracy normalization."""
    import string
    text = str(text).translate(str.maketrans("", "", string.punctuation))
    text = text.strip().lower()
    if stem:
        try:
            from nltk.stem.porter import PorterStemmer
            text = PorterStemmer().stem(text)
        except Exception: pass
    return text


def _legal_balanced_accuracy(generations, answers):
    """Port of evaluate_exact_match_balanced_accuracy() — sklearn balanced acc
    on normalized strings. Falls back to plain accuracy if sklearn missing."""
    norm_gen = [_legal_normalize(g) for g in generations]
    norm_ans = [_legal_normalize(a) for a in answers]
    try:
        from sklearn.metrics import balanced_accuracy_score
        return balanced_accuracy_score(norm_ans, norm_gen)
    except ImportError:
        # Fallback: plain accuracy
        if not norm_ans: return 0.0
        return sum(1 for g, a in zip(norm_gen, norm_ans) if g == a) / len(norm_ans)


def _score_legal_numeric(pred_text, gold_text):
    """Match HazyResearch/legalbench/evaluation.py:evaluate_sara_numeric_acc.
    Strip commas/dollars, regex first integer, accept if within 10% of gold."""
    import re as _re
    def _to_int(s):
        s = str(s).replace(",", "").replace("$", "").replace(".", "")
        m = _re.search(r"\d+", s)
        return int(m.group(0)) if m else None
    pred = _to_int(pred_text)
    gold = _to_int(gold_text)
    if pred is None or gold is None or gold == 0:
        return False
    return abs(pred / gold - 1.0) < 0.1


def _legal_template(spec):
    """Lazy-load the prompt template — `_load_prompt` reads PROMPTS_DIR which
    isn't finalized until main() parses --prompts-dir."""
    return _load_prompt(spec["template_path"])


@register("legal")
def eval_legal(model, tokenizer, use_chat, n=None):
    """LegalBench (canonical eval) — diversity_1..6 + sara_entailment + sara_numeric.

    Uses HazyResearch/legalbench's official author-written 6-shot demos
    (downloaded as base_prompt.txt per task) and the canonical evaluate()
    function: balanced_accuracy on normalized strings for diversity_* and
    sara_entailment; ±10% tolerance for sara_numeric. Generation-based:
    model emits a full string answer, we normalize+match (no log-lik fallback).

    n is total cap distributed evenly across the 8 subtasks.
    """
    from datasets import load_dataset
    n = _clamp_n(n)
    tasks = list(LEGAL_TASKS.keys())
    # Pre-load all subsets to compute sizes
    loaded = {}
    for task in tasks:
        try:
            loaded[task] = load_dataset("nguha/legalbench", task, split="test")
        except Exception as e:
            print(f"  [legal:{task}] load failed: {e}")
            loaded[task] = None
    sizes = [len(loaded[t]) if loaded[t] is not None else 0 for t in tasks]
    caps = _distribute_n(n, len(tasks), sizes)

    results = {}
    for task, cap in zip(tasks, caps):
        spec = LEGAL_TASKS[task]
        ds = loaded[task]
        if ds is None or len(ds) == 0:
            results[task] = {"error": "load failed or empty"}
            continue
        _ensure_legal_template(spec)
        ans_col = spec["answer_col"]
        if ans_col not in ds.column_names:
            ans_col = "label" if "label" in ds.column_names else ds.column_names[-1]
        score_type = spec.get("score_type", "yesno")

        generations, answers = [], []
        for ex in _subsample(list(ds), cap):
            text = ex.get(spec["input_col"])
            gold = ex.get(ans_col)
            if text is None or gold is None: continue
            try:
                prompt = _load_prompt(spec["template_path"]).format(input=text)
            except Exception:
                continue
            # Canonical: full generation, then normalize+exact-match
            max_tok = 20 if score_type == "numeric" else 10
            pred = generate_chat(model, tokenizer, prompt, max_new_tokens=max_tok,
                                 use_chat=use_chat)
            _dbg_log_gen(f"legal:{task}", prompt, gold, pred)
            generations.append(pred)
            answers.append(gold)

        if not generations:
            results[task] = {"error": "no valid examples"}
            continue
        if score_type == "numeric":
            corrects = [1 if _score_legal_numeric(g, a) else 0
                        for g, a in zip(generations, answers)]
            acc = sum(corrects) / len(corrects)
        else:
            acc = _legal_balanced_accuracy(generations, answers)
        results[task] = {"accuracy": acc, "total": len(generations)}
        print(f"  [legal:{task}]: balanced_acc={acc:.3f} over {len(generations)} samples")

    per_task_accs = [v["accuracy"] for v in results.values() if "accuracy" in v]
    macro = sum(per_task_accs) / len(per_task_accs) if per_task_accs else 0.0
    all_total = sum(v.get("total", 0) for v in results.values() if "total" in v)
    results["overall"] = {"accuracy": macro, "total": all_total}
    print(f"  [legal] macro balanced_acc: {macro:.3f} over {all_total} samples")
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
        ds = load_dataset("ZhengyanShi/StepGame", split="test")
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


# legal_gen merged into legal — the LegalBench canonical eval IS gen-based
# (balanced_accuracy on normalized strings), no separate gen variant needed.


# zebra_mc_gen merged into zebra_mc — fused scorer returns both metrics in one pass.


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
    QDep values (0 through 5) from theories that require up to 5-hop reasoning."""
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


def _proofwriter_few_shot_block(mode):
    """Produce ProofWriter few-shot block. Samples K balanced demos from
    `meta-test.jsonl` of a depth different from the one being evaluated
    (we eval on depth-5; demos come from depth-3). This avoids overlap with
    the test sweep while keeping demos in the canonical AllenAI format.

    Falls back to the static prompts/ .txt file if raw data isn't available.
    mode is 'OWA' or 'CWA'.
    """
    cache_key = f"proofwriter_{mode}"
    if cache_key in _train_ice_cache:
        return _train_ice_cache[cache_key]
    if not _USE_TRAIN_ICES:
        block = _load_prompt(f"proofwriter_{mode.lower()}_fewshot.txt")
        _train_ice_cache[cache_key] = block
        return block

    root = _find_proofwriter_raw()
    if root is None:
        block = _load_prompt(f"proofwriter_{mode.lower()}_fewshot.txt")
        _train_ice_cache[cache_key] = block
        return block

    # Try meta-train.jsonl first (canonical), fall back to meta-test.jsonl from
    # depth-3 (we eval on depth-5 so no overlap).
    p = None
    for depth_dir in ("depth-3", "depth-2", "depth-1"):
        for fname in ("meta-train.jsonl", "meta-test.jsonl"):
            cand = os.path.join(root, mode, depth_dir, fname)
            if os.path.isfile(cand):
                p = cand
                break
        if p: break
    if p is None:
        block = _load_prompt(f"proofwriter_{mode.lower()}_fewshot.txt")
        _train_ice_cache[cache_key] = block
        return block

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
                if ans is None: continue
                if mode == "CWA" and ans not in ("True", "False"): continue
                if label_counts[ans] >= want_per_label: continue
                label_counts[ans] += 1
                examples.append((theory, q.get("question", ""), ans))
                if len(examples) >= _TRAIN_ICE_K: break
    closed = " (closed-world assumption)" if mode == "CWA" else ""
    tf_u = "true or false" if mode == "CWA" else "True, False, or Unknown"
    blocks = [
        f"Facts and rules:\n{theory}\n"
        f"Question: Based only on the facts and rules above{closed}, "
        f"is the following statement {tf_u}?\n"
        f"Statement: {qtext}\nAnswer: {ans}"
        for theory, qtext, ans in examples[:_TRAIN_ICE_K]
    ]
    block = "\n\n".join(blocks) + "\n\n"
    _train_ice_cache[cache_key] = block
    return block


# proofwriter_gen + proofwriter_cwa_gen are merged into proofwriter / proofwriter_cwa
# (see fused-scorer impl below). Each example now does ONE forward pass and the
# adapter returns BOTH loglik_accuracy and gen_accuracy in its results dict.


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
            ds = load_dataset("ZhengyanShi/StepGame", split="train")
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
@register("proofwriter")
def eval_proofwriter(model, tokenizer, use_chat, n=None):
    """ProofWriter OWA 3-class (True/False/Unknown). Single forward pass per
    example returns BOTH log-lik and constrained-greedy gen accuracy.
    Per-depth breakdown; n is total cap distributed evenly across depths.
    """
    n = _clamp_n(n)
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
    depths = [d for d in sorted(by_depth.keys()) if d is not None and d <= 5]
    sizes = [len(by_depth[d]) for d in depths]
    caps = _distribute_n(n, len(depths), sizes)

    # Content-free prompt for Contextual Calibration (Zhao et al. 2021)
    cf_prompt = (f"Facts and rules:\nN/A\n\n"
                 f"Based only on the facts and rules above, is the following "
                 f"statement True, False, or Unknown (meaning it can be neither "
                 f"proven nor disproven)?\nN/A\n"
                 f"Answer with only True, False, or Unknown.")
    baseline_lp = _content_free_baseline(model, tokenizer, cf_prompt,
                                          ["True", "False", "Unknown"], use_chat)
    print(f"  [proofwriter] content-free baseline logprobs: "
          f"True={baseline_lp['True']:.3f} False={baseline_lp['False']:.3f} Unknown={baseline_lp['Unknown']:.3f}")

    results = {}
    for depth, cap in zip(depths, caps):
        ll_c = gen_c = cal_c = total = 0
        by_class = {"True": [0, 0], "False": [0, 0], "Unknown": [0, 0]}
        for ex in _subsample(by_depth[depth], cap):
            answer = ex["answer"]
            prompt = (f"Facts and rules:\n{ex['theory']}\n\n"
                      f"Based only on the facts and rules above, is the following "
                      f"statement True, False, or Unknown (meaning it can be neither "
                      f"proven nor disproven)?\n{ex['question']}\n"
                      f"Answer with only True, False, or Unknown.")
            out = _score_choices_fused(model, tokenizer, prompt,
                                       ["True", "False", "Unknown"], use_chat,
                                       baseline_logprobs=baseline_lp)
            _dbg_log_loglik(f"proofwriter:depth_{depth}", prompt, answer,
                            out["loglik_pred"], out["logprobs"])
            by_class[answer][1] += 1
            if out["loglik_pred"] == answer:
                ll_c += 1
                by_class[answer][0] += 1
            if out["gen_pred"] == answer: gen_c += 1
            if out["calibrated_pred"] == answer: cal_c += 1
            total += 1
        results[depth] = {
            "loglik_accuracy":     ll_c/total if total else 0.0,
            "gen_accuracy":        gen_c/total if total else 0.0,
            "calibrated_accuracy": cal_c/total if total else 0.0,
            "total": total,
            "accuracy": ll_c/total if total else 0.0,
            "per_class_loglik": {
                cls: {"correct": c[0], "total": c[1],
                      "accuracy": c[0]/c[1] if c[1] else 0.0}
                for cls, c in by_class.items()},
        }
        print(f"  [proofwriter] depth={depth}: ll={ll_c}/{total}={ll_c/max(total,1):.1%}  "
              f"gen={gen_c}/{total}={gen_c/max(total,1):.1%}  "
              f"cal={cal_c}/{total}={cal_c/max(total,1):.1%}")
    all_ll = sum(r["loglik_accuracy"] * r["total"] for r in results.values())
    all_gen = sum(r["gen_accuracy"] * r["total"] for r in results.values())
    all_cal = sum(r["calibrated_accuracy"] * r["total"] for r in results.values())
    all_t = sum(r["total"] for r in results.values())
    results["overall"] = {"loglik_accuracy":     all_ll/all_t if all_t else 0.0,
                           "gen_accuracy":        all_gen/all_t if all_t else 0.0,
                           "calibrated_accuracy": all_cal/all_t if all_t else 0.0,
                           "total": all_t,
                           "accuracy": all_ll/all_t if all_t else 0.0,
                           "baseline_logprobs": baseline_lp}
    return results


@register("proofwriter_cwa")
def eval_proofwriter_cwa(model, tokenizer, use_chat, n=None):
    """ProofWriter CWA binary (True/False). Single forward pass returns BOTH
    log-lik and constrained-greedy gen accuracy. n is total cap distributed
    across depths."""
    n = _clamp_n(n)
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
    depths = [d for d in sorted(by_depth.keys()) if d is not None and d <= 5]
    sizes = [len(by_depth[d]) for d in depths]
    caps = _distribute_n(n, len(depths), sizes)

    cf_prompt = (f"Facts and rules:\nN/A\n\n"
                 f"Based only on the facts and rules above, is the following "
                 f"statement true or false? Assume anything not provable is False.\n"
                 f"N/A\nAnswer with only True or False.")
    baseline_lp = _content_free_baseline(model, tokenizer, cf_prompt,
                                          ["True", "False"], use_chat)
    print(f"  [proofwriter_cwa] baseline: True={baseline_lp['True']:.3f} False={baseline_lp['False']:.3f}")

    results = {}
    for depth, cap in zip(depths, caps):
        ll_c = gen_c = cal_c = total = 0
        by_class = {"True": [0, 0], "False": [0, 0]}
        for ex in _subsample(by_depth[depth], cap):
            answer = ex["answer"]
            prompt = (f"Facts and rules:\n{ex['theory']}\n\n"
                      f"Based only on the facts and rules above, is the following "
                      f"statement true or false? Assume anything not provable is False.\n"
                      f"{ex['question']}\nAnswer with only True or False.")
            out = _score_choices_fused(model, tokenizer, prompt,
                                       ["True", "False"], use_chat,
                                       baseline_logprobs=baseline_lp)
            _dbg_log_loglik(f"proofwriter_cwa:depth_{depth}", prompt, answer,
                            out["loglik_pred"], out["logprobs"])
            by_class[answer][1] += 1
            if out["loglik_pred"] == answer:
                ll_c += 1
                by_class[answer][0] += 1
            if out["gen_pred"] == answer: gen_c += 1
            if out["calibrated_pred"] == answer: cal_c += 1
            total += 1
        results[depth] = {
            "loglik_accuracy":     ll_c/total if total else 0.0,
            "gen_accuracy":        gen_c/total if total else 0.0,
            "calibrated_accuracy": cal_c/total if total else 0.0,
            "total": total,
            "accuracy": ll_c/total if total else 0.0,
            "per_class_loglik": {
                cls: {"correct": c[0], "total": c[1],
                      "accuracy": c[0]/c[1] if c[1] else 0.0}
                for cls, c in by_class.items()},
        }
        print(f"  [proofwriter_cwa] depth={depth}: ll={ll_c}/{total}={ll_c/max(total,1):.1%}  "
              f"gen={gen_c}/{total}={gen_c/max(total,1):.1%}  "
              f"cal={cal_c}/{total}={cal_c/max(total,1):.1%}")
    all_ll = sum(r["loglik_accuracy"] * r["total"] for r in results.values())
    all_gen = sum(r["gen_accuracy"] * r["total"] for r in results.values())
    all_cal = sum(r["calibrated_accuracy"] * r["total"] for r in results.values())
    all_t = sum(r["total"] for r in results.values())
    results["overall"] = {"loglik_accuracy":     all_ll/all_t if all_t else 0.0,
                           "gen_accuracy":        all_gen/all_t if all_t else 0.0,
                           "calibrated_accuracy": all_cal/all_t if all_t else 0.0,
                           "total": all_t,
                           "accuracy": all_ll/all_t if all_t else 0.0,
                           "baseline_logprobs": baseline_lp}
    return results


# ==========================================================
# FOLIO (Han et al., EMNLP 2024)
# ==========================================================
# 3-class FOL deduction (True/False/Uncertain). Paper protocol: 8-shot demos +
# generation, extract "The answer is X". We provide both log-lik and gen variants.
# Dataset: yale-nlp/FOLIO (gated; users must accept license on HF first).

_FOLIO_FEW_SHOT_CACHE = None

def _folio_few_shot_block(tokenizer, k=8):
    """Build a k-shot demo block from FOLIO train split.
    Demos formatted as: Premises, Conclusion, Answer."""
    global _FOLIO_FEW_SHOT_CACHE
    if _FOLIO_FEW_SHOT_CACHE is not None:
        return _FOLIO_FEW_SHOT_CACHE
    from datasets import load_dataset
    try:
        ds = load_dataset("yale-nlp/FOLIO", split="train")
    except Exception as e:
        try:
            ds = load_dataset("tasksource/folio", split="train")
        except Exception:
            print(f"  [folio] Could not load train split for few-shot demos: {e}")
            _FOLIO_FEW_SHOT_CACHE = ""
            return ""
    # Pick balanced demos (try to cover all 3 labels)
    demos = []
    seen = set()
    for ex in ds:
        lbl = str(ex.get("label", "")).strip().capitalize().replace("Unknown", "Uncertain")
        if lbl not in FOLIO_LABELS or lbl in seen and len(seen) < 3:
            continue
        seen.add(lbl)
        prem = ex.get("premises", [])
        if isinstance(prem, list):
            prem_text = "\n".join(prem)
        else:
            prem_text = str(prem)
        conc = ex.get("conclusion", "")
        demos.append(f"Premises:\n{prem_text}\nConclusion: {conc}\nAnswer: {lbl}")
        if len(demos) >= k:
            break
    _FOLIO_FEW_SHOT_CACHE = "\n\n".join(demos) + "\n\n" if demos else ""
    return _FOLIO_FEW_SHOT_CACHE


def _folio_format_test(ex):
    prem = ex.get("premises", [])
    if isinstance(prem, list):
        prem_text = "\n".join(prem)
    else:
        prem_text = str(prem)
    return (f"Premises:\n{prem_text}\n"
            f"Conclusion: {ex.get('conclusion','')}\nAnswer:")


def _folio_load(split="validation"):
    from datasets import load_dataset
    for repo in ("yale-nlp/FOLIO", "tasksource/folio"):
        try:
            return load_dataset(repo, split=split)
        except Exception:
            continue
    return None


@register("folio")
def eval_folio(model, tokenizer, use_chat, n=None):
    """FOLIO — canonical 8-shot from train (Han et al. 2024 protocol).
    Single forward pass returns BOTH log-lik and constrained-greedy gen
    accuracy over {True, False, Uncertain}."""
    n = _clamp_n(n)
    ds = _folio_load("validation")
    if ds is None: return {"error": "FOLIO load failed (try `huggingface-cli login`)"}
    print(f"  [folio] loaded {len(ds)} validation examples")
    fs = _folio_few_shot_block(tokenizer, k=8)
    cf_prompt = fs + _folio_format_test({"premises": ["N/A"], "conclusion": "N/A"})
    baseline_lp = _content_free_baseline(model, tokenizer, cf_prompt, FOLIO_LABELS, use_chat)
    print(f"  [folio] baseline: " + " ".join(f"{l}={baseline_lp[l]:.3f}" for l in FOLIO_LABELS))

    ll_c = gen_c = cal_c = total = 0
    by_label = {l: [0, 0] for l in FOLIO_LABELS}
    for ex in _subsample(list(ds), n):
        gold_raw = str(ex.get("label", "")).strip().capitalize()
        gold = gold_raw.replace("Unknown", "Uncertain")
        if gold not in FOLIO_LABELS: continue
        prompt = fs + _folio_format_test(ex)
        out = _score_choices_fused(model, tokenizer, prompt, FOLIO_LABELS, use_chat,
                                   baseline_logprobs=baseline_lp)
        _dbg_log_loglik("folio", prompt, gold, out["loglik_pred"], out["logprobs"])
        by_label[gold][1] += 1
        if out["loglik_pred"] == gold:
            ll_c += 1
            by_label[gold][0] += 1
        if out["gen_pred"] == gold: gen_c += 1
        if out["calibrated_pred"] == gold: cal_c += 1
        total += 1
    return {"loglik_accuracy":     ll_c/total if total else 0.0,
            "gen_accuracy":        gen_c/total if total else 0.0,
            "calibrated_accuracy": cal_c/total if total else 0.0,
            "total": total,
            "accuracy": ll_c/total if total else 0.0,
            "per_label_loglik": {l: (c/t if t else 0.0) for l,(c,t) in by_label.items()},
            "baseline_logprobs": baseline_lp}


# ==========================================================
# LogiQA (Liu et al., IJCAI 2020)
# ==========================================================
# 4-way multi-choice from civil servant exams. lm-eval-harness uses
# log-likelihood as canonical (OpenLLM Leaderboard convention).

def _logiqa_load():
    """Load LogiQA test set. hails/agieval-logiqa-en is the only parquet-based
    mirror still loadable (script-based EleutherAI/logiqa was deprecated)."""
    from datasets import load_dataset
    for repo in ("hails/agieval-logiqa-en",):
        try:
            return load_dataset(repo, split="test")
        except Exception:
            continue
    return None


def _logiqa_format(ex):
    """Format LogiQA prompt. agieval-logiqa-en provides a pre-formatted `query`
    field already ending with 'A: Among A through D, the answer is' — use it
    directly to match canonical AGIEval evaluation. Falls back to building from
    fields for other dataset versions."""
    q = ex.get("query")
    if q and "the answer is" in q.lower():
        return q  # canonical agieval format, no extra trailing space
    ctx = ex.get("context") or ex.get("passage", "")
    qtext = q or ex.get("question", "")
    opts = ex.get("options") or ex.get("choices", [])
    if isinstance(opts, str):
        try:
            import ast
            opts = ast.literal_eval(opts)
        except Exception: opts = []
    if isinstance(opts, dict):
        opts = [opts.get(k, "") for k in sorted(opts.keys())]
    while len(opts) < 4: opts.append("")
    return (f"Passage: {ctx}\nQuestion: {qtext}\nChoices:\n"
            f"A. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\nAnswer:")


def _logiqa_gold_letter(ex):
    """Normalize gold to A/B/C/D. Handles agieval `gold` (list of int index)
    plus older fields like `correct_option`, `label`, `answer`."""
    g = ex.get("gold")
    if g is None: g = ex.get("correct_option")
    if g is None: g = ex.get("label")
    if g is None: g = ex.get("answer")
    if isinstance(g, str):
        try:
            import ast
            g = ast.literal_eval(g)
        except Exception: pass
    if isinstance(g, list) and g:
        g = g[0]
    if isinstance(g, int): return "ABCD"[g] if 0 <= g < 4 else None
    if isinstance(g, str):
        g = g.strip().upper()
        if g in "ABCD": return g
        if g.isdigit() and int(g) < 4: return "ABCD"[int(g)]
    return None


@register("logiqa")
def eval_logiqa(model, tokenizer, use_chat, n=None):
    """LogiQA — single forward pass per example, returns BOTH log-lik and
    constrained-greedy gen accuracy (single-token A/B/C/D labels).
    0-shot, lm-eval-harness convention."""
    n = _clamp_n(n)
    ds = _logiqa_load()
    if ds is None: return {"error": "LogiQA load failed"}
    print(f"  [logiqa] loaded {len(ds)} test examples")
    cf_ex = {"query": ("N/A\nQ: N/A Answer Choices: (A)N/A (B)N/A (C)N/A (D)N/A\n"
                       "A: Among A through D, the answer is")}
    cf_prompt = _logiqa_format(cf_ex)
    baseline_lp = _content_free_baseline(model, tokenizer, cf_prompt, LOGIQA_LABELS, use_chat)
    print(f"  [logiqa] baseline: " + " ".join(f"{l}={baseline_lp[l]:.3f}" for l in LOGIQA_LABELS))

    ll_c = gen_c = cal_c = total = 0
    for ex in _subsample(list(ds), n):
        gold = _logiqa_gold_letter(ex)
        if gold is None: continue
        prompt = _logiqa_format(ex)
        out = _score_choices_fused(model, tokenizer, prompt, LOGIQA_LABELS, use_chat,
                                   baseline_logprobs=baseline_lp)
        _dbg_log_loglik("logiqa", prompt, gold, out["loglik_pred"], out["logprobs"])
        if out["loglik_pred"] == gold: ll_c += 1
        if out["gen_pred"] == gold: gen_c += 1
        if out["calibrated_pred"] == gold: cal_c += 1
        total += 1
    return {"loglik_accuracy":     ll_c/total if total else 0.0,
            "gen_accuracy":        gen_c/total if total else 0.0,
            "calibrated_accuracy": cal_c/total if total else 0.0,
            "total": total,
            "accuracy": ll_c/total if total else 0.0,
            "baseline_logprobs": baseline_lp}


# ==========================================================
# RuleTaker (Clark et al., IJCAI 2020) — predecessor of ProofWriter
# ==========================================================
# 2-class True/False (entailment) per depth D0-D5.

def _ruletaker_load_by_depth():
    from datasets import load_dataset
    by_depth = {}
    try:
        ds = load_dataset("tasksource/ruletaker", split="test")
    except Exception as e:
        try:
            ds = load_dataset("hitachi-nlp/ruletaker", split="test")
        except Exception as e2:
            print(f"  [ruletaker] load failed: {e}; {e2}")
            return None
    cfg_field = "config" if "config" in ds.column_names else None
    for ex in ds:
        d = ex.get(cfg_field, "depth-?") if cfg_field else "depth-?"
        by_depth.setdefault(str(d), []).append(ex)
    return by_depth


def _ruletaker_format(ex):
    return (f"Rules and facts:\n{ex.get('context','')}\n\n"
            f"Statement: {ex.get('question','')}\n"
            f"Based only on the rules and facts above, is the statement true or false?\n"
            f"Answer:")


def _ruletaker_gold(ex):
    lbl = str(ex.get("label", "")).strip().lower()
    if lbl in ("true", "entailment", "1"): return "True"
    if lbl in ("false", "not entailment", "not_entailment", "0"): return "False"
    return None


@register("ruletaker")
def eval_ruletaker(model, tokenizer, use_chat, n=None):
    """RuleTaker — single forward pass per example returns BOTH log-lik and
    constrained-greedy gen accuracy over {True, False}. Per-depth breakdown;
    n is total cap distributed evenly across depths."""
    n = _clamp_n(n)
    by_depth = _ruletaker_load_by_depth()
    if by_depth is None: return {"error": "load failed"}
    cfgs = sorted(by_depth.keys())
    sizes = [len(by_depth[c]) for c in cfgs]
    caps = _distribute_n(n, len(cfgs), sizes)

    cf_ex = {"context": "N/A", "question": "N/A"}
    cf_prompt = _ruletaker_format(cf_ex)
    baseline_lp = _content_free_baseline(model, tokenizer, cf_prompt, RULETAKER_LABELS, use_chat)
    print(f"  [ruletaker] baseline: True={baseline_lp['True']:.3f} False={baseline_lp['False']:.3f}")

    results = {}
    for cfg, cap in zip(cfgs, caps):
        ll_c = gen_c = cal_c = total = 0
        for ex in _subsample(by_depth[cfg], cap):
            gold = _ruletaker_gold(ex)
            if gold is None: continue
            prompt = _ruletaker_format(ex)
            out = _score_choices_fused(model, tokenizer, prompt, RULETAKER_LABELS, use_chat,
                                       baseline_logprobs=baseline_lp)
            _dbg_log_loglik(f"ruletaker:{cfg}", prompt, gold, out["loglik_pred"], out["logprobs"])
            if out["loglik_pred"] == gold: ll_c += 1
            if out["gen_pred"] == gold: gen_c += 1
            if out["calibrated_pred"] == gold: cal_c += 1
            total += 1
        results[cfg] = {"loglik_accuracy":     ll_c/total if total else 0.0,
                        "gen_accuracy":        gen_c/total if total else 0.0,
                        "calibrated_accuracy": cal_c/total if total else 0.0,
                        "total": total,
                        "accuracy": ll_c/total if total else 0.0}
        print(f"  [ruletaker] {cfg}: ll={ll_c}/{total}={ll_c/max(total,1):.1%}  "
              f"gen={gen_c}/{total}={gen_c/max(total,1):.1%}  "
              f"cal={cal_c}/{total}={cal_c/max(total,1):.1%}")
    all_ll = sum(r["loglik_accuracy"] * r["total"] for r in results.values())
    all_gen = sum(r["gen_accuracy"] * r["total"] for r in results.values())
    all_cal = sum(r["calibrated_accuracy"] * r["total"] for r in results.values())
    all_t = sum(r["total"] for r in results.values())
    results["overall"] = {"loglik_accuracy":     all_ll/all_t if all_t else 0.0,
                           "gen_accuracy":        all_gen/all_t if all_t else 0.0,
                           "calibrated_accuracy": all_cal/all_t if all_t else 0.0,
                           "total": all_t,
                           "accuracy": all_ll/all_t if all_t else 0.0,
                           "baseline_logprobs": baseline_lp}
    return results


# ruletaker_gen merged into ruletaker (single forward pass, both metrics)


# ==========================================================
# CLUTRR (Sinha et al., EMNLP 2019)
# ==========================================================
# 21-class kinship inference. Per-k breakdown (k=2..10).

def _clutrr_load():
    """Load CLUTRR test split. kendrivp/CLUTRR_v1_extracted is the only
    actively-maintained mirror with all task splits intact (k=2..10)."""
    from datasets import load_dataset
    for repo in ("kendrivp/CLUTRR_v1_extracted", "kliang5/CLUTRR_huggingface_dataset"):
        try:
            return load_dataset(repo, split="test")
        except Exception:
            continue
    return None


def _clutrr_extract_k(ex):
    """k-hop count from task field (e.g. 'task_1.5' -> 5)."""
    t = ex.get("task_name") or ex.get("task", "")
    try:
        return int(t.split(".")[-1])
    except Exception:
        return 0


def _clutrr_format(ex):
    story = ex.get("clean_story") or ex.get("story", "")
    q = ex.get("query")
    if isinstance(q, (list, tuple)):
        a, b = q[0], q[1]
    elif isinstance(q, str) and "(" in q:
        # parse "('A', 'B')"
        import ast
        try:
            t = ast.literal_eval(q)
            a, b = t[0], t[1]
        except Exception:
            a, b = "?", "?"
    else:
        a, b = "?", "?"
    return (f"{story}\n"
            f"Question: How is {b} related to {a}?\n"
            f"Answer:")


@register("clutrr")
def eval_clutrr(model, tokenizer, use_chat, n=None):
    """CLUTRR log-lik over 21 kinship labels, per-k breakdown."""
    n = _clamp_n(n)
    ds = _clutrr_load()
    if ds is None: return {"error": "CLUTRR load failed"}
    by_k = {}
    for ex in ds:
        by_k.setdefault(_clutrr_extract_k(ex), []).append(ex)
    print(f"  [clutrr] loaded {len(ds)} examples across k={sorted(by_k.keys())}")
    ks = [k for k in sorted(by_k.keys()) if k > 0]
    sizes = [len(by_k[k]) for k in ks]
    caps = _distribute_n(n, len(ks), sizes)

    cf_ex = {"clean_story": "N/A", "query": "('A', 'B')"}
    cf_prompt = _clutrr_format(cf_ex)
    baseline_lp = _content_free_baseline(model, tokenizer, cf_prompt, CLUTRR_LABELS, use_chat)

    results = {}
    for k, cap in zip(ks, caps):
        ll_c = gen_c = cal_c = total = 0
        for ex in _subsample(by_k[k], cap):
            gold = str(ex.get("target_text") or ex.get("target") or ex.get("answer") or "").strip().lower()
            if gold not in CLUTRR_LABELS: continue
            prompt = _clutrr_format(ex)
            out = _score_choices_fused(model, tokenizer, prompt, CLUTRR_LABELS, use_chat,
                                       baseline_logprobs=baseline_lp)
            _dbg_log_loglik(f"clutrr:k={k}", prompt, gold, out["loglik_pred"], out["logprobs"])
            if out["loglik_pred"] == gold: ll_c += 1
            if out["gen_pred"] == gold: gen_c += 1
            if out["calibrated_pred"] == gold: cal_c += 1
            total += 1
        results[f"k={k}"] = {"loglik_accuracy":     ll_c/total if total else 0.0,
                              "gen_accuracy":        gen_c/total if total else 0.0,
                              "calibrated_accuracy": cal_c/total if total else 0.0,
                              "total": total,
                              "accuracy": ll_c/total if total else 0.0}
        print(f"  [clutrr] k={k}: ll={ll_c}/{total}={ll_c/max(total,1):.1%}  "
              f"gen={gen_c}/{total}={gen_c/max(total,1):.1%}  "
              f"cal={cal_c}/{total}={cal_c/max(total,1):.1%}")
    all_ll = sum(v["loglik_accuracy"] * v["total"] for v in results.values())
    all_gen = sum(v["gen_accuracy"] * v["total"] for v in results.values())
    all_cal = sum(v["calibrated_accuracy"] * v["total"] for v in results.values())
    all_t = sum(v["total"] for v in results.values())
    results["overall"] = {"loglik_accuracy":     all_ll/all_t if all_t else 0.0,
                           "gen_accuracy":        all_gen/all_t if all_t else 0.0,
                           "calibrated_accuracy": all_cal/all_t if all_t else 0.0,
                           "total": all_t,
                           "accuracy": all_ll/all_t if all_t else 0.0,
                           "baseline_logprobs": baseline_lp}
    return results


# clutrr_gen merged into clutrr (single forward pass, both metrics)


# ==========================================================
# PrOntoQA-OOD (Saparov et al., NeurIPS 2023)
# ==========================================================
# 8-shot CoT, generation, parse final True/False from CoT chain.
# Loads from extracted JSON files in DATA_DIR/prontoqa_ood/.

def _prontoqa_ood_ensure_data():
    """Auto-download generated_ood_data.zip from asaparov/prontoqa GitHub on
    first call and extract into $DATA_DIR/prontoqa_ood/."""
    import os, urllib.request, zipfile
    if DATA_DIR is None: return None
    base = os.path.join(DATA_DIR, "prontoqa_ood")
    if os.path.isdir(base) and any(f.endswith(".json") for f in os.listdir(base)):
        return base
    os.makedirs(base, exist_ok=True)
    zip_path = os.path.join(base, "generated_ood_data.zip")
    url = "https://github.com/asaparov/prontoqa/raw/main/generated_ood_data.zip"
    try:
        print(f"  [prontoqa_ood] downloading {url} (~5MB)")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(base)
        os.remove(zip_path)
        return base
    except Exception as e:
        print(f"  [prontoqa_ood] download failed: {e}")
        return None


def _prontoqa_ood_load():
    """Load all PrOntoQA-OOD JSON variants. Returns dict: {variant: examples}.
    Auto-downloads from GitHub on first call."""
    import os, json, glob
    base = _prontoqa_ood_ensure_data()
    if base is None: return None
    out = {}
    for fpath in sorted(glob.glob(os.path.join(base, "*.json"))):
        name = os.path.splitext(os.path.basename(fpath))[0]
        try:
            d = json.load(open(fpath))
            out[name] = d
        except Exception: pass
    return out if out else None


def _prontoqa_ood_gold_label(query, chain):
    """Derive the gold True/False label for a PrOntoQA-OOD example.

    The query is "Prove: <statement>." and the chain_of_thought is a sequence
    of derivation steps. If the final step matches the queried statement (proof
    succeeded), label is True; otherwise the proof failed/contradicted → False.
    Handles "ProofByContra" variants where the chain ends with a contradiction
    derivation and the answer is also derived by chain match.
    """
    if not chain or not query: return "False"
    target = query.strip()
    for prefix in ("Prove:", "prove:"):
        if target.startswith(prefix):
            target = target[len(prefix):].strip()
            break
    target = target.rstrip(".").strip().lower()
    last = chain[-1].strip().rstrip(".").strip().lower()
    return "True" if last == target else "False"


def _prontoqa_ood_iter_examples(d):
    """Iterate over (prompt, gold_chain, gold_label) tuples from one variant.
    NO COT: demos show only Q + final True/False label (not the reasoning chain).
    Deviates from canonical 8-shot CoT protocol (Saparov 2023); we strip CoT
    from demos to avoid conflating prompting capability with model reasoning."""
    for ex_key in d:
        ex = d[ex_key]
        in_context = []
        for k in sorted(ex.keys()):
            if k.startswith("in_context_example"):
                ic = ex[k]
                ic_label = _prontoqa_ood_gold_label(ic.get("query", ""),
                                                     ic.get("chain_of_thought", []))
                in_context.append(f"Q: {ic['question']}\n{ic.get('query','')}\n"
                                  f"A: {ic_label}")
        test = ex.get("test_example", {})
        prompt = "\n\n".join(in_context) + "\n\n" + \
                 f"Q: {test.get('question','')}\n{test.get('query','')}\nA:"
        gold_chain = test.get("chain_of_thought", [])
        gold_label = _prontoqa_ood_gold_label(test.get("query", ""), gold_chain)
        yield prompt, gold_chain, gold_label


@register("prontoqa_ood")
def eval_prontoqa_ood(model, tokenizer, use_chat, n=None):
    """PrOntoQA-OOD — 8-shot WITHOUT CoT. Single forward pass returns BOTH
    log-lik and constrained-greedy gen accuracy over single-token True/False.
    n is total cap distributed across the 79 variants.

    Deviation from canonical protocol: paper uses 8-shot CoT (Saparov 2023);
    we strip the reasoning chain from demos and the model emits only True/False
    directly. This avoids conflating prompting capability with model reasoning.
    """
    n = _clamp_n(n)
    variants = _prontoqa_ood_load()
    if variants is None:
        return {"error": "PrOntoQA-OOD data not found at $DATA_DIR/prontoqa_ood/"}
    vnames = sorted(variants.keys())
    # We don't know exact per-variant size without iterating; assume ~100/variant
    sizes = [100] * len(vnames)
    caps = _distribute_n(n, len(vnames), sizes)
    print(f"  [prontoqa_ood] loaded {len(vnames)} variants, ~{caps[0]} samples/variant "
          f"(~{sum(caps)} total)")
    results = {}
    for vname, cap in zip(vnames, caps):
        examples = list(_prontoqa_ood_iter_examples(variants[vname]))
        sampled = _subsample(examples, cap)
        # Per-variant baseline: take the demo block (shared across examples in
        # this variant) + a content-free test query.
        first_prompt = examples[0][0] if examples else ""
        # The demo block ends at the last "\n\nQ:" before the final test query.
        # Replace test-query content with N/A to get content-free version.
        demo_end = first_prompt.rfind("\n\nQ: ")
        cf_prompt = (first_prompt[:demo_end + 2] + "Q: N/A\nN/A\nA:") if demo_end > 0 else "Q: N/A\nN/A\nA:"
        baseline_lp = _content_free_baseline(model, tokenizer, cf_prompt, ["True", "False"], use_chat)

        ll_c = gen_c = cal_c = total = 0
        for prompt, gold_chain, gold_label in sampled:
            out = _score_choices_fused(model, tokenizer, prompt, ["True", "False"], use_chat,
                                       baseline_logprobs=baseline_lp)
            _dbg_log_loglik(f"prontoqa_ood:{vname}", prompt, gold_label,
                            out["loglik_pred"], out["logprobs"])
            if out["loglik_pred"] == gold_label: ll_c += 1
            if out["gen_pred"] == gold_label: gen_c += 1
            if out["calibrated_pred"] == gold_label: cal_c += 1
            total += 1
        results[vname] = {"loglik_accuracy":     ll_c/total if total else 0.0,
                          "gen_accuracy":        gen_c/total if total else 0.0,
                          "calibrated_accuracy": cal_c/total if total else 0.0,
                          "total": total,
                          "accuracy": ll_c/total if total else 0.0}
    all_ll = sum(v["loglik_accuracy"] * v["total"] for v in results.values())
    all_gen = sum(v["gen_accuracy"] * v["total"] for v in results.values())
    all_cal = sum(v["calibrated_accuracy"] * v["total"] for v in results.values())
    all_t = sum(v["total"] for v in results.values())
    results["overall"] = {"loglik_accuracy":     all_ll/all_t if all_t else 0.0,
                           "gen_accuracy":        all_gen/all_t if all_t else 0.0,
                           "calibrated_accuracy": all_cal/all_t if all_t else 0.0,
                           "total": all_t,
                           "accuracy": all_ll/all_t if all_t else 0.0}
    print(f"  [prontoqa_ood] overall: ll={all_ll/max(all_t,1):.1%}  "
          f"gen={all_gen/max(all_t,1):.1%}  cal={all_cal/max(all_t,1):.1%}  (n={all_t})")
    return results


# ==========================================================
# LogicBench (Parmar et al., ACL 2024)
# ==========================================================
# Zero-shot CoT trigger, 25 inference rules (BQA Yes/No or MCQA).

def _logicbench_ensure_repo():
    """Clone Mihir3009/LogicBench (canonical author repo) to $DATA_DIR.
    Returns the local repo root or None on failure."""
    import os, subprocess
    if DATA_DIR is None: return None
    base = os.path.join(DATA_DIR, "logicbench", "LogicBench")
    if os.path.isdir(os.path.join(base, "data")):
        return base
    parent = os.path.dirname(base)
    os.makedirs(parent, exist_ok=True)
    try:
        subprocess.check_call(
            ["git", "clone", "--depth=1",
             "https://github.com/Mihir3009/LogicBench.git", base],
            cwd=parent, timeout=120)
        return base
    except Exception as e:
        print(f"  [logicbench] clone failed: {e}")
        return None


def _logicbench_load_bqa():
    """Load all LogicBench(Eval) BQA data_instances.json into list of
    {context, qa_pairs[{question, answer}], logic_type, axiom}."""
    import os, json, glob
    base = _logicbench_ensure_repo()
    if base is None: return None
    pat = os.path.join(base, "data", "LogicBench(Eval)", "BQA", "*", "*",
                       "data_instances.json")
    out = []
    for fpath in sorted(glob.glob(pat)):
        try:
            d = json.load(open(fpath))
        except Exception: continue
        for s in d.get("samples", []):
            out.append({"context": s.get("context", ""),
                        "qa_pairs": s.get("qa_pairs", []),
                        "logic_type": d.get("type", ""),
                        "axiom": d.get("axiom", "")})
    print(f"  [logicbench:BQA] loaded {len(out)} samples")
    return out if out else None


def _logicbench_load_mcqa():
    """Load LogicBench(Eval) MCQA data. MCQA samples are flat (no qa_pairs
    nesting): each sample has {context, question, choices, answer}. We wrap
    as a single-element qa_pairs list to match the BQA iter pattern."""
    import os, json, glob
    base = _logicbench_ensure_repo()
    if base is None: return None
    pat = os.path.join(base, "data", "LogicBench(Eval)", "MCQA", "*", "*",
                       "data_instances.json")
    out = []
    for fpath in sorted(glob.glob(pat)):
        try:
            d = json.load(open(fpath))
        except Exception: continue
        for s in d.get("samples", []):
            out.append({"context": s.get("context", ""),
                        "qa_pairs": [{"question": s.get("question", ""),
                                       "choices": s.get("choices", {}),
                                       "answer": s.get("answer", "")}],
                        "logic_type": d.get("type", ""),
                        "axiom": d.get("axiom", "")})
    print(f"  [logicbench:MCQA] loaded {len(out)} samples")
    return out if out else None


def _logicbench_format_bqa(ex, cot=True):
    ctx = ex.get("context", "")
    q = ex.get("question", "")
    cot_trigger = "Let's think step by step. " if cot else ""
    return (f"Context: {ctx}\n"
            f"Question: {q}\n"
            f"Answer the question ONLY in 'yes' or 'no'.\n"
            f"{cot_trigger}Answer:")


def _logicbench_iter_bqa(items):
    """Each item has multiple qa_pairs."""
    for ex in items:
        ctx = ex.get("context", "")
        for qp in ex.get("qa_pairs", []):
            yield {"context": ctx, "question": qp.get("question", ""),
                   "answer": qp.get("answer", "")}


_LOGICBENCH_BQA_FS = None
_LOGICBENCH_MCQA_FS = None

def _logicbench_few_shot_block_bqa(items, k=3):
    """Build k-shot Direct demo block for BQA. Picks the first k yes/no-balanced
    examples from the dataset; demos are returned as a string AND the set of
    used (context, question) pairs to exclude from the test sweep.

    Matches Parmar et al. 2024's "Few-shot Direct" condition (3-shot, no CoT).
    """
    global _LOGICBENCH_BQA_FS
    if _LOGICBENCH_BQA_FS is not None:
        return _LOGICBENCH_BQA_FS
    half = max(1, k // 2)
    picked = {"yes": [], "no": []}
    used_keys = set()
    for ex in _logicbench_iter_bqa(items):
        gold = _yesno(str(ex.get("answer", "")))
        if not gold or len(picked[gold]) >= (k - half if gold == "no" else half):
            continue
        if sum(len(v) for v in picked.values()) >= k:
            break
        picked[gold].append(ex)
        used_keys.add((ex["context"], ex["question"]))
    demos = []
    for ex in picked["yes"] + picked["no"]:
        demos.append(_logicbench_format_bqa(ex, cot=False) + " " +
                     _yesno(str(ex.get("answer", ""))))
    _LOGICBENCH_BQA_FS = ("\n\n".join(demos) + "\n\n", used_keys)
    return _LOGICBENCH_BQA_FS


@register("logicbench_bqa")
def eval_logicbench_bqa(model, tokenizer, use_chat, n=None):
    """LogicBench BQA — canonical 3-shot Direct (Parmar et al. 2024 FS-Direct).
    One forward pass per example, returns BOTH log-lik and constrained-greedy
    gen accuracy over single-token Yes/No labels."""
    n = _clamp_n(n)
    items = _logicbench_load_bqa()
    if items is None: return {"error": "load failed"}
    fs_block, demo_keys = _logicbench_few_shot_block_bqa(items, k=3)
    all_examples = [ex for ex in _logicbench_iter_bqa(items)
                     if (ex["context"], ex["question"]) not in demo_keys
                     and _yesno(str(ex.get("answer", "")))]
    sampled = _subsample(all_examples, n)

    cf_ex = {"context": "N/A", "question": "N/A"}
    cf_prompt = fs_block + _logicbench_format_bqa(cf_ex, cot=False)
    baseline_lp = _content_free_baseline(model, tokenizer, cf_prompt, ["Yes", "No"], use_chat)
    print(f"  [logicbench_bqa] baseline: Yes={baseline_lp['Yes']:.3f} No={baseline_lp['No']:.3f}")

    ll_c = gen_c = cal_c = total = 0
    for ex in sampled:
        gold = _yesno(str(ex.get("answer", "")))
        prompt = fs_block + _logicbench_format_bqa(ex, cot=False)
        out = _score_choices_fused(model, tokenizer, prompt, ["Yes", "No"], use_chat,
                                   baseline_logprobs=baseline_lp)
        ll_pred = out["loglik_pred"].lower()
        gen_pred = out["gen_pred"].lower()
        cal_pred = (out["calibrated_pred"] or "").lower()
        _dbg_log_loglik("logicbench_bqa", prompt, gold, ll_pred, out["logprobs"])
        if ll_pred == gold: ll_c += 1
        if gen_pred == gold: gen_c += 1
        if cal_pred == gold: cal_c += 1
        total += 1
    return {"loglik_accuracy":     ll_c/total if total else 0.0,
            "gen_accuracy":        gen_c/total if total else 0.0,
            "calibrated_accuracy": cal_c/total if total else 0.0,
            "total": total,
            "accuracy": ll_c/total if total else 0.0,
            "baseline_logprobs": baseline_lp}


def _logicbench_format_mcqa(ex, cot=True):
    ctx = ex.get("context", "")
    q = ex.get("question", "")
    choices = ex.get("choices", {}) or {}
    cs = "\n".join(f"({chr(64+i)}) {choices.get(f'choice_{i}','')}"
                   for i in range(1, 5) if f"choice_{i}" in choices)
    cot_trigger = "Let's think step by step. " if cot else ""
    return (f"Context: {ctx}\n"
            f"Question: {q}\n"
            f"Choices:\n{cs}\n"
            f"{cot_trigger}Answer:")


def _logicbench_iter_mcqa(items):
    for ex in items:
        ctx = ex.get("context", "")
        for qp in ex.get("qa_pairs", []):
            yield {"context": ctx, "question": qp.get("question", ""),
                   "choices": qp.get("choices", {}),
                   "answer": qp.get("answer", "")}


def _logicbench_mcqa_gold_letter(ex):
    g = str(ex.get("answer", "")).strip().lower()
    if "choice_" in g:
        try: return "ABCD"[int(g.split("_")[-1]) - 1]
        except Exception: pass
    return None


def _logicbench_few_shot_block_mcqa(items, k=3):
    """k-shot Direct demos for MCQA — picks first k examples covering distinct
    gold letters where possible. Returns (block_string, used_keys)."""
    global _LOGICBENCH_MCQA_FS
    if _LOGICBENCH_MCQA_FS is not None:
        return _LOGICBENCH_MCQA_FS
    picked = []
    seen_letters = set()
    used_keys = set()
    for ex in _logicbench_iter_mcqa(items):
        gold = _logicbench_mcqa_gold_letter(ex)
        if gold is None: continue
        # Prefer letter diversity, but accept duplicates if needed
        if len(picked) >= k: break
        if gold in seen_letters and len(picked) + (k - len(seen_letters)) > k:
            continue
        seen_letters.add(gold)
        picked.append((ex, gold))
        used_keys.add((ex["context"], ex["question"]))
    demos = [_logicbench_format_mcqa(ex, cot=False) + " " + gold
             for ex, gold in picked]
    _LOGICBENCH_MCQA_FS = ("\n\n".join(demos) + "\n\n", used_keys)
    return _LOGICBENCH_MCQA_FS


@register("logicbench_mcqa")
def eval_logicbench_mcqa(model, tokenizer, use_chat, n=None):
    """LogicBench MCQA — canonical 3-shot Direct (Parmar et al. 2024 FS-Direct).
    Single forward pass returns BOTH log-lik and constrained-greedy gen
    accuracy over single-token A/B/C/D labels."""
    n = _clamp_n(n)
    items = _logicbench_load_mcqa()
    if items is None: return {"error": "load failed"}
    fs_block, demo_keys = _logicbench_few_shot_block_mcqa(items, k=3)
    all_examples = []
    for ex in _logicbench_iter_mcqa(items):
        if (ex["context"], ex["question"]) in demo_keys: continue
        if _logicbench_mcqa_gold_letter(ex) is None: continue
        all_examples.append(ex)
    sampled = _subsample(all_examples, n)

    cf_ex = {"context": "N/A", "question": "N/A",
             "choices": {"choice_1": "N/A", "choice_2": "N/A",
                         "choice_3": "N/A", "choice_4": "N/A"}}
    cf_prompt = fs_block + _logicbench_format_mcqa(cf_ex, cot=False)
    baseline_lp = _content_free_baseline(model, tokenizer, cf_prompt, LOGIQA_LABELS, use_chat)
    print(f"  [logicbench_mcqa] baseline: " +
          " ".join(f"{l}={baseline_lp[l]:.3f}" for l in LOGIQA_LABELS))

    ll_c = gen_c = cal_c = total = 0
    for ex in sampled:
        gold = _logicbench_mcqa_gold_letter(ex)
        prompt = fs_block + _logicbench_format_mcqa(ex, cot=False)
        out = _score_choices_fused(model, tokenizer, prompt, LOGIQA_LABELS, use_chat,
                                   baseline_logprobs=baseline_lp)
        _dbg_log_loglik("logicbench_mcqa", prompt, gold, out["loglik_pred"], out["logprobs"])
        if out["loglik_pred"] == gold: ll_c += 1
        if out["gen_pred"] == gold: gen_c += 1
        if out["calibrated_pred"] == gold: cal_c += 1
        total += 1
    return {"loglik_accuracy":     ll_c/total if total else 0.0,
            "gen_accuracy":        gen_c/total if total else 0.0,
            "calibrated_accuracy": cal_c/total if total else 0.0,
            "total": total,
            "accuracy": ll_c/total if total else 0.0,
            "baseline_logprobs": baseline_lp}


# ==========================================================
# MultiLogiEval (Patel et al., EMNLP 2024)
# ==========================================================
# Zero-shot CoT, Yes/No, per-depth d1-d5 × {prop, FOL, NM}.
# Loads from cloned GitHub repo (auto-downloads to $DATA_DIR/multilogieval).

def _multilogieval_ensure_data():
    import os, subprocess
    if DATA_DIR is None: return None
    base = os.path.join(DATA_DIR, "multilogieval", "Multi-LogiEval")
    if os.path.isdir(os.path.join(base, "data")): return base
    parent = os.path.dirname(base)
    os.makedirs(parent, exist_ok=True)
    try:
        subprocess.check_call(
            ["git", "clone", "--depth=1",
             "https://github.com/Mihir3009/Multi-LogiEval.git", base],
            cwd=parent, timeout=120)
        return base
    except Exception as e:
        print(f"  [multilogieval] clone failed: {e}")
        return None


def _multilogieval_load_all():
    """Load all (depth, logic, rule, examples) tuples from cloned repo."""
    import os, glob, json
    base = _multilogieval_ensure_data()
    if base is None: return None
    out = []
    for ddir in sorted(glob.glob(os.path.join(base, "data", "d*_Data"))):
        depth = os.path.basename(ddir).split("_")[0]  # d1, d2, ...
        for logic_dir in sorted(os.listdir(ddir)):
            ldir = os.path.join(ddir, logic_dir)
            if not os.path.isdir(ldir): continue
            for fpath in sorted(glob.glob(os.path.join(ldir, "*.json"))):
                rule = os.path.splitext(os.path.basename(fpath))[0]
                try:
                    items = json.load(open(fpath))
                    if isinstance(items, dict):
                        items = items.get("samples", []) or list(items.values())
                    out.append((depth, logic_dir, rule, items))
                except Exception: pass
    return out


@register("multilogieval")
def eval_multilogieval(model, tokenizer, use_chat, n=None):
    """Multi-LogiEval — zero-shot Direct (no CoT). Single forward pass per
    example returns BOTH log-lik and constrained-greedy gen accuracy over
    Yes/No labels. n is total cap distributed across (depth, logic) buckets.

    Deviation from canonical: Patel et al. 2024 evaluate with zero-shot CoT
    only. We strip the CoT trigger since it conflicts with our paper framing.
    """
    n = _clamp_n(n)
    data = _multilogieval_load_all()
    if data is None: return {"error": "Multi-LogiEval data load failed"}
    bucket = {}
    for depth, logic, rule, items in data:
        bucket.setdefault((depth, logic), []).extend(items)
    keys = sorted(bucket.keys())
    sizes = [len(bucket[k]) for k in keys]
    caps = _distribute_n(n, len(keys), sizes)

    cf_prompt = ("Given the context and question, answer ONLY in 'yes' or 'no'.\n\n"
                 "Context: N/A\nQuestion: N/A\nAnswer:")
    baseline_lp = _content_free_baseline(model, tokenizer, cf_prompt, ["Yes", "No"], use_chat)
    print(f"  [multilogieval] baseline: Yes={baseline_lp['Yes']:.3f} No={baseline_lp['No']:.3f}")

    results = {}
    for (depth, logic), cap in zip(keys, caps):
        items = bucket[(depth, logic)]
        ll_c = gen_c = cal_c = total = 0
        for ex in _subsample(items, cap):
            ctx = ex.get("context", "")
            q = ex.get("question", "")
            gold = str(ex.get("answer", "")).strip().lower()
            if gold not in ("yes", "no"): continue
            prompt = (f"Given the context and question, answer ONLY in 'yes' or 'no'.\n\n"
                      f"Context: {ctx}\nQuestion: {q}\nAnswer:")
            out = _score_choices_fused(model, tokenizer, prompt, ["Yes", "No"], use_chat,
                                       baseline_logprobs=baseline_lp)
            ll_pred = out["loglik_pred"].lower()
            gen_pred = out["gen_pred"].lower()
            cal_pred = (out["calibrated_pred"] or "").lower()
            _dbg_log_loglik(f"multilogieval:{depth}_{logic}", prompt, gold, ll_pred,
                            out["logprobs"])
            if ll_pred == gold: ll_c += 1
            if gen_pred == gold: gen_c += 1
            if cal_pred == gold: cal_c += 1
            total += 1
        if total:
            results[f"{depth}_{logic}"] = {"loglik_accuracy":     ll_c/total,
                                             "gen_accuracy":        gen_c/total,
                                             "calibrated_accuracy": cal_c/total,
                                             "total": total,
                                             "accuracy": ll_c/total}
            print(f"  [multilogieval] {depth}_{logic}: ll={ll_c}/{total}={ll_c/total:.1%}  "
                  f"gen={gen_c}/{total}={gen_c/total:.1%}  "
                  f"cal={cal_c}/{total}={cal_c/total:.1%}")
    all_ll = sum(v["loglik_accuracy"] * v["total"] for v in results.values())
    all_gen = sum(v["gen_accuracy"] * v["total"] for v in results.values())
    all_cal = sum(v["calibrated_accuracy"] * v["total"] for v in results.values())
    all_t = sum(v["total"] for v in results.values())
    results["overall"] = {"loglik_accuracy":     all_ll/all_t if all_t else 0.0,
                           "gen_accuracy":        all_gen/all_t if all_t else 0.0,
                           "calibrated_accuracy": all_cal/all_t if all_t else 0.0,
                           "total": all_t,
                           "accuracy": all_ll/all_t if all_t else 0.0,
                           "baseline_logprobs": baseline_lp}
    return results


# ==========================================================
# NLGraph (Wang et al., NeurIPS 2023 Spotlight)
# ==========================================================
# 8 graph-algorithm tasks in NL. Per-task generation + regex. 0-shot here.

NLGRAPH_TASKS = ["connectivity", "cycle", "shortest_path", "flow",
                 "hamilton", "matching", "topology", "GNN"]


def _nlgraph_ensure_repo():
    """Clone the official Arthur-Heng/NLGraph repo to $DATA_DIR/nlgraph/NLGraph
    on first call. Returns the local path to the NLGraph/ subdir, or None on
    failure."""
    import os, subprocess
    if DATA_DIR is None: return None
    base = os.path.join(DATA_DIR, "nlgraph", "NLGraph")
    if os.path.isdir(os.path.join(base, "NLGraph", "connectivity")):
        return os.path.join(base, "NLGraph")
    parent = os.path.dirname(base)
    os.makedirs(parent, exist_ok=True)
    try:
        subprocess.check_call(
            ["git", "clone", "--depth=1",
             "https://github.com/Arthur-Heng/NLGraph.git", base],
            cwd=parent, timeout=120)
        return os.path.join(base, "NLGraph")
    except Exception as e:
        print(f"  [nlgraph] clone failed: {e}")
        return None


def _nlgraph_load():
    """Load all 8 NLGraph tasks from the official repo. Returns a list of
    {'task','question','answer','difficulty'} dicts (1000 total). Falls back
    to the (subset) tasksource/nlgraph HF mirror if the clone fails."""
    import os, json
    base = _nlgraph_ensure_repo()
    if base is None:
        from datasets import load_dataset
        try:
            return list(load_dataset("tasksource/nlgraph", split="test"))
        except Exception as e:
            print(f"  [nlgraph] HF fallback load failed: {e}")
            return None
    out = []
    for task in NLGRAPH_TASKS:
        fpath = os.path.join(base, task, "test.json")
        if not os.path.isfile(fpath): continue
        try:
            d = json.load(open(fpath))
        except Exception as e:
            print(f"  [nlgraph:{task}] load failed: {e}")
            continue
        items = d.values() if isinstance(d, dict) else d
        for ex in items:
            out.append({"task": task,
                        "question": ex.get("question", ""),
                        "answer": ex.get("answer", ""),
                        "difficulty": ex.get("difficulty", "")})
    print(f"  [nlgraph] loaded {len(out)} examples across {len(NLGRAPH_TASKS)} tasks")
    return out


_NLGRAPH_KSHOT_CACHE = {}

def _nlgraph_kshot_prompt(task):
    """Load the canonical k-shot Direct prompt (Wang et al. 2023) for a task.
    Prefers the local cloned NLGraph repo (populated by _nlgraph_ensure_repo);
    falls back to GitHub download only if the local file is missing."""
    if task in _NLGRAPH_KSHOT_CACHE:
        return _NLGRAPH_KSHOT_CACHE[task]
    # Try local cloned repo first
    text = None
    base = _nlgraph_ensure_repo()
    if base is not None:
        p = os.path.join(base, task, "prompt", "k-shot-prompt.txt")
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    text = f.read()
            except Exception: pass
    if text is None:
        import urllib.request
        url = (f"https://raw.githubusercontent.com/Arthur-Heng/NLGraph/main/"
               f"NLGraph/{task}/prompt/k-shot-prompt.txt")
        try:
            text = urllib.request.urlopen(url, timeout=20).read().decode("utf-8")
        except Exception as e:
            print(f"  [nlgraph:{task}] k-shot prompt fetch failed: {e}; falling back to 0-shot")
            text = ""
    if text and not text.endswith("\n\n"):
        text = text.rstrip("\n") + "\n\n"
    _NLGRAPH_KSHOT_CACHE[task] = text
    return text


def _nlgraph_parse_edges(question_text):
    """Parse '(i,j)' edges from an NLGraph question. Returns list of (u,v) ints."""
    import re as _re
    return [(int(a), int(b)) for a, b in
            _re.findall(r"\((\d+)\s*,\s*(\d+)\)", question_text)]


def _nlgraph_parse_query_pair(question_text):
    """Extract the queried node pair from connectivity-style questions, e.g.
    'Is there a path between node 8 and node 2?' → (8, 2)."""
    import re as _re
    m = _re.search(r"node\s+(\d+).*?node\s+(\d+)", question_text)
    if m: return int(m.group(1)), int(m.group(2))
    return None


def _nlgraph_score_per_task(task, pred_text, gold_text, question_text=""):
    """Canonical per-task scoring matching Arthur-Heng/NLGraph evaluation/*.py.

    For boolean tasks (connectivity/cycle/hamilton): match yes/no AND, where
    applicable, validate the structured answer (path, etc.) per canonical eval.
    """
    import re as _re
    pl = pred_text.lower()
    gl = str(gold_text).lower()

    def _has_yes(text):
        return "the answer is yes" in text or _re.search(r"\byes\b", text[:50]) is not None

    def _has_no(text):
        return ("the answer is no" in text or
                _re.search(r"\b(no|not)\b", text[:50]) is not None)

    if task == "connectivity":
        # Canonical: connectivity.py uses substring "the answer is yes" OR
        # "there is a path between node X and node Y" for yes; inverse for no.
        gold_yes = "yes" in gl
        pair = _nlgraph_parse_query_pair(question_text)
        path_phrase = ""
        if pair:
            path_phrase = f"there is a path between node {pair[0]} and node {pair[1]}"
        if gold_yes:
            return 1.0 if (("the answer is yes" in pl) or
                           (path_phrase and path_phrase in pl)) else 0.0
        # gold_no
        return 1.0 if (("the answer is no" in pl) or
                       (path_phrase and path_phrase not in pl and
                        ("the answer is yes" not in pl))) else 0.0

    if task == "cycle":
        return 1.0 if (_has_yes(pl) if "yes" in gl else _has_no(pl)) else 0.0

    if task == "hamilton":
        # Canonical hamilton.py validates the path: parse "the path can be: X,Y,Z..."
        # check length == #nodes, each consecutive pair is an edge, no duplicates.
        gold_yes = "yes" in gl
        if not gold_yes:
            return 1.0 if _has_no(pl) else 0.0
        # gold says yes — model must output a valid Hamiltonian path
        edges = _nlgraph_parse_edges(question_text)
        if not edges: return 0.0
        nodes = set()
        for u, v in edges: nodes.add(u); nodes.add(v)
        n = len(nodes)
        # Find "the path can be" or "the path is" then parse subsequent ints
        m = _re.search(r"the path (?:can be|is)[:\s]*", pl)
        if not m: return 0.0
        seq_str = pl[m.end():m.end() + 400]
        path = [int(x) for x in _re.findall(r"\d+", seq_str)][:n]
        if len(path) != n: return 0.0
        # All nodes distinct
        if len(set(path)) != n: return 0.0
        # Each consecutive pair is an edge (undirected)
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v)); edge_set.add((v, u))
        for i in range(n - 1):
            if (path[i], path[i+1]) not in edge_set: return 0.0
        return 1.0

    if task == "shortest_path":
        # Canonical shortest_path.py: compare extracted "weight of N" values.
        gold_w = _re.search(r"weight of (\d+)", gl)
        pred_w = _re.search(r"weight of (\d+)|total weight[: ]+(\d+)", pl)
        if gold_w and pred_w:
            gw = int(gold_w.group(1))
            pw = int(pred_w.group(1) or pred_w.group(2))
            return 1.0 if pw == gw else 0.0
        return 0.0

    if task == "flow":
        # Canonical flow.py: exact-match the maximum-flow value.
        gold_v = _re.search(r"is\s+(\d+)", gl)
        pred_v = _re.search(r"is\s+(\d+)|flow[: ]+(\d+)", pl)
        if gold_v and pred_v:
            gv = int(gold_v.group(1))
            pv = int(pred_v.group(1) or pred_v.group(2))
            return 1.0 if pv == gv else 0.0
        return 0.0

    if task == "matching":
        # Canonical matching.py: extract "N applicants" count, exact match.
        gold_n = _re.search(r"(\d+)\s+applicants", gl)
        pred_n = _re.search(r"(\d+)\s+applicants", pl)
        if gold_n and pred_n and int(pred_n.group(1)) == int(gold_n.group(1)):
            return 1.0
        return 0.0

    if task == "topology":
        # Canonical topology.py: extract integer ordering, validate consistency
        # with the partial-order constraints. Simplified here to exact-prefix
        # match against gold ordering.
        gold_seq = _re.findall(r"\d+", gl)
        pred_seq = _re.findall(r"\d+", pl[:500])
        if gold_seq and pred_seq[:len(gold_seq)] == gold_seq: return 1.0
        return 0.0

    if task == "GNN":
        # Canonical gnn.py: per-node embedding exact match. Approximation: check
        # all "node K: [a,b]" lines in pred match gold.
        gold_pairs = dict(_re.findall(r"node\s*(\d+)\s*:\s*\[(\d+\s*,\s*\d+)\]", gl))
        pred_pairs = dict(_re.findall(r"node\s*(\d+)\s*:\s*\[(\d+\s*,\s*\d+)\]", pl))
        if not gold_pairs: return 0.0
        norm = lambda s: s.replace(" ", "")
        correct = sum(1 for k, v in gold_pairs.items()
                      if k in pred_pairs and norm(pred_pairs[k]) == norm(v))
        return correct / len(gold_pairs)

    return 0.0


@register("nlgraph_gen")
def eval_nlgraph_gen(model, tokenizer, use_chat, n=None):
    """NLGraph generation per-task — canonical 4-shot Direct (Wang et al. 2023).

    Uses the official `k-shot-prompt.txt` from Arthur-Heng/NLGraph for each task
    (4 demos, no CoT). Scoring uses the canonical per-task scorer (path
    validation for hamilton, weight match for shortest_path, etc.) ported from
    Arthur-Heng/NLGraph evaluation/*.py. n is total cap distributed across the
    8 tasks.
    """
    n = _clamp_n(n)
    ds = _nlgraph_load()
    if ds is None: return {"error": "load failed"}
    by_task = {}
    for ex in ds:
        by_task.setdefault(ex.get("task", "?"), []).append(ex)
    tasks = sorted(by_task.keys())
    sizes = [len(by_task[t]) for t in tasks]
    caps = _distribute_n(n, len(tasks), sizes)

    results = {}
    for task, cap in zip(tasks, caps):
        group = _subsample(by_task[task], cap)
        score_sum = 0.0; total = 0
        fs_block = _nlgraph_kshot_prompt(task)
        for ex in group:
            question = ex.get("question", "")
            prompt = fs_block + question
            gold = ex.get("answer", "")
            pred = generate_chat(model, tokenizer, prompt, max_new_tokens=200,
                                 use_chat=use_chat)
            score = _nlgraph_score_per_task(task, pred, gold, question)
            _dbg_log_gen(f"nlgraph:{task}", prompt, gold, pred, f"score={score:.2f}")
            score_sum += score
            total += 1
        avg = score_sum / total if total else 0.0
        results[task] = {"score_sum": score_sum, "total": total, "accuracy": avg}
        print(f"  [nlgraph_gen] {task}: {score_sum:.1f}/{total} = {avg:.1%}")
    all_s = sum(v["score_sum"] for v in results.values())
    all_t = sum(v["total"] for v in results.values())
    results["overall"] = {"score_sum": all_s, "total": all_t,
                          "accuracy": all_s/all_t if all_t else 0.0}
    return results


# ==========================================================
# Few-shot variants of the 0-shot benchmarks.
#
# Rationale: the primary adapters use 0-shot for benchmarks where that's either
# canonical (logiqa/AGIEval) or no canonical LLM method exists (clutrr, ruletaker,
# multilogieval, zebra_mc). A few-shot variant is provided alongside to test
# whether format-teaching helps the model escape class-collapse.
#
# Demos for each benchmark are built by seeded random sampling from the same
# pool as eval (excluded from the test sweep). Prompts are identical to the
# 0-shot versions but with K demos prepended.
# ==========================================================


def _fewshot_block_from_pool(pool, k, prompt_fn, gold_fn, seed=1234, balance_by_gold=True):
    """Return (demo_block_str, used_keys_set). `prompt_fn(ex)` produces the
    per-example prompt ending with 'Answer:' (or similar); `gold_fn(ex)` produces
    the gold label string. If balance_by_gold, picks demos round-robin over
    distinct gold values.
    """
    import random as _random
    rng = _random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    picked, seen_golds = [], []
    for ex in shuffled:
        g = gold_fn(ex)
        if g is None: continue
        if balance_by_gold:
            if g in seen_golds and len(seen_golds) < k and len(picked) + (k - len(seen_golds)) > k:
                continue
            if g not in seen_golds: seen_golds.append(g)
        picked.append((ex, g))
        if len(picked) >= k: break
    blocks = [f"{prompt_fn(ex)} {g}" for ex, g in picked]
    used_keys = {id(ex) for ex, _ in picked}  # use id() since structures are unhashable
    return "\n\n".join(blocks) + ("\n\n" if blocks else ""), used_keys


@register("ruletaker_fs")
def eval_ruletaker_fs(model, tokenizer, use_chat, n=None):
    """RuleTaker — 5-shot Direct variant. Same prompt + scoring as `ruletaker`
    but with 5 balanced True/False demos from the eval pool, excluded from
    the test sweep."""
    n = _clamp_n(n)
    by_depth = _ruletaker_load_by_depth()
    if by_depth is None: return {"error": "load failed"}
    cfgs = sorted(by_depth.keys())

    # Build demo pool from a mix across depths (so demos cover curriculum range)
    all_pool = [ex for cfg in cfgs for ex in by_depth[cfg]]
    fs_block, demo_ids = _fewshot_block_from_pool(all_pool, k=5,
        prompt_fn=_ruletaker_format, gold_fn=_ruletaker_gold)

    sizes = [sum(1 for ex in by_depth[c] if id(ex) not in demo_ids) for c in cfgs]
    caps = _distribute_n(n, len(cfgs), sizes)

    cf_ex = {"context": "N/A", "question": "N/A"}
    cf_prompt = fs_block + _ruletaker_format(cf_ex)
    baseline_lp = _content_free_baseline(model, tokenizer, cf_prompt, RULETAKER_LABELS, use_chat)
    print(f"  [ruletaker_fs] 5-shot baseline: True={baseline_lp['True']:.3f} False={baseline_lp['False']:.3f}")

    results = {}
    for cfg, cap in zip(cfgs, caps):
        pool = [ex for ex in by_depth[cfg] if id(ex) not in demo_ids]
        ll_c = gen_c = cal_c = total = 0
        for ex in _subsample(pool, cap):
            gold = _ruletaker_gold(ex)
            if gold is None: continue
            prompt = fs_block + _ruletaker_format(ex)
            out = _score_choices_fused(model, tokenizer, prompt, RULETAKER_LABELS, use_chat,
                                       baseline_logprobs=baseline_lp)
            _dbg_log_loglik(f"ruletaker_fs:{cfg}", prompt, gold, out["loglik_pred"], out["logprobs"])
            if out["loglik_pred"] == gold: ll_c += 1
            if out["gen_pred"] == gold: gen_c += 1
            if out["calibrated_pred"] == gold: cal_c += 1
            total += 1
        results[cfg] = {"loglik_accuracy": ll_c/total if total else 0.0,
                        "gen_accuracy":    gen_c/total if total else 0.0,
                        "calibrated_accuracy": cal_c/total if total else 0.0,
                        "total": total,
                        "accuracy": ll_c/total if total else 0.0}
        print(f"  [ruletaker_fs] {cfg}: ll={ll_c}/{total}={ll_c/max(total,1):.1%}  "
              f"gen={gen_c}/{total}={gen_c/max(total,1):.1%}  "
              f"cal={cal_c}/{total}={cal_c/max(total,1):.1%}")
    all_ll = sum(r["loglik_accuracy"] * r["total"] for r in results.values())
    all_gen = sum(r["gen_accuracy"] * r["total"] for r in results.values())
    all_cal = sum(r["calibrated_accuracy"] * r["total"] for r in results.values())
    all_t = sum(r["total"] for r in results.values())
    results["overall"] = {"loglik_accuracy": all_ll/all_t if all_t else 0.0,
                           "gen_accuracy":    all_gen/all_t if all_t else 0.0,
                           "calibrated_accuracy": all_cal/all_t if all_t else 0.0,
                           "total": all_t,
                           "accuracy": all_ll/all_t if all_t else 0.0,
                           "baseline_logprobs": baseline_lp}
    return results


@register("clutrr_fs")
def eval_clutrr_fs(model, tokenizer, use_chat, n=None):
    """CLUTRR — 5-shot Direct variant. Demos sampled with diverse k-hop
    settings and kinship labels; excluded from test sweep."""
    n = _clamp_n(n)
    ds = _clutrr_load()
    if ds is None: return {"error": "CLUTRR load failed"}
    by_k = {}
    for ex in ds:
        by_k.setdefault(_clutrr_extract_k(ex), []).append(ex)

    gold_fn = lambda ex: str(ex.get("target_text") or ex.get("target") or ex.get("answer") or "").strip().lower() or None
    all_pool = [ex for k in sorted(by_k.keys()) if k > 0 for ex in by_k[k]]
    fs_block, demo_ids = _fewshot_block_from_pool(all_pool, k=5,
        prompt_fn=_clutrr_format, gold_fn=gold_fn)

    ks = [k for k in sorted(by_k.keys()) if k > 0]
    sizes = [sum(1 for ex in by_k[k] if id(ex) not in demo_ids) for k in ks]
    caps = _distribute_n(n, len(ks), sizes)

    cf_ex = {"clean_story": "N/A", "query": "('A', 'B')"}
    cf_prompt = fs_block + _clutrr_format(cf_ex)
    baseline_lp = _content_free_baseline(model, tokenizer, cf_prompt, CLUTRR_LABELS, use_chat)

    results = {}
    for k, cap in zip(ks, caps):
        pool = [ex for ex in by_k[k] if id(ex) not in demo_ids]
        ll_c = gen_c = cal_c = total = 0
        for ex in _subsample(pool, cap):
            gold = gold_fn(ex)
            if gold not in CLUTRR_LABELS: continue
            prompt = fs_block + _clutrr_format(ex)
            out = _score_choices_fused(model, tokenizer, prompt, CLUTRR_LABELS, use_chat,
                                       baseline_logprobs=baseline_lp)
            _dbg_log_loglik(f"clutrr_fs:k={k}", prompt, gold, out["loglik_pred"], out["logprobs"])
            if out["loglik_pred"] == gold: ll_c += 1
            if out["gen_pred"] == gold: gen_c += 1
            if out["calibrated_pred"] == gold: cal_c += 1
            total += 1
        results[f"k={k}"] = {"loglik_accuracy": ll_c/total if total else 0.0,
                              "gen_accuracy":    gen_c/total if total else 0.0,
                              "calibrated_accuracy": cal_c/total if total else 0.0,
                              "total": total,
                              "accuracy": ll_c/total if total else 0.0}
        print(f"  [clutrr_fs] k={k}: ll={ll_c}/{total}={ll_c/max(total,1):.1%}  "
              f"cal={cal_c}/{total}={cal_c/max(total,1):.1%}")
    all_ll = sum(v["loglik_accuracy"] * v["total"] for v in results.values())
    all_gen = sum(v["gen_accuracy"] * v["total"] for v in results.values())
    all_cal = sum(v["calibrated_accuracy"] * v["total"] for v in results.values())
    all_t = sum(v["total"] for v in results.values())
    results["overall"] = {"loglik_accuracy": all_ll/all_t if all_t else 0.0,
                           "gen_accuracy":    all_gen/all_t if all_t else 0.0,
                           "calibrated_accuracy": all_cal/all_t if all_t else 0.0,
                           "total": all_t,
                           "accuracy": all_ll/all_t if all_t else 0.0,
                           "baseline_logprobs": baseline_lp}
    return results


@register("logiqa_fs")
def eval_logiqa_fs(model, tokenizer, use_chat, n=None):
    """LogiQA — 3-shot Direct variant (deviates from AGIEval 0-shot canonical).
    Demos sampled with diverse A/B/C/D gold letters from the test split,
    excluded from the eval sweep."""
    n = _clamp_n(n)
    ds = _logiqa_load()
    if ds is None: return {"error": "LogiQA load failed"}
    all_pool = list(ds)
    fs_block, demo_ids = _fewshot_block_from_pool(all_pool, k=3,
        prompt_fn=_logiqa_format, gold_fn=_logiqa_gold_letter)

    pool = [ex for ex in all_pool if id(ex) not in demo_ids]

    cf_ex = {"query": ("N/A\nQ: N/A Answer Choices: (A)N/A (B)N/A (C)N/A (D)N/A\n"
                       "A: Among A through D, the answer is")}
    cf_prompt = fs_block + _logiqa_format(cf_ex)
    baseline_lp = _content_free_baseline(model, tokenizer, cf_prompt, LOGIQA_LABELS, use_chat)
    print(f"  [logiqa_fs] 3-shot baseline: " + " ".join(f"{l}={baseline_lp[l]:.3f}" for l in LOGIQA_LABELS))

    ll_c = gen_c = cal_c = total = 0
    for ex in _subsample(pool, n):
        gold = _logiqa_gold_letter(ex)
        if gold is None: continue
        prompt = fs_block + _logiqa_format(ex)
        out = _score_choices_fused(model, tokenizer, prompt, LOGIQA_LABELS, use_chat,
                                   baseline_logprobs=baseline_lp)
        _dbg_log_loglik("logiqa_fs", prompt, gold, out["loglik_pred"], out["logprobs"])
        if out["loglik_pred"] == gold: ll_c += 1
        if out["gen_pred"] == gold: gen_c += 1
        if out["calibrated_pred"] == gold: cal_c += 1
        total += 1
    return {"loglik_accuracy": ll_c/total if total else 0.0,
            "gen_accuracy":    gen_c/total if total else 0.0,
            "calibrated_accuracy": cal_c/total if total else 0.0,
            "total": total,
            "accuracy": ll_c/total if total else 0.0,
            "baseline_logprobs": baseline_lp}


@register("multilogieval_fs")
def eval_multilogieval_fs(model, tokenizer, use_chat, n=None):
    """Multi-LogiEval — 3-shot Direct variant. Demos sampled balanced yes/no
    from the same pool, excluded from test sweep."""
    n = _clamp_n(n)
    data = _multilogieval_load_all()
    if data is None: return {"error": "Multi-LogiEval data load failed"}
    bucket = {}
    for depth, logic, rule, items in data:
        bucket.setdefault((depth, logic), []).extend(items)

    def _fmt(ex):
        return (f"Given the context and question, answer ONLY in 'yes' or 'no'.\n\n"
                f"Context: {ex.get('context','')}\nQuestion: {ex.get('question','')}\nAnswer:")
    def _gold(ex):
        g = str(ex.get('answer','')).strip().lower()
        return g.capitalize() if g in ('yes','no') else None

    all_pool = [ex for items in bucket.values() for ex in items]
    fs_block, demo_ids = _fewshot_block_from_pool(all_pool, k=3,
        prompt_fn=_fmt, gold_fn=_gold)

    keys = sorted(bucket.keys())
    sizes = [sum(1 for ex in bucket[k] if id(ex) not in demo_ids) for k in keys]
    caps = _distribute_n(n, len(keys), sizes)

    cf_prompt = fs_block + ("Given the context and question, answer ONLY in 'yes' or 'no'.\n\n"
                             "Context: N/A\nQuestion: N/A\nAnswer:")
    baseline_lp = _content_free_baseline(model, tokenizer, cf_prompt, ["Yes", "No"], use_chat)
    print(f"  [multilogieval_fs] 3-shot baseline: Yes={baseline_lp['Yes']:.3f} No={baseline_lp['No']:.3f}")

    results = {}
    for (depth, logic), cap in zip(keys, caps):
        pool = [ex for ex in bucket[(depth, logic)] if id(ex) not in demo_ids]
        ll_c = gen_c = cal_c = total = 0
        for ex in _subsample(pool, cap):
            gold = str(ex.get('answer','')).strip().lower()
            if gold not in ("yes", "no"): continue
            prompt = fs_block + _fmt(ex)
            out = _score_choices_fused(model, tokenizer, prompt, ["Yes", "No"], use_chat,
                                       baseline_logprobs=baseline_lp)
            ll_pred = out["loglik_pred"].lower()
            gen_pred = out["gen_pred"].lower()
            cal_pred = (out["calibrated_pred"] or "").lower()
            _dbg_log_loglik(f"multilogieval_fs:{depth}_{logic}", prompt, gold, ll_pred, out["logprobs"])
            if ll_pred == gold: ll_c += 1
            if gen_pred == gold: gen_c += 1
            if cal_pred == gold: cal_c += 1
            total += 1
        if total:
            results[f"{depth}_{logic}"] = {"loglik_accuracy": ll_c/total,
                                             "gen_accuracy": gen_c/total,
                                             "calibrated_accuracy": cal_c/total,
                                             "total": total,
                                             "accuracy": ll_c/total}
    all_ll = sum(v["loglik_accuracy"] * v["total"] for v in results.values())
    all_gen = sum(v["gen_accuracy"] * v["total"] for v in results.values())
    all_cal = sum(v["calibrated_accuracy"] * v["total"] for v in results.values())
    all_t = sum(v["total"] for v in results.values())
    results["overall"] = {"loglik_accuracy": all_ll/all_t if all_t else 0.0,
                           "gen_accuracy":    all_gen/all_t if all_t else 0.0,
                           "calibrated_accuracy": all_cal/all_t if all_t else 0.0,
                           "total": all_t,
                           "accuracy": all_ll/all_t if all_t else 0.0,
                           "baseline_logprobs": baseline_lp}
    return results


@register("zebra_mc_fs")
def eval_zebra_mc_fs(model, tokenizer, use_chat, n=None):
    """ZebraLogic mc_mode — 3-shot Direct variant. Demos sampled from the
    dataset (excluded from eval). Choice set is per-example so we use per-example
    calibration (same as `zebra_mc`)."""
    from datasets import load_dataset
    n = _clamp_n(n)
    try:
        ds = load_dataset("WildEval/ZebraLogic", "mc_mode", split="test")
    except Exception as e:
        return {"error": f"load_dataset failed: {e}"}
    print(f"  [zebra_mc_fs] loaded {len(ds)} questions")

    def extract_size(ex_id):
        try: return ex_id.split("-")[2].replace("x", "*")
        except Exception: return "?"

    by_size = {}
    for ex in ds:
        by_size.setdefault(extract_size(ex["id"]), []).append(ex)

    # Build 3 demos from smaller puzzles (2*2, 2*3) to minimize prompt length
    def _zebra_fmt(ex): return f"{ex['puzzle']}\n\nQuestion: {ex['question']}\nAnswer:"
    demo_pool = (by_size.get("2*2", []) or []) + (by_size.get("2*3", []) or [])
    # Pick 3 diverse demos
    import random as _random
    rng = _random.Random(1234)
    rng.shuffle(demo_pool)
    demos = []
    seen_answers = []
    for ex in demo_pool:
        a = str(ex["answer"])
        if a in seen_answers and len(demos) + 1 < 3: continue
        seen_answers.append(a)
        demos.append(f"{_zebra_fmt(ex)} {a}")
        if len(demos) >= 3: break
    fs_block = ("\n\n".join(demos) + "\n\n") if demos else ""
    demo_ids = {id(ex) for ex in demo_pool[:3]}

    sizes = sorted(by_size.keys(), key=lambda s: tuple(int(x) for x in s.split("*")) if "*" in s else (99,99))
    size_counts = [sum(1 for ex in by_size[s] if id(ex) not in demo_ids) for s in sizes]
    caps = _distribute_n(n, len(sizes), size_counts)

    results = {}
    for size, cap in zip(sizes, caps):
        pool = [ex for ex in by_size[size] if id(ex) not in demo_ids]
        ll_c = gen_c = cal_c = total = 0
        for ex in _subsample(pool, cap):
            prompt = fs_block + _zebra_fmt(ex)
            choices = [str(c) for c in ex["choices"]]
            cf_prompt = fs_block + f"N/A\n\nQuestion: N/A\nAnswer:"
            baseline_lp = _content_free_baseline(model, tokenizer, cf_prompt, choices, use_chat)
            out = _score_choices_fused(model, tokenizer, prompt, choices, use_chat,
                                       baseline_logprobs=baseline_lp)
            _dbg_log_loglik(f"zebra_mc_fs:{size}", prompt, ex["answer"],
                            out["loglik_pred"], out["logprobs"])
            if out["loglik_pred"] == ex["answer"]: ll_c += 1
            if out["gen_pred"] == ex["answer"]: gen_c += 1
            if out["calibrated_pred"] == ex["answer"]: cal_c += 1
            total += 1
        results[size] = {"loglik_accuracy": ll_c/total if total else 0.0,
                          "gen_accuracy":    gen_c/total if total else 0.0,
                          "calibrated_accuracy": cal_c/total if total else 0.0,
                          "total": total,
                          "accuracy": ll_c/total if total else 0.0}
        print(f"  [zebra_mc_fs] size={size:>5} n={total:>3}  ll={ll_c/max(total,1):.1%}  "
              f"cal={cal_c/max(total,1):.1%}")
    all_ll = sum(v["loglik_accuracy"] * v["total"] for v in results.values())
    all_gen = sum(v["gen_accuracy"] * v["total"] for v in results.values())
    all_cal = sum(v["calibrated_accuracy"] * v["total"] for v in results.values())
    all_t = sum(v["total"] for v in results.values())
    results["overall"] = {"loglik_accuracy": all_ll/all_t if all_t else 0.0,
                           "gen_accuracy":    all_gen/all_t if all_t else 0.0,
                           "calibrated_accuracy": all_cal/all_t if all_t else 0.0,
                           "total": all_t,
                           "accuracy": all_ll/all_t if all_t else 0.0}
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

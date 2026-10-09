"""Data-stream invariance check for the dead-code removal of 2026-10-09 (CPU only, no GPU, no flash-attn).

Prints sha256 hashes of
  (a) 16 search examples from NaturalLanguageGraphGenerator(768, seed=1234) at alpha_for_lookahead(16, 768) and 16
      at alpha_for_lookahead(128, 768), both with max_lookahead=128: one hash per example over
      input_text + '\n'.join(output_texts), plus one hash over the whole set;
  (b) input_ids / labels / position_ids / cu_seqlens of PackedSequenceDataset items 0, 1, 2 at stage 1 and at
      stage 5, the dataset built exactly as tuning_nl.main() builds it for the paper config (Qwen/Qwen3-0.6B
      tokenizer, batch_size 48, max_input_size 768, --linear_lookahead --base_lookahead 8 --lookahead_step 8
      --max_lookahead 128, seed 1234, no pretrain mix, no chat template, NL_PACKING=hf).
Only the hash lines go to stdout; import-time and dataset chatter goes to stderr, so
    cd nl && /home/huan2073/.conda/envs/search/bin/python bench/test_datastream.py > bench/test_datastream.reference.txt
    cd nl && /home/huan2073/.conda/envs/search/bin/python bench/test_datastream.py | diff - bench/test_datastream.reference.txt
must be byte-identical before and after a refactor that is not supposed to touch the data stream.
Constructor keywords that a refactor removed (num_shots, store_examples, store_cap) are passed only when the
PackedSequenceDataset signature still has them, so the same file runs against both versions.
"""
import contextlib
import hashlib
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NL_DIR = os.path.abspath(os.path.join(HERE, ".."))
os.chdir(NL_DIR)  # nl_generator compiles / checks generator.cpp relative to the cwd
sys.path.insert(0, NL_DIR)
os.environ.setdefault("NL_PACKING", "hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_OUT = sys.stdout
with contextlib.redirect_stdout(sys.stderr):
    from transformers import AutoTokenizer
    from nl_generator import NaturalLanguageGraphGenerator
    from tuning_nl import PackedSequenceDataset, alpha_for_lookahead

CACHE_DIR = "/scratch/gautschi/huan2073/model_cache"
MODEL = "Qwen/Qwen3-0.6B"
N = 768
MAX_L = 128


def emit(line: str) -> None:
    print(line, file=_OUT)


def sha(data) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def part_a() -> None:
    gen = NaturalLanguageGraphGenerator(N, seed=1234)
    for L in (16, MAX_L):
        alpha = alpha_for_lookahead(L, N)
        batch = gen.generate_batch("search", batch_size=16, alpha=alpha, max_lookahead=MAX_L)
        assert len(batch) == 16, len(batch)
        blobs = []
        for i, ex in enumerate(batch):
            blob = ex.input_text + "\n".join(ex.output_texts)
            blobs.append(blob)
            emit(f"[A] L={L} alpha={alpha:.6f} example={i:02d} n_out={len(ex.output_texts)} sha256={sha(blob)}")
        emit(f"[A] L={L} all16 sha256={sha(chr(0).join(blobs))}")


def build_dataset(tokenizer) -> PackedSequenceDataset:
    task_kwargs = {"max_lookahead": MAX_L, "fixed_vocab": False, "vocab_pool": "none"}
    kwargs = dict(
        task="search",
        tokenizer=tokenizer,
        batch_size=48,
        stage=1,
        n_stages=16,
        base_alpha=0.1,
        max_alpha=1.0,
        max_input_size=N,
        reserved_inputs=set(),
        seed=1234,
        resume_step=0,
        linear_lookahead=True,
        base_lookahead=8,
        lookahead_step=8,
        breadth=2,
        shuffled_mixture=None,
        epoch_size=1_000_000_000,
        mix_pretrain_data=None,
        mix_pretrain_subset=None,
        mix_pretrain_ratio=0.1,
        mix_pretrain_max_len=2048,
        mix_pretrain_cache_dir=os.path.join(os.environ.get("SCRATCH", "/tmp"), "pretrain_cache"),
        use_chat_template=False,
        **task_kwargs,
    )
    params = inspect.signature(PackedSequenceDataset.__init__).parameters
    if "num_shots" in params:
        kwargs["num_shots"] = 0
    if "store_examples" in params:
        kwargs["store_examples"] = False
    if "store_cap" in params:
        kwargs["store_cap"] = 1000
    return PackedSequenceDataset(**kwargs)


def part_b() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE_DIR, trust_remote_code=True,
                                              local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    with contextlib.redirect_stdout(sys.stderr):
        ds = build_dataset(tokenizer)
    for stage in (1, 5):
        ds.stage = stage
        for idx in (0, 1, 2):
            with contextlib.redirect_stdout(sys.stderr):
                item = ds[idx]
            cu = item["cu_seqlens_list"][0]
            emit(f"[B] stage={stage} target_L={ds._stage_target_lookahead()} alpha={ds._stage_alpha():.6f} "
                 f"item={idx} num_sequences={item['num_sequences']} tokens={item['input_ids'].shape[1]}")
            for key in ("input_ids", "labels", "position_ids"):
                t = item[key]
                emit(f"[B] stage={stage} item={idx} {key} shape={tuple(t.shape)} "
                     f"sha256={sha(t.contiguous().numpy().tobytes())}")
            emit(f"[B] stage={stage} item={idx} cu_seqlens len={len(cu)} last={cu[-1]} "
                 f"sha256={sha(','.join(str(int(c)) for c in cu))}")


if __name__ == "__main__":
    part_a()
    part_b()
    emit("[DONE] test_datastream")

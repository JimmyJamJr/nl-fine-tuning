# Retired packing-parity bench scripts (2026-10-09)

These scripts drove the old-vs-new packing parity sign-off and the sparse-head validation of 2026-10-09, when
`nl/tuning_nl.py` moved from the hand-rolled packed layer loop to transformers' built-in packed flash-attention
path (`PackedSequenceTrainer._hf_forward_body`). They are kept for the record only and **no longer run**: the
legacy loop (`nl/legacy_varlen_forward.py`, selected with `NL_PACKING=custom`) was deleted after the sign-off,
so the `custom_*` arms cannot be launched, and the `NL_DEBUG_PARAM_SYNC` / `NL_PARITY_DUMP` hooks whose
`[PARAM-SYNC]` lines and `parity_dump/` files the compare scripts parse were removed from the trainer together
with the `ParamSyncDebugCallback`.

| file | role in the sign-off |
| --- | --- |
| `packing_bench.sbatch` | 4xH100 job driving `bench/run_arm.sh` over the arms `custom_fa2 custom_fa3 hf_fa3 hf_fa2` at L=16 and L=128 (parity + speed). |
| `after_arms.sh` | post-refactor arms inside a borrowed allocation: deterministic parity arms (old vs new, same kernel family, `NL_DEBUG_PARAM_SYNC=1`) then speed arms. |
| `sparse_arms.sh` | sparse-head (`NL_HEAD=sparse`) deterministic and speed arms against the full-head `hf_*` arms. |
| `param_sync_check.sbatch` | 2-GPU, 4-step empirical check that `compute_loss` bypasses DDP's reducer (ranks DIVERGED from step 1). |
| `sparse_1gpu.sbatch` | 1-GPU full-head vs sparse-head arms (`NGPUS=1`, `TAG=_<head>_1g`), rank-independence check against the 4-GPU `hf_fa2_det` rank 0. |
| `compare_packing_bench.py` | tokens/step, step time, TFLOP/s, peak memory and per-step loss parity across arms; parses the `[PARAM-SYNC]` lines. |
| `compare_sparse_1gpu.py` | loss / checksum / accuracy comparison for the 1-GPU sparse-head arms. |

At the time these ran, `bench/run_arm.sh` took an arm named `<packing>_<kernel>` and exported `NL_PACKING`
from it; the surviving `nl/bench/run_arm.sh` now takes just the kernel (`bash bench/run_arm.sh <fa2|fa3> <L>
[STEPS]`), names jobs `pbench_<kernel>[_det][<TAG>]_L<L>`, and its `DETERMINISTIC=1` branch exports only
`FLASH_ATTENTION_DETERMINISTIC` and `CUBLAS_WORKSPACE_CONFIG`. The pre-refactor trainer is archived next to this
directory as `archive/tuning_nl_pre_hfpacked_20261009.py`. Outputs of the original runs, where still present,
are under `$SCRATCH/nl_output/search/job_pbench_*` and `nl/bench/log_*.txt` / `nl/bench/mem_*.csv`.

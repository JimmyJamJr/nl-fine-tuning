#!/bin/bash
# ============================================================================
# Periodic GCS sync daemon for the curriculum training run.
# Mirrors the local run dir to gs://jackierwzhang-purdue-research-curriculum/...
# AND prunes local copies after successful upload to keep disk usage bounded.
# ============================================================================
#
# Behavior (rolling sync + local pruning):
#   - Always sync top-level files (loss_history*, plots, *.json, training.log)
#   - Always sync ALL stage_checkpoints/  (small, kept forever locally + remote)
#   - Sync persistent_checkpoints/ then PRUNE LOCAL copies older than the last
#     KEEP_LOCAL_PERSIST (default 2). Remote keeps everything (or last
#     KEEP_REMOTE_PERSIST if set; default unlimited).
#   - For checkpoint-N/ dirs: upload last KEEP_REMOTE_CKPT (default 2),
#     keep last KEEP_LOCAL_CKPT (default 3) locally, prune older locally + remotely.
#
# Local pruning happens ONLY after a successful upload of the keep-set, so we
# never delete a checkpoint that isn't safely in GCS.
#
# Run alongside the training job (same machine, different process).
# Stops itself when the training process is gone or RUN_DIR has a STOP_SYNC flag.
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="${RUN_DIR:-/home/ray/scratch/nl_output/search/job_local_20260423_230739_L256_BS96_noGC}"
GCS_DEST="${GCS_DEST:-gs://jackierwzhang-purdue-research-curriculum/qwen3-0.6b-curr-L256-step16/job_local_20260423_230739_L256_BS96_noGC}"
SYNC_INTERVAL_SEC="${SYNC_INTERVAL_SEC:-180}"
KEEP_LOCAL_CKPT="${KEEP_LOCAL_CKPT:-3}"
KEEP_REMOTE_CKPT="${KEEP_REMOTE_CKPT:-2}"
KEEP_LOCAL_PERSIST="${KEEP_LOCAL_PERSIST:-2}"
KEEP_REMOTE_PERSIST="${KEEP_REMOTE_PERSIST:-0}"   # 0 = unlimited
TRAIN_PID="${TRAIN_PID:-}"

export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$REPO_ROOT/gcs-bucket-sa.json}"

# shellcheck disable=SC1091
CONDA_BASE="${CONDA_BASE:-/home/ray/anaconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate base

echo "[sync] start $(date)"
echo "[sync]   RUN_DIR=$RUN_DIR"
echo "[sync]   DEST=$GCS_DEST"
echo "[sync]   every=${SYNC_INTERVAL_SEC}s  keep_local_ckpt=$KEEP_LOCAL_CKPT  keep_remote_ckpt=$KEEP_REMOTE_CKPT"
echo "[sync]   keep_local_persist=$KEEP_LOCAL_PERSIST  keep_remote_persist=$KEEP_REMOTE_PERSIST"

python - <<'PY'
import os, sys, time, json, shutil, datetime, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage

run_dir          = Path(os.environ["RUN_DIR"])
gcs_url          = os.environ["GCS_DEST"]
interval         = int(os.environ.get("SYNC_INTERVAL_SEC", "180"))
keep_local_ckpt  = int(os.environ.get("KEEP_LOCAL_CKPT", "3"))
keep_remote_ckpt = int(os.environ.get("KEEP_REMOTE_CKPT", "2"))
keep_local_persist  = int(os.environ.get("KEEP_LOCAL_PERSIST", "2"))
keep_remote_persist = int(os.environ.get("KEEP_REMOTE_PERSIST", "0"))
train_pid        = int(os.environ.get("TRAIN_PID", "0") or 0)

assert gcs_url.startswith("gs://")
bucket_name, _, prefix_root = gcs_url[len("gs://"):].partition("/")
prefix_root = prefix_root.rstrip("/") + "/"

client = storage.Client()
bucket = client.bucket(bucket_name)

TOP_LEVEL_GLOBS = ["*.json", "*.jsonl", "*.png", "*.log"]
STAGE_DIR   = "stage_checkpoints"
PERSIST_DIR = "persistent_checkpoints"
CKPT_GLOB   = "checkpoint-*"

# Local mtime cache so we don't re-upload unchanged files
local_state = {}

def sig(local: Path):
    if not local.exists() or local.is_dir():
        return None
    return (local.stat().st_size, local.stat().st_mtime_ns)

def upload_one(local: Path, blob_name: str):
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local))

def maybe_upload_pool(plan, label=""):
    """Upload a list of (Path, blob_name) tuples in parallel; return (uploaded, skipped, bytes, errors)."""
    uploaded = skipped = bytes_up = 0
    errors = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = []
        for local, blob_name in plan:
            s = sig(local)
            if s is None:
                continue
            if local_state.get(str(local)) == s:
                skipped += 1
                continue
            futs.append((local, blob_name, s, ex.submit(upload_one, local, blob_name)))
        for local, blob_name, s, fut in futs:
            try:
                fut.result()
                local_state[str(local)] = s
                uploaded += 1
                bytes_up += s[0]
            except Exception as e:
                errors.append((blob_name, str(e)))
    return uploaded, skipped, bytes_up, errors

def list_remote_dirs_under(subprefix: str):
    """List immediate child dir names under prefix_root + subprefix."""
    found = set()
    full = prefix_root + (subprefix.rstrip("/") + "/" if subprefix else "")
    for blob in client.list_blobs(bucket, prefix=full):
        rel = blob.name[len(full):]
        first = rel.split("/", 1)[0]
        if first:
            found.add(first)
    return sorted(found)

def delete_remote_dir(name_or_path: str):
    """Delete all blobs under prefix_root + name_or_path."""
    n = 0
    full = prefix_root + name_or_path.rstrip("/") + "/"
    for blob in client.list_blobs(bucket, prefix=full):
        blob.delete()
        n += 1
    return n

def parse_step(dirname: str):
    """Extract step number from 'checkpoint-N' or 'step_N' or 'stage_K_step_N_LM'."""
    import re
    m = re.search(r'(\d+)', dirname)
    return int(m.group(1)) if m else -1

def should_stop():
    if (run_dir / "STOP_SYNC").exists():
        return True
    if train_pid > 0:
        try:
            os.kill(train_pid, 0)
        except ProcessLookupError:
            return True
    return False

def disk_usage_gb(path: str) -> float:
    """Return GiB used at path (du -sb)."""
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], stderr=subprocess.DEVNULL).decode().split()[0]
        return int(out) / (1024**3)
    except Exception:
        return -1.0

def sync_once():
    if not run_dir.exists():
        print(f"[sync] run_dir not yet present: {run_dir}", flush=True)
        return

    # ----- 1) plan top-level + stage uploads -----
    top_plan = []
    for pat in TOP_LEVEL_GLOBS:
        for f in run_dir.glob(pat):
            if f.is_file():
                top_plan.append((f, prefix_root + f.name))

    stage_plan = []
    sd = run_dir / STAGE_DIR
    if sd.exists():
        for f in sd.rglob("*"):
            if f.is_file():
                stage_plan.append((f, prefix_root + str(f.relative_to(run_dir))))

    # ----- 2) persistent_checkpoints (sort by step, sync all, prune locally) -----
    persist_plan = []
    persist_dirs = []
    pd = run_dir / PERSIST_DIR
    if pd.exists():
        persist_dirs = sorted([d for d in pd.iterdir() if d.is_dir()],
                              key=lambda p: parse_step(p.name))
        for d in persist_dirs:
            for f in d.rglob("*"):
                if f.is_file():
                    persist_plan.append((f, prefix_root + str(f.relative_to(run_dir))))

    # ----- 3) checkpoint-N/ (sort by step) -----
    ckpt_dirs = sorted(run_dir.glob(CKPT_GLOB),
                       key=lambda p: parse_step(p.name))
    keep_local_ckpt_set = ckpt_dirs[-keep_local_ckpt:] if ckpt_dirs else []
    keep_remote_ckpt_set = ckpt_dirs[-keep_remote_ckpt:] if ckpt_dirs else []
    ckpt_plan = []
    for d in keep_remote_ckpt_set:
        for f in d.rglob("*"):
            if f.is_file():
                ckpt_plan.append((f, prefix_root + str(f.relative_to(run_dir))))

    # ----- 4) UPLOAD all -----
    t0 = time.time()
    upl = skp = byt = 0
    for label, plan in [("top", top_plan), ("stage", stage_plan),
                        ("persist", persist_plan), ("ckpt", ckpt_plan)]:
        u, s, b, errs = maybe_upload_pool(plan, label)
        upl += u; skp += s; byt += b
        for n, e in errs:
            print(f"[sync][ERR] {label}: {n}: {e}", flush=True)

    # ----- 5) PRUNE remote checkpoint-N (rolling) -----
    pruned_remote_blobs = 0
    keep_local_names  = {d.name for d in keep_local_ckpt_set}
    keep_remote_names = {d.name for d in keep_remote_ckpt_set}
    for n in list_remote_dirs_under(""):
        if n.startswith("checkpoint-") and n not in keep_remote_names:
            try:
                pruned_remote_blobs += delete_remote_dir(n)
            except Exception as e:
                print(f"[sync][prune-ERR] remote {n}: {e}", flush=True)

    # ----- 6) PRUNE remote persistent_checkpoints/ if KEEP_REMOTE_PERSIST > 0 -----
    if keep_remote_persist > 0:
        remote_persist = list_remote_dirs_under(PERSIST_DIR)
        # Sort by step number
        remote_persist_sorted = sorted(remote_persist, key=parse_step)
        keep_remote_persist_set = set(remote_persist_sorted[-keep_remote_persist:])
        for n in remote_persist_sorted:
            if n not in keep_remote_persist_set:
                try:
                    pruned_remote_blobs += delete_remote_dir(f"{PERSIST_DIR}/{n}")
                except Exception as e:
                    print(f"[sync][prune-ERR] remote persist {n}: {e}", flush=True)

    # ----- 7) PRUNE LOCAL: checkpoint-N/ -----
    pruned_local_dirs = 0
    pruned_local_bytes = 0
    for d in ckpt_dirs:
        if d not in keep_local_ckpt_set and d.exists():
            sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            try:
                shutil.rmtree(d)
                pruned_local_dirs += 1
                pruned_local_bytes += sz
                print(f"[sync] local-pruned {d.name} ({sz/1e9:.2f} GB)", flush=True)
            except Exception as e:
                print(f"[sync][prune-ERR] local {d}: {e}", flush=True)

    # ----- 8) PRUNE LOCAL: persistent_checkpoints/ -----
    if persist_dirs:
        keep_local_persist_set = set(persist_dirs[-keep_local_persist:])
        for d in persist_dirs:
            if d not in keep_local_persist_set and d.exists():
                sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                try:
                    shutil.rmtree(d)
                    pruned_local_dirs += 1
                    pruned_local_bytes += sz
                    print(f"[sync] local-pruned persist/{d.name} ({sz/1e9:.2f} GB)", flush=True)
                except Exception as e:
                    print(f"[sync][prune-ERR] local persist {d}: {e}", flush=True)

    # ----- 9) Manifest -----
    latest_step = max([parse_step(d.name) for d in keep_local_ckpt_set], default=0)
    rd_gb = disk_usage_gb(run_dir)
    manifest = {
        "synced_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "rolling+local-prune",
        "run_dir": str(run_dir),
        "dest": gcs_url,
        "latest_checkpoint_step": latest_step,
        "kept_checkpoints_local":  [d.name for d in keep_local_ckpt_set],
        "kept_checkpoints_remote": [d.name for d in keep_remote_ckpt_set],
        "kept_persist_local":  [d.name for d in (persist_dirs[-keep_local_persist:] if persist_dirs else [])],
        "uploaded_files":   upl,
        "skipped_files":    skp,
        "bytes_uploaded":   byt,
        "pruned_remote_blobs": pruned_remote_blobs,
        "pruned_local_dirs":   pruned_local_dirs,
        "pruned_local_bytes":  pruned_local_bytes,
        "run_dir_size_gb":  round(rd_gb, 2),
    }
    bucket.blob(prefix_root + "sync_manifest.json").upload_from_string(
        json.dumps(manifest, indent=2), content_type="application/json")
    print(f"[sync] {datetime.datetime.now().strftime('%H:%M:%S')}  "
          f"uploaded={upl:4d} skipped={skp:5d} "
          f"bytes_up={byt/1e6:7.1f}MB "
          f"pruned_remote={pruned_remote_blobs:3d} "
          f"pruned_local={pruned_local_dirs:2d}({pruned_local_bytes/1e9:.2f}GB) "
          f"run_dir={rd_gb:.2f}GB "
          f"local_ckpt={manifest['kept_checkpoints_local']} "
          f"elapsed={time.time()-t0:.1f}s", flush=True)


while True:
    if should_stop():
        print("[sync] stop requested or train pid gone -- final sync then exit", flush=True)
        try: sync_once()
        except Exception as e: print(f"[sync][ERR final] {e}", flush=True)
        break
    try:
        sync_once()
    except Exception as e:
        print(f"[sync][ERR] {e}", flush=True)
    time.sleep(interval)

print(f"[sync] exit {datetime.datetime.now()}")
PY

#!/usr/bin/env bash
# Periodic sync of a training run dir to GCS.
#
# Two modes:
#   MIRROR=1 (default)        : full recursive rsync of RUN_DIR -> DEST.
#                               Anything that's ever appeared locally stays in GCS,
#                               even after the trainer rotates it locally.
#   MIRROR=0                  : "rolling" mode. Top-level artifacts rsync'd every
#                               run, only the newest KEEP_N=2 rolling checkpoints
#                               uploaded; older rolling checkpoints on GCS are
#                               pruned. Persistent dirs (pflops_checkpoints/,
#                               persistent_checkpoints/, stage_checkpoints/) are
#                               uploaded append-only — never deleted from GCS even
#                               if they disappear locally (so safe local cleanup
#                               does not nuke remote copies).
#
# Auth:
#   If SA_KEY is set or ./gcs-bucket-sa.json (next to this script) exists, the
#   service account is activated automatically. Set SA_KEY="" to disable.
#
# Local cleanup (CLEANUP=1):
#   After a successful sync iteration, walks the persistent dirs (default:
#   pflops_checkpoints/ persistent_checkpoints/ stage_checkpoints/) and for each
#   inner sub-dir verifies every file is present on GCS with matching size. Only
#   if ALL files match does it `rm -rf` the local sub-dir. Top-level state files,
#   the rolling checkpoints in the keep list, and any sub-dir with even one
#   unverified file are left untouched.
#
# Usage:
#   one-shot:   RUN_DIR=/path/to/run  ./sync_to_gcs.sh
#   loop+roll:  LOOP=1 INTERVAL_SEC=600 MIRROR=0 CLEANUP=1 \
#                  RUN_DIR=/path/to/run ./sync_to_gcs.sh
#
# Per-run lock + log: derived from basename(RUN_DIR) so multiple loops can coexist.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="/home/ray/google-cloud-sdk/bin:$PATH"

RUN_DIR="${RUN_DIR:?must set RUN_DIR=/abs/path/to/run}"
RUN_NAME=$(basename "$RUN_DIR")
BUCKET="${BUCKET:-jackierwzhang-purdue-research-curriculum}"
DEST_PREFIX="${DEST_PREFIX:-${RUN_NAME}}"
DEST="gs://${BUCKET}/${DEST_PREFIX}"
TRAIN_LOG="${TRAIN_LOG:-}"
SYNC_LOG="${SYNC_LOG:-/tmp/nl_gcs_sync_${RUN_NAME}.log}"
LOCK="${LOCK:-/tmp/nl_gcs_sync_${RUN_NAME}.lock}"
PID_FILE="${PID_FILE:-/tmp/nl_gcs_sync_${RUN_NAME}.pid}"
KEEP_N="${KEEP_N:-2}"
INTERVAL_SEC="${INTERVAL_SEC:-1800}"
MIRROR="${MIRROR:-1}"
CLEANUP="${CLEANUP:-0}"
CLEANUP_DIRS="${CLEANUP_DIRS:-pflops_checkpoints persistent_checkpoints stage_checkpoints}"

# Default service-account key: ./gcs-bucket-sa.json next to this script.
SA_KEY_DEFAULT="$SCRIPT_DIR/gcs-bucket-sa.json"
SA_KEY="${SA_KEY-$SA_KEY_DEFAULT}"

log() { echo "[$(date -u +%FT%TZ)] $*" >> "$SYNC_LOG"; }

ensure_auth() {
  if [ -n "$SA_KEY" ] && [ -f "$SA_KEY" ]; then
    local sa_email
    sa_email=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['client_email'])" "$SA_KEY" 2>/dev/null || true)
    local active
    active=$(gcloud config get-value account 2>/dev/null || true)
    if [ -n "$sa_email" ] && [ "$active" != "$sa_email" ]; then
      log "activating service account $sa_email"
      gcloud auth activate-service-account --key-file="$SA_KEY" 2>>"$SYNC_LOG" >>"$SYNC_LOG" || {
        log "WARN: gcloud auth activate-service-account failed"
      }
    fi
    export GOOGLE_APPLICATION_CREDENTIALS="$SA_KEY"
  fi
}

upload_log() {
  if [ -n "$TRAIN_LOG" ] && [ -f "$TRAIN_LOG" ]; then
    log "upload training log ($TRAIN_LOG)"
    gcloud storage cp "$TRAIN_LOG" "$DEST/training.log" 2>> "$SYNC_LOG" >> "$SYNC_LOG"
  fi
}

write_manifest() {
  local latest_step="${1:-null}"
  local kept_json="${2:-[]}"
  local mode_str="${3:-mirror}"
  local cleaned_json="${4:-[]}"
  local tmp=$(mktemp)
  cat > "$tmp" <<EOF
{
  "synced_at_utc": "$(date -u +%FT%TZ)",
  "mode": "$mode_str",
  "run_dir": "$RUN_DIR",
  "dest": "$DEST",
  "latest_checkpoint_step": $latest_step,
  "kept_checkpoints": $kept_json,
  "locally_cleaned": $cleaned_json
}
EOF
  gcloud storage cp "$tmp" "$DEST/sync_manifest.json" 2>> "$SYNC_LOG" >> "$SYNC_LOG"
  rm -f "$tmp"
}

run_once_mirror() {
  log "=== mirror sync start (dest=$DEST) ==="
  gcloud storage rsync "$RUN_DIR" "$DEST" --recursive 2>> "$SYNC_LOG" | tail -25 >> "$SYNC_LOG"
  upload_log
  local latest_local
  latest_local=$(ls -1d "$RUN_DIR"/checkpoint-* 2>/dev/null \
      | sed 's:.*/checkpoint-::' | sort -n | tail -1)
  [ -z "$latest_local" ] && latest_local="null"
  write_manifest "$latest_local" "[]" "mirror" "[]"
  log "=== mirror sync done ==="
}

# Build the gcloud-style exclude regex that covers rolling checkpoint dirs,
# the script-managed log + manifest, and every persistent dir (so the persistent
# dirs are handled by their own no-delete rsyncs and never get pruned remotely
# when they're cleaned up locally).
build_exclude_regex() {
  local re='(^|/)checkpoint-[0-9]+(/|$)|^training\.log$|^sync_manifest\.json$'
  for d in $CLEANUP_DIRS; do
    re="$re|(^|/)${d}(/|$)"
  done
  echo "$re"
}

run_once_rolling() {
  log "=== rolling sync start (dest=$DEST keep=$KEEP_N) ==="
  local exclude_re
  exclude_re=$(build_exclude_regex)

  # 1) Mirror top-level + small files; persistent dirs and rolling ckpt dirs
  #    are handled separately below. --delete-unmatched-destination-objects
  #    keeps the remote a true mirror of local for everything not excluded.
  gcloud storage rsync "$RUN_DIR" "$DEST" --recursive \
      --exclude="$exclude_re" \
      --delete-unmatched-destination-objects \
      2>> "$SYNC_LOG" | tail -20 >> "$SYNC_LOG"

  # 2) Persistent dirs — append-only on GCS (no --delete-* flag).
  for d in $CLEANUP_DIRS; do
    if [ -d "$RUN_DIR/$d" ]; then
      log "rsync persistent $d/"
      gcloud storage rsync "$RUN_DIR/$d" "$DEST/$d" --recursive \
          2>> "$SYNC_LOG" | tail -10 >> "$SYNC_LOG"
    fi
  done

  # 3) Top KEEP_N rolling checkpoints, plus prune older ones from GCS.
  local KEEP_CKPTS=()
  local KEEP_NAMES=()
  mapfile -t KEEP_CKPTS < <(ls -1dt "$RUN_DIR"/checkpoint-* 2>/dev/null | head -n "$KEEP_N")
  if [ "${#KEEP_CKPTS[@]}" -eq 0 ]; then
    log "no local checkpoints found"
  else
    for ck in "${KEEP_CKPTS[@]}"; do KEEP_NAMES+=("$(basename "$ck")"); done
    log "keep ckpts: ${KEEP_NAMES[*]}"
    for ck in "${KEEP_CKPTS[@]}"; do
      local name=$(basename "$ck")
      log "rsync $name"
      gcloud storage rsync "$ck" "$DEST/$name" --recursive \
          2>> "$SYNC_LOG" | tail -10 >> "$SYNC_LOG"
    done
    log "prune stale rolling ckpts on GCS"
    local GCS_CKPTS
    GCS_CKPTS=$(gcloud storage ls "$DEST/" 2>/dev/null \
        | grep -oE 'checkpoint-[0-9]+/?$' | sed 's:/$::' | sort -u)
    for gck in $GCS_CKPTS; do
      local keep=0
      for k in "${KEEP_NAMES[@]}"; do
        [ "$gck" = "$k" ] && { keep=1; break; }
      done
      if [ "$keep" -eq 0 ]; then
        log "delete stale $gck"
        gcloud storage rm -r "$DEST/$gck" --quiet 2>> "$SYNC_LOG" || true
      fi
    done
  fi

  upload_log

  # 4) Optional safe local cleanup of persistent dirs.
  local CLEANED=()
  if [ "$CLEANUP" = "1" ]; then
    safe_cleanup_local CLEANED
  fi

  local latest_step="null"
  if [ "${#KEEP_CKPTS[@]}" -gt 0 ]; then
    latest_step=$(basename "${KEEP_CKPTS[0]}" | sed 's/checkpoint-//')
  fi
  local kept_json="[]"
  if [ "${#KEEP_NAMES[@]}" -gt 0 ]; then
    kept_json="[$(printf '"%s",' "${KEEP_NAMES[@]}" | sed 's/,$//')]"
  fi
  local cleaned_json="[]"
  if [ "${#CLEANED[@]}" -gt 0 ]; then
    cleaned_json="[$(printf '"%s",' "${CLEANED[@]}" | sed 's/,$//')]"
  fi
  write_manifest "$latest_step" "$kept_json" "rolling" "$cleaned_json"
  log "=== rolling sync done ==="
}

# safe_cleanup_local <out_array_name>
# For each top-level entry under each $CLEANUP_DIRS dir locally, verify every
# file has a same-size object on GCS, and only then `rm -rf` the local copy.
# Appends cleaned relative paths to the named bash array.
safe_cleanup_local() {
  local -n _out=$1
  log "--- safe local cleanup ---"

  # Build a temp index of GCS objects -> size, scoped to $DEST.
  local idx; idx=$(mktemp)
  # `gcloud storage ls -l --recursive` prints lines like:
  #   "    1234  2026-04-26T20:00:00Z  gs://bucket/prefix/path/to/file"
  # plus a trailing TOTAL line. We extract size + path-relative-to-DEST.
  gcloud storage ls -l --recursive "$DEST/" 2>/dev/null \
    | awk -v dest="$DEST/" '
        {
          path=$NF
          if (index(path, dest) != 1) next
          size=$1
          if (size !~ /^[0-9]+$/) next
          rel=substr(path, length(dest)+1)
          if (rel == "") next
          print rel"\t"size
        }
      ' > "$idx"
  local n_remote
  n_remote=$(wc -l < "$idx" || echo 0)
  log "remote index: $n_remote files under $DEST"

  for d in $CLEANUP_DIRS; do
    local base="$RUN_DIR/$d"
    [ -d "$base" ] || continue
    # Iterate top-level entries inside the persistent dir (e.g. pflops_5362).
    local entry
    while IFS= read -r entry; do
      [ -z "$entry" ] && continue
      local rel_top="$d/$(basename "$entry")"
      _verify_and_remove "$entry" "$rel_top" "$idx" _out
    done < <(find "$base" -mindepth 1 -maxdepth 1 \( -type d -o -type f \) -print)
  done

  rm -f "$idx"
  log "--- cleanup done (${#_out[@]} entries removed locally) ---"
}

# _verify_and_remove <local_path> <rel_top> <idx_file> <out_array_name>
_verify_and_remove() {
  local local_path="$1" rel_top="$2" idx="$3"
  local -n _arr=$4
  local missing=0 nfiles=0 mismatch=""
  if [ -f "$local_path" ]; then
    local lsz; lsz=$(stat -c '%s' "$local_path")
    local rsz; rsz=$(awk -F'\t' -v p="$rel_top" '$1==p{print $2; exit}' "$idx")
    if [ -z "$rsz" ] || [ "$rsz" != "$lsz" ]; then
      log "  skip $rel_top  (local=$lsz remote=${rsz:-MISSING})"
      return 0
    fi
    nfiles=1
  else
    while IFS= read -r -d '' f; do
      nfiles=$((nfiles+1))
      local relfile="${f#$RUN_DIR/}"
      local lsz; lsz=$(stat -c '%s' "$f")
      local rsz; rsz=$(awk -F'\t' -v p="$relfile" '$1==p{print $2; exit}' "$idx")
      if [ -z "$rsz" ] || [ "$rsz" != "$lsz" ]; then
        mismatch="$relfile (local=$lsz remote=${rsz:-MISSING})"
        missing=1
        break
      fi
    done < <(find "$local_path" -type f -print0)
    if [ "$missing" = 1 ]; then
      log "  skip $rel_top  ($mismatch)"
      return 0
    fi
  fi
  log "  delete $rel_top  ($nfiles files verified on GCS)"
  rm -rf "$local_path"
  _arr+=("$rel_top")
}

run_once() {
  exec 9>"$LOCK"
  if ! flock -n 9; then
    log "another sync already running; skip"
    return 0
  fi
  if [ ! -d "$RUN_DIR" ]; then
    log "RUN_DIR missing: $RUN_DIR"
    return 1
  fi
  ensure_auth
  if [ "$MIRROR" = "1" ]; then
    run_once_mirror
  else
    run_once_rolling
  fi
}

if [ "${LOOP:-0}" = "1" ]; then
  echo "$$" > "$PID_FILE"
  log ">>> loop start (run=$RUN_NAME mirror=$MIRROR cleanup=$CLEANUP interval=${INTERVAL_SEC}s pid=$$ dest=$DEST)"
  trap 'log "<<< loop received signal, exiting"; rm -f "$PID_FILE"; exit 0' INT TERM
  while true; do
    run_once || log "iteration failed (non-fatal, continuing)"
    sleep "$INTERVAL_SEC"
  done
else
  run_once
fi

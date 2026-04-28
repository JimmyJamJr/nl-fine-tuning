#!/usr/bin/env bash
# Periodic sync of a training run dir to GCS.
#
# Auth: uses GOOGLE_APPLICATION_CREDENTIALS if set, otherwise auto-picks the
# repo-local gcs-bucket-sa.json. Activates that service-account up front so
# callers don't have to.
#
# GCS layout (matches existing convention in
# gs://jackierwzhang-purdue-research-curriculum/):
#   gs://<BUCKET>/<MODEL_TAG>/<RUN_NAME>/...
#     ^ bucket            ^ kebab-case model+run-flavor (e.g. qwen17b-6pct-dolci-stage32-L48)
#                                       ^ basename(RUN_DIR), already includes the JOB_ID timestamp
#                                         (e.g. job_local_20260426_172317_qwen17b_6pct_L48)
# RUN_NAME is taken from basename(RUN_DIR); the JOB_ID timestamp baked into it
# satisfies the "subdir should be timestamped" requirement.
#
# Two modes:
#   MIRROR=1            : full recursive rsync of RUN_DIR -> DEST.
#                         Anything that's ever appeared locally stays in GCS,
#                         even after the trainer rotates it locally.
#   MIRROR=0 (default)  : "rolling" mode. All non-checkpoint artifacts (loss
#                         history, plots, persistent_checkpoints/, stage_checkpoints/,
#                         logs, manifest) are mirrored. Rolling checkpoint-NNNN
#                         dirs are uploaded, then only the KEEP_N highest-step
#                         dirs are kept on GCS. Older local checkpoint-* dirs
#                         are pruned after a successful upload + quiet period
#                         so the local disk does not fill.
#
# Defaults (match what the user wants for the 6%-mix L=48 resume run):
#   INTERVAL_SEC=600           every 10 minutes
#   KEEP_N=2                   only keep last 2 rolling checkpoints on GCS
#   PRUNE_LOCAL_CHECKPOINTS=1  safely prune older local checkpoint-* dirs after upload
#   LOCAL_PRUNE_MIN_AGE_MIN=10 skip pruning a dir if files were modified within last 10m
#                              (Trainer may still be writing)
#
# Usage:
#   one-shot:   RUN_DIR=/path/to/run MODEL_TAG=qwen17b-6pct-dolci-stage32-L48 ./sync_to_gcs.sh
#   loop 10m:   LOOP=1 INTERVAL_SEC=600 RUN_DIR=/path/to/run MODEL_TAG=... ./sync_to_gcs.sh
#   full mirror: MIRROR=1 RUN_DIR=... MODEL_TAG=... ./sync_to_gcs.sh
#   override:    DEST_PREFIX=my-experiment/run-a RUN_DIR=... ./sync_to_gcs.sh
#
# Per-run lock + log: derived from basename(RUN_DIR) so multiple loops can coexist.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
  if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "GOOGLE_APPLICATION_CREDENTIALS points to missing file: $GOOGLE_APPLICATION_CREDENTIALS" >&2
    exit 1
  fi
  export GOOGLE_APPLICATION_CREDENTIALS
elif [ -f "$SCRIPT_DIR/gcs-bucket-sa.json" ]; then
  export GOOGLE_APPLICATION_CREDENTIALS="$SCRIPT_DIR/gcs-bucket-sa.json"
fi

if [ -d "/home/ray/google-cloud-sdk/bin" ]; then
  export PATH="/home/ray/google-cloud-sdk/bin:$PATH"
fi
command -v gcloud >/dev/null 2>&1 || { echo "gcloud not on PATH" >&2; exit 1; }

# Activate the service account once up-front; idempotent + silent.
if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
  gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS" --quiet >/dev/null 2>&1 || true
fi

RUN_DIR="${RUN_DIR:?must set RUN_DIR=/abs/path/to/run}"
RUN_NAME=$(basename "$RUN_DIR")
BUCKET="${BUCKET:-jackierwzhang-purdue-research-curriculum}"
# DEST is gs://<BUCKET>/<MODEL_TAG>/<RUN_NAME>. Either DEST_PREFIX (full
# "<top-dir>/<sub-dir>") or MODEL_TAG (just the top-level model-flavor dir)
# must be provided so we don't silently back up to a wrong/stale folder.
if [ -z "${DEST_PREFIX:-}" ]; then
  if [ -z "${MODEL_TAG:-}" ]; then
    echo "Must set MODEL_TAG (e.g. qwen17b-6pct-dolci-stage32-L48) or DEST_PREFIX (e.g. <model-tag>/<run-name>)." >&2
    echo "MODEL_TAG should be a kebab-case top-level dir matching others in gs://${BUCKET}/" >&2
    exit 1
  fi
  DEST_PREFIX="${MODEL_TAG}/${RUN_NAME}"
fi
DEST="gs://${BUCKET}/${DEST_PREFIX}"
TRAIN_LOG="${TRAIN_LOG:-}"
SYNC_LOG="${SYNC_LOG:-/tmp/nl_gcs_sync_${RUN_NAME}.log}"
LOCK="${LOCK:-/tmp/nl_gcs_sync_${RUN_NAME}.lock}"
PID_FILE="${PID_FILE:-/tmp/nl_gcs_sync_${RUN_NAME}.pid}"
KEEP_N="${KEEP_N:-2}"
INTERVAL_SEC="${INTERVAL_SEC:-600}"
MIRROR="${MIRROR:-0}"
PRUNE_LOCAL_CHECKPOINTS="${PRUNE_LOCAL_CHECKPOINTS:-1}"
LOCAL_PRUNE_MIN_AGE_MIN="${LOCAL_PRUNE_MIN_AGE_MIN:-10}"

log() { echo "[$(date -u +%FT%TZ)] $*" >> "$SYNC_LOG"; }

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
  local tmp=$(mktemp)
  cat > "$tmp" <<EOF
{
  "synced_at_utc": "$(date -u +%FT%TZ)",
  "mode": "$mode_str",
  "run_dir": "$RUN_DIR",
  "dest": "$DEST",
  "latest_checkpoint_step": $latest_step,
  "kept_checkpoints": $kept_json
}
EOF
  gcloud storage cp "$tmp" "$DEST/sync_manifest.json" 2>> "$SYNC_LOG" >> "$SYNC_LOG"
  rm -f "$tmp"
}

run_once_mirror() {
  log "=== mirror sync start (dest=$DEST) ==="
  gcloud storage rsync "$RUN_DIR" "$DEST" --recursive --no-ignore-symlinks \
      2>> "$SYNC_LOG" | tail -25 >> "$SYNC_LOG"
  upload_log
  # Latest local checkpoint step (best-effort, for the manifest)
  local latest_local
  latest_local=$(ls -1d "$RUN_DIR"/checkpoint-* 2>/dev/null \
      | sed 's:.*/checkpoint-::' | sort -n | tail -1)
  [ -z "$latest_local" ] && latest_local="null"
  write_manifest "$latest_local" "[]" "mirror"
  log "=== mirror sync done ==="
}

# True if GCS prefix looks like a complete HF Trainer checkpoint (after rsync).
gcs_checkpoint_has_weights() {
  local ckpt_url="${1%/}"
  gcloud storage ls "${ckpt_url}/**" 2>/dev/null | grep -q 'model\.safetensors'
}

run_once_rolling() {
  log "=== rolling sync start (dest=$DEST keep=$KEEP_N local_prune=${PRUNE_LOCAL_CHECKPOINTS}) ==="
  # Mirror top-level files (and persistent_checkpoints/, stage_checkpoints/),
  # excluding the rolling checkpoint-NNNN dirs (managed below) and excluding
  # the script-managed training.log + sync_manifest.json from any deletion.
  # --delete-unmatched-destination-objects ensures the remote is a true mirror
  # of the local for everything that's not excluded.
  gcloud storage rsync "$RUN_DIR" "$DEST" --recursive --no-ignore-symlinks \
      --exclude='(^|/)checkpoint-[0-9]+(/|$)|^training\.log$|^sync_manifest\.json$' \
      --delete-unmatched-destination-objects \
      2>> "$SYNC_LOG" | tail -20 >> "$SYNC_LOG"

  local KEEP_NAMES=()
  local ALL_STEPS=()
  mapfile -t ALL_STEPS < <(ls -1d "$RUN_DIR"/checkpoint-* 2>/dev/null | sed 's:.*/checkpoint-::' | sort -n)

  if [ "${#ALL_STEPS[@]}" -eq 0 ]; then
    log "no local checkpoints found (skip checkpoint upload / GCS+local prune)"
  else
    # KEEP = highest KEEP_N steps among local dirs (Trainer-compatible).
    local n=${#ALL_STEPS[@]}
    local start=$((n > KEEP_N ? n - KEEP_N : 0))
    local i s
    for ((i = start; i < n; i++)); do
      s="${ALL_STEPS[$i]}"
      [ -z "$s" ] && continue
      [ -d "$RUN_DIR/checkpoint-$s" ] || continue
      KEEP_NAMES+=("checkpoint-$s")
    done
    log "keep ckpts (by step): ${KEEP_NAMES[*]}"

    # Upload every local checkpoint so we never delete local-only state before GCS has a copy.
    local uploaded
    uploaded=$(mktemp)
    for s in "${ALL_STEPS[@]}"; do
      [ -z "$s" ] && continue
      [ -d "$RUN_DIR/checkpoint-$s" ] || continue
      log "rsync checkpoint-$s -> GCS"
      gcloud storage rsync "$RUN_DIR/checkpoint-$s" "$DEST/checkpoint-$s" --recursive --no-ignore-symlinks \
          2>> "$SYNC_LOG" | tail -15 >> "$SYNC_LOG"
      if [ "${PIPESTATUS[0]}" -eq 0 ]; then
        printf '%s\n' "$s" >> "$uploaded"
      else
        log "WARN rsync failed for checkpoint-$s (will not local-prune this dir)"
      fi
    done

    # Safe local disk prune: only non-KEEP dirs, only after successful upload + quiescence + GCS verify.
    if [ "$PRUNE_LOCAL_CHECKPOINTS" = "1" ]; then
      for s in "${ALL_STEPS[@]}"; do
        [ -z "$s" ] && continue
        local ck="$RUN_DIR/checkpoint-$s"
        [ -d "$ck" ] || continue
        local is_keep=0 k
        for k in "${KEEP_NAMES[@]}"; do
          [ "checkpoint-$s" = "$k" ] && { is_keep=1; break; }
        done
        [ "$is_keep" = "1" ] && continue
        if ! grep -qx "$s" "$uploaded" 2>/dev/null; then
          log "skip local prune checkpoint-$s (upload did not succeed this run)"
          continue
        fi
        if find "$ck" -type f -mmin -"$LOCAL_PRUNE_MIN_AGE_MIN" -print -quit 2>/dev/null | grep -q .; then
          log "skip local prune checkpoint-$s (files modified within ${LOCAL_PRUNE_MIN_AGE_MIN}m)"
          continue
        fi
        if ! gcs_checkpoint_has_weights "$DEST/checkpoint-$s"; then
          log "skip local prune checkpoint-$s (GCS missing model.safetensors)"
          continue
        fi
        log "local prune checkpoint-$s (copy verified on GCS)"
        rm -rf -- "$ck" 2>> "$SYNC_LOG" || log "WARN rm -rf failed: $ck"
      done
    fi
    rm -f "$uploaded"

    log "prune stale ckpts on GCS"
    local GCS_CKPTS
    GCS_CKPTS=$(gcloud storage ls "$DEST/" 2>/dev/null \
        | grep -oE 'checkpoint-[0-9]+' | sort -u)
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

  local latest_step="null"
  if [ "${#KEEP_NAMES[@]}" -gt 0 ]; then
    latest_step=$(printf '%s\n' "${KEEP_NAMES[@]}" | sed 's/checkpoint-//' | sort -n | tail -1)
  fi
  local kept_json="[]"
  if [ "${#KEEP_NAMES[@]}" -gt 0 ]; then
    kept_json="[$(printf '"%s",' "${KEEP_NAMES[@]}" | sed 's/,$//')]"
  fi
  write_manifest "$latest_step" "$kept_json" "rolling"
  log "=== rolling sync done ==="
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
  if [ "$MIRROR" = "1" ]; then
    run_once_mirror
  else
    run_once_rolling
  fi
}

if [ "${LOOP:-0}" = "1" ]; then
  echo "$$" > "$PID_FILE"
  log ">>> loop start (run=$RUN_NAME mirror=$MIRROR interval=${INTERVAL_SEC}s pid=$$)"
  trap 'log "<<< loop received signal, exiting"; rm -f "$PID_FILE"; exit 0' INT TERM
  while true; do
    run_once || log "iteration failed (non-fatal, continuing)"
    sleep "$INTERVAL_SEC"
  done
else
  run_once
fi

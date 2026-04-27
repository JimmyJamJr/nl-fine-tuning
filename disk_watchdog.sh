#!/bin/bash
# ============================================================================
# Disk-pressure watchdog: monitors local artifact size for the training run
# and the overall pod ephemeral storage. Acts before the kernel kills the pod.
# ============================================================================
#
# Thresholds (override via env):
#   - RUN_DIR_SOFT_GB (default 30): warn + nudge sync daemon to prune harder
#   - RUN_DIR_HARD_GB (default 50): SIGINT training so it gets a graceful save+exit
#   - POD_HARD_GB    (default 400): hard pod-level cap; SIGINT training
#
# Action policy:
#   - If RUN_DIR_HARD_GB hit: signal training to exit gracefully (SIGINT
#     triggers HF Trainer save_on_signal). Sync daemon will then push the
#     final checkpoint to GCS before it stops.
#   - If RUN_DIR_SOFT_GB hit: log warning + drop a HARD_PRUNE_NOW flag the
#     sync daemon picks up to immediately prune to the strictest keep set.
# ============================================================================

set -uo pipefail

RUN_DIR="${RUN_DIR:-/home/ray/scratch/nl_output/search/job_local_20260423_230739_L256_BS96_noGC}"
TRAIN_PID="${TRAIN_PID:-}"
SYNC_PID="${SYNC_PID:-}"
RUN_DIR_SOFT_GB="${RUN_DIR_SOFT_GB:-30}"
RUN_DIR_HARD_GB="${RUN_DIR_HARD_GB:-50}"
POD_HARD_GB="${POD_HARD_GB:-400}"
INTERVAL_SEC="${INTERVAL_SEC:-60}"

echo "[watchdog] start $(date)"
echo "[watchdog]   RUN_DIR=$RUN_DIR"
echo "[watchdog]   TRAIN_PID=$TRAIN_PID  SYNC_PID=$SYNC_PID"
echo "[watchdog]   thresholds: run-dir soft=${RUN_DIR_SOFT_GB}GB hard=${RUN_DIR_HARD_GB}GB  pod hard=${POD_HARD_GB}GB"
echo "[watchdog]   interval=${INTERVAL_SEC}s"

train_alive() {
    [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null
}

run_dir_gb() {
    if [ -d "$RUN_DIR" ]; then
        du -sb "$RUN_DIR" 2>/dev/null | awk '{printf "%.2f", $1/1024/1024/1024}'
    else
        echo "0.00"
    fi
}

pod_used_gb() {
    df -B1 /home/ray 2>/dev/null | awk 'NR==2 {printf "%.2f", $3/1024/1024/1024}'
}

graceful_kill_train() {
    if train_alive; then
        # Find the actual torchrun python workers and signal the bash launcher
        echo "[watchdog] SIGINT -> training pid $TRAIN_PID (graceful save+exit)"
        kill -INT "$TRAIN_PID" 2>/dev/null || true
        # Also signal child python processes directly so HF Trainer's
        # save-on-signal path runs even if bash didn't forward.
        for pid in $(pgrep -P "$TRAIN_PID" 2>/dev/null); do
            kill -INT "$pid" 2>/dev/null || true
        done
        # Signal torchrun's worker pythons too
        pgrep -f "tuning_nl.py" | while read -r p; do
            kill -INT "$p" 2>/dev/null || true
        done
    fi
}

while train_alive; do
    rd_gb=$(run_dir_gb)
    pod_gb=$(pod_used_gb)
    ts=$(date +%H:%M:%S)
    echo "[watchdog] $ts run_dir=${rd_gb}GB  pod=${pod_gb}GB  (soft=${RUN_DIR_SOFT_GB} hard=${RUN_DIR_HARD_GB} pod=${POD_HARD_GB})"

    # awk-based float compare: returns 1 if $1 > $2, else 0
    gt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'; }

    if gt "$rd_gb" "$RUN_DIR_HARD_GB"; then
        echo "[watchdog][HARD] run_dir ${rd_gb}GB > ${RUN_DIR_HARD_GB}GB -- killing training"
        touch "$RUN_DIR/HARD_PRUNE_NOW" 2>/dev/null || true
        graceful_kill_train
        sleep 30
        break
    elif gt "$pod_gb" "$POD_HARD_GB"; then
        echo "[watchdog][HARD] pod ${pod_gb}GB > ${POD_HARD_GB}GB -- killing training"
        touch "$RUN_DIR/HARD_PRUNE_NOW" 2>/dev/null || true
        graceful_kill_train
        sleep 30
        break
    elif gt "$rd_gb" "$RUN_DIR_SOFT_GB"; then
        echo "[watchdog][SOFT] run_dir ${rd_gb}GB > ${RUN_DIR_SOFT_GB}GB -- nudging sync to prune"
        touch "$RUN_DIR/HARD_PRUNE_NOW" 2>/dev/null || true
    fi

    sleep "$INTERVAL_SEC"
done

echo "[watchdog] exit $(date) (training pid $TRAIN_PID gone or thresholds tripped)"

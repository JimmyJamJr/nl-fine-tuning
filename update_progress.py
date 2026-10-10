#!/usr/bin/env python3
"""Sync Experiment 1 results from GCS and generate PROGRESS.md.

Why: Keep `/usr/local/google/home/jzrw/nl-fine-tuning-exp1/PROGRESS.md` and
`results/job_exp1_*/` synchronized with live training metrics and grounded ETAs
from `nl-exp1-qwen06b` every 15 minutes (and after stage transitions).
"""

from dataclasses import dataclass
import datetime
import json
import os
import re
import subprocess
import time
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True, kw_only=True)
class ArmSpec:
  arm_label: str
  job_id: str
  mode_desc: str
  wave_desc: str


@dataclass(frozen=True, kw_only=True)
class ArmLiveMetrics:
  status: str
  gpu_slot: str
  step: int
  stage: Optional[int]
  stage_start_step: int
  lookahead: Optional[int]
  rolling_full_acc: Optional[float]
  rolling_first_acc: Optional[float]
  recent_loss: Optional[float]
  cumulative_pflops: float
  achieved_tflops: Optional[float]
  wall_time_hours: float
  eta_stage_weighted_hours: Optional[float]
  eta_pflops_ref_hours: Optional[float]
  eta_display: str


@dataclass(frozen=True, kw_only=True)
class StageEvalRecord:
  arm_label: str
  job_id: str
  stage: int
  step: int
  effective_l: int
  alpha_training: float
  tf_loss: float
  greedy_first_hard: float
  greedy_full_hard: float
  greedy_first_stage: Optional[float]
  greedy_full_stage: Optional[float]


ARMS: Tuple[ArmSpec, ...] = (
    ArmSpec(
        arm_label="Full-FT",
        job_id="exp1_qwen06b_fullft",
        mode_desc="Full FT",
        wave_desc="0,1,2,3 (On-Demand 4x A100: nl-exp1-qwen06b, asia-southeast1-c)",
    ),
    ArmSpec(
        arm_label="LoRA `R=8`",
        job_id="exp1_qwen06b_lora_r8",
        mode_desc="`--use_lora --lora_rank 8` (`alpha=16, dropout=0.1`)",
        wave_desc="0,1,2,3 (Spot 4x H100: nl-exp1-spot-r8, us-east4-a)",
    ),
    ArmSpec(
        arm_label="LoRA `R=64`",
        job_id="exp1_qwen06b_lora_r64",
        mode_desc="`--use_lora --lora_rank 64` (`alpha=128, dropout=0.1`)",
        wave_desc="0,1,2,3 (Spot 4x H100: nl-exp1-spot-r64, us-east4-a)",
    ),
    ArmSpec(
        arm_label="LoRA `R=256`",
        job_id="exp1_qwen06b_lora_r256",
        mode_desc="`--use_lora --lora_rank 256` (`alpha=512, dropout=0.1`)",
        wave_desc="0,1,2,3 (Spot 4x H100: nl-exp1-spot-r256, us-east4-b)",
    ),
)

WORKTREE_ROOT = "/usr/local/google/home/jzrw/nl-fine-tuning-exp1"
RESULTS_ROOT = os.path.join(WORKTREE_ROOT, "results")
PROGRESS_MD_PATH = os.path.join(WORKTREE_ROOT, "PROGRESS.md")
GCS_RESULTS_URI = "gs://research-sandbox-88180-nl-experiments/exp1/results"

# Sum of stage indices k=1..16 (L_k = 8*k) for stage-weighted curriculum progress
TOTAL_STAGE_UNITS = 136.0
# Collaborator reference total compute for Qwen3-0.6B step=8 cold-start to L=128
REF_TOTAL_PFLOPS = 82000.0

# Regex matching `[Stage 1/16 L=8] step 50 | loss_avg=1.2345(50) | First=85.00% | Full=45.00% | ... | 123.4 TFLOPs/s | Stage time=0.4m | ... | toks/step=39850`
STEP_LOG_RE = re.compile(
    r"\[Stage\s+(\d+)/(\d+)\s+L=(\d+)\]\s+step\s+(\d+)\s+\|\s+"
    r"loss_avg=([0-9.]+)\(\d+\)\s+\|\s+First=([0-9.]+)%\s+\|\s+Full=([0-9.]+)%"
    r"(?:.*?\|\s+([0-9.]+)\s+TFLOPs/s)?"
    r"(?:.*?\|\s+Stage time=([0-9.]+)m)?"
    r"(?:.*?\|\s+toks/step=([0-9]+))?"
)


def sync_from_gcs() -> None:
  """Rsync lightweight experiment artifacts from GCS into local results/."""
  os.makedirs(RESULTS_ROOT, exist_ok=True)
  ls_proc = subprocess.run(
      ["gcloud", "storage", "ls", GCS_RESULTS_URI],
      capture_output=True,
      text=True,
      check=False,
  )
  if ls_proc.returncode != 0 or not ls_proc.stdout.strip():
    return

  # Retry up to 3 times in case the VM's 60-second sync daemon overwrites an object generation mid-download.
  for _ in range(3):
    proc = subprocess.run(
        [
            "gcloud",
            "storage",
            "rsync",
            "-r",
            "--exclude=.*\\.(safetensors|bin|pt|pth)$|.*checkpoint-.*|.*/final/.*",
            GCS_RESULTS_URI,
            RESULTS_ROOT,
        ],
        check=False,
    )
    if proc.returncode == 0:
      break
    time.sleep(3)


def load_loss_records(job_dir: str) -> List[Dict[str, object]]:
  """Load loss records from loss_history.jsonl (authoritative live log) or fallback to loss_history.json."""
  json_path = os.path.join(job_dir, "loss_history.json")
  jsonl_path = os.path.join(job_dir, "loss_history.jsonl")
  records: List[Dict[str, object]] = []
  if os.path.isfile(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if line:
          try:
            records.append(json.loads(line))
          except json.JSONDecodeError:
            continue
  if not records and os.path.isfile(json_path):
    try:
      with open(json_path, "r", encoding="utf-8") as f:
        json_records = json.load(f)
      if isinstance(json_records, list):
        records = json_records
    except (OSError, json.JSONDecodeError):
      pass
  return records


def compute_eta(
    status: str,
    stage: Optional[int],
    step: int,
    stage_start_step: int,
    rolling_full_acc: Optional[float],
    cumulative_pflops: float,
    wall_time_hours: float,
    recent_pflops_per_hour: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], str]:
  """Compute stage-weighted ETA and 82k-PFLOPs reference ETA for an arm."""
  if status == "Completed":
    return 0.0, 0.0, "0.0h (Done)"
  if status == "Queued":
    return None, None, "Queued (~55-150h run)"
  if stage is None or wall_time_hours <= 0.002:
    return None, None, "Starting (measuring rate...)"

  # 1. Stage-weighted ETA using lookahead units sum_{k=1}^{16} k = 136
  completed_units = float((stage - 1) * stage) / 2.0
  steps_in_stage = max(1, step - stage_start_step)
  acc_frac = 0.0
  if rolling_full_acc is not None:
    acc_frac = max(0.0, min(0.92, (rolling_full_acc - 40.0) / (98.0 - 40.0)))
  step_frac = min(0.90, steps_in_stage / 1000.0)
  # Respect min_steps_per_stage=200 (and typical ~300-475 steps in Stage 1) so early high accuracy at step <200 does not overstate progress.
  if stage == 1:
    stage_frac = max(0.02, min(0.92, steps_in_stage / 350.0))
  elif steps_in_stage < 200:
    stage_frac = max(0.05, min(acc_frac, steps_in_stage / 200.0))
  else:
    stage_frac = max(0.05, max(acc_frac, step_frac))
  effective_units = completed_units + float(stage) * stage_frac
  remaining_units = max(0.0, TOTAL_STAGE_UNITS - effective_units)
  eta_stage_wt = wall_time_hours * (remaining_units / max(effective_units, 0.02))

  # 2. Reference 82,000 PFLOPs ETA based on achieved PFLOPs/hour rate.
  # Why: If an arm migrated from slower hardware (4x A100) to faster hardware (4x H100),
  # use the measured recent PFLOPs/hour rate for remaining work and adjust stage-wt ETA proportionally.
  avg_pflops_per_hour = cumulative_pflops / wall_time_hours if wall_time_hours > 0 else 0.0
  effective_pflops_per_hour = avg_pflops_per_hour
  if recent_pflops_per_hour is not None and recent_pflops_per_hour > 0:
    if avg_pflops_per_hour > 0 and recent_pflops_per_hour > 1.25 * avg_pflops_per_hour:
      eta_stage_wt = eta_stage_wt * (avg_pflops_per_hour / recent_pflops_per_hour)
      effective_pflops_per_hour = recent_pflops_per_hour

  if effective_pflops_per_hour > 0:
    remaining_pflops = max(0.0, REF_TOTAL_PFLOPS - cumulative_pflops)
    eta_pflops_ref = remaining_pflops / effective_pflops_per_hour
  else:
    eta_pflops_ref = eta_stage_wt

  eta_str = f"~{eta_stage_wt:.1f}h (stage-wt) / ~{eta_pflops_ref:.1f}h (82k PFLOPs ref)"
  return eta_stage_wt, eta_pflops_ref, eta_str


def parse_arm_metrics(spec: ArmSpec) -> Tuple[ArmLiveMetrics, List[StageEvalRecord]]:
  """Parse live metrics and stage evaluation history for a single arm."""
  job_dir = os.path.join(RESULTS_ROOT, f"job_{spec.job_id}")
  eval_records: List[StageEvalRecord] = []

  if not os.path.isdir(job_dir):
    return (
        ArmLiveMetrics(
            status="Queued",
            gpu_slot=spec.wave_desc,
            step=0,
            stage=None,
            stage_start_step=0,
            lookahead=None,
            rolling_full_acc=None,
            rolling_first_acc=None,
            recent_loss=None,
            cumulative_pflops=0.0,
            achieved_tflops=None,
            wall_time_hours=0.0,
            eta_stage_weighted_hours=None,
            eta_pflops_ref_hours=None,
            eta_display="Queued (~55-150h run)",
        ),
        eval_records,
    )

  slot_file = os.path.join(job_dir, "slot_info.txt")
  gpu_slot = spec.wave_desc
  if os.path.isfile(slot_file):
    try:
      with open(slot_file, "r", encoding="utf-8") as f:
        slot_txt = f.read().strip()
      if slot_txt and "Wave 1" not in slot_txt:
        gpu_slot = slot_txt
    except OSError:
      pass

  flops_per_token = 6.0 * 596049920.0
  plot_meta_path = os.path.join(job_dir, "plot_metadata.json")
  if os.path.isfile(plot_meta_path):
    try:
      with open(plot_meta_path, "r", encoding="utf-8") as f:
        pm = json.load(f)
      if pm.get("flops_per_token"):
        flops_per_token = float(pm["flops_per_token"])
    except (OSError, json.JSONDecodeError, ValueError):
      pass

  loss_records = load_loss_records(job_dir)
  step = 0
  stage: Optional[int] = None
  stage_start_step = 0
  lookahead: Optional[int] = None
  recent_loss: Optional[float] = None
  achieved_tflops: Optional[float] = None
  wall_time_hours = 0.0
  total_tokens = 0

  for r in loss_records:
    s = int(r.get("step", 0) or 0)
    t = int(r.get("tokens", 0) or 0)
    total_tokens += t
    if s >= step:
      step = s
      stage = int(r.get("stage", 1) or 1)
      if r.get("effective_L") is not None:
        lookahead = int(r["effective_L"])
      if r.get("loss") is not None:
        recent_loss = float(r["loss"])
      if r.get("achieved_tflops") is not None and 0 < float(r["achieved_tflops"]) < 500.0:
        achieved_tflops = float(r["achieved_tflops"])
      if r.get("wall_time") is not None:
        wall_time_hours = float(r["wall_time"]) / 3600.0

  # Compute live step-delta TFLOP/s and PFLOPs/h over the most recent 30 steps.
  # Why: Mid-stage checkpoint resumes reset stage_start_time in tuning_nl.py, causing transient >500 TFLOP/s
  # values in train.log/loss_history.jsonl; step-delta timing gives exact hardware throughput.
  recent_sparse_tflops: Optional[float] = None
  recent_pflops_per_hour: Optional[float] = None
  if len(loss_records) >= 5:
    tail_recs = loss_records[-31:]
    dt_sum = 0.0
    sparse_flops_sum = 0.0
    nominal_flops_sum = 0.0
    n_total = 596049920.0
    n_head = 155582464.0
    for prev_r, curr_r in zip(tail_recs[:-1], tail_recs[1:]):
      s_prev = int(prev_r.get("step", 0) or 0)
      s_curr = int(curr_r.get("step", 0) or 0)
      wt_prev = float(prev_r.get("wall_time", 0.0) or 0.0)
      wt_curr = float(curr_r.get("wall_time", 0.0) or 0.0)
      dt = wt_curr - wt_prev
      if s_curr == s_prev + 1 and 0.2 < dt < 60.0:
        toks = float(curr_r.get("tokens", 0) or 0)
        hrows = float(curr_r.get("head_rows", 0) or 0)
        dt_sum += dt
        sparse_flops_sum += 6.0 * (n_total - n_head) * toks + 6.0 * n_head * hrows
        nominal_flops_sum += toks * flops_per_token
    if dt_sum > 5.0:
      recent_sparse_tflops = (sparse_flops_sum / dt_sum) / 1e12
      recent_pflops_per_hour = (nominal_flops_sum / 1e15) / (dt_sum / 3600.0)

  cumulative_pflops = (total_tokens * flops_per_token) / 1e15

  rolling_first_acc: Optional[float] = None
  rolling_full_acc: Optional[float] = None
  status = "Starting" if os.path.isfile(os.path.join(job_dir, "train.log")) else "Queued"

  train_log_path = os.path.join(job_dir, "train.log")
  last_log_tflops_valid = False
  if os.path.isfile(train_log_path):
    try:
      with open(train_log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
      for line in lines:
        m = STEP_LOG_RE.search(line)
        if m:
          status = "Running"
          log_stage = int(m.group(1))
          log_l = int(m.group(3))
          log_step = int(m.group(4))
          log_loss = float(m.group(5))
          log_first = float(m.group(6))
          log_full = float(m.group(7))
          log_tflops = float(m.group(8)) if m.group(8) else None
          log_stage_mins = float(m.group(9)) if m.group(9) else None
          log_toks_per_step = int(m.group(10)) if m.group(10) else None
          step = log_step
          stage = log_stage
          lookahead = log_l
          recent_loss = log_loss
          if wall_time_hours <= 0.0 and log_stage_mins is not None:
            wall_time_hours = log_stage_mins / 60.0
          if cumulative_pflops <= 0.0 and log_toks_per_step is not None:
            cumulative_pflops = (log_step * log_toks_per_step * 4.0 * flops_per_token) / 1e15
          rolling_first_acc = log_first
          rolling_full_acc = log_full
          if log_tflops is not None and log_tflops < 500.0:
            achieved_tflops = log_tflops
            last_log_tflops_valid = True
          else:
            last_log_tflops_valid = False
        elif "[CURRICULUM] Resuming from step " in line:
          res_m = re.search(r"\[CURRICULUM\] Resuming from step (\d+)", line)
          if res_m:
            step = int(res_m.group(1))
            status = "Running"
        elif " -> Advanced to stage " in line:
          adv_m = re.search(r"Advanced to stage (\d+).*?L=(\d+)", line)
          if adv_m:
            stage = int(adv_m.group(1))
            lookahead = int(adv_m.group(2))
            stage_start_step = step
        elif "[DONE] Training/evaluation complete." in line or "[FINISHED] Curriculum complete" in line:
          status = "Completed"
        elif "RuntimeError:" in line or "CUDA out of memory" in line or "Traceback (most recent call last):" in line:
          if status != "Completed":
            status = "Error (check train.log)"
    except OSError:
      pass

  if recent_sparse_tflops is not None:
    achieved_tflops = recent_sparse_tflops

  curr_state_path = os.path.join(job_dir, "curriculum_state.json")
  if os.path.isfile(curr_state_path):
    try:
      with open(curr_state_path, "r", encoding="utf-8") as f:
        cs = json.load(f)
      if cs.get("stage") is not None and (stage is None or int(cs["stage"]) >= stage):
        stage = int(cs["stage"])
        lookahead = 8 * stage
      if cs.get("stage_start_step") is not None:
        stage_start_step = int(cs["stage_start_step"])
      if cs.get("wall_time_offset") is not None:
        wt_hrs = float(cs["wall_time_offset"]) / 3600.0
        if wt_hrs > wall_time_hours:
          wall_time_hours = wt_hrs
    except (OSError, json.JSONDecodeError, ValueError):
      pass

  stage_eval_path = os.path.join(job_dir, "stage_eval_history.json")
  if os.path.isfile(stage_eval_path):
    try:
      with open(stage_eval_path, "r", encoding="utf-8") as f:
        se_list = json.load(f)
      for item in se_list:
        eval_records.append(
            StageEvalRecord(
                arm_label=spec.arm_label,
                job_id=spec.job_id,
                stage=int(item["stage"]),
                step=int(item["step"]),
                effective_l=int(item.get("effective_L") or (8 * int(item["stage"]))),
                alpha_training=float(item.get("alpha_training", 0.0)),
                tf_loss=float(item.get("tf_loss", 0.0)),
                greedy_first_hard=float(item.get("greedy_first", 0.0)),
                greedy_full_hard=float(item.get("greedy_full", 0.0)),
                greedy_first_stage=(
                    float(item["stage_greedy_first"])
                    if item.get("stage_greedy_first") is not None
                    else None
                ),
                greedy_full_stage=(
                    float(item["stage_greedy_full"])
                    if item.get("stage_greedy_full") is not None
                    else None
                ),
            )
        )
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
      pass

  eta_stage_wt, eta_pflops_ref, eta_str = compute_eta(
      status=status,
      stage=stage,
      step=step,
      stage_start_step=stage_start_step,
      rolling_full_acc=rolling_full_acc,
      cumulative_pflops=cumulative_pflops,
      wall_time_hours=wall_time_hours,
      recent_pflops_per_hour=recent_pflops_per_hour,
  )

  return (
      ArmLiveMetrics(
          status=status,
          gpu_slot=gpu_slot,
          step=step,
          stage=stage,
          stage_start_step=stage_start_step,
          lookahead=lookahead,
          rolling_full_acc=rolling_full_acc,
          rolling_first_acc=rolling_first_acc,
          recent_loss=recent_loss,
          cumulative_pflops=cumulative_pflops,
          achieved_tflops=achieved_tflops,
          wall_time_hours=wall_time_hours,
          eta_stage_weighted_hours=eta_stage_wt,
          eta_pflops_ref_hours=eta_pflops_ref,
          eta_display=eta_str,
      ),
      eval_records,
  )


def ensure_spot_vms_running() -> None:
  """Verify Spot H100 VMs (nl-exp1-spot-r8, nl-exp1-spot-r64, nl-exp1-spot-r256) are running; auto-restart or migrate if preempted."""
  provision_script = "/usr/local/google/home/jzrw/experimental/users/jzrw/exp1_scripts/provision_spot_h100.sh"
  if not os.path.isfile(provision_script):
    return
  for vm_name, job_id, rank, wait_ckpt in (
      ("nl-exp1-spot-r8", "exp1_qwen06b_lora_r8", "8", "checkpoint-3000"),
      ("nl-exp1-spot-r64", "exp1_qwen06b_lora_r64", "64", ""),
      ("nl-exp1-spot-r256", "exp1_qwen06b_lora_r256", "256", ""),
  ):
    cmd = [provision_script, vm_name, job_id, rank]
    if wait_ckpt:
      cmd.append(wait_ckpt)
    subprocess.run(cmd, check=False)


def cleanup_completed_vms() -> None:
  """Sync final checkpoints and immediately delete any Experiment 1 GCE VM whose assigned arm has completed."""
  vm_assignments = (
      ("nl-exp1-qwen06b", "exp1_qwen06b_fullft"),
      ("nl-exp1-spot-r8", "exp1_qwen06b_lora_r8"),
      ("nl-exp1-spot-r64", "exp1_qwen06b_lora_r64"),
      ("nl-exp1-spot-r256", "exp1_qwen06b_lora_r256"),
  )
  for vm_name, job_id in vm_assignments:
    local_job_dir = os.path.join(RESULTS_ROOT, f"job_{job_id}")
    final_metrics_path = os.path.join(local_job_dir, "final_metrics.json")
    if not os.path.isfile(final_metrics_path):
      continue

    # Verify GCS has final/*.safetensors and stage_checkpoints/ before deleting the VM
    gcs_final_uri = f"gs://research-sandbox-88180-nl-experiments/exp1/job_{job_id}/final"
    gcs_stage_uri = f"gs://research-sandbox-88180-nl-experiments/exp1/job_{job_id}/stage_checkpoints"
    ls_final = subprocess.run(
        ["gcloud", "storage", "ls", gcs_final_uri],
        capture_output=True,
        text=True,
        check=False,
    )
    if ls_final.returncode != 0 or ".safetensors" not in ls_final.stdout:
      continue

    # For nl-exp1-qwen06b, also ensure exp1_qwen06b_lora_r8 has already handed off checkpoint-3000
    if vm_name == "nl-exp1-qwen06b":
      ls_r8 = subprocess.run(
          [
              "gcloud",
              "storage",
              "ls",
              "gs://research-sandbox-88180-nl-experiments/exp1/job_exp1_qwen06b_lora_r8/latest_ckpt_name.txt",
          ],
          capture_output=True,
          text=True,
          check=False,
      )
      if ls_r8.returncode != 0:
        continue

    # Sync final/ and stage_checkpoints/ locally into results/job_<job_id>/ before deleting VM
    os.makedirs(os.path.join(local_job_dir, "final"), exist_ok=True)
    os.makedirs(os.path.join(local_job_dir, "stage_checkpoints"), exist_ok=True)
    subprocess.run(
        ["gcloud", "storage", "rsync", "-r", gcs_final_uri, os.path.join(local_job_dir, "final")],
        check=False,
    )
    subprocess.run(
        ["gcloud", "storage", "rsync", "-r", gcs_stage_uri, os.path.join(local_job_dir, "stage_checkpoints")],
        check=False,
    )

    # Check if VM still exists and delete it immediately
    vm_list = subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--project=research-sandbox-88180",
            f"--filter=name={vm_name}",
            "--format=value(zone)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    zone = vm_list.stdout.strip().splitlines()[0] if vm_list.stdout.strip() else ""
    if zone:
      print(f"[CLEANUP] {job_id} is complete and verified in GCS/local. Deleting VM {vm_name} in {zone}...")
      subprocess.run(
          [
              "gcloud",
              "compute",
              "instances",
              "delete",
              vm_name,
              f"--zone={zone}",
              "--project=research-sandbox-88180",
              "--quiet",
          ],
          check=False,
      )


def render_progress_md(
    metrics_by_arm: List[Tuple[ArmSpec, ArmLiveMetrics]],
    all_evals: List[StageEvalRecord],
) -> str:
  now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
  lines: List[str] = [
      "# Experiment 1: Qwen3-0.6B LoRA vs Full Fine-Tuning (Step Size 8)",
      "",
      "**Base commit:** `da7b19c` (`main`)",
      "**Branch:** `jackierwzhang/exp1-qwen06b-lora-vs-fullft`",
      "**VMs:** `nl-exp1-qwen06b` (`a2-highgpu-8g`, On-Demand A100-40GB, `asia-southeast1-c`), `nl-exp1-spot-r8` (`a3-highgpu-4g`, 4x H100-80GB Spot, `us-east4-a`), `nl-exp1-spot-r64` (`a3-highgpu-4g`, 4x H100-80GB Spot, `us-east4-a`), `nl-exp1-spot-r256` (`a3-highgpu-4g`, 4x H100-80GB Spot, `us-east4-b`)",
      f"**Last updated:** {now_utc}",
      "",
      "## Shared Configuration",
      "",
      "| Parameter | Value |",
      "| :--- | :--- |",
      "| Model | `Qwen/Qwen3-0.6B` |",
      "| Task | `search` |",
      "| Seed | `1234` |",
      "| Topology | 4 GPUs per run (`NL_DDP_GRAD_AVERAGE` unset): 1x 4x A100-SXM4-40GB (`Full-FT`) + 3x 4x H100-80GB Spot (`LoRA R=8` migrating at `checkpoint-3000`, `LoRA R=64`, `LoRA R=256`) |",
      "| Batch size / Grad accum | `48` per GPU, `gradient_accumulation_steps=4` |",
      "| Learning rate | `5e-5` (constant, 0 warmup) |",
      "| Curriculum | `base_lookahead=8`, `lookahead_step=8`, `n_stages=16` (`L=8..128`), `--linear_lookahead` |",
      "| Gate | `accuracy_threshold=0.98`, `accuracy_window=800`, `check_every=25`, `min_steps_per_stage=200` |",
      "| Context & Alpha | `max_input_size=768`, `max_lookahead=128`, `base_alpha=0.1`, `max_alpha=1.0` |",
      "| Loss & Kernels | `first_token_soft_weight=0.0`, `ce_chunk_size=4096`, `NL_ATTN_KERNEL=fa2`, `--use_liger`, `--gradient_checkpointing` |",
      "| Evals & Checkpoints | `--do_stage_eval`, `eval_samples=500`, `eval_every_steps=0`, `save_steps=500`, `save_total_limit=2`, `persist_every=0` |",
      "",
      "## Run Status",
      "",
      "| Arm | Job ID | Mode | GPUs | Status | Step | Stage / $L$ | Rolling Full Acc | Rolling First Acc | Loss | PFLOPs | TFLOP/s | Wall Time | Est. Remaining |",
      "| :--- | :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
  ]

  for spec, m in metrics_by_arm:
    stage_str = f"{m.stage} (`L={m.lookahead}`)" if m.stage is not None and m.lookahead is not None else "-"
    full_str = f"{m.rolling_full_acc:.2f}%" if m.rolling_full_acc is not None else "-"
    first_str = f"{m.rolling_first_acc:.2f}%" if m.rolling_first_acc is not None else "-"
    loss_str = f"{m.recent_loss:.4f}" if m.recent_loss is not None else "-"
    tflops_str = f"{m.achieved_tflops:.1f}" if m.achieved_tflops is not None else "-"
    lines.append(
        f"| {spec.arm_label} | `{spec.job_id}` | {spec.mode_desc} | {m.gpu_slot} | "
        f"{m.status} | {m.step} | {stage_str} | {full_str} | {first_str} | "
        f"{loss_str} | {m.cumulative_pflops:.2f} | {tflops_str} | {m.wall_time_hours:.2f}h | {m.eta_display} |"
    )

  # Compute overall concurrent 4-arm completion range across the 4 active 4-GPU slots
  running_arms = [m for _, m in metrics_by_arm if m.eta_stage_weighted_hours is not None and m.eta_pflops_ref_hours is not None]
  if running_arms:
    min_stage = min(m.eta_stage_weighted_hours for m in running_arms if m.eta_stage_weighted_hours is not None)
    max_stage = max(m.eta_stage_weighted_hours for m in running_arms if m.eta_stage_weighted_hours is not None)
    min_pflops = min(m.eta_pflops_ref_hours for m in running_arms if m.eta_pflops_ref_hours is not None)
    max_pflops = max(m.eta_pflops_ref_hours for m in running_arms if m.eta_pflops_ref_hours is not None)
    lines.extend([
        "",
        f"> **ETA Methodology & Overall Experiment 1 Completion (All 4 Arms Running Concurrently on 4x A100 + 12x Spot H100):** "
        f"`stage-wt` scales current wall time by remaining lookahead units ($\\sum_{{k=1}}^{{16}} k = 136$); "
        f"`82k PFLOPs ref` divides remaining compute against the collaborator's `Qwen3-0.6B` `step=8` cold-start reference (`82,000 PFLOPs` to `L=128`) by the arm's measured `PFLOPs/h` rate. "
        f"Fastest arm completion is estimated in **~{min_stage:.1f}h (stage-wt) / ~{min_pflops:.1f}h (82k PFLOPs ref)**; "
        f"full 4-arm concurrent completion is bounded by the slowest arm at **~{max_stage:.1f}h (stage-wt) to ~{max_pflops:.1f}h (82k PFLOPs ref)**.",
    ])

  lines.extend([
      "",
      "## Stage Evaluation Summary (`stage_eval_history.json`)",
      "",
  ])

  if not all_evals:
    lines.append("_Stage evaluation metrics populate here as stages complete._")
  else:
    lines.extend([
        "| Arm | Stage | Step | $L$ | Stage $\\alpha$ | TF Loss ($\\alpha=1.0$) | Greedy First ($\\alpha=1.0$) | Greedy Full ($\\alpha=1.0$) | Greedy First (Stage $\\alpha$) | Greedy Full (Stage $\\alpha$) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for ev in all_evals:
      gf_stage = f"{ev.greedy_first_stage * 100:.2f}%" if ev.greedy_first_stage is not None else "-"
      gfull_stage = f"{ev.greedy_full_stage * 100:.2f}%" if ev.greedy_full_stage is not None else "-"
      lines.append(
          f"| {ev.arm_label} | {ev.stage} | {ev.step} | {ev.effective_l} | "
          f"{ev.alpha_training:.3f} | {ev.tf_loss:.4f} | "
          f"{ev.greedy_first_hard * 100:.2f}% | {ev.greedy_full_hard * 100:.2f}% | "
          f"{gf_stage} | {gfull_stage} |"
      )

  lines.append("")
  return "\n".join(lines)


def main() -> int:
  ensure_spot_vms_running()
  sync_from_gcs()
  cleanup_completed_vms()
  metrics_by_arm: List[Tuple[ArmSpec, ArmLiveMetrics]] = []
  all_evals: List[StageEvalRecord] = []
  for spec in ARMS:
    m, evals = parse_arm_metrics(spec)
    metrics_by_arm.append((spec, m))
    all_evals.extend(evals)

  md_content = render_progress_md(metrics_by_arm, all_evals)
  with open(PROGRESS_MD_PATH, "w", encoding="utf-8") as f:
    f.write(md_content)
  print(f"Updated {PROGRESS_MD_PATH}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

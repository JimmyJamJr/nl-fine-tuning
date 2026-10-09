"""CPU-only resume-compatibility check for curriculum_state.json (2026-10-09 removal pass 2).

The per-stage LR-schedule family (--stage_schedule, --lr_reset_on_stage, --plateau_spike, ...) used to write three
extra keys into curriculum_state.json: lr_reset_step, batch_increase_count, plateau_last_spike_step. The family is
gone, but checkpoints written before the removal still carry those keys, so this test:
  * writes a checkpoint-1500/curriculum_state.json in the OLD format (all three legacy keys present) next to a
    loss_history.json, and calls tuning_nl._try_restore_curriculum_state on it with stub dataset/curriculum objects;
  * asserts it returns True, restores stage / stage_start_step / wall_time_offset / samples / tokens / the trainer
    deques exactly, truncates the loss history to the checkpoint step, and does NOT set any of the legacy attributes;
  * asserts _save_curriculum_state now writes a file without the legacy keys, and that re-reading it also works.
Run: /home/huan2073/.conda/envs/search/bin/python bench/test_curriculum_state_compat.py   (no GPU)
"""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.setdefault("NL_PACKING", "hf")

from tuning_nl import _save_curriculum_state, _try_restore_curriculum_state  # noqa: E402

LEGACY_KEYS = ("lr_reset_step", "batch_increase_count", "plateau_last_spike_step")


def _fresh_curriculum():
    # Only the attributes _try_restore_curriculum_state reads or writes on a FirstTokenCurriculum.
    return SimpleNamespace(stage_start_step=0, wall_time_offset=0.0, samples_this_stage=0, tokens_this_stage=0,
                           loss_history=[], resume_points=[], _cumulative_tokens=0, _mark_next_as_resume=False,
                           _last_persist_step=0, stage_eval_history=[])


def main():
    with tempfile.TemporaryDirectory() as run_dir:
        ckpt = os.path.join(run_dir, "checkpoint-1500")
        os.makedirs(ckpt)
        old_state = {
            "stage": 7, "stage_start_step": 1200, "wall_time_offset": 345.6,
            "samples_this_stage": 4800, "tokens_this_stage": 1234567,
            # legacy keys from the removed schedule family
            "lr_reset_step": 1200, "batch_increase_count": 2, "plateau_last_spike_step": 900,
            "first_token_correct": [1, 0, 1], "full_word_correct": [1, 1, 0], "recent_losses": [0.5, 0.4],
        }
        with open(os.path.join(ckpt, "curriculum_state.json"), "w") as f:
            json.dump(old_state, f)
        history = [
            {"step": 10, "loss": 1.0, "tokens": 100, "head_rows": 5, "head": "sparse", "wall_time": 5.0},
            {"step": 1500, "loss": 0.5, "tokens": 100, "head_rows": 5, "head": "sparse", "wall_time": 300.0},
            {"step": 1600, "loss": 0.4, "tokens": 100, "head_rows": 5, "head": "sparse", "wall_time": 320.0},
        ]
        with open(os.path.join(run_dir, "loss_history.json"), "w") as f:
            json.dump(history, f)

        dataset = SimpleNamespace(stage=1)
        curriculum = _fresh_curriculum()
        ok = _try_restore_curriculum_state(ckpt, dataset, curriculum)
        assert ok is True, "restore of an old-format curriculum_state.json returned False"
        assert dataset.stage == 7, dataset.stage
        assert curriculum.stage_start_step == 1200, curriculum.stage_start_step
        assert abs(curriculum.wall_time_offset - 345.6) < 1e-9, curriculum.wall_time_offset
        assert curriculum.samples_this_stage == 4800 and curriculum.tokens_this_stage == 1234567
        assert curriculum._restored_trainer_state == {
            "first_token_correct": [1, 0, 1], "full_word_correct": [1, 1, 0], "recent_losses": [0.5, 0.4]}
        for k in LEGACY_KEYS:
            assert not hasattr(curriculum, k), f"legacy key {k} was restored onto the curriculum"
        assert [h["step"] for h in curriculum.loss_history] == [10, 1500], curriculum.loss_history
        assert curriculum._cumulative_tokens == 200
        assert curriculum.resume_points == [1500] and curriculum._mark_next_as_resume is True
        assert curriculum._last_persist_step == 1500

        # The writer no longer emits the legacy keys, and its output round-trips through the same reader.
        _save_curriculum_state(ckpt, dataset.stage, curriculum.stage_start_step, curriculum.wall_time_offset,
                               first_token_correct=[1], full_word_correct=[0], recent_losses=[0.3],
                               samples_this_stage=11, tokens_this_stage=22)
        with open(os.path.join(ckpt, "curriculum_state.json")) as f:
            new_state = json.load(f)
        assert not any(k in new_state for k in LEGACY_KEYS), new_state
        assert new_state["stage"] == 7 and new_state["samples_this_stage"] == 11 and new_state["tokens_this_stage"] == 22
        dataset2, curriculum2 = SimpleNamespace(stage=1), _fresh_curriculum()
        assert _try_restore_curriculum_state(ckpt, dataset2, curriculum2) is True
        assert dataset2.stage == 7 and curriculum2.samples_this_stage == 11 and curriculum2.tokens_this_stage == 22

    print("OK: old-format curriculum_state.json (with lr_reset_step / batch_increase_count / plateau_last_spike_step) "
          "restores cleanly; new files omit those keys and round-trip.")


if __name__ == "__main__":
    main()

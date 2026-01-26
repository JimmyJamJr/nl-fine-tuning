import multiprocessing
from typing import Optional, Set, Dict, Any, Iterator, List, Tuple
import torch
from torch.utils.data import IterableDataset

from nl_generator import NaturalLanguageGraphGenerator

class SyntheticNL(IterableDataset):
    def __init__(
        self,
        *,
        max_input_size: int = 256,
        max_lookahead: int = 100,
        seed: Optional[int] = None,
        task: str = "search",
        stage: int = 4,
        reserved_inputs: Optional[Set[str]] = None,
        world_size: int = 1,
        rank: int = 0,
        start_index: int = 0,
        **task_kwargs,
    ):
        self.task = task
        self.max_input_size = int(max_input_size)
        self.max_lookahead = int(max_lookahead)
        self.seed = seed
        self._stage = multiprocessing.Value("i", int(stage))
        self._current_index = multiprocessing.Value("i", int(start_index))
        self.reserved_inputs = reserved_inputs or set()
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.start_index = int(start_index)
        self.task_kwargs = task_kwargs

    @property
    def stage(self) -> int:
        return int(self._stage.value)

    @stage.setter
    def stage(self, value: int) -> None:
        self._stage.value = int(value)

    @property
    def current_index(self) -> int:
        return int(self._current_index.value)

    def current_alpha(self) -> float:
        return float(self.stage) / float(self.max_lookahead)

    def increment_stage(self, step: int = 1) -> None:
        self.stage = self.stage + int(step)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0

        base_seed = (self.seed or 0) + self.rank * 9973 + worker_id * 997

        i = self.start_index

        while True:
            stage = self.stage
            alpha = self.current_alpha()
            max_attempts = 500
            for attempt in range(max_attempts):
                sample_seed = (base_seed + i * 1_000_003 + attempt * 104729) & 0x7FFFFFFF

                gen = NaturalLanguageGraphGenerator(self.max_input_size, seed=sample_seed)

                batch = gen.generate_batch(
                    self.task,
                    batch_size=1,
                    reserved_inputs=self.reserved_inputs,
                    alpha=alpha,
                    max_lookahead=self.max_lookahead,
                    **self.task_kwargs,
                )

                if not batch or not batch[0] or not batch[0].output_texts:
                    continue

                ex = batch[0]
                if ex.input_text in self.reserved_inputs:
                    continue

                yield {
                    "input_text": ex.input_text,
                    "output_texts": list(ex.output_texts),
                    "alpha": alpha,
                    "stage": stage,
                    "seed": int(sample_seed),
                }
                break
            else:
                raise RuntimeError(
                    f"[SyntheticNL] Failed to generate valid sample after {max_attempts} attempts "
                    f"(rank={self.rank}, worker={worker_id}, stage={stage}, alpha={alpha:.4f})."
                )

            i += 1
            self._current_index.value = i


def build_heldout_set(
    *,
    size: int,
    max_input_size: int,
    max_lookahead: int,
    seed: int = 12345,
    task: str = "search",
    reserved_inputs: Optional[Set[str]] = None,
    max_attempts_per_example: int = 2000,
    **task_kwargs,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    avoid = reserved_inputs or set()

    heldout: List[Dict[str, Any]] = []
    heldout_inputs: Set[str] = set()

    alpha = 1.0
    stage = max_lookahead

    i = 0
    while len(heldout) < size:
        base_seed = seed + i * 1_000_003

        for attempt in range(max_attempts_per_example):
            sample_seed = (base_seed + attempt * 104729) & 0x7FFFFFFF

            gen = NaturalLanguageGraphGenerator(max_input_size, seed=sample_seed)
            batch = gen.generate_batch(
                task,
                batch_size=1,
                reserved_inputs=avoid,
                alpha=alpha,
                max_lookahead=max_lookahead,
                **task_kwargs,
            )

            if not batch or not batch[0] or not batch[0].output_texts:
                continue

            ex = batch[0]

            if ex.input_text in avoid:
                continue
            if ex.input_text in heldout_inputs:
                continue

            row = {
                "input_text": ex.input_text,
                "output_texts": list(ex.output_texts),
                "alpha": alpha,
                "stage": stage,
                "seed": int(sample_seed),
            }
            heldout.append(row)
            heldout_inputs.add(ex.input_text)
            break
        else:
            raise RuntimeError(
                f"[build_heldout_set] Failed to fill heldout slot {len(heldout)+1}/{size} "
                f"after {max_attempts_per_example} attempts."
            )

        i += 1

    return heldout, heldout_inputs

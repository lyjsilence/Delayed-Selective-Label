from __future__ import annotations

"""Finite-cell Beta--Bernoulli potential outcomes with common random numbers."""

from dataclasses import dataclass
import hashlib
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .timing import DelayKernel


def _derived_seed(environment_seed: int, stream: str) -> int:
    payload = (str(int(environment_seed)) + "::" + stream).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def method_seed(environment_seed: int, method: str) -> int:
    """Align the sole stochastic stream for the SIV/Count identification pair."""

    group = "siv_count_pair" if method in {"siv", "count_only"} else method
    return _derived_seed(environment_seed, "policy::" + group)


@dataclass(frozen=True)
class PotentialOutcomes:
    environment_seed: int
    probabilities: np.ndarray
    cells: np.ndarray
    uniforms: np.ndarray
    outcomes: np.ndarray
    delays: np.ndarray
    warm_successes: np.ndarray
    warm_total: int

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.probabilities,
            self.cells,
            self.uniforms,
            self.outcomes,
            self.delays,
            self.warm_successes,
        ):
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()


def draw_cell_probabilities(
    seed: int,
    cells: int = 20,
    cost: float = 0.70,
    strata: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> np.ndarray:
    if cells != 20:
        raise ValueError("the finite-cell design requires K=20")
    rng = np.random.default_rng(_derived_seed(seed, "cell_probabilities"))
    if strata is None:
        intervals = {
            "clearly_profitable": (max(cost + 0.10, 0.80), 0.95),
            "clearly_unprofitable": (0.30, min(cost - 0.12, 0.58)),
            "near_boundary": (cost - 0.06, cost + 0.06),
        }
    else:
        expected = ("clearly_profitable", "clearly_unprofitable", "near_boundary")
        if set(strata) != set(expected):
            raise ValueError("strata must contain exactly the three registered groups")
        intervals = {}
        expected_fractions = dict(
            clearly_profitable=0.30, clearly_unprofitable=0.30, near_boundary=0.40
        )
        for name in expected:
            fraction = float(strata[name]["fraction"])
            interval = tuple(float(value) for value in strata[name]["interval"])
            if not np.isclose(fraction, expected_fractions[name]) or len(interval) != 2:
                raise ValueError("stratum fractions/intervals differ from the K=20 design")
            if not 0.0 <= interval[0] < interval[1] <= 1.0:
                raise ValueError("invalid probability interval for " + name)
            intervals[name] = interval
    profitable = rng.uniform(*intervals["clearly_profitable"], 6)
    unprofitable = rng.uniform(*intervals["clearly_unprofitable"], 6)
    boundary = rng.uniform(*intervals["near_boundary"], 8)
    values = np.concatenate([profitable, unprofitable, boundary])
    rng.shuffle(values)
    return values


def generate_potentials(
    environment_seed: int,
    kernel: DelayKernel,
    *,
    horizon: int = 5000,
    cells: int = 20,
    cost: float = 0.70,
    warm_total: int = 2,
    strata: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> PotentialOutcomes:
    if horizon < 1 or warm_total < 0:
        raise ValueError("horizon must be positive and warm_total non-negative")
    probabilities = draw_cell_probabilities(environment_seed, cells, cost, strata)
    arrival_rng = np.random.default_rng(_derived_seed(environment_seed, "arrivals"))
    outcome_rng = np.random.default_rng(_derived_seed(environment_seed, "outcomes"))
    delay_rng = np.random.default_rng(_derived_seed(environment_seed, "delays"))
    warm_rng = np.random.default_rng(_derived_seed(environment_seed, "warm_labels"))
    arrival_cells = arrival_rng.integers(0, cells, int(horizon), dtype=np.int64)
    uniforms = outcome_rng.random(int(horizon))
    outcomes = (uniforms < probabilities[arrival_cells]).astype(np.int8)
    delays = kernel.sample(delay_rng, int(horizon))
    warm_successes = warm_rng.binomial(int(warm_total), probabilities).astype(np.int64)
    return PotentialOutcomes(
        int(environment_seed),
        probabilities,
        arrival_cells,
        uniforms,
        outcomes,
        delays,
        warm_successes,
        int(warm_total),
    )


def seed_partitions(
    development: Sequence[int], validation: Sequence[int], frozen: Sequence[int]
) -> Dict[str, Tuple[int, ...]]:
    partitions = {
        "development": tuple(int(x) for x in development),
        "validation": tuple(int(x) for x in validation),
        "frozen": tuple(int(x) for x in frozen),
    }
    sets = {name: set(values) for name, values in partitions.items()}
    if any(
        sets[left].intersection(sets[right])
        for left, right in (
            ("development", "validation"),
            ("development", "frozen"),
            ("validation", "frozen"),
        )
    ):
        raise ValueError("development, validation, and frozen seeds must be disjoint")
    return partitions

from __future__ import annotations

"""Potential outcomes coupled to v10 exogenous arrival traces."""

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Optional, Sequence

import numpy as np

from siv_core.timing import DelayKernel

from .arrivals import ArrivalSpec, ArrivalTrace, _update_array, derived_seed, generate_arrivals


@dataclass(frozen=True)
class CellPopulation:
    probabilities: np.ndarray
    boundary_mask: np.ndarray
    stratum: np.ndarray

    def fingerprint(self) -> str:
        digest = hashlib.sha256(b"siv_v10.cell_population.v1")
        _update_array(digest, "probabilities", self.probabilities)
        _update_array(digest, "boundary_mask", self.boundary_mask)
        _update_array(digest, "stratum", self.stratum)
        return digest.hexdigest()


def draw_cell_population(
    seed: int,
    *,
    cells: int = 20,
    cost: float = 0.70,
    strata: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> CellPopulation:
    """Draw the registered 6/6/8 finite-cell population with retained labels."""

    if int(cells) != 20:
        raise ValueError("the v10 finite-cell population prespecifies K=20")
    expected = ("clearly_profitable", "clearly_unprofitable", "near_boundary")
    if strata is None:
        intervals = {
            "clearly_profitable": (max(float(cost) + 0.10, 0.80), 0.95),
            "clearly_unprofitable": (0.30, min(float(cost) - 0.12, 0.58)),
            "near_boundary": (float(cost) - 0.06, float(cost) + 0.06),
        }
    else:
        if set(strata) != set(expected):
            raise ValueError("strata must contain exactly the three registered groups")
        required_fraction = dict(
            clearly_profitable=0.30, clearly_unprofitable=0.30, near_boundary=0.40
        )
        intervals = {}
        for name in expected:
            fraction = float(strata[name]["fraction"])
            interval = tuple(float(x) for x in strata[name]["interval"])
            if not np.isclose(fraction, required_fraction[name]) or len(interval) != 2:
                raise ValueError("stratum fractions/intervals differ from the K=20 design")
            if not 0.0 <= interval[0] < interval[1] <= 1.0:
                raise ValueError("invalid probability interval for " + name)
            intervals[name] = interval

    rng = np.random.default_rng(derived_seed(seed, "cell_probabilities"))
    values = np.concatenate(
        [
            rng.uniform(*intervals["clearly_profitable"], 6),
            rng.uniform(*intervals["clearly_unprofitable"], 6),
            rng.uniform(*intervals["near_boundary"], 8),
        ]
    )
    labels = np.asarray([0] * 6 + [1] * 6 + [2] * 8, dtype=np.int8)
    permutation = rng.permutation(int(cells))
    values = values[permutation]
    labels = labels[permutation]
    return CellPopulation(probabilities=values, boundary_mask=(labels == 2), stratum=labels)


def draw_fixed_grid_population(
    seed: int, values: Sequence[float], *, cost: float = 0.70, boundary_half_width: float = 0.06
) -> CellPopulation:
    """Assign a fixed probability grid to cells using a seeded permutation."""

    grid = np.asarray(values, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 1:
        raise ValueError("fixed grid must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(grid)) or np.any(grid < 0.0) or np.any(grid > 1.0):
        raise ValueError("fixed-grid probabilities must lie in [0, 1]")
    if not np.isfinite(boundary_half_width) or boundary_half_width < 0.0:
        raise ValueError("boundary_half_width must be finite and non-negative")
    rng = np.random.default_rng(derived_seed(seed, "fixed_grid_permutation"))
    values_permuted = grid[rng.permutation(grid.size)]
    boundary = np.abs(values_permuted - float(cost)) <= float(boundary_half_width) + 1e-12
    labels = np.where(boundary, 2, np.where(values_permuted > float(cost), 0, 1)).astype(np.int8)
    return CellPopulation(probabilities=values_permuted, boundary_mask=boundary, stratum=labels)


def draw_population_from_mapping(
    seed: int, definition: Mapping[str, object], *, cost: float = 0.70
) -> CellPopulation:
    """Compile a cell population from the registered v10 probability schema."""

    design_type = str(definition["type"])
    if design_type == "fixed_grid_seeded_cell_permutation":
        return draw_fixed_grid_population(
            seed,
            definition["values"],
            cost=float(cost),
            boundary_half_width=float(definition.get("boundary_half_width", 0.06)),
        )
    if design_type == "seeded_stratified_uniform":
        strata = definition["strata"]
        expected_counts = {
            "clearly_profitable": 6,
            "clearly_unprofitable": 6,
            "near_boundary": 8,
        }
        converted = {}
        for name, expected_count in expected_counts.items():
            if int(strata[name]["count"]) != expected_count:
                raise ValueError("registered stratified design requires counts 6/6/8")
            converted[name] = {
                "fraction": expected_count / 20.0,
                "interval": strata[name]["interval"],
            }
        return draw_cell_population(seed, cells=20, cost=float(cost), strata=converted)
    raise ValueError("unsupported registered probability design: " + design_type)


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
    arrival_trace: ArrivalTrace
    population_fingerprint: str

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        metadata = {
            "schema": "siv_v10.potential_outcomes.v1",
            "environment_seed": int(self.environment_seed),
            "warm_total": int(self.warm_total),
            "arrival_fingerprint": self.arrival_trace.fingerprint(),
            "population_fingerprint": str(self.population_fingerprint),
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        for name, value in (
            ("probabilities", self.probabilities),
            ("cells", self.cells),
            ("uniforms", self.uniforms),
            ("outcomes", self.outcomes),
            ("delays", self.delays),
            ("warm_successes", self.warm_successes),
        ):
            _update_array(digest, name, value)
        return digest.hexdigest()


def generate_potentials(
    environment_seed: int,
    kernel: DelayKernel,
    *,
    arrival_spec: ArrivalSpec,
    horizon: int = 5000,
    cells: int = 20,
    cost: float = 0.70,
    warm_total: int = 2,
    strata: Optional[Mapping[str, Mapping[str, object]]] = None,
    probability_design: Optional[Mapping[str, object]] = None,
    outcome_design: str = "iid",
    delay_design: str = "iid",
) -> PotentialOutcomes:
    """Generate a method-independent v10 potential-outcome tape."""

    if int(horizon) < 1 or int(warm_total) < 0:
        raise ValueError("horizon must be positive and warm_total non-negative")
    if probability_design is not None and strata is not None:
        raise ValueError("provide probability_design or strata, not both")
    if probability_design is None:
        population = draw_cell_population(
            environment_seed, cells=int(cells), cost=float(cost), strata=strata
        )
    else:
        population = draw_population_from_mapping(
            environment_seed, probability_design, cost=float(cost)
        )
        if population.probabilities.size != int(cells):
            raise ValueError("probability design size does not equal cells")
    arrivals = generate_arrivals(
        environment_seed,
        horizon=int(horizon),
        cells=int(cells),
        spec=arrival_spec,
        boundary_mask=population.boundary_mask,
    )
    delay_rng = np.random.default_rng(derived_seed(environment_seed, "delays"))
    warm_rng = np.random.default_rng(derived_seed(environment_seed, "warm_labels"))
    if outcome_design == "iid":
        outcome_rng = np.random.default_rng(derived_seed(environment_seed, "outcomes"))
        uniforms = outcome_rng.random(int(horizon))
    elif outcome_design == "stratified_per_cell":
        uniforms = np.empty(int(horizon), dtype=np.float64)
        for cell in range(int(cells)):
            positions = np.flatnonzero(arrivals.cells == cell)
            count = len(positions)
            if count == 0:
                continue
            rng = np.random.default_rng(
                derived_seed(environment_seed, "outcomes::stratified_cell::" + str(cell))
            )
            points = (np.arange(count, dtype=float) + rng.random(count)) / count
            uniforms[positions] = points[rng.permutation(count)]
    else:
        raise ValueError("outcome_design must be iid or stratified_per_cell")
    outcomes = (uniforms < population.probabilities[arrivals.cells]).astype(np.int8)
    if delay_design == "iid":
        delays = kernel.sample(delay_rng, int(horizon))
    elif delay_design == "stratified_per_cell":
        if not hasattr(kernel, "delays") or not hasattr(kernel, "probabilities"):
            raise ValueError("stratified_per_cell delays require a finite mixture kernel")
        support = np.asarray(kernel.delays, dtype=np.int64)
        mixture_probabilities = np.asarray(kernel.probabilities, dtype=float)
        cumulative = np.cumsum(mixture_probabilities)
        cumulative[-1] = 1.0
        delays = np.empty(int(horizon), dtype=np.int64)
        for cell in range(int(cells)):
            positions = np.flatnonzero(arrivals.cells == cell)
            count = len(positions)
            if count == 0:
                continue
            rng = np.random.default_rng(
                derived_seed(environment_seed, "delays::stratified_cell::" + str(cell))
            )
            points = (np.arange(count, dtype=float) + rng.random(count)) / count
            categories = np.searchsorted(cumulative, points, side="right")
            delays[positions] = support[categories[rng.permutation(count)]]
    else:
        raise ValueError("delay_design must be iid or stratified_per_cell")
    warm_successes = warm_rng.binomial(int(warm_total), population.probabilities).astype(np.int64)
    return PotentialOutcomes(
        environment_seed=int(environment_seed),
        probabilities=population.probabilities,
        cells=arrivals.cells,
        uniforms=uniforms,
        outcomes=outcomes,
        delays=delays,
        warm_successes=warm_successes,
        warm_total=int(warm_total),
        arrival_trace=arrivals,
        population_fingerprint=population.fingerprint(),
    )

from __future__ import annotations

"""Exogenous arrival processes for delayed feedback.

The Markov process is a sticky-refresh chain. Its refresh dwell has an exact
geometric interpretation and its stationary marginal is the configured base
cell distribution, including optional boundary-cell enrichment.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Optional

import numpy as np


def derived_seed(environment_seed: int, stream: str) -> int:
    """Derive a stable, named 32-bit stream seed from an environment seed."""

    payload = ("v10::" + str(int(environment_seed)) + "::" + str(stream)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def _update_array(digest, name: str, value: np.ndarray) -> None:
    """Add an array to a digest with unambiguous dtype and shape framing."""

    array = np.asarray(value)
    if array.dtype.kind in "iufc":
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    array = np.ascontiguousarray(array)
    header = {
        "name": str(name),
        "dtype": array.dtype.str,
        "shape": [int(x) for x in array.shape],
    }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    payload = array.tobytes(order="C")
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


@dataclass(frozen=True)
class ArrivalSpec:
    """Configuration for i.i.d. or sticky-refresh arrivals.

    ``mean_dwell`` is the mean number of steps between exogenous refreshes. It
    is used only in ``markov_bursty`` mode. A boundary weight of one leaves the
    cell marginal uniform; values above one enrich boundary cells.
    """

    mode: str = "uniform"
    mean_dwell: float = 1.0
    boundary_weight: float = 1.0
    hotspot_persistence: Optional[float] = None
    hotspot_mass: Optional[float] = None
    hotspot_size: int = 1
    switch_excludes_current: bool = True
    background: str = "uniform_over_non_hotspot_cells"

    def __post_init__(self) -> None:
        allowed = {
            "uniform",
            "iid_weighted",
            "markov_bursty",
            "balanced_blocks",
            "reshuffled_balanced_blocks",
            "markov_modulated_hotspot",
        }
        if self.mode not in allowed:
            raise ValueError("unsupported arrival mode: " + str(self.mode))
        if not np.isfinite(self.mean_dwell) or self.mean_dwell < 1.0:
            raise ValueError("mean_dwell must be finite and at least one")
        if not np.isfinite(self.boundary_weight) or self.boundary_weight <= 0.0:
            raise ValueError("boundary_weight must be finite and positive")
        if self.mode == "uniform" and not np.isclose(self.mean_dwell, 1.0):
            raise ValueError("uniform mode requires mean_dwell=1")
        if self.mode == "uniform" and not np.isclose(self.boundary_weight, 1.0):
            raise ValueError("uniform mode requires boundary_weight=1")
        if self.mode == "iid_weighted" and not np.isclose(self.mean_dwell, 1.0):
            raise ValueError("iid_weighted mode requires mean_dwell=1")
        if self.mode in {"balanced_blocks", "reshuffled_balanced_blocks"}:
            if int(self.mean_dwell) != self.mean_dwell:
                raise ValueError("balanced block length must be an integer")
            if not np.isclose(self.boundary_weight, 1.0):
                raise ValueError("balanced blocks require boundary_weight=1")
        if self.mode == "markov_modulated_hotspot":
            if not np.isclose(self.mean_dwell, 1.0):
                raise ValueError("hotspot mode does not use mean_dwell")
            if not np.isclose(self.boundary_weight, 1.0):
                raise ValueError("hotspot mode requires boundary_weight=1 for uniform exposure")
            if self.hotspot_persistence is None or not (
                0.0 <= float(self.hotspot_persistence) < 1.0
            ):
                raise ValueError("hotspot_persistence must lie in [0, 1)")
            if self.hotspot_mass is None or not (0.0 <= float(self.hotspot_mass) <= 1.0):
                raise ValueError("hotspot_mass must lie in [0, 1]")
            if int(self.hotspot_size) != self.hotspot_size or int(self.hotspot_size) < 1:
                raise ValueError("hotspot_size must be a positive integer")
            if self.background != "uniform_over_non_hotspot_cells":
                raise ValueError("only uniform_over_non_hotspot_cells is supported")
        elif (
            self.hotspot_persistence is not None
            or self.hotspot_mass is not None
            or int(self.hotspot_size) != 1
        ):
            raise ValueError("hotspot parameters are valid only in hotspot mode")

    @classmethod
    def negative_control(cls) -> "ArrivalSpec":
        return cls(mode="uniform", mean_dwell=1.0, boundary_weight=1.0)

    @classmethod
    def bursty(cls, mean_dwell: float, boundary_weight: float = 1.0) -> "ArrivalSpec":
        return cls(mode="markov_bursty", mean_dwell=mean_dwell, boundary_weight=boundary_weight)

    @classmethod
    def iid_weighted(cls, boundary_weight: float) -> "ArrivalSpec":
        return cls(mode="iid_weighted", mean_dwell=1.0, boundary_weight=boundary_weight)

    @classmethod
    def balanced_blocks(cls, block_length: int) -> "ArrivalSpec":
        return cls(mode="balanced_blocks", mean_dwell=int(block_length), boundary_weight=1.0)

    @classmethod
    def reshuffled_balanced_blocks(cls, block_length: int) -> "ArrivalSpec":
        """Balanced contiguous blocks with a fresh cell order every full cycle."""

        return cls(
            mode="reshuffled_balanced_blocks", mean_dwell=int(block_length), boundary_weight=1.0
        )

    @classmethod
    def hotspot(
        cls,
        persistence: float,
        hotspot_mass: float,
        switch_excludes_current: bool = True,
        background: str = "uniform_over_non_hotspot_cells",
        hotspot_size: int = 1,
    ) -> "ArrivalSpec":
        return cls(
            mode="markov_modulated_hotspot",
            hotspot_persistence=float(persistence),
            hotspot_mass=float(hotspot_mass),
            hotspot_size=int(hotspot_size),
            switch_excludes_current=bool(switch_excludes_current),
            background=str(background),
        )

    @classmethod
    def pair_hotspot(
        cls, persistence: float, hotspot_mass: float, switch_excludes_current: bool = True
    ) -> "ArrivalSpec":
        """Latent hotspot over disjoint pairs, with a uniform cell marginal."""

        return cls.hotspot(persistence, hotspot_mass, switch_excludes_current, hotspot_size=2)

    @property
    def refresh_probability(self) -> float:
        if self.mode == "markov_modulated_hotspot":
            return 1.0 - float(self.hotspot_persistence)
        return 1.0 if self.mode in {"uniform", "iid_weighted"} else 1.0 / float(self.mean_dwell)

    @property
    def is_uniform_negative_control(self) -> bool:
        return self.mode == "uniform" and self.boundary_weight == 1.0


def cell_probabilities(
    cells: int, boundary_mask: Optional[np.ndarray], boundary_weight: float
) -> np.ndarray:
    """Return normalized cell probabilities after optional boundary weighting."""

    if int(cells) < 1:
        raise ValueError("cells must be positive")
    if boundary_mask is None:
        boundary = np.zeros(int(cells), dtype=bool)
    else:
        boundary = np.asarray(boundary_mask, dtype=bool)
        if boundary.shape != (int(cells),):
            raise ValueError("boundary_mask must have shape (cells,)")
    weights = np.ones(int(cells), dtype=np.float64)
    weights[boundary] *= float(boundary_weight)
    return weights / weights.sum()


@dataclass(frozen=True)
class ArrivalTrace:
    environment_seed: int
    spec: ArrivalSpec
    cells: np.ndarray
    base_probabilities: np.ndarray
    boundary_mask: np.ndarray
    selection_uniforms: np.ndarray
    refresh_uniforms: np.ndarray
    refreshed: np.ndarray
    hotspots: Optional[np.ndarray] = None
    hotspot_state_uniforms: Optional[np.ndarray] = None
    hotspot_persistence_uniforms: Optional[np.ndarray] = None
    emission_uniforms: Optional[np.ndarray] = None
    background_uniforms: Optional[np.ndarray] = None
    member_uniforms: Optional[np.ndarray] = None

    def transition_matrix(self) -> np.ndarray:
        """Return the exact Markov transition matrix implied by this trace."""

        if self.spec.mode == "markov_modulated_hotspot":
            raise ValueError("use hotspot_transition_matrix for latent-hotspot arrivals")
        if self.spec.mode in {"balanced_blocks", "reshuffled_balanced_blocks"}:
            raise ValueError("balanced blocks are periodic rather than a homogeneous Markov chain")
        k = int(self.base_probabilities.size) // int(self.spec.hotspot_size)
        q = float(self.spec.refresh_probability)
        return (1.0 - q) * np.eye(k) + q * np.tile(self.base_probabilities, (k, 1))

    def hotspot_transition_matrix(self) -> np.ndarray:
        """Return the exact latent-hotspot transition matrix."""

        if self.spec.mode != "markov_modulated_hotspot":
            raise ValueError("trace is not a latent-hotspot process")
        cells = int(self.base_probabilities.size)
        size = int(self.spec.hotspot_size)
        if cells % size:
            raise ValueError("cells must be divisible by hotspot_size")
        k = cells // size
        persistence = float(self.spec.hotspot_persistence)
        if self.spec.switch_excludes_current:
            transition = np.full((k, k), (1.0 - persistence) / (k - 1))
            np.fill_diagonal(transition, persistence)
            return transition
        return persistence * np.eye(k) + (1.0 - persistence) * np.full((k, k), 1.0 / k)

    def hotspot_emission_matrix(self) -> np.ndarray:
        """Return P(arrival cell | latent hotspot) for every hotspot."""

        if self.spec.mode != "markov_modulated_hotspot":
            raise ValueError("trace is not a latent-hotspot process")
        cells = int(self.base_probabilities.size)
        size = int(self.spec.hotspot_size)
        if cells % size:
            raise ValueError("cells must be divisible by hotspot_size")
        states = cells // size
        mass = float(self.spec.hotspot_mass)
        if cells <= size:
            raise ValueError("hotspot background requires at least one non-hotspot cell")
        emission = np.full((states, cells), (1.0 - mass) / (cells - size))
        for state in range(states):
            start = state * size
            emission[state, start : start + size] = mass / size
        return emission

    def stationary_exposure(self) -> np.ndarray:
        """Return the analytic stationary cell-exposure distribution."""

        if self.spec.mode == "markov_modulated_hotspot":
            states = self.hotspot_emission_matrix().shape[0]
            return np.full(states, 1.0 / states) @ self.hotspot_emission_matrix()
        return self.base_probabilities.copy()

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        spec_metadata = {
            "mode": self.spec.mode,
            "mean_dwell": self.spec.mean_dwell,
            "boundary_weight": self.spec.boundary_weight,
        }
        if self.spec.mode == "markov_modulated_hotspot":
            spec_metadata.update(
                {
                    "hotspot_persistence": self.spec.hotspot_persistence,
                    "hotspot_mass": self.spec.hotspot_mass,
                    "switch_excludes_current": self.spec.switch_excludes_current,
                    "background": self.spec.background,
                }
            )
            if int(self.spec.hotspot_size) != 1:
                spec_metadata["hotspot_size"] = int(self.spec.hotspot_size)
        metadata = {
            "schema": (
                "siv_v10.arrival_trace.v2"
                if self.spec.mode == "markov_modulated_hotspot"
                else "siv_v10.arrival_trace.v1"
            ),
            "environment_seed": int(self.environment_seed),
            "spec": spec_metadata,
        }
        encoded = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        for name, value in (
            ("cells", self.cells),
            ("base_probabilities", self.base_probabilities),
            ("boundary_mask", self.boundary_mask),
            ("selection_uniforms", self.selection_uniforms),
            ("refresh_uniforms", self.refresh_uniforms),
            ("refreshed", self.refreshed),
        ):
            _update_array(digest, name, value)
        if self.spec.mode == "markov_modulated_hotspot":
            for name, value in (
                ("hotspots", self.hotspots),
                ("hotspot_state_uniforms", self.hotspot_state_uniforms),
                ("hotspot_persistence_uniforms", self.hotspot_persistence_uniforms),
                ("emission_uniforms", self.emission_uniforms),
                ("background_uniforms", self.background_uniforms),
            ):
                _update_array(digest, name, value)
            if int(self.spec.hotspot_size) != 1:
                _update_array(digest, "member_uniforms", self.member_uniforms)
        return digest.hexdigest()


def _categorical_from_uniforms(uniforms: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, uniforms, side="right").astype(np.int64)


def _uniform_cells_excluding(uniforms: np.ndarray, excluded: np.ndarray, cells: int) -> np.ndarray:
    if int(cells) < 2:
        raise ValueError("excluding one cell requires at least two cells")
    candidates = np.floor(np.asarray(uniforms) * (int(cells) - 1)).astype(np.int64)
    excluded = np.asarray(excluded, dtype=np.int64)
    return candidates + (candidates >= excluded).astype(np.int64)


def _uniform_cells_excluding_block(
    uniforms: np.ndarray, block: np.ndarray, block_size: int, cells: int
) -> np.ndarray:
    """Map uniforms to cells outside one contiguous equal-sized hotspot block."""

    if int(block_size) < 1 or int(cells) <= int(block_size):
        raise ValueError("a hotspot block must leave at least one background cell")
    starts = np.asarray(block, dtype=np.int64) * int(block_size)
    candidates = np.floor(np.asarray(uniforms) * (int(cells) - int(block_size))).astype(np.int64)
    return candidates + (candidates >= starts).astype(np.int64) * int(block_size)


def arrival_spec_from_mapping(definition) -> ArrivalSpec:
    """Compile one arrival definition from the registered v10 YAML schema."""

    family = str(definition["family"])
    if family == "iid_uniform":
        return ArrivalSpec.negative_control()
    if family == "iid_weighted":
        return ArrivalSpec.iid_weighted(float(definition["boundary_weight"]))
    if family == "balanced_blocks":
        return ArrivalSpec.balanced_blocks(int(definition["block_length"]))
    if family == "reshuffled_balanced_blocks":
        return ArrivalSpec.reshuffled_balanced_blocks(int(definition["block_length"]))
    if family == "markov_modulated_hotspot":
        return ArrivalSpec.hotspot(
            persistence=float(definition["hotspot_persistence"]),
            hotspot_mass=float(definition["hotspot_mass"]),
            switch_excludes_current=bool(definition.get("switch_excludes_current_hotspot", True)),
            background=str(definition.get("background", "uniform_over_non_hotspot_cells")),
            hotspot_size=int(definition.get("hotspot_size", 1)),
        )
    raise ValueError("unsupported registered arrival family: " + family)


def generate_arrivals(
    environment_seed: int,
    *,
    horizon: int,
    cells: int,
    spec: ArrivalSpec,
    boundary_mask: Optional[np.ndarray] = None,
) -> ArrivalTrace:
    """Generate one reusable CRN arrival trace.

    Candidate categorical draws and refresh indicators always use separate
    named streams. Thus every policy can share the same environment tape
    without method-dependent random-number consumption.
    """

    if int(horizon) < 1:
        raise ValueError("horizon must be positive")
    if not isinstance(spec, ArrivalSpec):
        raise TypeError("spec must be an ArrivalSpec")
    if boundary_mask is None:
        boundary = np.zeros(int(cells), dtype=bool)
    else:
        boundary = np.asarray(boundary_mask, dtype=bool).copy()
        if boundary.shape != (int(cells),):
            raise ValueError("boundary_mask must have shape (cells,)")

    probabilities = cell_probabilities(int(cells), boundary, spec.boundary_weight)
    if spec.mode in {"balanced_blocks", "reshuffled_balanced_blocks"}:
        block = int(spec.mean_dwell)
        order_rng = np.random.default_rng(
            derived_seed(environment_seed, "arrivals::balanced_block_order")
        )
        block_index = np.arange(int(horizon), dtype=np.int64) // block
        if spec.mode == "balanced_blocks":
            order = order_rng.permutation(int(cells))
            arrival_cells = order[block_index % int(cells)]
        else:
            block_count = int(block_index[-1]) + 1
            cycle_count = (block_count + int(cells) - 1) // int(cells)
            cycle_orders = np.vstack(
                [order_rng.permutation(int(cells)) for _ in range(cycle_count)]
            )
            arrival_cells = cycle_orders[block_index // int(cells), block_index % int(cells)]
        refreshed = np.zeros(int(horizon), dtype=bool)
        refreshed[::block] = True
        empty = np.empty(0, dtype=np.float64)
        return ArrivalTrace(
            environment_seed=int(environment_seed),
            spec=spec,
            cells=arrival_cells,
            base_probabilities=probabilities,
            boundary_mask=boundary,
            selection_uniforms=empty,
            refresh_uniforms=empty.copy(),
            refreshed=refreshed,
        )
    if spec.mode == "markov_modulated_hotspot":
        size = int(spec.hotspot_size)
        if int(cells) <= size or int(cells) % size:
            raise ValueError(
                "latent-hotspot cells must be divisible by hotspot_size and leave background"
            )
        states = int(cells) // size
        probabilities = np.full(int(cells), 1.0 / int(cells))
        state_rng = np.random.default_rng(derived_seed(environment_seed, "arrivals::hotspot_state"))
        persistence_rng = np.random.default_rng(
            derived_seed(environment_seed, "arrivals::hotspot_persistence")
        )
        emission_rng = np.random.default_rng(derived_seed(environment_seed, "arrivals::emission"))
        background_rng = np.random.default_rng(
            derived_seed(environment_seed, "arrivals::background")
        )
        hotspot_state_uniforms = state_rng.random(int(horizon))
        hotspot_persistence_uniforms = persistence_rng.random(int(horizon))
        emission_uniforms = emission_rng.random(int(horizon))
        background_uniforms = background_rng.random(int(horizon))
        if size == 1:
            member_uniforms = np.empty(0, dtype=np.float64)
        else:
            member_rng = np.random.default_rng(
                derived_seed(environment_seed, "arrivals::hotspot_member")
            )
            member_uniforms = member_rng.random(int(horizon))
        hotspots = np.empty(int(horizon), dtype=np.int64)
        refreshed = np.zeros(int(horizon), dtype=bool)
        hotspots[0] = min(int(hotspot_state_uniforms[0] * states), states - 1)
        refreshed[0] = True
        for t in range(1, int(horizon)):
            if hotspot_persistence_uniforms[t] < float(spec.hotspot_persistence):
                hotspots[t] = hotspots[t - 1]
            else:
                refreshed[t] = True
                if spec.switch_excludes_current:
                    hotspots[t] = _uniform_cells_excluding(
                        hotspot_state_uniforms[t : t + 1], hotspots[t - 1 : t], states
                    )[0]
                else:
                    hotspots[t] = min(int(hotspot_state_uniforms[t] * states), states - 1)
        background_cells = _uniform_cells_excluding_block(
            background_uniforms, hotspots, size, int(cells)
        )
        member_cells = (
            hotspots
            if size == 1
            else hotspots * size + np.minimum((member_uniforms * size).astype(np.int64), size - 1)
        )
        arrival_cells = np.where(
            emission_uniforms < float(spec.hotspot_mass), member_cells, background_cells
        ).astype(np.int64)
        empty_float = np.empty(0, dtype=np.float64)
        return ArrivalTrace(
            environment_seed=int(environment_seed),
            spec=spec,
            cells=arrival_cells,
            base_probabilities=probabilities,
            boundary_mask=boundary,
            selection_uniforms=empty_float,
            refresh_uniforms=empty_float.copy(),
            refreshed=refreshed,
            hotspots=hotspots,
            hotspot_state_uniforms=hotspot_state_uniforms,
            hotspot_persistence_uniforms=hotspot_persistence_uniforms,
            emission_uniforms=emission_uniforms,
            background_uniforms=background_uniforms,
            member_uniforms=member_uniforms,
        )

    select_rng = np.random.default_rng(derived_seed(environment_seed, "arrivals::selection"))
    refresh_rng = np.random.default_rng(derived_seed(environment_seed, "arrivals::refresh"))
    selection_uniforms = select_rng.random(int(horizon))
    refresh_uniforms = refresh_rng.random(int(horizon))
    candidates = _categorical_from_uniforms(selection_uniforms, probabilities)

    if spec.mode in {"uniform", "iid_weighted"}:
        refreshed = np.ones(int(horizon), dtype=bool)
        arrival_cells = candidates.copy()
    else:
        refreshed = refresh_uniforms < spec.refresh_probability
        refreshed[0] = True
        arrival_cells = np.empty(int(horizon), dtype=np.int64)
        arrival_cells[0] = candidates[0]
        for t in range(1, int(horizon)):
            arrival_cells[t] = candidates[t] if refreshed[t] else arrival_cells[t - 1]

    return ArrivalTrace(
        environment_seed=int(environment_seed),
        spec=spec,
        cells=arrival_cells,
        base_probabilities=probabilities,
        boundary_mask=boundary,
        selection_uniforms=selection_uniforms,
        refresh_uniforms=refresh_uniforms,
        refreshed=refreshed,
    )

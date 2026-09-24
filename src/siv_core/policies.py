from __future__ import annotations

"""Pure-information finite-cell policies; no risk machinery is permitted here."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .information import beta_variance, exact_one_step_decision_value, marginal_information_value
from .timing import DelayKernel


@dataclass(frozen=True)
class PendingLabel:
    cell: int
    origin_time: int


class FiniteCellPolicy:
    def __init__(
        self,
        name: str,
        cells: int,
        cost: float,
        seed: int,
        warm_successes: np.ndarray,
        warm_total: int,
    ) -> None:
        self.name = str(name)
        self.cells = int(cells)
        self.cost = float(cost)
        successes = np.asarray(warm_successes, dtype=float)
        self.alpha = 1.0 + successes
        self.beta = 1.0 + float(warm_total) - successes
        self.pending = [[] for _ in range(self.cells)]  # type: List[List[PendingLabel]]
        self.rng = np.random.default_rng(int(seed))
        self.arrivals = np.zeros(self.cells, dtype=np.int64)

    def posterior_mean(self, cell: int) -> float:
        return float(self.alpha[cell] / (self.alpha[cell] + self.beta[cell]))

    def schedule(self, cell: int, origin_time: int) -> None:
        self.pending[int(cell)].append(PendingLabel(int(cell), int(origin_time)))

    def observe(self, cell: int, outcome: int, origin_time: int) -> None:
        cell = int(cell)
        for index, item in enumerate(self.pending[cell]):
            if item.origin_time == int(origin_time):
                self.pending[cell].pop(index)
                break
        else:
            raise RuntimeError("matured label was not present in the pending queue")
        self.alpha[cell] += int(outcome == 1)
        self.beta[cell] += int(outcome == 0)

    def decide(self, cell: int, time: int) -> int:
        del time
        return int(self.posterior_mean(int(cell)) > self.cost)

    def diagnostics(self, cell: int, time: int) -> Dict[str, float]:
        del time
        return {
            "mean": self.posterior_mean(int(cell)),
            "miv": 0.0,
            "score": self.posterior_mean(int(cell)),
        }


class MaturedGreedy(FiniteCellPolicy):
    pass


class DelayedUCB(FiniteCellPolicy):
    def __init__(self, *args, exploration: float = 0.8, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.exploration = float(exploration)
        self.total_arrivals = 0

    def diagnostics(self, cell: int, time: int) -> Dict[str, float]:
        cell = int(cell)
        mean = self.posterior_mean(cell)
        matured_labels = max(1.0, self.alpha[cell] + self.beta[cell] - 2.0)
        bonus = self.exploration * np.sqrt(
            2.0 * np.log(2.0 + max(time, self.total_arrivals)) / matured_labels
        )
        return {"mean": mean, "miv": 0.0, "score": float(mean + bonus)}

    def decide(self, cell: int, time: int) -> int:
        cell = int(cell)
        self.arrivals[cell] += 1
        self.total_arrivals += 1
        return int(self.diagnostics(cell, time)["score"] > self.cost)


class DelayedThompsonSampling(FiniteCellPolicy):
    def decide(self, cell: int, time: int) -> int:
        del time
        cell = int(cell)
        draw = self.rng.beta(self.alpha[cell], self.beta[cell])
        return int(draw > self.cost)


class ScheduledInformationPolicy(FiniteCellPolicy):
    """SIV and CountOnly share this code path; ``age_aware`` is the only switch."""

    def __init__(
        self,
        *args,
        kernel: DelayKernel,
        planning_window: int = 25,
        information_coefficient: float = 2.0,
        age_aware: bool = True,
        true_kernel: Optional[DelayKernel] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kernel = kernel
        self.true_kernel = kernel if true_kernel is None else true_kernel
        self.planning_window = int(planning_window)
        self.information_coefficient = float(information_coefficient)
        self.age_aware = bool(age_aware)
        self._diagnostic_cache_key = None
        self._diagnostic_cache = None

    def maturation_probabilities(
        self,
        cell: int,
        time: int,
        kernel: Optional[DelayKernel] = None,
        age_aware: Optional[bool] = None,
    ) -> np.ndarray:
        active_kernel = self.kernel if kernel is None else kernel
        use_ages = self.age_aware if age_aware is None else bool(age_aware)
        if use_ages:
            ages = [max(0, int(time) - item.origin_time) for item in self.pending[int(cell)]]
        else:
            ages = [0] * len(self.pending[int(cell)])
        return np.asarray(
            [active_kernel.maturation_probability(age, self.planning_window) for age in ages],
            dtype=float,
        )

    def miv(
        self,
        cell: int,
        time: int,
        kernel: Optional[DelayKernel] = None,
        age_aware: Optional[bool] = None,
    ) -> float:
        active_kernel = self.kernel if kernel is None else kernel
        probabilities = self.maturation_probabilities(cell, time, kernel, age_aware)
        new_probability = active_kernel.maturation_probability(0, self.planning_window)
        return marginal_information_value(
            self.alpha[int(cell)], self.beta[int(cell)], probabilities, new_probability
        )

    def diagnostics(self, cell: int, time: int) -> Dict[str, float]:
        cell = int(cell)
        cache_key = (cell, int(time), float(self.cost))
        if self._diagnostic_cache_key == cache_key:
            return dict(self._diagnostic_cache)
        mean = self.posterior_mean(cell)
        miv = self.miv(cell, time)
        oracle_miv = self.miv(cell, time, self.true_kernel, age_aware=True)
        age_aware_miv = self.miv(cell, time, self.kernel, age_aware=True)
        count_only_miv = self.miv(cell, time, self.kernel, age_aware=False)
        score = mean + self.information_coefficient * np.sqrt(max(miv, 0.0))
        age_aware_score = mean + self.information_coefficient * np.sqrt(max(age_aware_miv, 0.0))
        count_only_score = mean + self.information_coefficient * np.sqrt(max(count_only_miv, 0.0))
        diagnostics = {
            "mean": mean,
            "miv": miv,
            "oracle_miv": oracle_miv,
            "age_aware_miv": age_aware_miv,
            "count_only_miv": count_only_miv,
            "same_state_miv_gap": abs(age_aware_miv - count_only_miv),
            "age_aware_score": float(age_aware_score),
            "count_only_score": float(count_only_score),
            "same_state_score_gap": float(abs(age_aware_score - count_only_score)),
            "same_state_score_midpoint": float(0.5 * (age_aware_score + count_only_score)),
            "score": float(score),
        }
        self._diagnostic_cache_key = cache_key
        self._diagnostic_cache = diagnostics
        return dict(diagnostics)

    def decide(self, cell: int, time: int) -> int:
        return int(self.diagnostics(cell, time)["score"] > self.cost)

    def schedule(self, cell: int, origin_time: int) -> None:
        self._diagnostic_cache_key = None
        self._diagnostic_cache = None
        super().schedule(cell, origin_time)

    def observe(self, cell: int, outcome: int, origin_time: int) -> None:
        self._diagnostic_cache_key = None
        self._diagnostic_cache = None
        super().observe(cell, outcome, origin_time)


class CountOnly(ScheduledInformationPolicy):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["age_aware"] = False
        super().__init__(*args, **kwargs)


class OracleSIV(ScheduledInformationPolicy):
    """Compatibility alias for true-kernel SIV; do not report beside an identical SIV."""

    pass


class FittedSIV(ScheduledInformationPolicy):
    pass


class SurvivalEM(FiniteCellPolicy):
    """Pending-count pseudo-information baseline; posterior remains matured-only."""

    def __init__(
        self,
        *args,
        kernel: DelayKernel,
        planning_window: int = 25,
        exploration: float = 0.8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kernel = kernel
        self.planning_window = int(planning_window)
        self.exploration = float(exploration)

    def diagnostics(self, cell: int, time: int) -> Dict[str, float]:
        cell = int(cell)
        mean = self.posterior_mean(cell)
        pseudo_information = sum(
            self.kernel.maturation_probability(
                max(0, int(time) - item.origin_time), self.planning_window
            )
            for item in self.pending[cell]
        )
        effective = self.alpha[cell] + self.beta[cell] + pseudo_information
        score = mean + self.exploration * np.sqrt(mean * (1.0 - mean) / (effective + 1.0))
        return {
            "mean": mean,
            "miv": 0.0,
            "pseudo_information": float(pseudo_information),
            "score": float(score),
        }

    def decide(self, cell: int, time: int) -> int:
        return int(self.diagnostics(cell, time)["score"] > self.cost)


class DelayedSelectiveBootstrap(FiniteCellPolicy):
    def __init__(self, *args, ensemble_size: int = 31, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ensemble_size = int(ensemble_size)
        self.ensemble_alpha = np.tile(self.alpha, (self.ensemble_size, 1))
        self.ensemble_beta = np.tile(self.beta, (self.ensemble_size, 1))

    def observe(self, cell: int, outcome: int, origin_time: int) -> None:
        weights = self.rng.poisson(1.0, self.ensemble_size)
        self.ensemble_alpha[:, int(cell)] += weights * int(outcome == 1)
        self.ensemble_beta[:, int(cell)] += weights * int(outcome == 0)
        super().observe(cell, outcome, origin_time)

    def decide(self, cell: int, time: int) -> int:
        del time
        cell = int(cell)
        member = int(self.rng.integers(0, self.ensemble_size))
        draw = self.rng.beta(self.ensemble_alpha[member, cell], self.ensemble_beta[member, cell])
        return int(draw > self.cost)


class OneStepBayesInformationHeuristic(FiniteCellPolicy):
    """Immediate-label myopic information heuristic, not a delayed-policy oracle."""

    def __init__(self, *args, information_coefficient: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.information_coefficient = float(information_coefficient)

    def diagnostics(self, cell: int, time: int) -> Dict[str, float]:
        del time
        cell = int(cell)
        mean = self.posterior_mean(cell)
        value = exact_one_step_decision_value(self.alpha[cell], self.beta[cell], self.cost)
        return {
            "mean": mean,
            "miv": value,
            "score": float(mean + self.information_coefficient * value),
        }

    def decide(self, cell: int, time: int) -> int:
        return int(self.diagnostics(cell, time)["score"] > self.cost)


# Backward-compatible import name. New reports must use the non-oracle label.
OneStepBayesOracle = OneStepBayesInformationHeuristic

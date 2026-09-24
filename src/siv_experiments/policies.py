from __future__ import annotations

"""Policies that prevent elapsed feedback time from inducing information starvation."""

from typing import Dict

import numpy as np

from siv_design.priors import ContinuousPriorScheduledPolicy


class MonotoneScheduledInformationPolicy(ContinuousPriorScheduledPolicy):
    """Add only the positive elapsed-time increment to a count-only MIV floor.

    At a matched posterior and pending-count state, ``timing_gain_scale=0`` is
    CountOnly. For every non-negative scale, the policy score is no smaller
    than the CountOnly score. This matched-state safeguard prevents anticipated
    future feedback from suppressing exploration under selective labels.
    """

    def __init__(self, *args, timing_gain_scale: float = 1.0, **kwargs) -> None:
        kwargs["age_aware"] = True
        super().__init__(*args, **kwargs)
        if not np.isfinite(timing_gain_scale) or float(timing_gain_scale) < 0.0:
            raise ValueError("timing_gain_scale must be finite and non-negative")
        self.timing_gain_scale = float(timing_gain_scale)
        self._diagnostic_cache_key = None
        self._diagnostic_cache = None

    def diagnostics(self, cell: int, time: int) -> Dict[str, float]:
        cell = int(cell)
        cache_key = (cell, int(time))
        if self._diagnostic_cache_key == cache_key:
            return dict(self._diagnostic_cache)
        mean = self.posterior_mean(cell)
        age_miv = self.miv(cell, time, self.kernel, age_aware=True)
        count_miv = self.miv(cell, time, self.kernel, age_aware=False)
        oracle_miv = self.miv(cell, time, self.true_kernel, age_aware=True)
        positive_increment = max(float(age_miv) - float(count_miv), 0.0)
        effective_miv = float(count_miv) + self.timing_gain_scale * positive_increment
        score = mean + self.information_coefficient * np.sqrt(max(effective_miv, 0.0))
        age_score = mean + self.information_coefficient * np.sqrt(max(age_miv, 0.0))
        count_score = mean + self.information_coefficient * np.sqrt(max(count_miv, 0.0))
        diagnostics = {
            "mean": float(mean),
            "miv": float(effective_miv),
            "raw_age_aware_miv": float(age_miv),
            "positive_timing_miv_increment": float(positive_increment),
            "oracle_miv": float(oracle_miv),
            "age_aware_miv": float(age_miv),
            "count_only_miv": float(count_miv),
            "same_state_miv_gap": float(abs(age_miv - count_miv)),
            "age_aware_score": float(age_score),
            "count_only_score": float(count_score),
            "same_state_score_gap": float(abs(age_score - count_score)),
            "same_state_score_midpoint": float(0.5 * (age_score + count_score)),
            "score": float(score),
        }
        self._diagnostic_cache_key = cache_key
        self._diagnostic_cache = diagnostics
        return dict(diagnostics)

    def schedule(self, cell: int, origin_time: int) -> None:
        self._diagnostic_cache_key = None
        self._diagnostic_cache = None
        super().schedule(cell, origin_time)

    def observe(self, cell: int, outcome: int, origin_time: int) -> None:
        self._diagnostic_cache_key = None
        self._diagnostic_cache = None
        super().observe(cell, outcome, origin_time)


class SurvivalFlooredMonotoneSIV(MonotoneScheduledInformationPolicy):
    """Survival-weighted exploration plus a non-negative elapsed-time increment."""

    def __init__(self, *args, survival_exploration: float = 0.6, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not np.isfinite(survival_exploration) or float(survival_exploration) < 0.0:
            raise ValueError("survival_exploration must be finite and non-negative")
        self.survival_exploration = float(survival_exploration)

    def diagnostics(self, cell: int, time: int) -> Dict[str, float]:
        cell = int(cell)
        cache_key = ("survival_floor", cell, int(time))
        if self._diagnostic_cache_key == cache_key:
            return dict(self._diagnostic_cache)
        mean = self.posterior_mean(cell)
        age_miv = self.miv(cell, time, self.kernel, age_aware=True)
        count_miv = self.miv(cell, time, self.kernel, age_aware=False)
        oracle_miv = self.miv(cell, time, self.true_kernel, age_aware=True)
        pseudo_information = sum(
            self.kernel.maturation_probability(
                max(0, int(time) - item.origin_time), self.planning_window
            )
            for item in self.pending[cell]
        )
        effective = self.alpha[cell] + self.beta[cell] + pseudo_information
        survival_bonus = self.survival_exploration * np.sqrt(
            mean * (1.0 - mean) / (effective + 1.0)
        )
        positive_sqrt_increment = max(
            np.sqrt(max(age_miv, 0.0)) - np.sqrt(max(count_miv, 0.0)), 0.0
        )
        timing_bonus = (
            self.timing_gain_scale * self.information_coefficient * positive_sqrt_increment
        )
        score = mean + survival_bonus + timing_bonus
        survival_score = mean + survival_bonus
        age_score = mean + self.information_coefficient * np.sqrt(max(age_miv, 0.0))
        count_score = mean + self.information_coefficient * np.sqrt(max(count_miv, 0.0))
        diagnostics = {
            "mean": float(mean),
            "miv": float(age_miv),
            "raw_age_aware_miv": float(age_miv),
            "positive_timing_miv_increment": float(max(age_miv - count_miv, 0.0)),
            "positive_sqrt_timing_increment": float(positive_sqrt_increment),
            "pseudo_information": float(pseudo_information),
            "survival_bonus": float(survival_bonus),
            "timing_bonus": float(timing_bonus),
            "survival_floor_score": float(survival_score),
            "oracle_miv": float(oracle_miv),
            "age_aware_miv": float(age_miv),
            "count_only_miv": float(count_miv),
            "same_state_miv_gap": float(abs(age_miv - count_miv)),
            "age_aware_score": float(age_score),
            "count_only_score": float(count_score),
            "same_state_score_gap": float(abs(age_score - count_score)),
            "same_state_score_midpoint": float(0.5 * (age_score + count_score)),
            "score": float(score),
        }
        self._diagnostic_cache_key = cache_key
        self._diagnostic_cache = diagnostics
        return dict(diagnostics)

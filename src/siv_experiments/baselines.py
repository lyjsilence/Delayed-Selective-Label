from __future__ import annotations

"""Continuous-prior delayed-feedback baselines."""

from typing import Dict

import numpy as np
from scipy.stats import beta as beta_distribution

from siv_core.policies import FiniteCellPolicy


class ContinuousPriorPolicy(FiniteCellPolicy):
    def __init__(self, *args, prior_alpha: np.ndarray, prior_beta: np.ndarray, **kwargs):
        alpha = np.asarray(prior_alpha, dtype=float)
        beta = np.asarray(prior_beta, dtype=float)
        if alpha.shape != beta.shape or alpha.ndim != 1:
            raise ValueError("continuous prior shapes must be aligned vectors")
        if np.any(alpha <= 0.0) or np.any(beta <= 0.0):
            raise ValueError("continuous prior shapes must be positive")
        kwargs["warm_successes"] = np.zeros(alpha.size, dtype=int)
        kwargs["warm_total"] = 0
        super().__init__(*args, **kwargs)
        self.alpha = alpha.copy()
        self.beta = beta.copy()


class DelayedBayesUCB(ContinuousPriorPolicy):
    def __init__(self, *args, posterior_quantile: float = 0.95, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not 0.5 < float(posterior_quantile) < 1.0:
            raise ValueError("posterior_quantile must lie strictly between .5 and 1")
        self.posterior_quantile = float(posterior_quantile)

    def diagnostics(self, cell: int, time: int) -> Dict[str, float]:
        del time
        cell = int(cell)
        score = beta_distribution.ppf(self.posterior_quantile, self.alpha[cell], self.beta[cell])
        return {"mean": self.posterior_mean(cell), "miv": 0.0, "score": float(score)}

    def decide(self, cell: int, time: int) -> int:
        return int(self.diagnostics(cell, time)["score"] > self.cost)


class TemperedDelayedThompsonSampling(ContinuousPriorPolicy):
    def __init__(self, *args, temperature: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not np.isfinite(temperature) or float(temperature) <= 0.0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = float(temperature)

    def decide(self, cell: int, time: int) -> int:
        del time
        cell = int(cell)
        draw = self.rng.beta(
            self.alpha[cell] / self.temperature, self.beta[cell] / self.temperature
        )
        return int(draw > self.cost)

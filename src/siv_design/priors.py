from __future__ import annotations

"""Continuous underwriting-score priors for v10 adaptive development."""

from dataclasses import dataclass
import hashlib
from typing import Tuple

import numpy as np

from siv_core.policies import ScheduledInformationPolicy

from .arrivals import derived_seed


def continuous_underwriting_prior(
    seed: int, probabilities: np.ndarray, *, strength: float = 4.0, logit_noise_sd: float = 0.35
) -> Tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=float)
    if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
        raise ValueError("probabilities must lie strictly inside (0, 1)")
    if strength <= 0.0 or logit_noise_sd < 0.0:
        raise ValueError("prior strength must be positive and noise non-negative")
    rng = np.random.default_rng(derived_seed(int(seed), "underwriting_prior"))
    logits = np.log(probabilities / (1.0 - probabilities))
    noisy = logits + rng.normal(0.0, float(logit_noise_sd), len(probabilities))
    means = 1.0 / (1.0 + np.exp(-noisy))
    return float(strength) * means, float(strength) * (1.0 - means)


def prior_fingerprint(alpha: np.ndarray, beta: np.ndarray) -> str:
    digest = hashlib.sha256(b"siv_v10.continuous_prior.v1")
    digest.update(np.ascontiguousarray(alpha, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(beta, dtype="<f8").tobytes())
    return digest.hexdigest()


class ContinuousPriorScheduledPolicy(ScheduledInformationPolicy):
    """Shared SIV/Count policy state initialized by continuous Beta shapes."""

    def __init__(self, *args, prior_alpha: np.ndarray, prior_beta: np.ndarray, **kwargs):
        alpha = np.asarray(prior_alpha, dtype=float)
        beta = np.asarray(prior_beta, dtype=float)
        if alpha.shape != beta.shape or np.any(alpha <= 0.0) or np.any(beta <= 0.0):
            raise ValueError("continuous prior shapes must be positive and aligned")
        kwargs["warm_successes"] = np.zeros(len(alpha), dtype=int)
        kwargs["warm_total"] = 0
        super().__init__(*args, **kwargs)
        self.alpha = alpha.copy()
        self.beta = beta.copy()

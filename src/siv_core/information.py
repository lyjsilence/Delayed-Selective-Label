from __future__ import annotations

"""Exact finite-cell Beta--Bernoulli scheduled-information calculations."""

from typing import Iterable

import numpy as np


def beta_variance(alpha: float, beta: float) -> float:
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("Beta shapes must be positive")
    total = alpha + beta
    return float(alpha * beta / (total**2 * (total + 1.0)))


def poisson_binomial_pmf(probabilities: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(probabilities), dtype=float)
    if (
        values.ndim != 1
        or not np.all(np.isfinite(values))
        or np.any((values < 0.0) | (values > 1.0))
    ):
        raise ValueError("maturation probabilities must lie in [0, 1]")
    pmf = np.asarray([1.0])
    for probability in values:
        updated = np.zeros(len(pmf) + 1)
        updated[:-1] += pmf * (1.0 - probability)
        updated[1:] += pmf * probability
        pmf = updated
    return pmf


def scheduled_variance(alpha: float, beta: float, probabilities: Iterable[float]) -> float:
    pmf = poisson_binomial_pmf(probabilities)
    total = alpha + beta
    remaining = total / (total + np.arange(len(pmf), dtype=float))
    return float(beta_variance(alpha, beta) * np.dot(pmf, remaining))


def marginal_information_value(
    alpha: float, beta: float, probabilities: Iterable[float], new_label_probability: float = 1.0
) -> float:
    """Expected reduction in squared-error Bayes risk from one scheduled label."""

    if not 0.0 <= new_label_probability <= 1.0:
        raise ValueError("new-label probability must lie in [0, 1]")
    pmf = poisson_binomial_pmf(probabilities)
    total = alpha + beta
    matured = np.arange(len(pmf), dtype=float)
    reductions = beta_variance(alpha, beta) * total / ((total + matured) * (total + matured + 1.0))
    return float(new_label_probability * np.dot(pmf, reductions))


def exact_one_step_decision_value(alpha: float, beta: float, cost: float) -> float:
    """Myopic value of observing one immediate Bernoulli label before acting."""

    total = alpha + beta
    mean = alpha / total
    after_success = max((alpha + 1.0) / (total + 1.0) - cost, 0.0)
    after_failure = max(alpha / (total + 1.0) - cost, 0.0)
    return float(mean * after_success + (1.0 - mean) * after_failure - max(mean - cost, 0.0))

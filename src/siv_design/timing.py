from __future__ import annotations

"""Delay kernels normalized to an exact positive-integer mean."""

from dataclasses import dataclass
from math import erf, exp, log, sqrt
from typing import Dict

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from siv_core.timing import DiscreteMixtureKernel, GeometricKernel


def _conditional(kernel, elapsed: int, within: int) -> float:
    if elapsed < 0 or within < 0:
        raise ValueError("elapsed and within must be non-negative")
    if within == 0:
        return 0.0
    denominator = kernel.survival(int(elapsed))
    if denominator <= 1e-15:
        return 1.0
    return float(np.clip(1.0 - kernel.survival(int(elapsed + within)) / denominator, 0.0, 1.0))


def _integer_mean_from_survival(survival, *, tolerance: float = 1e-12) -> float:
    total = 0.0
    elapsed = 0
    while elapsed < 100000:
        value = float(survival(elapsed))
        total += value
        if elapsed > 100 and value < tolerance:
            break
        elapsed += 1
    else:
        raise RuntimeError("integer-mean survival sum did not converge")
    return total


def _lognormal_integer_mean(mu_log: float, sigma_log: float) -> float:
    """Accurate E[ceil(X)] for the registered log-normal tail."""

    ages = np.arange(1, 1000001, dtype=float)
    survival = ndtr((float(mu_log) - np.log(ages)) / float(sigma_log))
    if survival[-1] > 1e-12:
        raise RuntimeError("log-normal discrete-mean grid is too short")
    return float(1.0 + survival.sum())


@dataclass(frozen=True)
class DiscreteMeanWeibullKernel:
    mean: float
    shape: float
    scale: float

    @classmethod
    def compile(cls, target_mean: float, shape: float):
        target_mean, shape = float(target_mean), float(shape)
        if target_mean <= 1.0 or shape <= 0.0:
            raise ValueError("target mean must exceed one and shape must be positive")

        def integer_mean(scale):
            return _integer_mean_from_survival(
                lambda elapsed: exp(-((float(elapsed) / scale) ** shape))
            )

        scale = brentq(
            lambda value: integer_mean(value) - target_mean,
            0.05,
            target_mean * 20.0,
            xtol=1e-12,
            rtol=1e-12,
        )
        return cls(target_mean, shape, float(scale))

    @property
    def discrete_mean(self) -> float:
        return _integer_mean_from_survival(self.survival)

    def survival(self, elapsed: int) -> float:
        if elapsed < 0:
            raise ValueError("elapsed must be non-negative")
        return float(np.exp(-((float(elapsed) / self.scale) ** self.shape)))

    def maturation_probability(self, elapsed: int, within: int) -> float:
        return _conditional(self, elapsed, within)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return np.maximum(1, np.ceil(self.scale * rng.weibull(self.shape, int(size)))).astype(
            np.int64
        )


@dataclass(frozen=True)
class DiscreteMeanLogNormalKernel:
    mean: float
    sigma_log: float
    mu_log: float

    @classmethod
    def compile(cls, target_mean: float, sigma_log: float):
        target_mean, sigma_log = float(target_mean), float(sigma_log)
        if target_mean <= 1.0 or sigma_log <= 0.0:
            raise ValueError("target mean must exceed one and sigma must be positive")

        def integer_mean(mu):
            return _lognormal_integer_mean(mu, sigma_log)

        # The continuous-mean solution is a close bracket center. One log unit
        # above it multiplies the mean by e and is safely above the integer
        # target without creating an unnecessarily extreme numerical tail.
        center = log(target_mean) - 0.5 * sigma_log**2
        mu = brentq(
            lambda value: integer_mean(value) - target_mean,
            center - 3.0,
            center + 1.0,
            xtol=1e-12,
            rtol=1e-12,
        )
        return cls(target_mean, sigma_log, float(mu))

    @property
    def discrete_mean(self) -> float:
        return _lognormal_integer_mean(self.mu_log, self.sigma_log)

    def survival(self, elapsed: int) -> float:
        if elapsed < 0:
            raise ValueError("elapsed must be non-negative")
        if elapsed == 0:
            return 1.0
        z = (log(float(elapsed)) - self.mu_log) / (self.sigma_log * sqrt(2.0))
        return float(0.5 * (1.0 - erf(z)))

    def maturation_probability(self, elapsed: int, within: int) -> float:
        return _conditional(self, elapsed, within)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return np.maximum(1, np.ceil(rng.lognormal(self.mu_log, self.sigma_log, int(size)))).astype(
            np.int64
        )


@dataclass(frozen=True)
class DiscreteWeibullMixtureKernel:
    """Two smooth Weibull components with an exact discrete mixture mean."""

    early_weight: float
    early: DiscreteMeanWeibullKernel
    late: DiscreteMeanWeibullKernel

    @classmethod
    def compile(
        cls,
        *,
        early_weight: float,
        early_mean: float,
        early_shape: float,
        late_mean: float,
        late_shape: float,
    ):
        weight = float(early_weight)
        if not 0.0 < weight < 1.0:
            raise ValueError("early_weight must lie strictly between zero and one")
        return cls(
            early_weight=weight,
            early=DiscreteMeanWeibullKernel.compile(early_mean, early_shape),
            late=DiscreteMeanWeibullKernel.compile(late_mean, late_shape),
        )

    @property
    def discrete_mean(self) -> float:
        return float(
            self.early_weight * self.early.discrete_mean
            + (1.0 - self.early_weight) * self.late.discrete_mean
        )

    def survival(self, elapsed: int) -> float:
        return float(
            self.early_weight * self.early.survival(elapsed)
            + (1.0 - self.early_weight) * self.late.survival(elapsed)
        )

    def maturation_probability(self, elapsed: int, within: int) -> float:
        return _conditional(self, elapsed, within)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        count = int(size)
        choose_early = rng.random(count) < self.early_weight
        early_delays = self.early.sample(rng, count)
        late_delays = self.late.sample(rng, count)
        return np.where(choose_early, early_delays, late_delays).astype(np.int64)


def compile_kernel(definition: Dict[str, object]):
    family = str(definition["family"]).lower()
    if family == "geometric":
        return GeometricKernel(float(definition.get("mean", 50.0)))
    if family == "weibull":
        return DiscreteMeanWeibullKernel.compile(
            float(definition["target_discrete_mean"]), float(definition["shape"])
        )
    if family == "lognormal":
        return DiscreteMeanLogNormalKernel.compile(
            float(definition["target_discrete_mean"]), float(definition["sigma_log"])
        )
    if family == "weibull_mixture":
        early = definition["early"]
        late = definition["late"]
        return DiscreteWeibullMixtureKernel.compile(
            early_weight=float(definition["early_weight"]),
            early_mean=float(early["target_discrete_mean"]),
            early_shape=float(early["shape"]),
            late_mean=float(late["target_discrete_mean"]),
            late_shape=float(late["shape"]),
        )
    if family == "mixture":
        return DiscreteMixtureKernel(
            tuple(definition["delays"]), tuple(definition["probabilities"])
        )
    raise ValueError("unsupported v10 kernel family: " + family)

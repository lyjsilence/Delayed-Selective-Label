from __future__ import annotations

"""Matched-mean delay kernels and elapsed-feedback-time diagnostics."""

from dataclasses import dataclass
from math import erf, gamma, log, sqrt
from typing import Dict, Optional, Protocol, Sequence

import numpy as np


class DelayKernel(Protocol):
    mean: float

    def survival(self, elapsed: int) -> float: ...
    def maturation_probability(self, elapsed: int, within: int) -> float: ...
    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray: ...


def _validate_time(elapsed: int, within: int) -> None:
    if elapsed < 0 or within < 0:
        raise ValueError("elapsed and within must be non-negative")


def _conditional(kernel: DelayKernel, elapsed: int, within: int) -> float:
    _validate_time(elapsed, within)
    if within == 0:
        return 0.0
    denominator = kernel.survival(elapsed)
    if denominator <= 1e-15:
        return 1.0
    return float(np.clip(1.0 - kernel.survival(elapsed + within) / denominator, 0.0, 1.0))


@dataclass(frozen=True)
class GeometricKernel:
    """Geometric delay on {1, 2, ...}; ``mean`` is exact."""

    mean: float = 50.0
    cap: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mean < 1.0 or (self.cap is not None and self.cap < 1):
            raise ValueError("a positive-support geometric mean must be at least one")

    @property
    def probability(self) -> float:
        return 1.0 / self.mean

    def survival(self, elapsed: int) -> float:
        if elapsed < 0:
            raise ValueError("elapsed must be non-negative")
        if self.cap is not None and elapsed >= self.cap:
            return 0.0
        return float((1.0 - self.probability) ** elapsed)

    def maturation_probability(self, elapsed: int, within: int) -> float:
        _validate_time(elapsed, within)
        if self.cap is not None:
            return _conditional(self, elapsed, within)
        return float(1.0 - (1.0 - self.probability) ** within)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        draws = rng.geometric(self.probability, int(size)).astype(np.int64)
        return np.minimum(draws, self.cap) if self.cap is not None else draws


@dataclass(frozen=True)
class WeibullKernel:
    """Ceiled continuous Weibull delay with a matched continuous-time mean."""

    mean: float = 50.0
    shape: float = 2.0
    cap: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mean <= 0.0 or self.shape <= 0.0 or (self.cap is not None and self.cap < 1):
            raise ValueError("Weibull parameters must be positive")

    @property
    def scale(self) -> float:
        return self.mean / gamma(1.0 + 1.0 / self.shape)

    def survival(self, elapsed: int) -> float:
        if elapsed < 0:
            raise ValueError("elapsed must be non-negative")
        if self.cap is not None and elapsed >= self.cap:
            return 0.0
        return float(np.exp(-((float(elapsed) / self.scale) ** self.shape)))

    def maturation_probability(self, elapsed: int, within: int) -> float:
        return _conditional(self, elapsed, within)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        draws = np.maximum(1, np.ceil(self.scale * rng.weibull(self.shape, int(size)))).astype(
            np.int64
        )
        return np.minimum(draws, self.cap) if self.cap is not None else draws


@dataclass(frozen=True)
class LogNormalKernel:
    """Ceiled log-normal delay parameterized by its arithmetic mean."""

    mean: float = 50.0
    sigma_log: float = 1.0
    cap: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mean <= 0.0 or self.sigma_log <= 0.0 or (self.cap is not None and self.cap < 1):
            raise ValueError("log-normal parameters must be positive")

    @property
    def mu_log(self) -> float:
        return log(self.mean) - 0.5 * self.sigma_log**2

    def survival(self, elapsed: int) -> float:
        if elapsed < 0:
            raise ValueError("elapsed must be non-negative")
        if self.cap is not None and elapsed >= self.cap:
            return 0.0
        if elapsed == 0:
            return 1.0
        z = (log(float(elapsed)) - self.mu_log) / (self.sigma_log * sqrt(2.0))
        return float(0.5 * (1.0 - erf(z)))

    def maturation_probability(self, elapsed: int, within: int) -> float:
        return _conditional(self, elapsed, within)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        draws = rng.lognormal(self.mu_log, self.sigma_log, int(size))
        values = np.maximum(1, np.ceil(draws)).astype(np.int64)
        return np.minimum(values, self.cap) if self.cap is not None else values


@dataclass(frozen=True)
class DiscreteMixtureKernel:
    """Finite delay mixture, useful for a high-TI matched-mean environment."""

    delays: Sequence[int]
    probabilities: Sequence[float]

    def __post_init__(self) -> None:
        delays = np.asarray(self.delays, dtype=int)
        probabilities = np.asarray(self.probabilities, dtype=float)
        if delays.ndim != 1 or probabilities.shape != delays.shape or len(delays) == 0:
            raise ValueError("delays and probabilities must be equal-length vectors")
        if (
            np.any(delays < 1)
            or np.any(probabilities < 0.0)
            or not np.isclose(probabilities.sum(), 1.0)
        ):
            raise ValueError("invalid finite delay mixture")

    @property
    def mean(self) -> float:
        return float(np.dot(np.asarray(self.delays, dtype=float), self.probabilities))

    def survival(self, elapsed: int) -> float:
        if elapsed < 0:
            raise ValueError("elapsed must be non-negative")
        return float(np.sum(np.asarray(self.probabilities)[np.asarray(self.delays) > elapsed]))

    def maturation_probability(self, elapsed: int, within: int) -> float:
        return _conditional(self, elapsed, within)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.choice(np.asarray(self.delays, dtype=np.int64), int(size), p=self.probabilities)


def make_kernel(definition: Dict[str, object]) -> DelayKernel:
    family = str(definition["family"]).lower()
    if family == "geometric":
        return GeometricKernel(
            float(definition.get("mean", 50.0)),
            None if definition.get("cap") is None else int(definition["cap"]),
        )
    if family == "weibull":
        return WeibullKernel(
            float(definition.get("mean", 50.0)),
            float(definition["shape"]),
            None if definition.get("cap") is None else int(definition["cap"]),
        )
    if family == "lognormal":
        return LogNormalKernel(
            float(definition.get("mean", 50.0)),
            float(definition["sigma_log"]),
            None if definition.get("cap") is None else int(definition["cap"]),
        )
    if family == "mixture":
        return DiscreteMixtureKernel(
            tuple(definition["delays"]), tuple(definition["probabilities"])
        )
    raise ValueError("unknown delay family: " + family)


def fit_kernel(family: str, delays: Sequence[int]) -> DelayKernel:
    """Outcome-free method-of-moments fit of a supported delay family."""

    values = np.asarray(delays, dtype=float)
    if values.ndim != 1 or len(values) < 20 or np.any(values < 1):
        raise ValueError("delay calibration requires at least 20 positive observations")
    family = family.lower()
    mean = float(values.mean())
    if family == "geometric":
        return GeometricKernel(max(1.0, mean))
    if family == "lognormal":
        logged = np.log(values)
        sigma = float(np.clip(logged.std(ddof=1), 0.05, 2.5))
        fitted_mean = float(np.exp(logged.mean() + 0.5 * sigma**2))
        return LogNormalKernel(fitted_mean, sigma)
    if family == "weibull":
        cv2 = float(values.var(ddof=1) / max(mean**2, 1e-12))
        lower, upper = 0.2, 10.0
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            implied = gamma(1.0 + 2.0 / middle) / gamma(1.0 + 1.0 / middle) ** 2 - 1.0
            if implied > cv2:
                lower = middle
            else:
                upper = middle
        return WeibullKernel(mean, 0.5 * (lower + upper))
    raise ValueError("fitting is not implemented for family " + family)


def pending_age_weights(kernel: DelayKernel, max_elapsed: Optional[int] = None) -> np.ndarray:
    """Stationary age distribution of an item conditional on still being pending."""

    if max_elapsed is None:
        cap = getattr(kernel, "cap", None)
        max_elapsed = (
            int(cap) - 1 if cap is not None else max(500, int(np.ceil(20.0 * kernel.mean)))
        )
    survival = np.asarray([kernel.survival(age) for age in range(max_elapsed + 1)], dtype=float)
    if survival.sum() <= 0.0:
        raise ValueError("kernel has no positive pending mass")
    return survival / survival.sum()


def timing_informativeness(
    kernel: DelayKernel, within: int, max_elapsed: Optional[int] = None
) -> float:
    """TI_s: pending-age-weighted SD of conditional maturation probability."""

    weights = pending_age_weights(kernel, max_elapsed)
    values = np.asarray(
        [kernel.maturation_probability(age, within) for age in range(len(weights))], dtype=float
    )
    center = float(np.dot(weights, values))
    return float(np.sqrt(np.dot(weights, (values - center) ** 2)))


def monte_carlo_timing_informativeness(
    kernel: DelayKernel, within: int, *, seed: int = 0, draws: int = 200000
) -> float:
    """Monte Carlo TI using empirical exposure counts from sampled delays."""

    delays = kernel.sample(np.random.default_rng(seed), int(draws))
    counts = np.bincount(delays, minlength=int(delays.max()) + 1)
    # An item of delay d contributes once at every pending age 0,...,d-1.
    exposure = np.cumsum(counts[::-1])[::-1][1:]
    ages = np.arange(len(exposure), dtype=int)
    weights = exposure.astype(float) / exposure.sum()
    values = np.asarray([kernel.maturation_probability(int(age), within) for age in ages])
    center = float(np.dot(weights, values))
    return float(np.sqrt(np.dot(weights, (values - center) ** 2)))

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr


@dataclass(frozen=True)
class LogNormalDelay:
    mu: float
    sigma: float
    mean: float
    mean_error_bound: float

    def survival(self, elapsed):
        if elapsed < 0:
            raise ValueError("elapsed must be nonnegative")
        return 1.0 if elapsed == 0 else float(ndtr((self.mu - np.log(elapsed)) / self.sigma))

    def maturation_probability(self, elapsed, within):
        if elapsed < 0 or within < 0:
            raise ValueError("times must be nonnegative")
        if within == 0:
            return 0.0
        denominator = self.survival(elapsed)
        return (
            1.0
            if denominator <= 1e-15
            else float(np.clip(1 - self.survival(elapsed + within) / denominator, 0, 1))
        )

    def sample(self, rng, size):
        return np.maximum(1, np.ceil(rng.lognormal(self.mu, self.sigma, size))).astype(np.int64)


@dataclass(frozen=True)
class Mixture:
    fast: LogNormalDelay
    slow: LogNormalDelay
    probability: float = 0.25

    @property
    def mean(self):
        return self.probability * self.fast.mean + (1 - self.probability) * self.slow.mean

    def survival(self, elapsed):
        return self.probability * self.fast.survival(elapsed) + (
            1 - self.probability
        ) * self.slow.survival(elapsed)

    def maturation_probability(self, elapsed, within):
        if elapsed < 0 or within < 0:
            raise ValueError("Negative time")
        if within == 0:
            return 0.0
        denominator = self.survival(elapsed)
        return (
            1.0
            if denominator <= 1e-15
            else float(np.clip(1 - self.survival(elapsed + within) / denominator, 0, 1))
        )

    def sample(self, rng, size):
        fast = rng.random(size) < self.probability
        noise = rng.standard_normal(size)
        continuous = np.exp(
            np.where(fast, self.fast.mu, self.slow.mu)
            + np.where(fast, self.fast.sigma, self.slow.sigma) * noise
        )
        # Observe a continuous arrival at the first integer decision time after it.
        return np.maximum(1, np.ceil(continuous)).astype(np.int64)


class CachedDelay:
    def __init__(self, kernel):
        self.kernel = kernel
        self.queries = {}

    def maturation_probability(self, elapsed, within):
        key = (elapsed, within)
        if key not in self.queries:
            self.queries[key] = self.kernel.maturation_probability(elapsed, within)
        return self.queries[key]

    def __deepcopy__(self, memo):
        return self

    def __getattr__(self, name):
        return getattr(self.kernel, name)

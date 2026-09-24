from __future__ import annotations

"""Independent historical-delay calibration for deployable SIV policies."""

import hashlib
from typing import Tuple

import numpy as np

from siv_core.timing import DelayKernel, DiscreteMixtureKernel
from siv_design.arrivals import derived_seed


def empirical_discrete_kernel(delays: np.ndarray) -> DiscreteMixtureKernel:
    """Fit a finite discrete delay kernel by empirical frequencies."""

    values = np.asarray(delays)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("calibration delays must be a non-empty one-dimensional array")
    if values.dtype.kind not in "iu":
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError("calibration delays must be finite integers")
    values = values.astype(np.int64)
    if np.any(values < 1):
        raise ValueError("calibration delays must be positive")
    support, counts = np.unique(values, return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    return DiscreteMixtureKernel(
        tuple(int(x) for x in support), tuple(float(x) for x in probabilities)
    )


def fit_independent_delay_kernel(
    environment_seed: int, true_kernel: DelayKernel, calibration_size: int
) -> Tuple[DiscreteMixtureKernel, np.ndarray, str]:
    """Draw and fit a pre-decision delay tape on a named independent RNG stream."""

    if int(calibration_size) != calibration_size or int(calibration_size) < 1:
        raise ValueError("calibration_size must be a positive integer")
    rng = np.random.default_rng(derived_seed(int(environment_seed), "historical_delay_calibration"))
    delays = np.asarray(true_kernel.sample(rng, int(calibration_size)), dtype=np.int64)
    fitted = empirical_discrete_kernel(delays)
    digest = hashlib.sha256(b"siv_v11.historical_delay_calibration.v1")
    digest.update(np.ascontiguousarray(delays, dtype="<i8").tobytes())
    return fitted, delays, digest.hexdigest()

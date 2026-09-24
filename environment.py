"""Generate the fixed Table 1 environment using independent random streams."""

import hashlib
import numpy as np


def rng_for(seed, stream):
    # Keep the published experiment's stream namespace for exact reproducibility.
    payload = f"v10::{int(seed)}::{stream}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)
    return np.random.default_rng(value)


def sample_delays(rng, size, cfg):
    fast = rng.random(size) < cfg["fast_probability"]
    noise = rng.standard_normal(size)
    return np.maximum(
        1,
        np.ceil(
            np.exp(
                np.where(fast, cfg["mu_fast"], cfg["mu_slow"]) + cfg["sigma"] * noise
            )
        ),
    ).astype(np.int64)


def prepare(seed, cfg):
    env = cfg["environment"]
    horizon, cells = env["horizon"], env["cells"]
    permutation = rng_for(seed, "fixed_grid_permutation").permutation(cells)

    def population(center):
        return np.asarray(
            env["unchanged_low"]
            + np.linspace(
                center - env["boundary_half_band"],
                center + env["boundary_half_band"],
                env["boundary_cells"],
            ).tolist()
            + env["unchanged_high"]
        )[permutation]

    prior = population(env["historical_boundary_center"])
    probabilities = population(env["current_boundary_center"])
    cumulative = np.cumsum(np.full(cells, 1.0 / cells))
    cumulative[-1] = 1.0
    profiles = np.searchsorted(
        cumulative, rng_for(seed, "arrivals::selection").random(horizon), side="right"
    ).astype(np.int64)
    outcomes = (
        rng_for(seed, "outcomes").random(horizon) < probabilities[profiles]
    ).astype(np.int8)
    delays = sample_delays(rng_for(seed, "delays"), horizon, cfg["delay"])
    calibration = sample_delays(
        rng_for(seed, "historical_delay_calibration"),
        env["calibration_size"],
        cfg["delay"],
    )
    support, counts = np.unique(calibration, return_counts=True)
    weights = counts.astype(float) / counts.sum()

    def survival(age):
        return float(np.sum(weights[support > age]))

    window = cfg["policy"]["planning_window"]
    maturation = np.asarray(
        [
            (
                1.0
                if survival(age) <= 1e-15
                else float(
                    np.clip(1.0 - survival(age + window) / survival(age), 0.0, 1.0)
                )
            )
            for age in range(horizon)
        ]
    )
    return (
        profiles,
        outcomes,
        delays,
        np.full(horizon, env["cost"]),
        prior * env["prior_strength"],
        (1 - prior) * env["prior_strength"],
        maturation,
        probabilities,
    )

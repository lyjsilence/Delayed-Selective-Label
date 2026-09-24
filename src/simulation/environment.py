"""Shared IID potential outcomes, priors, costs, and delay calibration."""

import numpy as np
from siv_design import ArrivalSpec, generate_potentials, draw_fixed_grid_population
from siv_design.arrivals import derived_seed
from siv_experiments.calibration import fit_independent_delay_kernel


def prepare(seed, kernel, cfg, horizon=None):
    env = cfg["environment"]
    t = horizon or env["horizon"]
    width = env["boundary_half_band"]
    historical = (
        env["unchanged_low"]
        + np.linspace(
            env["historical_boundary_center"] - width,
            env["historical_boundary_center"] + width,
            env["boundary_cells"],
        ).tolist()
        + env["unchanged_high"]
    )
    current = (
        env["unchanged_low"]
        + np.linspace(
            env["current_boundary_center"] - width,
            env["current_boundary_center"] + width,
            env["boundary_cells"],
        ).tolist()
        + env["unchanged_high"]
    )
    tape = generate_potentials(
        seed,
        kernel,
        arrival_spec=ArrivalSpec.negative_control(),
        horizon=t,
        cells=env["cells"],
        cost=env["reference_cost"],
        warm_total=0,
        probability_design={
            "type": "fixed_grid_seeded_cell_permutation",
            "values": current,
            "boundary_half_width": 0.14,
        },
        outcome_design=env["outcome_design"],
        delay_design=env["delay_design"],
    )
    prior = draw_fixed_grid_population(
        seed, historical, cost=env["reference_cost"], boundary_half_width=0.14
    ).probabilities
    alpha, beta = env["prior_strength"] * prior, env["prior_strength"] * (1 - prior)
    costs = np.full(t, env["cost"], dtype=float)
    fitted, calibration, fingerprint = fit_independent_delay_kernel(
        seed, kernel, env["calibration_size"]
    )
    return tape, alpha, beta, costs, fitted, fingerprint

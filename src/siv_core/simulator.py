from __future__ import annotations

"""Event-ordered pure-SIV simulator and learning-curve metrics."""

from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import beta as beta_distribution

from .dgp import PotentialOutcomes
from .information import beta_variance, exact_one_step_decision_value
from .policies import FiniteCellPolicy, ScheduledInformationPolicy


CostSchedule = Union[float, Sequence[float], np.ndarray, Callable[[int], float]]


def _materialize_costs(costs: Optional[CostSchedule], horizon: int, default: float) -> np.ndarray:
    """Return a validated, method-independent per-arrival cost tape.

    Callbacks are evaluated exactly once for each time index before the run starts.
    A materialized array is preferable for common-random-number experiments because
    the same tape can then be passed to every policy.
    """

    if costs is None:
        values = np.full(int(horizon), float(default), dtype=float)
    elif callable(costs):
        values = np.asarray([costs(time) for time in range(int(horizon))], dtype=float)
    elif np.isscalar(costs):
        values = np.full(int(horizon), float(costs), dtype=float)
    else:
        values = np.asarray(costs, dtype=float)
        if values.ndim != 1 or values.size != int(horizon):
            raise ValueError("costs must be scalar or a one-dimensional tape of length horizon")
        values = values.copy()
    if values.ndim != 1 or values.size != int(horizon):
        raise ValueError("cost callback must produce exactly one scalar per arrival")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("costs must be finite and lie in [0, 1]")
    return values


def run_policy(
    policy: FiniteCellPolicy,
    potentials: PotentialOutcomes,
    checkpoints: Iterable[int] = (1000, 2500, 5000, 10000),
    costs: Optional[CostSchedule] = None,
) -> Tuple[Dict[str, object], List[Dict[str, float]]]:
    horizon = len(potentials.cells)
    cost_tape = _materialize_costs(costs, horizon, policy.cost)
    if horizon:
        policy.cost = float(cost_tape[0])
    checkpoints = sorted(set(int(min(horizon, point)) for point in checkpoints if point > 0))
    if horizon not in checkpoints:
        checkpoints.append(horizon)
    due = defaultdict(list)
    utility = regret = latent_squared_error = brier_loss = 0.0
    approvals = near_correct = near_total = redundant = 0
    oracle_agreement = bayes_agreement = 0
    miv_errors = []  # type: List[float]
    miv_margin_ratios = []  # type: List[float]
    same_state_miv_gaps = []  # type: List[float]
    same_state_score_gap_ratios = []  # type: List[float]
    miv_values = np.zeros(horizon, dtype=float)
    actions = np.zeros(horizon, dtype=np.int8)
    identified = np.full(policy.cells, np.nan)
    curves = []  # type: List[Dict[str, float]]

    def update_identification(indices, identification_time):
        for index in indices:
            if not np.isnan(identified[index]):
                continue
            lower = beta_distribution.ppf(0.025, policy.alpha[index], policy.beta[index])
            upper = beta_distribution.ppf(0.975, policy.alpha[index], policy.beta[index])
            if upper < policy.cost or lower > policy.cost:
                identified[index] = identification_time

    update_identification(range(policy.cells), 0)

    for time in range(horizon):
        # Cost is exogenous and observed at decision time.  Setting it before
        # maturation diagnostics keeps every calculation on the same boundary.
        policy.cost = float(cost_tape[time])
        # The event ordering is intentional: only labels whose due time has arrived update.
        matured_now = due.pop(time, [])
        for cell, outcome, origin in matured_now:
            policy.observe(cell, outcome, origin)
        update_identification(set(cell for cell, _, _ in matured_now), time + 1)

        cell = int(potentials.cells[time])
        probability = float(potentials.probabilities[cell])
        diagnostics = policy.diagnostics(cell, time)
        action = int(policy.decide(cell, time))
        actions[time] = action
        known_oracle = int(probability > policy.cost)
        bayes_value = exact_one_step_decision_value(
            policy.alpha[cell], policy.beta[cell], policy.cost
        )
        bayes_oracle = int(policy.posterior_mean(cell) + bayes_value > policy.cost)
        oracle_agreement += int(action == known_oracle)
        bayes_agreement += int(action == bayes_oracle)
        near = abs(probability - policy.cost) <= 0.06
        near_total += int(near)
        near_correct += int(near and action == known_oracle)
        prediction = policy.posterior_mean(cell)
        latent_squared_error += (prediction - probability) ** 2
        brier_loss += (prediction - float(potentials.outcomes[time])) ** 2
        if isinstance(policy, ScheduledInformationPolicy):
            miv_values[time] = float(diagnostics["miv"])
            miv_errors.append(abs(float(diagnostics["miv"]) - float(diagnostics["oracle_miv"])))
            miv_margin_ratios.append(
                float(diagnostics["miv"]) / max(abs(prediction - policy.cost), 1e-8)
            )
            same_state_miv_gaps.append(float(diagnostics["same_state_miv_gap"]))
            same_state_score_gap_ratios.append(
                float(diagnostics["same_state_score_gap"])
                / max(abs(float(diagnostics["same_state_score_midpoint"]) - policy.cost), 1e-8)
            )

        if action:
            approvals += 1
            redundant += int(probability <= policy.cost)
            utility += float(potentials.outcomes[time]) - policy.cost
            due_time = time + int(potentials.delays[time])
            policy.schedule(cell, time)
            if due_time < horizon:
                due[due_time].append((cell, int(potentials.outcomes[time]), time))
        regret += max(probability - policy.cost, 0.0) - action * (probability - policy.cost)

        if time + 1 in checkpoints:
            curves.append(
                {
                    "time": float(time + 1),
                    "cumulative_utility": float(utility),
                    "cumulative_regret": float(regret),
                    "approval_rate": float(approvals / (time + 1)),
                    "brier_score": float(brier_loss / (time + 1)),
                    "latent_probability_mse": float(latent_squared_error / (time + 1)),
                    "posterior_bayes_risk": float(
                        np.mean(
                            [
                                beta_variance(policy.alpha[j], policy.beta[j])
                                for j in range(policy.cells)
                            ]
                        )
                    ),
                }
            )

    unprofitable_approval_rate = float(redundant / max(1, approvals))
    result = {
        "method": policy.name,
        "environment_seed": potentials.environment_seed,
        "horizon": horizon,
        "cumulative_utility": float(utility),
        "cumulative_regret": float(regret),
        "approval_rate": float(approvals / horizon),
        "unprofitable_approval_rate": unprofitable_approval_rate,
        # Retained only for exact compatibility with the preregistered column name.
        # Reports use ``unprofitable_approval_rate`` because the simulator cannot
        # identify an action's exploratory intent.
        "redundant_exploration_rate": unprofitable_approval_rate,
        "brier_score": float(brier_loss / horizon),
        "latent_probability_mse": float(latent_squared_error / horizon),
        "posterior_bayes_risk": float(
            np.mean([beta_variance(policy.alpha[j], policy.beta[j]) for j in range(policy.cells)])
        ),
        "median_posterior_variance": float(
            np.median([beta_variance(policy.alpha[j], policy.beta[j]) for j in range(policy.cells)])
        ),
        "near_boundary_decision_accuracy": float(near_correct / max(1, near_total)),
        "known_oracle_action_agreement": float(oracle_agreement / horizon),
        "bayes_oracle_action_agreement": float(bayes_agreement / horizon),
        "miv_absolute_error": float(np.mean(miv_errors)) if miv_errors else np.nan,
        "mean_miv": float(np.mean(miv_values)) if miv_errors else np.nan,
        "median_miv_to_absolute_reward_margin_ratio": (
            float(np.median(miv_margin_ratios)) if miv_margin_ratios else np.nan
        ),
        "mean_same_state_miv_gap": (
            float(np.mean(same_state_miv_gaps)) if same_state_miv_gaps else np.nan
        ),
        "median_same_state_score_gap_to_margin_ratio": (
            float(np.median(same_state_score_gap_ratios)) if same_state_score_gap_ratios else np.nan
        ),
        "median_timing_relevant_score_gap_to_margin_ratio": (
            float(
                np.median(
                    [
                        value
                        for value, gap in zip(same_state_score_gap_ratios, same_state_miv_gaps)
                        if gap > 1e-15
                    ]
                )
            )
            if any(gap > 1e-15 for gap in same_state_miv_gaps)
            else np.nan
        ),
        "timing_relevant_state_share": (
            float(np.mean(np.asarray(same_state_miv_gaps) > 1e-15))
            if same_state_miv_gaps
            else np.nan
        ),
        "mean_identification_time": float(
            np.nanmean(np.where(np.isnan(identified), horizon, identified))
        ),
        "action_trajectory": actions,
        "miv_trajectory": miv_values,
        "cost_trajectory": cost_tape.copy(),
        "potential_fingerprint": potentials.fingerprint(),
    }
    return result, curves

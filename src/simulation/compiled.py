"""Compiled evaluator for deterministic policies."""

import numpy as np
from numba import njit


@njit(cache=True)
def miv(alpha, beta, probs, newprob):
    pmf = np.zeros(len(probs) + 1)
    pmf[0] = 1.0
    for i in range(len(probs)):
        q = probs[i]
        for j in range(i + 1, 0, -1):
            pmf[j] = pmf[j] * (1 - q) + pmf[j - 1] * q
        pmf[0] *= 1 - q
    n = alpha + beta
    variance = alpha * beta / (n * n * (n + 1))
    value = 0.0
    for j in range(len(pmf)):
        value += pmf[j] * (variance * n / ((n + j) * (n + j + 1)))
    return newprob * value


@njit(cache=True)
def actions_for(
    cells, outcomes, delays, costs, prior_a, prior_b, maturation, rho, gamma, kappa, kind
):
    horizon = len(cells)
    alpha = prior_a.copy()
    beta = prior_b.copy()
    pending = np.zeros((len(alpha), horizon), dtype=np.int64)
    counts = np.zeros(len(alpha), dtype=np.int64)
    due_heads = np.full(horizon, -1, dtype=np.int64)
    due_next = np.full(horizon, -1, dtype=np.int64)
    actions = np.zeros(horizon, dtype=np.int8)
    for t in range(horizon):
        origin = due_heads[t]
        while origin >= 0:
            c = cells[origin]
            y = outcomes[origin]
            alpha[c] += y
            beta[c] += 1 - y
            for j in range(counts[c]):
                if pending[c, j] == origin:
                    for k in range(j, counts[c] - 1):
                        pending[c, k] = pending[c, k + 1]
                    counts[c] -= 1
                    break
            origin = due_next[origin]
        c = cells[t]
        n = alpha[c] + beta[c]
        mu = alpha[c] / n
        if kind == 1:
            score = mu + rho * np.sqrt(2 * np.log(2.0 + t + 1) / max(1.0, n - 2))
        elif kind == 4:
            value = (
                mu * max((alpha[c] + 1) / (n + 1) - costs[t], 0.0)
                + (1 - mu) * max(alpha[c] / (n + 1) - costs[t], 0.0)
                - max(mu - costs[t], 0.0)
            )
            score = mu + kappa * value
        elif kind == 5:
            score = mu
        else:
            probs = np.empty(counts[c])
            pseudo = 0.0
            for j in range(counts[c]):
                q = maturation[t - pending[c, j]]
                probs[j] = q
                pseudo += q
            bonus = rho * np.sqrt(mu * (1 - mu) / (n + pseudo + 1))
            delta = 0.0
            if gamma > 0:
                age = miv(alpha[c], beta[c], probs, maturation[0])
                count = miv(alpha[c], beta[c], np.full(counts[c], maturation[0]), maturation[0])
                delta = max(np.sqrt(max(age, 0.0)) - np.sqrt(max(count, 0.0)), 0.0)
            if kind == 2:
                score = mu + kappa * np.sqrt(max(miv(alpha[c], beta[c], probs, maturation[0]), 0.0))
            elif kind == 3:
                score = mu + kappa * np.sqrt(
                    max(
                        miv(alpha[c], beta[c], np.full(counts[c], maturation[0]), maturation[0]),
                        0.0,
                    )
                )
            else:
                score = mu + bonus + gamma * kappa * delta
        if score > costs[t]:
            actions[t] = 1
            pending[c, counts[c]] = t
            counts[c] += 1
            due = t + delays[t]
            if due < horizon:
                due_next[t] = due_heads[due]
                due_heads[due] = t
    return actions


def evaluate(tape, a, b, costs, kernel, spec, cache):
    params = spec["parameters"]
    method = spec["method"]
    horizon = len(tape.cells)
    if method == "delayed_ucb":
        window = 0
        rho = params["exploration"]
        gamma = 0.0
        kind = 1
    elif method == "tuned_survival":
        window = params["planning_window"]
        rho = params["exploration"]
        gamma = 0.0
        kind = 0
    elif method in ["raw_fitted_siv", "count_only"]:
        window = params["planning_window"]
        rho = 0.0
        gamma = 0.0
        kind = 2 if method == "raw_fitted_siv" else 3
    elif method in ["one_step_bayes", "matured_greedy"]:
        window = 0
        rho = 0.0
        gamma = 0.0
        kind = 4 if method == "one_step_bayes" else 5
    else:
        window = params["planning_window"]
        rho = params["survival_exploration"]
        gamma = params["timing_gain_scale"]
        kind = 0
    if window not in cache:
        cache[window] = (
            np.zeros(horizon)
            if kind in [1, 4, 5]
            else np.array([kernel.maturation_probability(age, window) for age in range(horizon)])
        )
    actions = actions_for(
        tape.cells,
        tape.outcomes,
        tape.delays,
        costs,
        a,
        b,
        cache[window],
        rho,
        gamma,
        params.get("information_coefficient", 2.0),
        kind,
    )
    probabilities = tape.probabilities[tape.cells]
    margin = probabilities - costs
    metrics = dict(
        total_utility=float(np.sum(actions * (tape.outcomes - costs))),
        total_pseudo_regret=float(np.sum(np.maximum(margin, 0) - actions * margin)),
        oracle_agreement=float(np.mean(actions == (margin > 0))),
        selection_rate=float(np.mean(actions)),
        unprofitable_selection_rate=float(
            np.sum(actions * (margin <= 0)) / max(1, np.sum(actions))
        ),
    )
    return metrics, actions

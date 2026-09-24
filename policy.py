"""SF-SIV information values and delayed-feedback decision loop."""

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
    cells, outcomes, delays, costs, prior_a, prior_b, maturation, rho, gamma, kappa
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
            count = miv(
                alpha[c], beta[c], np.full(counts[c], maturation[0]), maturation[0]
            )
            delta = max(np.sqrt(max(age, 0.0)) - np.sqrt(max(count, 0.0)), 0.0)
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

"""Seed-level totals and paired statistical summaries."""

from __future__ import annotations
import hashlib, json, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from siv_core.simulator import run_policy


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(str(temporary), str(path))


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary), str(path))


def ci95(values) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def one_sided_lower(values, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    return float(
        values.mean()
        - stats.t.ppf(1.0 - alpha, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )


def run_totals(policy, potentials, costs) -> dict[str, float]:
    horizon = len(potentials.cells)
    due = defaultdict(list)
    utility = 0.0
    regret = 0.0
    selections = 0
    unprofitable = 0
    oracle_agreement = 0
    for time in range(horizon):
        policy.cost = float(costs[time])
        for cell, outcome, origin in due.pop(time, []):
            policy.observe(cell, outcome, origin)
        cell = int(potentials.cells[time])
        probability = float(potentials.probabilities[cell])
        action = int(policy.decide(cell, time))
        oracle = int(probability > policy.cost)
        oracle_agreement += int(action == oracle)
        if action:
            selections += 1
            unprofitable += int(probability <= policy.cost)
            utility += float(potentials.outcomes[time]) - policy.cost
            policy.schedule(cell, time)
            due_time = time + int(potentials.delays[time])
            if due_time < horizon:
                due[due_time].append((cell, int(potentials.outcomes[time]), time))
        regret += max(probability - policy.cost, 0.0) - action * (probability - policy.cost)
    return {
        "total_utility": float(utility),
        "total_pseudo_regret": float(regret),
        "oracle_agreement": float(oracle_agreement / horizon),
        "selection_rate": float(selections / horizon),
        "unprofitable_selection_rate": float(unprofitable / max(1, selections)),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, part in frame.groupby("method"):
        utility = ci95(part["total_utility"])
        regret = ci95(part["total_pseudo_regret"])
        rows.append(
            {
                "method": method,
                "seeds": int(part["seed"].nunique()),
                "total_utility_mean": utility[0],
                "total_utility_ci_low": utility[1],
                "total_utility_ci_high": utility[2],
                "total_utility_sd": float(part["total_utility"].std(ddof=1)),
                "total_pseudo_regret_mean": regret[0],
                "total_pseudo_regret_ci_low": regret[1],
                "total_pseudo_regret_ci_high": regret[2],
                "total_pseudo_regret_sd": float(part["total_pseudo_regret"].std(ddof=1)),
                "oracle_agreement": float(part["oracle_agreement"].mean()),
                "selection_rate": float(part["selection_rate"].mean()),
                "unprofitable_selection_rate": float(part["unprofitable_selection_rate"].mean()),
                "realized_delay_mean": float(part["realized_delay_mean"].mean()),
                "realized_delay_sd": float(part["realized_delay_sd"].mean()),
                "realized_unique_delays": float(part["realized_unique_delays"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("total_utility_mean", ascending=False)


def paired_contrasts(config: dict, frame: pd.DataFrame) -> pd.DataFrame:
    main = config["statistics"]["main_method"]
    utility = frame.pivot(index="seed", columns="method", values="total_utility")
    regret = frame.pivot(index="seed", columns="method", values="total_pseudo_regret")
    baselines = sorted(set(utility.columns) - {main})
    alpha = float(config["statistics"]["familywise_alpha"]) / (2 * len(baselines))
    rows = []
    for baseline in baselines:
        utility_gain = utility[main] - utility[baseline]
        regret_reduction = regret[baseline] - regret[main]
        utility_ci = ci95(utility_gain)
        regret_ci = ci95(regret_reduction)
        rows.append(
            {
                "baseline": baseline,
                "seeds": int(len(utility_gain)),
                "total_utility_gain_mean": utility_ci[0],
                "total_utility_gain_ci_low": utility_ci[1],
                "total_utility_gain_ci_high": utility_ci[2],
                "total_utility_gain_bonferroni_lower": one_sided_lower(utility_gain, alpha),
                "total_regret_reduction_mean": regret_ci[0],
                "total_regret_reduction_ci_low": regret_ci[1],
                "total_regret_reduction_ci_high": regret_ci[2],
                "total_regret_reduction_bonferroni_lower": one_sided_lower(regret_reduction, alpha),
            }
        )
    return pd.DataFrame(rows)

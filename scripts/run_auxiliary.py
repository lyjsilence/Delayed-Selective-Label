from __future__ import annotations

"""Mechanism and robustness diagnostics for the reported simulation study.

The main benchmark is run separately.  Every inferential
row is first aggregated within an auxiliary environment seed.
"""

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation.kernels import Mixture, LogNormalDelay, CachedDelay

from siv_core import GeometricKernel, PotentialOutcomes
from siv_core.dgp import method_seed
from siv_core.simulator import run_policy
from siv_design import ArrivalSpec, draw_fixed_grid_population, generate_potentials
from siv_design.arrivals import derived_seed
from siv_experiments.calibration import empirical_discrete_kernel
from siv_experiments import SurvivalFlooredMonotoneSIV


CONFIG = ROOT / "configs" / "auxiliary.json"
RESULTS = ROOT / "results" / "diagnostics"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"


@lru_cache(None)
def population_kernel():
    definition = json.loads((ROOT / "configs/delay.json").read_text())
    return CachedDelay(
        Mixture(LogNormalDelay(**definition["fast"]), LogNormalDelay(**definition["slow"]))
    )


def cached_empirical(values):
    return CachedDelay(empirical_discrete_kernel(values))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def ci95(values) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    mean = float(x.mean())
    if len(x) < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return mean, mean - half, mean + half


def design():
    low = [0.48, 0.52, 0.56, 0.60, 0.62, 0.64]
    high = [0.82, 0.84, 0.86, 0.88, 0.90, 0.92]
    historical = low + np.linspace(0.668, 0.682, 8).tolist() + high
    current = low + np.linspace(0.793, 0.807, 8).tolist() + high
    return historical, {
        "type": "fixed_grid_seeded_cell_permutation",
        "values": current,
        "boundary_half_width": 0.14,
    }


@dataclass(frozen=True)
class MeanMatchedMixtureKernel:
    mixture_weight: float
    mean: float = 271.6398231074338

    def __post_init__(self):
        if not 0.0 <= float(self.mixture_weight) <= 1.0:
            raise ValueError("mixture_weight must lie in [0,1]")

    @property
    def geometric(self):
        return GeometricKernel(self.mean)

    @property
    def bimodal(self):
        return population_kernel()

    def survival(self, elapsed: int) -> float:
        lam = float(self.mixture_weight)
        return (1.0 - lam) * self.geometric.survival(elapsed) + lam * self.bimodal.survival(elapsed)

    def maturation_probability(self, elapsed: int, within: int) -> float:
        if elapsed < 0 or within < 0:
            raise ValueError("elapsed and within must be non-negative")
        if within == 0:
            return 0.0
        denominator = self.survival(elapsed)
        if denominator <= 1e-15:
            return 1.0
        return float(np.clip(1.0 - self.survival(elapsed + within) / denominator, 0.0, 1.0))

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        n = int(size)
        choose_bimodal = rng.random(n) < float(self.mixture_weight)
        geometric = rng.geometric(1.0 / self.mean, n).astype(np.int64)
        bimodal = self.bimodal.sample(rng, n)
        return np.where(choose_bimodal, bimodal, geometric).astype(np.int64)


def make_tape(seed: int, kernel, cfg, *, horizon: int):
    historical, probability_design = design()
    arrival = ArrivalSpec.negative_control()
    potentials = generate_potentials(
        seed,
        kernel,
        arrival_spec=arrival,
        horizon=horizon,
        cells=cfg["cells"],
        cost=cfg["reference_cost"],
        warm_total=0,
        probability_design=probability_design,
        outcome_design="iid",
        delay_design="iid",
    )
    prior = draw_fixed_grid_population(
        seed, historical, cost=cfg["reference_cost"], boundary_half_width=0.14
    ).probabilities
    alpha = cfg["prior_strength"] * prior
    beta = cfg["prior_strength"] * (1.0 - prior)
    costs = np.full(horizon, cfg["cost"], dtype=float)
    return potentials, alpha, beta, costs


def fitted_kernel(seed: int, cfg):
    true = population_kernel()
    calibration = true.sample(
        np.random.default_rng(derived_seed(seed, "auxiliary_calibration_delays")),
        cfg["calibration_size"],
    )
    return cached_empirical(calibration), calibration


def policy(
    seed: int,
    cfg,
    kernel,
    alpha,
    beta,
    *,
    rho: float,
    gamma: float,
    planning: int,
    name: str,
    true_kernel=None,
):
    return SurvivalFlooredMonotoneSIV(
        name=name,
        cells=cfg["cells"],
        cost=cfg["reference_cost"],
        seed=method_seed(seed, "auxiliary_matched_policy"),
        prior_alpha=alpha,
        prior_beta=beta,
        kernel=kernel,
        true_kernel=kernel if true_kernel is None else true_kernel,
        planning_window=int(planning),
        information_coefficient=cfg["information_coefficient"],
        timing_gain_scale=float(gamma),
        survival_exploration=float(rho),
    )


def metrics(result, potentials, costs) -> dict[str, float]:
    actions = np.asarray(result["action_trajectory"], dtype=np.int8)
    probability = potentials.probabilities[potentials.cells]
    selected = int(actions.sum())
    return {
        "utility_total": float(result["cumulative_utility"]),
        "regret_total": float(result["cumulative_regret"]),
        "selection_rate": float(actions.mean()),
        "unprofitable_selection_rate": float(
            np.sum(actions * (probability <= costs)) / max(1, selected)
        ),
        "oracle_agreement": float(np.mean(actions == (probability > costs))),
        "action_sha256": array_hash(actions),
    }


def fast_decision(p, cell: int, time: int, *, record_timing: bool = False):
    """Exact SF-SIV score without diagnostics that do not affect the action."""
    cell = int(cell)
    mean = p.posterior_mean(cell)
    pseudo_information = sum(
        p.kernel.maturation_probability(max(0, int(time) - item.origin_time), p.planning_window)
        for item in p.pending[cell]
    )
    effective = p.alpha[cell] + p.beta[cell] + pseudo_information
    survival_bonus = p.survival_exploration * np.sqrt(mean * (1.0 - mean) / (effective + 1.0))
    delta = 0.0
    if p.timing_gain_scale > 0.0 or record_timing:
        age_miv = p.miv(cell, time, p.kernel, age_aware=True)
        count_miv = p.miv(cell, time, p.kernel, age_aware=False)
        delta = max(np.sqrt(max(age_miv, 0.0)) - np.sqrt(max(count_miv, 0.0)), 0.0)
    score = mean + survival_bonus + p.timing_gain_scale * p.information_coefficient * delta
    return int(float(score) > p.cost), float(delta), float(mean + survival_bonus)


def run_fast(p, potentials, costs, *, record_timing: bool = False):
    horizon = len(potentials.cells)
    due = defaultdict(list)
    utility = regret = 0.0
    actions = np.zeros(horizon, dtype=np.int8)
    increments = np.zeros(horizon, dtype=float) if record_timing else None
    for time in range(horizon):
        p.cost = float(costs[time])
        for cell, outcome, origin in due.pop(time, []):
            p.observe(cell, outcome, origin)
        cell = int(potentials.cells[time])
        probability = float(potentials.probabilities[cell])
        action, delta, _ = fast_decision(p, cell, time, record_timing=record_timing)
        actions[time] = action
        if increments is not None:
            increments[time] = delta
        if action:
            utility += float(potentials.outcomes[time]) - p.cost
            p.schedule(cell, time)
            due_time = time + int(potentials.delays[time])
            if due_time < horizon:
                due[due_time].append((cell, int(potentials.outcomes[time]), time))
        regret += max(probability - p.cost, 0.0) - action * (probability - p.cost)
    return {
        "cumulative_utility": utility,
        "cumulative_regret": regret,
        "action_trajectory": actions,
    }, increments


def validate_fast_equivalence(cfg):
    true = population_kernel()
    seed = 1699999
    potentials, alpha, beta, costs = make_tape(seed, true, cfg, horizon=400)
    for rho, gamma in [
        (cfg["survival_exploration"], 0.0),
        (cfg["survival_exploration"], cfg["timing_gain_scale"]),
    ]:
        exact_p = policy(
            seed,
            cfg,
            true,
            alpha,
            beta,
            rho=rho,
            gamma=gamma,
            planning=cfg["planning_window"],
            name="equivalence",
            true_kernel=true,
        )
        fast_p = policy(
            seed,
            cfg,
            true,
            alpha,
            beta,
            rho=rho,
            gamma=gamma,
            planning=cfg["planning_window"],
            name="equivalence",
            true_kernel=true,
        )
        exact, _ = run_policy(exact_p, potentials, (400,), costs=costs)
        fast, _ = run_fast(fast_p, potentials, costs)
        if (
            not np.array_equal(exact["action_trajectory"], fast["action_trajectory"])
            or abs(exact["cumulative_utility"] - fast["cumulative_utility"]) > 1e-12
            or abs(exact["cumulative_regret"] - fast["cumulative_regret"]) > 1e-12
        ):
            raise RuntimeError("fast auxiliary stepper differs from the locked simulator")


def run_reference_states(seed: int, cfg, potentials, costs, alpha, beta, true_kernel):
    floor = policy(
        seed,
        cfg,
        true_kernel,
        alpha,
        beta,
        rho=cfg["survival_exploration"],
        gamma=0.0,
        planning=cfg["planning_window"],
        name="matched_floor",
        true_kernel=true_kernel,
    )
    due = defaultdict(list)
    rows = []
    start = cfg["mechanism"]["state_start"]
    stop = cfg["mechanism"]["state_stop"]
    for time in range(cfg["horizon"]):
        floor.cost = float(costs[time])
        for cell, outcome, origin in due.pop(time, []):
            floor.observe(cell, outcome, origin)
        cell = int(potentials.cells[time])
        floor_action, delta, floor_score = fast_decision(floor, cell, time, record_timing=True)
        full_action = int(
            float(floor_score) + cfg["timing_gain_scale"] * cfg["information_coefficient"] * delta
            > floor.cost
        )
        if start <= time <= stop:
            rows.append(
                {
                    "seed": seed,
                    "time": time,
                    "delta_i": delta,
                    "floor_action": floor_action,
                    "full_action": full_action,
                    "timing_induced_selection": int(full_action == 1 and floor_action == 0),
                    "reverse_switch": int(full_action == 0 and floor_action == 1),
                }
            )
        action = floor_action
        if action:
            floor.schedule(cell, time)
            due_time = time + int(potentials.delays[time])
            if due_time < len(potentials.cells):
                due[due_time].append((cell, int(potentials.outcomes[time]), time))
    frame = pd.DataFrame(rows)
    frame["quintile"] = pd.qcut(
        frame["delta_i"].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )
    if int(frame.reverse_switch.sum()) != 0:
        raise RuntimeError("nonnegative timing term produced a reverse action switch")
    return frame


def select_state_times(states: pd.DataFrame, count: int) -> list[int]:
    selected = []
    for label in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        values = states.loc[states.quintile == label].sort_values(["delta_i", "time"])
        indices = np.linspace(0, len(values) - 1, count + 2, dtype=int)[1:-1]
        selected.extend(values.iloc[indices].time.astype(int).tolist())
    return sorted(selected)


def branch_rollout(base_policy, base_due, potentials, costs, start: int, horizons, gamma: float):
    p = copy.deepcopy(base_policy)
    p.timing_gain_scale = float(gamma)
    p._diagnostic_cache_key = None
    p._diagnostic_cache = None
    due = copy.deepcopy(base_due)
    utility = 0.0
    regret = 0.0
    out = {}
    max_h = max(horizons)
    for offset, time in enumerate(range(start, start + max_h), start=1):
        p.cost = float(costs[time])
        for cell, outcome, origin in due.pop(time, []):
            p.observe(cell, outcome, origin)
        cell = int(potentials.cells[time])
        probability = float(potentials.probabilities[cell])
        action, _, _ = fast_decision(p, cell, time)
        if action:
            utility += float(potentials.outcomes[time]) - p.cost
            p.schedule(cell, time)
            due_time = time + int(potentials.delays[time])
            if due_time < len(potentials.cells):
                due[due_time].append((cell, int(potentials.outcomes[time]), time))
        regret += max(probability - p.cost, 0.0) - action * (probability - p.cost)
        if offset in horizons:
            out[int(offset)] = (utility, regret)
    return out


def continuation_values(seed: int, cfg, states, potentials, costs, alpha, beta, kernel):
    chosen = set(select_state_times(states, cfg["mechanism"]["states_per_quintile"]))
    quintile = dict(zip(states.time.astype(int), states.quintile.astype(str)))
    delta = dict(zip(states.time.astype(int), states.delta_i.astype(float)))
    floor = policy(
        seed,
        cfg,
        kernel,
        alpha,
        beta,
        rho=cfg["survival_exploration"],
        gamma=0.0,
        planning=cfg["planning_window"],
        name="state_reference",
        true_kernel=kernel,
    )
    due = defaultdict(list)
    rows = []
    horizons = tuple(int(x) for x in cfg["mechanism"]["continuation_horizons"])
    for time in range(cfg["horizon"]):
        floor.cost = float(costs[time])
        for cell, outcome, origin in due.pop(time, []):
            floor.observe(cell, outcome, origin)
        if time in chosen:
            full = branch_rollout(
                floor, due, potentials, costs, time, horizons, cfg["timing_gain_scale"]
            )
            matched = branch_rollout(floor, due, potentials, costs, time, horizons, 0.0)
            for horizon in horizons:
                rows.append(
                    {
                        "seed": seed,
                        "state_time": time,
                        "quintile": quintile[time],
                        "delta_i": delta[time],
                        "continuation_horizon": horizon,
                        "utility_gain_total": (full[horizon][0] - matched[horizon][0]),
                        "regret_reduction_total": (matched[horizon][1] - full[horizon][1]),
                    }
                )
        cell = int(potentials.cells[time])
        action, _, _ = fast_decision(floor, cell, time)
        if action:
            floor.schedule(cell, time)
            due_time = time + int(potentials.delays[time])
            if due_time < len(potentials.cells):
                due[due_time].append((cell, int(potentials.outcomes[time]), time))
    return rows


def mechanism_state_worker(seed: int, cfg):
    true = population_kernel()
    potentials, alpha, beta, costs = make_tape(
        seed, true, cfg, horizon=cfg["continuation_tape_horizon"]
    )
    states = run_reference_states(seed, cfg, potentials, costs, alpha, beta, true)
    continuation = continuation_values(seed, cfg, states, potentials, costs, alpha, beta, true)
    return seed, states, continuation


def lambda_worker(seed: int, cfg):
    rows = []
    for lam in cfg["mechanism"]["lambda_grid"]:
        kernel = MeanMatchedMixtureKernel(float(lam), cfg["delay"]["mean"])
        potentials, alpha, beta, costs = make_tape(seed, kernel, cfg, horizon=cfg["horizon"])
        floor_p = policy(
            seed,
            cfg,
            kernel,
            alpha,
            beta,
            rho=cfg["survival_exploration"],
            gamma=0.0,
            planning=cfg["planning_window"],
            name="matched_floor",
            true_kernel=kernel,
        )
        full_p = policy(
            seed,
            cfg,
            kernel,
            alpha,
            beta,
            rho=cfg["survival_exploration"],
            gamma=cfg["timing_gain_scale"],
            planning=cfg["planning_window"],
            name="sf_siv",
            true_kernel=kernel,
        )
        full_r, _ = run_fast(full_p, potentials, costs)
        floor_r, increments = run_fast(floor_p, potentials, costs, record_timing=True)
        fm, mm = metrics(full_r, potentials, costs), metrics(floor_r, potentials, costs)
        fa = np.asarray(full_r["action_trajectory"])
        ma = np.asarray(floor_r["action_trajectory"])
        rows.append(
            {
                "seed": seed,
                "lambda": float(lam),
                "utility_gain_total": fm["utility_total"] - mm["utility_total"],
                "regret_reduction_total": mm["regret_total"] - fm["regret_total"],
                "action_disagreement_rate": float(np.mean(fa != ma)),
                "mean_timing_increment": float(np.mean(increments)),
                "max_timing_increment": float(np.max(increments)),
                "full_utility_total": fm["utility_total"],
                "floor_utility_total": mm["utility_total"],
                "mean_delay_realized": float(np.mean(potentials.delays)),
                "potential_fingerprint": potentials.fingerprint(),
                "full_action_sha256": fm["action_sha256"],
                "floor_action_sha256": mm["action_sha256"],
            }
        )
    return seed, rows


def summarize_group(frame: pd.DataFrame, groups: list[str], values: list[str]):
    rows = []
    for keys, sub in frame.groupby(groups, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["seeds"] = int(sub.seed.nunique()) if "seed" in sub else len(sub)
        for value in values:
            mean, low, high = ci95(sub[value])
            row[value + "_mean"] = mean
            row[value + "_ci_low"] = low
            row[value + "_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def one_sided_p(values) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    sd = float(x.std(ddof=1))
    if sd == 0.0:
        return 0.0 if float(x.mean()) > 0.0 else 1.0
    statistic = float(x.mean() / (sd / np.sqrt(len(x))))
    return float(stats.t.sf(statistic, len(x) - 1))


def holm_adjust(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def attach_group_tests(summary, seed_frame, groups, value, *, family=None):
    result = summary.copy()
    lookup = {}
    for keys, sub in seed_frame.groupby(groups, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        lookup[keys] = one_sided_p(sub[value])
    result["p_one_sided"] = [lookup[tuple(row[g] for g in groups)] for _, row in result.iterrows()]
    if family is None:
        result["p_holm"] = holm_adjust(result.p_one_sided)
    else:
        result["p_holm"] = np.nan
        for _, indices in result.groupby(family, observed=True).groups.items():
            result.loc[indices, "p_holm"] = holm_adjust(
                result.loc[indices, "p_one_sided"].to_numpy()
            )
    return result


def run_mechanism(cfg):
    out = RESULTS / "simulation_mechanism"
    out.mkdir(parents=True, exist_ok=True)
    state_rows, continuation_rows = [], []
    true = population_kernel()
    mcfg = cfg["mechanism"]
    seeds = list(range(mcfg["seed_start"], mcfg["seed_start"] + mcfg["seed_count"]))
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = [pool.submit(mechanism_state_worker, seed, cfg) for seed in seeds]
        for future in as_completed(futures):
            seed, states, continuation = future.result()
            state_rows.append(states)
            continuation_rows.extend(continuation)
            print(f"mechanism states/continuations seed {seed}", flush=True)
    state_frame = pd.concat(state_rows, ignore_index=True)
    state_seed = (
        state_frame.groupby(["seed", "quintile"], observed=True)
        .agg(
            mean_delta_i=("delta_i", "mean"),
            timing_induced_selection_rate=("timing_induced_selection", "mean"),
            reverse_switch_rate=("reverse_switch", "mean"),
        )
        .reset_index()
    )
    continuation = pd.DataFrame(continuation_rows)
    continuation_seed = (
        continuation.groupby(["seed", "quintile", "continuation_horizon"], observed=True)
        .agg(
            utility_gain_total=("utility_gain_total", "mean"),
            regret_reduction_total=("regret_reduction_total", "mean"),
            mean_delta_i=("delta_i", "mean"),
            states=("state_time", "count"),
        )
        .reset_index()
    )
    state_summary = summarize_group(
        state_seed, ["quintile"], ["mean_delta_i", "timing_induced_selection_rate"]
    )
    cont_summary = summarize_group(
        continuation_seed,
        ["quintile", "continuation_horizon"],
        ["utility_gain_total", "regret_reduction_total", "mean_delta_i"],
    )
    state_summary = attach_group_tests(
        state_summary, state_seed, ["quintile"], "timing_induced_selection_rate"
    )
    cont_summary = attach_group_tests(
        cont_summary,
        continuation_seed,
        ["quintile", "continuation_horizon"],
        "utility_gain_total",
        family=["continuation_horizon"],
    )

    lambda_rows = []
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = [pool.submit(lambda_worker, seed, cfg) for seed in seeds]
        for future in as_completed(futures):
            seed, rows = future.result()
            lambda_rows.extend(rows)
            print(f"nonmemorylessness sweep seed {seed}", flush=True)
    lambda_frame = pd.DataFrame(lambda_rows)
    zero = lambda_frame.loc[lambda_frame["lambda"] == 0.0]
    if (
        zero.action_disagreement_rate.abs().max() > 1e-15
        or zero.utility_gain_total.abs().max() > 1e-12
        or zero.regret_reduction_total.abs().max() > 1e-12
        or zero.max_timing_increment.abs().max() > 1e-15
    ):
        raise RuntimeError("memoryless pathwise-null check failed")
    lambda_summary = summarize_group(
        lambda_frame,
        ["lambda"],
        [
            "utility_gain_total",
            "regret_reduction_total",
            "action_disagreement_rate",
            "mean_timing_increment",
            "mean_delay_realized",
        ],
    )
    lambda_summary = attach_group_tests(
        lambda_summary, lambda_frame, ["lambda"], "utility_gain_total"
    )
    for name, frame in {
        "state_seed_results.csv": state_seed,
        "state_summary.csv": state_summary,
        "continuation_state_results.csv": continuation,
        "continuation_seed_results.csv": continuation_seed,
        "continuation_summary.csv": cont_summary,
        "lambda_seed_results.csv": lambda_frame,
        "lambda_summary.csv": lambda_summary,
    }.items():
        frame.to_csv(out / name, index=False)
    state_summary.to_csv(TABLES / "simulation_matched_state_intervention.csv", index=False)
    cont_summary.to_csv(TABLES / "simulation_continuation_value.csv", index=False)
    lambda_summary.to_csv(TABLES / "simulation_nonmemorylessness_sweep.csv", index=False)
    plot_mechanism(state_summary, cont_summary, lambda_summary)
    write_manifest(
        out,
        cfg,
        "mechanism",
        [mcfg["seed_start"], mcfg["seed_start"] + mcfg["seed_count"] - 1],
        {
            "reverse_switches_zero": bool(state_seed.reverse_switch_rate.max() == 0),
            "memoryless_actions_pathwise_equal": True,
            "memoryless_utility_gain_zero": True,
            "memoryless_timing_increment_zero": True,
        },
    )


def plot_error(ax, x, frame, value, *, color="#2166ac", marker="o"):
    mean = frame[value + "_mean"].to_numpy(float)
    low = frame[value + "_ci_low"].to_numpy(float)
    high = frame[value + "_ci_high"].to_numpy(float)
    ax.errorbar(
        x,
        mean,
        yerr=np.vstack([mean - low, high - mean]),
        fmt=marker + "-",
        color=color,
        ecolor=color,
        capsize=3,
        lw=1.6,
        ms=4.5,
    )


def plot_mechanism(state, continuation, sweep):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    s = state.set_index("quintile").loc[labels].reset_index()
    fig, ax = plt.subplots(figsize=(3.1, 2.25))
    plot_error(ax, np.arange(5), s, "timing_induced_selection_rate")
    ax.set_xticks(np.arange(5), labels)
    ax.set_ylabel("Timing-induced selection")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / "simulation_mechanism_a.pdf", bbox_inches="tight")
    plt.close(fig)

    c = continuation.loc[continuation.continuation_horizon == 500].copy()
    c["quintile"] = pd.Categorical(c.quintile, labels, ordered=True)
    c = c.sort_values("quintile")
    fig, ax = plt.subplots(figsize=(3.1, 2.25))
    plot_error(ax, np.arange(5), c, "utility_gain_total", color="#b2182b")
    ax.axhline(0, color="0.35", lw=0.8, ls="--")
    ax.set_xticks(np.arange(5), labels)
    ax.set_ylabel("Continuation utility gain (total)")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / "simulation_mechanism_b.pdf", bbox_inches="tight")
    plt.close(fig)

    sw = sweep.sort_values("lambda")
    fig, ax = plt.subplots(figsize=(3.1, 2.25))
    plot_error(ax, sw["lambda"].to_numpy(float), sw, "utility_gain_total", color="#1b7837")
    ax.axhline(0, color="0.35", lw=0.8, ls="--")
    ax.set_xlabel(r"Non-memorylessness $\lambda$")
    ax.set_ylabel("Paired utility gain (total)")
    ax.set_xticks(sw["lambda"].to_numpy(float))
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / "simulation_mechanism_c.pdf", bbox_inches="tight")
    plt.close(fig)


def run_one_policy(
    seed, cfg, potentials, costs, alpha, beta, kernel, *, rho, gamma, planning, name, true_kernel
):
    p = policy(
        seed,
        cfg,
        kernel,
        alpha,
        beta,
        rho=rho,
        gamma=gamma,
        planning=planning,
        name=name,
        true_kernel=true_kernel,
    )
    result, _ = run_fast(p, potentials, costs)
    return metrics(result, potentials, costs)


def operational_worker(seed: int, cfg):
    true = population_kernel()
    potentials, alpha, beta, costs = make_tape(seed, true, cfg, horizon=cfg["horizon"])
    fitted, calibration = fitted_kernel(seed, cfg)
    components, grid, planning_rows = [], [], []
    cache = {}
    component_defs = [
        ("Exploitation", 0.0, 0.0),
        ("Floor", cfg["survival_exploration"], 0.0),
        ("Timing", 0.0, cfg["timing_gain_scale"]),
        ("Full SF-SIV", cfg["survival_exploration"], cfg["timing_gain_scale"]),
    ]
    for name, rho, gamma in component_defs:
        m = run_one_policy(
            seed,
            cfg,
            potentials,
            costs,
            alpha,
            beta,
            fitted,
            rho=rho,
            gamma=gamma,
            planning=cfg["planning_window"],
            name=name,
            true_kernel=true,
        )
        cache[(rho, gamma, cfg["planning_window"])] = m
        components.append(
            {
                "seed": seed,
                "component": name,
                "rho": rho,
                "gamma": gamma,
                "calibration_sha256": array_hash(calibration),
                "potential_fingerprint": potentials.fingerprint(),
                **m,
            }
        )
    for rho in cfg["robustness"]["rho_grid"]:
        floor_ref = None
        for gamma in [0.0] + list(cfg["robustness"]["gamma_grid"]):
            key = (float(rho), float(gamma), cfg["planning_window"])
            if key not in cache:
                cache[key] = run_one_policy(
                    seed,
                    cfg,
                    potentials,
                    costs,
                    alpha,
                    beta,
                    fitted,
                    rho=rho,
                    gamma=gamma,
                    planning=cfg["planning_window"],
                    name=f"rho{rho}_gamma{gamma}",
                    true_kernel=true,
                )
            if gamma == 0.0:
                floor_ref = cache[key]
            else:
                m = cache[key]
                grid.append(
                    {
                        "seed": seed,
                        "rho": rho,
                        "gamma": gamma,
                        "utility_gain_vs_same_rho_floor": m["utility_total"]
                        - floor_ref["utility_total"],
                        "regret_reduction_vs_same_rho_floor": floor_ref["regret_total"]
                        - m["regret_total"],
                        **m,
                    }
                )
    for planning in cfg["robustness"]["planning_grid"]:
        full = run_one_policy(
            seed,
            cfg,
            potentials,
            costs,
            alpha,
            beta,
            fitted,
            rho=cfg["survival_exploration"],
            gamma=cfg["timing_gain_scale"],
            planning=planning,
            name=f"full_s{planning}",
            true_kernel=true,
        )
        floor = run_one_policy(
            seed,
            cfg,
            potentials,
            costs,
            alpha,
            beta,
            fitted,
            rho=cfg["survival_exploration"],
            gamma=0.0,
            planning=planning,
            name=f"floor_s{planning}",
            true_kernel=true,
        )
        planning_rows.append(
            {
                "seed": seed,
                "planning_window": planning,
                "utility_gain_vs_matched_floor": full["utility_total"] - floor["utility_total"],
                "regret_reduction_vs_matched_floor": floor["regret_total"] - full["regret_total"],
                **full,
            }
        )
    return seed, components, grid, planning_rows


def planning_worker(seed: int, cfg):
    """Run only the horizon-matched planning sensitivity for one environment."""
    true = population_kernel()
    potentials, alpha, beta, costs = make_tape(seed, true, cfg, horizon=cfg["horizon"])
    fitted, _ = fitted_kernel(seed, cfg)
    rows = []
    for planning in cfg["robustness"]["planning_grid"]:
        full = run_one_policy(
            seed,
            cfg,
            potentials,
            costs,
            alpha,
            beta,
            fitted,
            rho=cfg["survival_exploration"],
            gamma=cfg["timing_gain_scale"],
            planning=planning,
            name=f"full_s{planning}",
            true_kernel=true,
        )
        floor = run_one_policy(
            seed,
            cfg,
            potentials,
            costs,
            alpha,
            beta,
            fitted,
            rho=cfg["survival_exploration"],
            gamma=0.0,
            planning=planning,
            name=f"floor_s{planning}",
            true_kernel=true,
        )
        rows.append(
            {
                "seed": seed,
                "planning_window": planning,
                "utility_gain_vs_matched_floor": full["utility_total"] - floor["utility_total"],
                "regret_reduction_vs_matched_floor": floor["regret_total"] - full["regret_total"],
                **full,
            }
        )
    return seed, rows


def kernel_worker(seed: int, cfg):
    true = population_kernel()
    potentials, alpha, beta, costs = make_tape(seed, true, cfg, horizon=cfg["horizon"])
    calibration = true.sample(
        np.random.default_rng(derived_seed(seed, "auxiliary_calibration_delays")),
        cfg["calibration_size"],
    )
    oracle_p = policy(
        seed,
        cfg,
        true,
        alpha,
        beta,
        rho=cfg["survival_exploration"],
        gamma=cfg["timing_gain_scale"],
        planning=cfg["planning_window"],
        name="oracle_kernel",
        true_kernel=true,
    )
    oracle_r, _ = run_fast(oracle_p, potentials, costs)
    oracle_m = metrics(oracle_r, potentials, costs)
    oracle_actions = np.asarray(oracle_r["action_trajectory"])
    rows = []
    for size in list(cfg["robustness"]["calibration_grid"]) + ["oracle"]:
        if size == "oracle":
            m, actions, error = oracle_m, oracle_actions, 0.0
        else:
            kernel = cached_empirical(calibration[: int(size)])
            p = policy(
                seed,
                cfg,
                kernel,
                alpha,
                beta,
                rho=cfg["survival_exploration"],
                gamma=cfg["timing_gain_scale"],
                planning=cfg["planning_window"],
                name=f"calibration_{size}",
                true_kernel=true,
            )
            r, _ = run_fast(p, potentials, costs)
            m = metrics(r, potentials, costs)
            actions = np.asarray(r["action_trajectory"])
            points = np.unique(calibration[: int(size)])
            error = max(
                abs(kernel.survival(int(t)) - true.survival(int(t)))
                for t in np.unique(np.concatenate([points, np.maximum(points - 1, 0)]))
            )
        rows.append(
            {
                "seed": seed,
                "calibration_size": str(size),
                "kernel_cdf_error": error,
                "action_disagreement_rate": float(np.mean(actions != oracle_actions)),
                **m,
                "oracle_action_sha256": oracle_m["action_sha256"],
            }
        )
    return seed, rows


def run_robustness(cfg):
    out = RESULTS / "simulation_robustness"
    out.mkdir(parents=True, exist_ok=True)
    rcfg = cfg["robustness"]
    components, grid, planning_rows = [], [], []
    seeds = list(range(rcfg["seed_start"], rcfg["seed_start"] + rcfg["seed_count"]))
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = [pool.submit(operational_worker, seed, cfg) for seed in seeds]
        for future in as_completed(futures):
            seed, crows, grows, prows = future.result()
            components.extend(crows)
            grid.extend(grows)
            planning_rows.extend(prows)
            print(f"operational robustness seed {seed}", flush=True)
    comp = pd.DataFrame(components)
    grid_frame = pd.DataFrame(grid)
    planning_frame = pd.DataFrame(planning_rows)
    comp_summary = summarize_group(
        comp,
        ["component"],
        ["utility_total", "regret_total", "selection_rate", "unprofitable_selection_rate"],
    )
    wide_u = comp.pivot(index="seed", columns="component", values="utility_total")
    wide_r = comp.pivot(index="seed", columns="component", values="regret_total")
    contrast_rows = []
    for comparator in ["Exploitation", "Floor", "Timing"]:
        du = wide_u["Full SF-SIV"] - wide_u[comparator]
        dr = wide_r[comparator] - wide_r["Full SF-SIV"]
        um, ul, uh = ci95(du)
        rm, rl, rh = ci95(dr)
        contrast_rows.append(
            {
                "comparator": comparator,
                "seeds": len(du),
                "utility_gain_mean": um,
                "utility_gain_ci_low": ul,
                "utility_gain_ci_high": uh,
                "regret_reduction_mean": rm,
                "regret_reduction_ci_low": rl,
                "regret_reduction_ci_high": rh,
                "utility_p_one_sided": one_sided_p(du),
                "regret_p_one_sided": one_sided_p(dr),
            }
        )
    comp_contrasts = pd.DataFrame(contrast_rows)
    comp_contrasts["utility_p_holm"] = holm_adjust(comp_contrasts.utility_p_one_sided)
    comp_contrasts["regret_p_holm"] = holm_adjust(comp_contrasts.regret_p_one_sided)
    grid_summary = summarize_group(
        grid_frame,
        ["rho", "gamma"],
        ["utility_gain_vs_same_rho_floor", "regret_reduction_vs_same_rho_floor", "utility_total"],
    )
    planning_summary = summarize_group(
        planning_frame,
        ["planning_window"],
        ["utility_gain_vs_matched_floor", "regret_reduction_vs_matched_floor", "utility_total"],
    )
    for name, frame in {
        "component_seed_results.csv": comp,
        "component_summary.csv": comp_summary,
        "component_paired_contrasts.csv": comp_contrasts,
        "rho_gamma_seed_results.csv": grid_frame,
        "rho_gamma_summary.csv": grid_summary,
        "planning_seed_results.csv": planning_frame,
        "planning_summary.csv": planning_summary,
    }.items():
        frame.to_csv(out / name, index=False)
    comp_summary.to_csv(TABLES / "simulation_component_ablation.csv", index=False)
    comp_contrasts.to_csv(TABLES / "simulation_component_contrasts.csv", index=False)
    grid_summary.to_csv(TABLES / "simulation_rho_gamma_sensitivity.csv", index=False)
    planning_summary.to_csv(TABLES / "simulation_planning_horizon_sensitivity.csv", index=False)
    plot_robustness(comp_summary, grid_summary, planning_summary)
    run_kernel_calibration(cfg)
    write_manifest(
        out,
        cfg,
        "robustness",
        [rcfg["seed_start"], rcfg["seed_start"] + rcfg["seed_count"] - 1],
        {
            "matched_component_code_path": True,
            "fixed_coefficients_no_retuning": True,
            "seed_disjointness": True,
        },
    )


def run_planning_sensitivity(cfg):
    out = RESULTS / "simulation_robustness"
    out.mkdir(parents=True, exist_ok=True)
    rcfg = cfg["robustness"]
    rows = []
    seeds = list(range(rcfg["seed_start"], rcfg["seed_start"] + rcfg["seed_count"]))
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = [pool.submit(planning_worker, seed, cfg) for seed in seeds]
        for future in as_completed(futures):
            seed, seed_rows = future.result()
            rows.extend(seed_rows)
            print(f"planning sensitivity seed {seed}", flush=True)
    frame = pd.DataFrame(rows)
    summary = summarize_group(
        frame,
        ["planning_window"],
        ["utility_gain_vs_matched_floor", "regret_reduction_vs_matched_floor", "utility_total"],
    )
    frame.to_csv(out / "planning_seed_results.csv", index=False)
    summary.to_csv(out / "planning_summary.csv", index=False)
    summary.to_csv(TABLES / "simulation_planning_horizon_sensitivity.csv", index=False)
    plot_planning_sensitivity(summary)
    write_manifest(
        out,
        cfg,
        "planning_sensitivity",
        [rcfg["seed_start"], rcfg["seed_start"] + rcfg["seed_count"] - 1],
        {
            "matched_floor_within_seed_and_horizon": True,
            "fixed_coefficients_no_retuning": True,
            "seed_disjointness": True,
        },
        filenames=["planning_seed_results.csv", "planning_summary.csv"],
        manifest_name="planning_manifest.json",
    )


def run_kernel_calibration(cfg):
    rcfg = cfg["robustness"]
    out = RESULTS / "kernel_calibration"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    seeds = list(
        range(rcfg["kernel_seed_start"], rcfg["kernel_seed_start"] + rcfg["kernel_seed_count"])
    )
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = [pool.submit(kernel_worker, seed, cfg) for seed in seeds]
        for future in as_completed(futures):
            seed, seed_rows = future.result()
            rows.extend(seed_rows)
            print(f"kernel calibration seed {seed}", flush=True)
    frame = pd.DataFrame(rows)
    order = [str(x) for x in rcfg["calibration_grid"]] + ["oracle"]
    frame["calibration_size"] = pd.Categorical(frame.calibration_size, order, ordered=True)
    summary = summarize_group(
        frame,
        ["calibration_size"],
        ["kernel_cdf_error", "action_disagreement_rate", "utility_total", "regret_total"],
    )
    frame.to_csv(out / "seed_results.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    summary.to_csv(TABLES / "kernel_calibration_sensitivity.csv", index=False)
    plot_kernel(summary)
    oracle_rows = frame.loc[frame.calibration_size.astype(str) == "oracle"]
    write_manifest(
        out,
        cfg,
        "kernel_calibration",
        [rcfg["kernel_seed_start"], rcfg["kernel_seed_start"] + rcfg["kernel_seed_count"] - 1],
        {
            "oracle_kernel_error_zero": bool(oracle_rows.kernel_cdf_error.max() == 0),
            "oracle_action_disagreement_zero": bool(
                oracle_rows.action_disagreement_rate.max() == 0
            ),
        },
    )


def plot_robustness(comp, grid, planning):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
        }
    )
    order = ["Exploitation", "Floor", "Timing", "Full SF-SIV"]
    c = comp.set_index("component").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    values = c.utility_total_mean.to_numpy(float)
    low = c.utility_total_ci_low.to_numpy(float)
    high = c.utility_total_ci_high.to_numpy(float)
    ax.bar(np.arange(4), values, color=["#bdbdbd", "#80cdc1", "#dfc27d", "#2166ac"])
    ax.errorbar(
        np.arange(4),
        values,
        yerr=np.vstack([values - low, high - values]),
        fmt="none",
        color="0.2",
        capsize=3,
    )
    ax.set_xticks(np.arange(4), ["Exploit", "Floor", "Timing", "Full"])
    ax.set_ylabel("Utility (total)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / "simulation_component_ablation.pdf", bbox_inches="tight")
    plt.close(fig)

    pivot = grid.pivot(
        index="rho", columns="gamma", values="utility_gain_vs_same_rho_floor_mean"
    ).sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    im = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-np.max(np.abs(pivot.to_numpy())),
        vmax=np.max(np.abs(pivot.to_numpy())),
    )
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), [f"{x:g}" for x in pivot.index])
    ax.set_xlabel(r"Timing scale $\gamma$")
    ax.set_ylabel(r"Floor $\rho$")
    fig.colorbar(im, ax=ax, label="Utility gain (total)", fraction=0.05, pad=0.03)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / "simulation_rho_gamma.pdf", bbox_inches="tight")
    plt.close(fig)

    plot_planning_sensitivity(planning)


def plot_planning_sensitivity(planning):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
        }
    )
    p = planning.sort_values("planning_window")
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    x = np.arange(len(p))
    plot_error(ax, x, p, "utility_gain_vs_matched_floor", color="#762a83")
    ax.axhline(0, color="0.35", lw=0.8, ls="--")

    ax.set_xticks(x, [f"{value:g}" for value in p.planning_window])
    ax.set_xlabel("Planning horizon")
    ax.set_ylabel("Timing gain (total)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / "simulation_planning_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_kernel(summary):
    order = ["25", "50", "100", "250", "500", "1000", "oracle"]
    s = summary.copy()
    s["calibration_size"] = pd.Categorical(s.calibration_size.astype(str), order, ordered=True)
    s = s.sort_values("calibration_size")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
    x = np.arange(len(s))
    plot_error(axes[0], x, s, "kernel_cdf_error", color="#b2182b")
    plot_error(axes[1], x, s, "action_disagreement_rate", color="#2166ac")
    for ax in axes:
        ax.set_xticks(x, ["25", "50", "100", "250", "500", "1k", "oracle"], rotation=30)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Kernel CDF error")
    axes[1].set_ylabel("Action disagreement")
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURES / "simulation_kernel_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def write_manifest(
    out: Path, cfg, suite: str, seed_range, checks, *, filenames=None, manifest_name="manifest.json"
):
    files = (
        sorted(p for p in out.glob("*.csv"))
        if filenames is None
        else [out / name for name in filenames]
    )
    manifest = {
        "suite": suite,
        "role": cfg["role"],
        "confirmatory": False,
        "tuning": cfg["tuning"],
        "independent_unit": "environment_seed",
        "seed_range": seed_range,
        "config_sha256": sha256(CONFIG),
        "effective_config": cfg,
        "script_sha256": sha256(Path(__file__)),
        "outputs": {p.name: sha256(p) for p in files},
        "self_checks": checks,
    }
    (out / manifest_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main():
    global CONFIG, RESULTS, TABLES, FIGURES
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--suite", choices=["mechanism", "planning", "robustness", "all"], default="all"
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--quick", action="store_true", help="Two seeds per suite for a smoke test")
    args = parser.parse_args()
    RESULTS = args.output / "diagnostics"
    TABLES = args.output / "tables"
    FIGURES = args.output / "figures"
    CONFIG = args.config.resolve()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.workers is not None:
        cfg["workers"] = args.workers
    if cfg["workers"] < 1:
        raise ValueError("workers must be positive")
    if args.quick:
        cfg["mechanism"]["seed_count"] = 2
        cfg["robustness"]["seed_count"] = 2
        cfg["robustness"]["kernel_seed_count"] = 2
    validate_fast_equivalence(cfg)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if args.suite in ("mechanism", "all"):
        run_mechanism(cfg)
    if args.suite == "planning":
        run_planning_sensitivity(cfg)
    if args.suite in ("robustness", "all"):
        run_robustness(cfg)


if __name__ == "__main__":
    main()

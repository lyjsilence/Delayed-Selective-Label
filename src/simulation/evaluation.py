"""Evaluate all methods on one paired environment seed."""

import copy
import numpy as np
import pandas as pd
from .environment import prepare
from .kernels import CachedDelay
from .policies import make_policy
from .statistics import run_totals
from .compiled import evaluate


def worker(job):
    stage, seed, kernel, cfg = job
    tape, a, b, costs, fitted, fingerprint = prepare(seed, kernel, cfg)
    fitted, true = CachedDelay(fitted), CachedDelay(kernel)
    cache = {}
    rows = []
    for name in list(cfg["methods"]) + ["matched_no_increment"]:
        runs = []
        for repeat in range(cfg["methods"].get(name, {}).get("policy_repeats", 1)):
            if name not in ["bayes_ucb", "tempered_ts", "online_bootstrap"]:
                key = "survival_floored_siv" if name == "matched_no_increment" else name
                params = copy.deepcopy(cfg["methods"][key])
                if name == "matched_no_increment":
                    params["timing_gain_scale"] = 0.0
                metrics, actions = evaluate(
                    tape, a, b, costs, fitted, dict(method=key, parameters=params), cache
                )
            else:
                p = make_policy(name, repeat, seed, a, b, fitted, true, cfg["methods"])
                actions = []
                original = p.decide

                def decide(cell, time):
                    action = original(cell, time)
                    actions.append(action)
                    return action

                p.decide = decide
                metrics = run_totals(p, tape, costs)
            rewards = np.asarray(actions) * (tape.outcomes - costs)
            for block in range(8):
                metrics[f"utility_block_{block+1}"] = float(
                    rewards[block * 500 : (block + 1) * 500].sum()
                )
            np.testing.assert_allclose(
                sum(metrics[f"utility_block_{j+1}"] for j in range(8)),
                metrics["total_utility"],
                atol=1e-9,
            )
            runs.append(metrics)
        rows.append(
            dict(
                stage=stage,
                seed=seed,
                method=name,
                **{m: float(np.mean([r[m] for r in runs])) for m in runs[0]},
                realized_delay_mean=float(tape.delays.mean()),
                realized_delay_sd=float(tape.delays.std()),
                realized_unique_delays=len(np.unique(tape.delays)),
                potential_fingerprint=tape.fingerprint(),
                calibration_fingerprint=fingerprint,
            )
        )
    return pd.DataFrame(rows)

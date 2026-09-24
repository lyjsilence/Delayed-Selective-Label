"""Reproduce the SF-SIV row of Table 1."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import numpy as np
import pandas as pd
from environment import prepare
from policy import actions_for

ROOT = Path(__file__).resolve().parent
METRICS = ["utility", "regret", "oracle_pct", "selected_pct", "unprofitable_pct"]


def run_seed(job):
    seed, cfg = job
    cells, outcomes, delays, costs, a, b, maturation, probabilities = prepare(seed, cfg)
    p = cfg["policy"]
    actions = actions_for(
        cells,
        outcomes,
        delays,
        costs,
        a,
        b,
        maturation,
        p["survival_exploration"],
        p["timing_gain_scale"],
        p["information_coefficient"],
    )
    margin = probabilities[cells] - costs
    return dict(
        seed=seed,
        utility=float(np.sum(actions * (outcomes - costs))),
        regret=float(np.sum(np.maximum(margin, 0) - actions * margin)),
        oracle_pct=float(100 * np.mean(actions == (margin > 0))),
        selected_pct=float(100 * np.mean(actions)),
        unprofitable_pct=float(
            100 * np.sum(actions * (margin <= 0)) / max(1, np.sum(actions))
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--quick", action="store_true", help="Two seeds for an execution check"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    cfg = json.loads((ROOT / "config.json").read_text())
    count = 2 if args.quick else cfg["seed_count"]
    jobs = [(seed, cfg) for seed in range(cfg["seed_start"], cfg["seed_start"] + count)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(run_seed, jobs), 1):
            rows.append(row)
            if i % 100 == 0 or i == count:
                print(f"Completed {i}/{count} seeds", flush=True)
    frame = pd.DataFrame(rows)
    summary = dict(method="SF-SIV", seeds=count)
    for field in METRICS:
        summary[field + "_mean"] = float(frame[field].mean())
        summary[field + "_sd"] = float(frame[field].std(ddof=1))
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "seed_results.csv", index=False)
    pd.DataFrame([summary]).to_csv(args.output / "table1.csv", index=False)
    row = (
        "SF-SIV & "
        + " & ".join(
            f'${summary[f+"_mean"]:.2f}\\pm{summary[f+"_sd"]:.2f}$' for f in METRICS
        )
        + r"\\"
    )
    (args.output / "table1_row.tex").write_text(row + "\n")
    print(row)


if __name__ == "__main__":
    main()

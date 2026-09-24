"""Run paired simulations with resumable, protocol-checked checkpoints."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import sys
import time

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation.evaluation import worker
from simulation.kernels import LogNormalDelay, Mixture
from simulation.statistics import atomic_csv, atomic_json, paired_contrasts, sha256, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--partition", choices=["all", "development", "validation", "test"], default="all"
    )
    parser.add_argument("--quick", action="store_true", help="Two seeds in each selected partition")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    cfg = yaml.safe_load((ROOT / "configs/main.yaml").read_text())
    kd = json.loads((ROOT / "configs/delay.json").read_text())
    law = Mixture(LogNormalDelay(**kd["fast"]), LogNormalDelay(**kd["slow"]), kd["probability"])
    out = args.output / "main"
    out.mkdir(parents=True, exist_ok=True)
    parts = {k: dict(v) for k, v in cfg["partitions"].items()}
    if args.quick:
        for v in parts.values():
            v["count"] = 2
    protocol = dict(
        configuration=cfg,
        delay=kd,
        partitions=parts,
        quick=args.quick,
        sources={str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src").rglob("*.py"))},
    )
    protocol = json.loads(json.dumps(protocol))
    lock = out / "protocol.json"
    if lock.exists() and json.loads(lock.read_text()) != protocol:
        raise RuntimeError(
            "Output directory belongs to another protocol; choose a new --output directory."
        )
    atomic_json(lock, protocol)
    selected = parts if args.partition == "all" else {args.partition: parts[args.partition]}
    jobs = []
    for stage, p in selected.items():
        folder = out / stage
        folder.mkdir(exist_ok=True)
        jobs.extend(
            (stage, seed, law, cfg)
            for seed in range(p["start"], p["start"] + p["count"])
            if not (folder / f"seed_{seed}.csv").exists()
        )
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            stage, seed, _, _ = futures[future]
            atomic_csv(out / stage / f"seed_{seed}.csv", future.result())
            if i % 20 == 0 or i == len(jobs):
                print(f"Completed {i}/{len(jobs)} environments", flush=True)
    frames = [pd.read_csv(p) for p in sorted(out.glob("*/seed_*.csv"))]
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["stage", "seed", "method"]).any():
        raise RuntimeError("Duplicate environment/method rows")
    summaries, contrasts = [], []
    for stage, part in data.groupby("stage"):
        summaries.append(summarize(part).assign(stage=stage))
        contrasts.append(
            paired_contrasts(cfg, part[part.method != "matched_no_increment"]).assign(stage=stage)
        )
    atomic_csv(out / "summary.csv", pd.concat(summaries, ignore_index=True))
    atomic_csv(out / "primary_contrasts.csv", pd.concat(contrasts, ignore_index=True))
    atomic_json(
        out / "completed.json",
        dict(
            rows=len(data),
            elapsed_seconds=time.perf_counter() - start,
            protocol_sha256=sha256(lock),
        ),
    )


if __name__ == "__main__":
    main()

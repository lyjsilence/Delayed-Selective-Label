"""Generate total-metric main figures and primary paired inference."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation import statistics as common

OUT = ROOT / "results/main"
FIG = ROOT / "results/figures"
NAMES = {
    "survival_floored_siv": "SF-SIV",
    "tuned_survival": "Survival",
    "bayes_ucb": "Bayes-UCB",
    "delayed_ucb": "Delayed-UCB",
    "raw_fitted_siv": "Raw SIV",
    "one_step_bayes": "One-step Bayes",
    "count_only": "Count-only",
    "tempered_ts": "TS",
    "online_bootstrap": "Bootstrap",
    "matured_greedy": "Greedy",
}


def main():
    cfg = yaml.safe_load((ROOT / "configs/main.yaml").read_text())
    data = pd.concat([pd.read_csv(p) for p in OUT.glob("*/seed_*.csv")], ignore_index=True)
    primary = data[data.method != "matched_no_increment"]
    rows = []
    for stage, part in primary.groupby("stage"):
        rows.append(common.paired_contrasts(cfg, part).assign(stage=stage))
    contrasts = pd.concat(rows, ignore_index=True)
    common.atomic_csv(OUT / "primary_contrasts.csv", contrasts)
    plt.rcParams.update(
        {"font.size": 11, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    )
    FIG.mkdir(parents=True, exist_ok=True)
    c = contrasts[contrasts.stage == "test"].sort_values("total_utility_gain_mean")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    y = np.arange(len(c))
    for field, offset, color, label in [
        ("total_utility_gain", -0.13, "#2166ac", "Utility"),
        ("total_regret_reduction", 0.13, "#d95f02", "Regret"),
    ]:
        v = c[field + "_mean"]
        err = np.vstack([v - c[field + "_ci_low"], c[field + "_ci_high"] - v])
        ax.errorbar(v, y + offset, xerr=err, fmt="o", markersize=3, color=color, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([NAMES[n] for n in c.baseline])
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("Paired total gain")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "main_a.pdf", bbox_inches="tight")
    plt.close(fig)
    summary = pd.read_csv(OUT / "summary.csv")
    s = summary[(summary.stage == "test") & (summary.method != "matched_no_increment")]
    fig, ax = plt.subplots(figsize=(3.7, 3.5))
    scatter = ax.scatter(
        s.total_pseudo_regret_mean,
        s.total_utility_mean,
        c=s.selection_rate,
        s=30 + 300 * s.unprofitable_selection_rate,
        cmap="viridis",
    )
    for _, r in s.iterrows():
        if r.method in [
            "survival_floored_siv",
            "bayes_ucb",
            "tuned_survival",
            "tempered_ts",
            "online_bootstrap",
            "matured_greedy",
        ]:
            ax.annotate(
                NAMES[r.method],
                (r.total_pseudo_regret_mean, r.total_utility_mean),
                xytext={
                    "survival_floored_siv": (8, 10),
                    "bayes_ucb": (28, 0),
                    "tuned_survival": (28, -17),
                    "tempered_ts": (8, -10),
                }.get(r.method, (4, 5)),
                textcoords="offset points",
                fontsize=8,
                arrowprops=(
                    dict(arrowstyle="-", linewidth=0.4)
                    if r.method in ["survival_floored_siv", "bayes_ucb", "tuned_survival"]
                    else None
                ),
            )
    ax.set_xlabel("Total pseudo-regret")
    ax.set_ylabel("Total utility")
    fig.tight_layout()
    fig.savefig(FIG / "main_b.pdf", bbox_inches="tight")
    plt.close(fig)
    test = primary[primary.stage == "test"]
    blocks = []
    for i in range(1, 9):
        pivot = test.pivot(index="seed", columns="method", values=f"utility_block_{i}")
        du = pivot.survival_floored_siv - pivot.tuned_survival
        m, l, h = common.ci95(du)
        blocks.append(dict(block=i, mean=m, low=l, high=h))
    b = pd.DataFrame(blocks)
    common.atomic_csv(OUT / "blocks.csv", b)
    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    ax.errorbar(
        b.block,
        b["mean"],
        yerr=np.vstack([b["mean"] - b.low, b.high - b["mean"]]),
        fmt="o-",
        capsize=3,
    )
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xticks(b.block)
    ax.set_xlabel("500-decision block")
    ax.set_ylabel("Total gain vs survival")
    fig.tight_layout()
    fig.savefig(FIG / "main_c.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3))
    for method in ["survival_floored_siv", "tuned_survival", "one_step_bayes", "count_only"]:
        values = (
            test[test.method == method][[f"utility_block_{j}" for j in range(1, 9)]]
            .to_numpy()
            .cumsum(axis=1)
        )
        means = values.mean(axis=0)
        errors = np.array([common.ci95(values[:, j]) for j in range(8)])
        ax.plot(np.arange(1, 9) * 500, means, label=NAMES[method])
        ax.fill_between(np.arange(1, 9) * 500, errors[:, 1], errors[:, 2], alpha=0.12)
    ax.set_xlabel("Decisions")
    ax.set_ylabel("Cumulative utility")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "cumulative.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3))
    stages = [x for x in ["development", "validation", "test"] if x in contrasts.stage.unique()]
    p = contrasts[contrasts.baseline == "tuned_survival"].set_index("stage").loc[stages]
    ax.errorbar(
        np.arange(len(stages)),
        p.total_utility_gain_mean,
        yerr=np.vstack(
            [
                p.total_utility_gain_mean - p.total_utility_gain_ci_low,
                p.total_utility_gain_ci_high - p.total_utility_gain_mean,
            ]
        ),
        fmt="o",
        capsize=4,
    )
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels(stages)
    ax.set_ylabel("Total gain vs survival")
    fig.tight_layout()
    fig.savefig(FIG / "stages.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Main artifacts generated with total metrics and 18 primary contrasts per stage.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    OUT = args.input / "main"
    FIG = args.input / "figures"
    main()

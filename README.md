# Delayed Selective Label

Simulation experiments for sequential decisions with delayed, selectively observed labels. The repository implements SF-SIV and nine comparison policies, together with mechanism, ablation, and sensitivity experiments.

All data are generated locally. No external datasets, lending records, or empirical-study code are included.

## Installation

Python 3.12 is the tested interpreter. The experiments run on CPU; a GPU is not needed.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Commands below are run from the repository root. Scripts locate their configuration files relative to the repository, rather than the current working directory.

## Quick check

```bash
python -m unittest discover -s tests -v
python scripts/run_main.py --quick --workers 2 --output results/quick
python scripts/run_auxiliary.py --quick --workers 2 --output results/quick
python scripts/plot_results.py --input results/quick
```

The quick runs use two seeds per partition or auxiliary suite. They check execution and are not estimates of the paper's reported performance. Numba compiles the deterministic evaluator on first use, so the first run takes longer.

## Reproduce the experiments

```bash
python scripts/run_main.py --workers 8
python scripts/run_auxiliary.py --workers 8
python scripts/plot_results.py
```

The main run evaluates all ten methods and a matched zero-timing-increment policy on 1,140 paired environments: 60 and 80 replication seeds, plus 1,000 test seeds. The auxiliary runner evaluates 40 mechanism seeds, 50 ablation/sensitivity seeds, and 50 independent calibration seeds.

`--workers` controls CPU concurrency. The main run saves each completed seed and resumes from those checkpoints. It refuses to reuse an output directory when the configuration or source inventory changes. Use a separate `--output` directory for modified experiments. Auxiliary suites overwrite their own outputs; do not run two auxiliary jobs into the same directory.

Individual suites can be run separately:

```bash
python scripts/run_main.py --partition test --workers 8
python scripts/run_auxiliary.py --suite mechanism --workers 8
python scripts/run_auxiliary.py --suite robustness --workers 8
```

The robustness suite includes component ablations, coefficient and planning-window sensitivity, and delay-kernel calibration. `--suite planning` runs only the planning-window study.

## Data-generating process

There are 4,000 decisions and 20 profiles. Profile arrivals are IID uniform; the decision cost is constant, `c[t] = 0.7`, for every arrival. Conditional on profile `j`, the potential label is Bernoulli with probability `p[j]`.

Each prior is `Beta(4 m[j], 4 (1-m[j]))`. Six low-success probabilities are `[0.48, 0.52, 0.56, 0.60, 0.62, 0.64]`; six high-success probabilities are `[0.82, 0.84, 0.86, 0.88, 0.90, 0.92]`. These profiles have correctly centered priors. The remaining eight profiles have equally spaced prior means in `[0.668, 0.682]` and true probabilities in `[0.793, 0.807]`. Profile indices are permuted by seed.

For every arrival, independently sample a fast/slow component and a standard normal variable:

```text
P(component = fast) = 0.25
P(component = slow) = 0.75
mu_fast = 2
mu_slow = 5.6
sigma = 0.75
D = ceil(exp(mu_component + sigma * Normal(0, 1)))
```

The rounded component means are approximately 10.289 and 358.757 decision steps; the mixture mean is 271.640. Delays have unbounded support and are independent of profiles and labels. A selected label is delivered after `D` steps. Labels due after the simulation horizon remain pending. Unselected labels are never disclosed to the policy.

The main policies estimate the delay distribution using 1,000 independent calibration delays. The population law is used by the evaluator and the explicitly identified oracle-kernel auxiliary comparator. The memoryless diagnostic compares the mixture with a geometric delay law of the same mean.

## Policies

The SF-SIV score is

```text
posterior_mean + rho * B + gamma * kappa * max(sqrt(I_sch) - sqrt(I_cnt), 0)
```

Here `B` is the survival-based uncertainty floor, and `I_sch` and `I_cnt` are the scheduled and count-reference information values. The action is one when the score exceeds the current cost. The reported coefficients are `rho=0.245`, `gamma=300`, `kappa=2`, and planning window `s=50`.

Comparators are Bayes-UCB, delayed UCB, survival-based scoring, count-only information, raw scheduled information, one-step Bayes information, tempered Thompson sampling, online bootstrap, and greedy exploitation. These are the specific Beta–Bernoulli implementations documented in the source, rather than claims to reproduce every variant of those algorithm families. Their fixed settings are in `configs/main.yaml`.

Policy settings were chosen during exploratory development and frozen before the reported reruns. The delay configuration was selected after exploratory comparisons on the same main seeds. The reported intervals quantify Monte Carlo uncertainty for this configuration and do not adjust for selection across delay configurations. This repository reproduces that fixed experiment; it does not implement the earlier search or establish globally optimal hyperparameters. No method is retuned on the reported test seeds.

## Metrics and inference

- **Cumulative utility:** `sum A[t] * (Y[t] - c[t])`.
- **Pseudo-regret:** `sum (max(p[profile[t]] - c[t], 0) - A[t] * (p[profile[t]] - c[t]))`.
- **Oracle agreement:** fraction of actions agreeing with `p[profile[t]] > c[t]`.
- **Unprofitable selections:** fraction of selected instances whose success probability does not exceed cost.

Methods share potential outcomes, arrivals, costs, delays, and calibration data within each environment seed. Randomized policies average five policy repeats before seed-level inference. Ordinary paired 95% intervals and one-sided Bonferroni lower bounds are reported separately. The primary family comprises nine utility and nine regret contrasts, with familywise level 0.05. Auxiliary comparisons use their documented Holm families or explicitly descriptive intervals.

The partitions named `development` and `validation` are fixed-policy replications in this release; neither selects parameters.

## Verification

The tests compare complete action trajectories from the compiled evaluator and the reference policies, verify constant costs, and check delay-law identities. All checks generate their inputs at runtime; no stored experiment results are needed.

## Layout

```text
configs/             Fixed main, delay, and auxiliary configurations
src/siv_core/        Information values, policy state, and reference simulator
src/siv_design/      Potential outcomes, arrivals, and priors
src/siv_experiments/ SF-SIV, comparison policies, and calibration
src/simulation/     Experiment assembly, delay mixture, fast evaluation, statistics
scripts/            Main/auxiliary runners, plots, and result checks
tests/              Delay-law and action-trajectory checks
```

Generated checkpoints, tables, and figures are written to `results/`, which is excluded from version control. Named random-stream identifiers are intentionally stable so that refactoring does not change the simulated data.

## Scope

The results concern this synthetic population, prior shift, and delay model. They do not imply uniform superiority over all data-generating processes. The private-data empirical study is outside this repository.

## Submission archive

```bash
python scripts/package_submission.py
```

This writes `dist/delayed-selective-label-simulation.zip`, containing source, configurations, tests, and this README. Experimental results, figures, local paths, credentials, and Git history are excluded.

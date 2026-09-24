# Delayed Selective Label

Minimal reproduction of the **SF-SIV row in Table 1**. This repository contains only the proposed method, its synthetic environment, and the table-generation entry point.

## Run

Tested with Python 3.12. CPU only; no GPU or external data is required.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --workers 8
```

For a two-seed execution check:

```bash
python run.py --quick --workers 2 --output results/quick
```

The full command runs 1,000 seeds (5700000–5700999), with 4,000 sequential decisions per seed. Outputs are generated locally in `results/`:

- `table1.csv`: means and sample standard deviations across seeds.
- `table1_row.tex`: the SF-SIV table row, ordered as utility, pseudo-regret, oracle agreement (%), selection rate (%), and unprofitable selections (%).
- `seed_results.csv`: the five metrics for each seed.

The quick run does not reproduce the paper's aggregate row. Numba compiles the decision loop on its first invocation. `--output` selects another output directory; a completed run overwrites files with the same names.

## Experiment

All parameters are in `config.json`. Profiles arrive IID uniformly among 20 profiles. Each potential label is Bernoulli with its profile's success probability. Cost is constant at **0.7**.

Each profile starts with `Beta(4*m, 4*(1-m))`. The six low-success and six high-success profiles have correctly centered priors. The eight middle profiles have prior means equally spaced in `[0.668, 0.682]` and success probabilities equally spaced in `[0.793, 0.807]`. Profile indices are permuted by seed.

Delay is `ceil(exp(mu + 0.75*Z))`, where `Z` is standard normal and `mu` equals **2** with probability 0.25 or **5.6** otherwise. Delays are independent of profiles and labels. A selected label becomes available at `t + D`; labels due beyond the horizon stay pending. Unselected labels are not observed. The policy estimates the delay law from 1,000 independently generated calibration delays.

The score is:

```text
posterior_mean + rho * B + gamma * kappa * max(sqrt(I_sch) - sqrt(I_cnt), 0)
```

`B` is the survival-weighted uncertainty floor; `I_sch` and `I_cnt` are the marginal information values using elapsed-time and count-reference maturation probabilities. Selection occurs when the score exceeds cost. The fixed parameters are `rho=0.245`, `gamma=300`, `kappa=2`, and planning window `s=50`. Due feedback is processed before each decision. The policy uses observed selected labels and pending selection ages, not unobserved labels or true success probabilities.

The evaluator uses the true success probabilities to compute pseudo-regret and oracle agreement. Utility is `sum A*(Y-c)`; pseudo-regret is `sum (max(p-c,0)-A*(p-c))`. The unprofitable-selection percentage uses selected instances as its denominator. Table entries use sample standard deviations, not standard errors.

The policy settings were fixed during earlier development. The delay configuration was chosen through exploratory comparisons on the same main seeds; this reproduction does not establish performance across other configurations.

## Files

```text
config.json     Fixed DGP, policy parameters, and seeds
environment.py Independent random streams and delay calibration
policy.py       SF-SIV information calculations and decision loop
run.py          Run the seeds and write the table row
tests/          Feedback-timing and information-value checks
```

Random-stream names are retained for reproducibility. Experimental outputs are not tracked. Baselines, auxiliary experiments, plots, empirical-study code, and private data are outside this release.

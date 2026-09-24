"""Construct policies with common priors and reproducible random streams."""

import numpy as np
from siv_core import (
    DelayedSelectiveBootstrap,
    DelayedUCB,
    MaturedGreedy,
    OneStepBayesInformationHeuristic,
    SurvivalEM,
)
from siv_core.dgp import method_seed
from siv_design import ContinuousPriorScheduledPolicy
from siv_experiments import (
    DelayedBayesUCB,
    TemperedDelayedThompsonSampling,
    SurvivalFlooredMonotoneSIV,
)


def _ordinary(cls, name, seed, alpha, beta, **kwargs):
    policy = cls(
        name=name,
        cells=len(alpha),
        cost=0.7,
        seed=method_seed(seed, name),
        warm_successes=np.zeros(len(alpha), dtype=int),
        warm_total=0,
        **kwargs,
    )
    policy.alpha, policy.beta = alpha.copy(), beta.copy()
    if isinstance(policy, DelayedSelectiveBootstrap):
        policy.ensemble_alpha = np.tile(policy.alpha, (policy.ensemble_size, 1))
        policy.ensemble_beta = np.tile(policy.beta, (policy.ensemble_size, 1))
    return policy


def make_policy(name, repeat, seed, alpha, beta, fitted, true, methods):
    p = methods[name]
    policy_name = name if repeat == 0 else name + "_repeat_" + str(repeat)
    policy_seed = method_seed(seed, policy_name)
    if name == "matured_greedy":
        return _ordinary(MaturedGreedy, policy_name, seed, alpha, beta)
    if name == "bayes_ucb":
        return DelayedBayesUCB(
            name=policy_name,
            cells=20,
            cost=0.7,
            seed=policy_seed,
            prior_alpha=alpha,
            prior_beta=beta,
            posterior_quantile=p["posterior_quantile"],
        )
    if name == "delayed_ucb":
        return _ordinary(DelayedUCB, policy_name, seed, alpha, beta, exploration=p["exploration"])
    if name == "tempered_ts":
        return TemperedDelayedThompsonSampling(
            name=policy_name,
            cells=20,
            cost=0.7,
            seed=policy_seed,
            prior_alpha=alpha,
            prior_beta=beta,
            temperature=p["temperature"],
        )
    if name == "online_bootstrap":
        return _ordinary(
            DelayedSelectiveBootstrap,
            policy_name,
            seed,
            alpha,
            beta,
            ensemble_size=p["ensemble_size"],
        )
    if name == "one_step_bayes":
        return _ordinary(
            OneStepBayesInformationHeuristic,
            policy_name,
            seed,
            alpha,
            beta,
            information_coefficient=p["information_coefficient"],
        )
    if name == "count_only":
        return ContinuousPriorScheduledPolicy(
            name=policy_name,
            cells=20,
            cost=0.7,
            seed=policy_seed,
            prior_alpha=alpha,
            prior_beta=beta,
            kernel=fitted,
            true_kernel=true,
            planning_window=p["planning_window"],
            information_coefficient=p["information_coefficient"],
            age_aware=False,
        )
    if name == "raw_fitted_siv":
        return ContinuousPriorScheduledPolicy(
            name=policy_name,
            cells=20,
            cost=0.7,
            seed=policy_seed,
            prior_alpha=alpha,
            prior_beta=beta,
            kernel=fitted,
            true_kernel=true,
            planning_window=p["planning_window"],
            information_coefficient=p["information_coefficient"],
            age_aware=True,
        )
    if name == "tuned_survival":
        return _ordinary(
            SurvivalEM,
            policy_name,
            seed,
            alpha,
            beta,
            kernel=fitted,
            planning_window=p["planning_window"],
            exploration=p["exploration"],
        )
    if name == "survival_floored_siv":
        return SurvivalFlooredMonotoneSIV(
            name=policy_name,
            cells=20,
            cost=0.7,
            seed=policy_seed,
            prior_alpha=alpha,
            prior_beta=beta,
            kernel=fitted,
            true_kernel=true,
            planning_window=p["planning_window"],
            information_coefficient=p["information_coefficient"],
            timing_gain_scale=p["timing_gain_scale"],
            survival_exploration=p["survival_exploration"],
        )
    raise ValueError("unregistered method: " + name)

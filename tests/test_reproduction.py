"""Check delay identities and compiled/reference action equivalence."""

from pathlib import Path
import json
import sys
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation.environment import prepare
from simulation.kernels import LogNormalDelay, Mixture, CachedDelay
from simulation.policies import make_policy
from simulation.compiled import evaluate
from simulation.statistics import run_totals


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = yaml.safe_load((ROOT / "configs/main.yaml").read_text())
        k = json.loads((ROOT / "configs/delay.json").read_text())
        cls.law = Mixture(
            LogNormalDelay(**k["fast"]), LogNormalDelay(**k["slow"]), k["probability"]
        )

    def test_compiled_action_equivalence(self):
        seed = 5700000
        tape, a, b, costs, fitted, _ = prepare(seed, self.law, self.cfg)
        np.testing.assert_array_equal(costs, np.full(len(costs), 0.7))
        fitted, true = CachedDelay(fitted), CachedDelay(self.law)
        cache = {}
        for name in [
            "matured_greedy",
            "one_step_bayes",
            "count_only",
            "raw_fitted_siv",
            "survival_floored_siv",
            "tuned_survival",
            "delayed_ucb",
        ]:
            p = make_policy(name, 0, seed, a, b, fitted, true, self.cfg["methods"])
            actions = []
            original = p.decide

            def decide(cell, time):
                action = original(cell, time)
                actions.append(action)
                return action

            p.decide = decide
            run_totals(p, tape, costs)
            _, compiled = evaluate(
                tape,
                a,
                b,
                costs,
                fitted,
                dict(method=name, parameters=self.cfg["methods"][name]),
                cache,
            )
            np.testing.assert_array_equal(actions, compiled, err_msg=name)

    def test_delay_law(self):
        self.assertEqual((self.law.fast.mu, self.law.slow.mu), (2, 5.6))
        self.assertAlmostEqual(self.law.mean, 271.6398231074338)
        for age in [0, 1, 10, 100, 1000]:
            expected = 1 - self.law.survival(age + 50) / self.law.survival(age)
            self.assertAlmostEqual(self.law.maturation_probability(age, 50), expected)


if __name__ == "__main__":
    unittest.main()

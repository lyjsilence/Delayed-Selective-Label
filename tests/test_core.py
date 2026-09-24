"""Check information values and feedback timing without stored results."""

import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy import miv, actions_for


class CoreTests(unittest.TestCase):
    def test_information_without_pending_labels(self):
        a, b, q = 2.0, 3.0, 0.4
        variance = a * b / ((a + b) ** 2 * (a + b + 1))
        self.assertAlmostEqual(miv(a, b, np.empty(0), q), q * variance / (a + b + 1))
        self.assertEqual(miv(a, b, np.array([0.2, 0.7]), 0.0), 0.0)

    def test_only_selected_delivered_labels_update_posterior(self):
        # The first accepted failure arrives at t=2, reducing the score below cost.
        cells = np.zeros(4, dtype=np.int64)
        outcomes = np.zeros(4, dtype=np.int8)
        delays = np.full(4, 2, dtype=np.int64)
        costs = np.full(4, 0.65)
        actions = actions_for(
            cells,
            outcomes,
            delays,
            costs,
            np.array([3.0]),
            np.array([1.0]),
            np.ones(4),
            0.0,
            0.0,
            2.0,
        )
        np.testing.assert_array_equal(actions, [1, 1, 0, 0])
        # No delivery within the horizon: latent labels cannot change actions.
        delays[:] = 10
        hidden = actions_for(
            cells,
            outcomes,
            delays,
            costs,
            np.array([3.0]),
            np.array([1.0]),
            np.ones(4),
            0.0,
            0.0,
            2.0,
        )
        np.testing.assert_array_equal(hidden, [1, 1, 1, 1])
        # Rejected instances never enter the delivery queue.
        rejected = actions_for(
            cells,
            np.ones(4, dtype=np.int8),
            np.ones(4, dtype=np.int64),
            costs,
            np.array([1.0]),
            np.array([3.0]),
            np.ones(4),
            0.0,
            0.0,
            2.0,
        )
        np.testing.assert_array_equal(rejected, [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()

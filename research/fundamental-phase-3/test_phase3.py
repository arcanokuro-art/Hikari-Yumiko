from dataclasses import replace
import unittest

import numpy as np

from residual import Residual, features, gate, examples
from environment import Profile, Memory, initial, legal, outcomes, planner
from experiment import environment_cases, previous, audit


class Phase3Tests(unittest.TestCase):
    def setUp(self):
        self.p = Profile((.45, .45, .10), (.55, .98), (.10, .10), (2, 1, 1), 3)

    def test_gate_and_index_value(self):
        root = initial(self.p)
        self.assertTrue(gate(root, self.p))
        q = planner(self.p)
        self.assertGreater(q(root, 3)[4], q(root, 1)[4])
        indexed = Memory((0, 1)).acquire(root, 4, self.p)
        self.assertFalse(gate(indexed, self.p))
        self.assertEqual(q(indexed, 1), q(indexed, 3))

    def test_gate_zero_residual_on_reachable_states(self):
        for p, _, _ in environment_cases(7, "id", 5, 32):
            plan = planner(p)
            queue = [initial(p)]
            while queue:
                s = queue.pop()
                if not gate(s, p):
                    np.testing.assert_allclose(plan(s, 1), plan(s, 3))
                for a in range(3, 6):
                    if legal(s, p)[a]:
                        queue.extend(child for _, child in outcomes(s, a, p))

    def test_budget_and_cost_intervention_keep_hidden_reports(self):
        baseline = environment_cases(7, "id", 3, 32)
        for scenario in ("id", "cost-shift"):
            other = environment_cases(7, scenario, 7, 32)
            for left, right in zip(baseline, other):
                self.assertEqual(left[1:], right[1:])
                self.assertEqual(replace(left[0], budget=7, costs=right[0].costs), right[0])

    def test_counterfactual_hidden_report_keeps_features_and_decision(self):
        state = initial(self.p)
        model = Residual(np.random.default_rng(7))
        a = Memory((0, 0)).acquire(state, 4, self.p)
        b = Memory((1, 1)).acquire(state, 4, self.p)
        self.assertEqual(features(a, self.p), features(b, self.p))
        self.assertEqual(model.choose(a, self.p), model.choose(b, self.p))

    def test_residual_targets_are_nonnegative_on_legal_acquisitions(self):
        x, y, mask = examples([self.p])
        self.assertEqual(x.shape[1], 20)
        self.assertEqual(y.shape[1], 3)
        self.assertGreater(y[mask].max(), 0)
        self.assertGreaterEqual(y[mask].min(), -1e-12)

    def test_oracle_has_no_decision_regret(self):
        cases = environment_cases(7, "id", 5, 32)
        d = audit(cases, previous.policy("oracle", None, np.random.default_rng(7)))
        self.assertEqual(d["positive_regret"], 0)
        self.assertEqual(d["regret_per_decision"], 0.)


if __name__ == "__main__":
    unittest.main()

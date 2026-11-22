import unittest
from dataclasses import replace

import numpy as np

from benchmark import (EPS, NetPolicy, best_stop, cases_for, diagnose, policy_value,
                       target_data)
from environment import Memory, initial, legal, planner


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.p, _, self.reports = cases_for(7, "two-required", 3, 1)[0]
        self.model = NetPolicy(np.random.default_rng(7))

    def test_two_acquisitions_strictly_needed(self):
        s, plan = initial(self.p), planner(self.p)
        self.assertEqual(max(plan(s, 0)), max(plan(s, 1)))
        self.assertGreater(max(plan(s, 2)), max(plan(s, 1)))
        self.assertEqual(int(np.argmax(plan(s, 3))), 4)

    def test_immediate_stop_best_without_signal(self):
        p = replace(self.p, fidelity=(.5, .5))
        s, plan = initial(p), planner(p)
        self.assertLess(max(plan(s, 3)[3:]), max(plan(s, 0)))
        self.assertAlmostEqual(policy_value(p, best_stop), max(plan(s, 3)))

    def test_masks_and_exact_terminal(self):
        p = replace(self.p, budget=0)
        s = initial(p)
        self.assertEqual(self.model.decide(s, p, -100), best_stop(s, p))
        s = initial(self.p)
        a = self.model.decide(s, self.p, -100)
        self.assertTrue(legal(s, self.p)[a])
        self.assertNotEqual(a, 5)

    def test_signed_gain_and_mask(self):
        _, gain, q, mask = target_data([self.p])
        self.assertTrue(np.all(gain[~mask] == 0))
        self.assertLess(gain[0, 0], 0)
        self.assertGreater(gain[0, 1], 0)
        self.assertAlmostEqual(q[0, 1] - gain[0, 1], max(planner(self.p)(initial(self.p), 0)))

    def test_unacquired_reports_do_not_affect_decision(self):
        s = initial(self.p)
        left, right = Memory((0, 0)), Memory((0, 2))
        a = left.acquire(s, 4, self.p)
        b = right.acquire(s, 4, self.p)
        self.assertEqual(a, b)
        self.assertEqual(self.model.decide(a, self.p), self.model.decide(b, self.p))

    def test_oracle_audit_and_classification(self):
        cases = cases_for(19, "two-required", 3, 17)
        audit = diagnose(cases, lambda s, p: int(np.argmax(planner(p)(s, 3))))
        self.assertEqual(audit["tasks_multiple"], len(cases))
        self.assertEqual(audit["useful_index"], len(cases))
        self.assertLessEqual(audit["regret_per_decision"], EPS)


if __name__ == "__main__":
    unittest.main()

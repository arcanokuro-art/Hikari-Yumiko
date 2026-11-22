from dataclasses import replace
import unittest

import numpy as np

from controller import Controller, features
from environment import Memory, Profile, belief, initial, legal, outcomes, planner, transition
from experiment import policy, rollout


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        self.p = Profile((.45, .45, .10), (.55, .98), (.1, .1), (2, 1, 1), 3)
        self.s = initial(self.p)

    def test_index_has_indirect_value(self):
        q = planner(self.p)
        self.assertLess(np.argmax(q(self.s, 1)), 3)
        self.assertEqual(np.argmax(q(self.s, 3)), 4)
        indexed = transition(self.s, 4, self.p)
        self.assertEqual(belief(self.s, self.p), belief(indexed, self.p))
        self.assertTrue(legal(indexed, self.p)[5])

    def test_blocked_verification_does_not_access_private_payload(self):
        mem = Memory((0, 1))
        with self.assertRaises(ValueError):
            mem.acquire(self.s, 5, self.p)
        self.assertEqual(mem.reads, [])

    def test_unobserved_payload_cannot_change_public_prefix(self):
        a, b = Memory((0, 0)), Memory((0, 1))
        for action in (4, 3):
            left = a.acquire(self.s, action, self.p)
            right = b.acquire(self.s, action, self.p)
            self.assertEqual(left, right)
            self.assertEqual(features(left, self.p), features(right, self.p))
            model = Controller(np.random.default_rng(7))
            self.assertEqual(model.select(left, self.p), model.select(right, self.p))
        s = a.acquire(self.s, 4, self.p)
        self.assertNotEqual(a.acquire(s, 5, self.p), b.acquire(s, 5, self.p))

    def test_posteriors_and_label_symmetry(self):
        s = transition(self.s, 3, self.p, 0)
        mirrored = transition(self.s, 3, self.p, 1)
        self.assertAlmostEqual(sum(belief(s, self.p)), 1)
        self.assertAlmostEqual(sum(w for w, _ in outcomes(self.s, 3, self.p)), 1)
        np.testing.assert_allclose(belief(s, self.p), np.asarray(belief(mirrored, self.p))[[1, 0, 2]])

    def test_uninformative_memory_is_skipped_by_planner(self):
        p = replace(self.p, fidelity=(.5, .5))
        self.assertLess(np.argmax(planner(p)(initial(p), 3)), 3)

    def test_neural_ablation_still_respects_budget_mask(self):
        p = replace(self.p, budget=0)
        model = Controller(np.random.default_rng(7), no_cost=True)
        answer, trace, cost = rollout(p, (0, 1), policy("no-cost", model, np.random.default_rng(7)))
        self.assertLess(answer, 3)
        self.assertEqual((trace, cost), ([], 0))


if __name__ == "__main__":
    unittest.main()

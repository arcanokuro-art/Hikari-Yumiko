import unittest
from dataclasses import replace

import numpy as np

from audit import alias_probe, broad_profiles, cases_for, get_policy, p4, score_decisions
from environment import initial


class AuditTests(unittest.TestCase):
    def test_hidden_reliability_requires_distinct_actions(self):
        worlds = alias_probe()['worlds']
        self.assertEqual(worlds[0]['optimal'], ['answer-0', 'answer-1'])
        self.assertEqual(worlds[1]['optimal'], ['local'])

    def test_read_is_mandatory_for_exact_two_step_in_special_case(self):
        p, _, _ = cases_for(7, 'two-required', 3, 1)[0]
        self.assertEqual(get_policy('myopic', {}, 7)(initial(p), p), 0)
        self.assertEqual(get_policy('two-step', {}, 7)(initial(p), p), 4)

    def test_intervention_does_not_unmask_operation(self):
        p, _, _ = cases_for(7, 'id', 3, 1)[0]
        p = replace(p, budget=0)
        model = p4.NetPolicy(np.random.default_rng(7))
        self.assertLess(get_policy('miope-veto', {'net': (model, 0.)}, 7)(initial(p), p), 3)

    def test_no_signal_terminal_audit_and_pairing(self):
        cases = cases_for(7, 'unhelpful', 3, 20)
        model = p4.NetPolicy(np.random.default_rng(7))
        d = score_decisions(cases, model, 100., 'unhelpful', 3, None)
        self.assertEqual(d['counts']['errors'], 0)
        across_budget = cases_for(7, 'unhelpful', 5, 20)
        self.assertEqual([x[1:] for x in cases], [x[1:] for x in across_budget])

    def test_coverage_preserves_profile_count(self):
        self.assertEqual(len(broad_profiles(np.random.default_rng(7), 20)), 20)


if __name__ == '__main__':
    unittest.main()

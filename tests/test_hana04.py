import unittest

import torch

from hikari.hana04 import HANA04, RELATIONS, STOP
from hikari.hana04_data import make_queries, query_patterns


def solve(facts, start, relations):
    lookup = {(relation, source): destination for source, relation, destination in facts}
    for relation in relations:
        if relation != STOP:
            start = lookup[relation, start]
    return start


class TestHANA04(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.generator = torch.Generator().manual_seed(11)

    def test_oracle_and_holdouts(self):
        for split, depth, expected_facts in (
            ("fresh", 1, 30), ("fresh", 3, 30), ("composition", 2, 30),
            ("composition", 3, 30), ("relation", 3, 30), ("length", 3, 54),
        ):
            data = make_queries(16, depth, split, generator=self.generator)
            self.assertEqual(data.facts.size(1), expected_facts)
            for facts, start, query, target in zip(*(field.tolist() for field in data)):
                used = query[:depth]
                self.assertEqual(len(set(used)), depth)
                self.assertEqual(solve(facts, start, query), target)
                self.assertEqual(solve(list(reversed(facts)), start, query), target)
                if split == "relation":
                    self.assertIn(RELATIONS - 1, used)
                if split == "composition":
                    self.assertNotIn(tuple(used), query_patterns(depth, "train"))

    def test_critical_edge_changes_answer_without_changing_query(self):
        graph, start, query, target = make_queries(1, 3, generator=self.generator)
        facts = graph[0].tolist()
        relations = query[0].tolist()
        predecessor = solve(facts, start.item(), relations[:-1])
        decisive = next(edge for edge in facts if edge[:2] == [predecessor, relations[-1]])
        alternative = next(edge for edge in facts if edge[1] == relations[-1] and edge is not decisive)
        decisive[2], alternative[2] = alternative[2], decisive[2]
        self.assertNotEqual(solve(facts, start.item(), relations), target.item())
        self.assertEqual(solve(facts, start.item(), relations[:-1]), predecessor)

    def test_only_active_rows_execute_extra_cycles(self):
        model = HANA04(width=8).eval()
        data = [make_queries(1, depth, generator=self.generator) for depth in (1, 2, 3)]
        facts, start, relations, _ = (torch.cat(field) for field in zip(*data))
        with torch.inference_mode():
            model.continue_gate.weight.zero_()
            model.continue_gate.weight[0, -4:-1] = torch.tensor([-1., 0.5, 1.5])
            model.continue_gate.weight[0, -1] = -1
            model.continue_gate.bias.zero_()
            sizes = []
            hook = model.core.register_forward_hook(lambda _, args, __: sizes.append(args[0].size(0)))
            try:
                scores, counts = model(facts, start, relations)
            finally:
                hook.remove()
            self.assertEqual(counts.tolist(), [1, 2, 3])
            self.assertEqual(sizes, [3, 2, 1])
            for i in range(3):
                single, cycles = model(facts[i:i+1], start[i:i+1], relations[i:i+1])
                torch.testing.assert_close(scores[i:i+1], single)
                self.assertEqual(cycles.item(), counts[i].item())

    def test_controller_gradient_and_memory_order_invariance(self):
        model = HANA04(width=8)
        facts, start, relations, _ = make_queries(8, 3, generator=self.generator)
        scores, log_routes = model(facts, start, relations)
        torch.testing.assert_close(log_routes.exp().sum(1), torch.ones(8))
        reordered, _ = model(facts.flip(1), start, relations)
        torch.testing.assert_close(scores, reordered)
        (-log_routes[:, 0].mean() + scores.square().mean()).backward()
        self.assertGreater(model.continue_gate.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.read_gate.weight.grad.abs().sum().item(), 0)
        no_memory = HANA04(width=8, use_memory=False)
        self.assertIsNone(no_memory.prepare(facts, start, relations)[2])

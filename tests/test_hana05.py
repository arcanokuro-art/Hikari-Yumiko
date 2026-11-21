import unittest

import torch

from hikari.hana05 import HANA05
from hikari.hana05_data import counterfactual, make_episode, patterns


def solve(edges, nodes, tags, start, relations, goal):
    bank = {(relation, source): destination for source, relation, destination in edges}
    attributes = dict(zip(nodes, tags))
    for step, relation in enumerate(relations, start=1):
        start = bank[relation, start]
        if attributes[start] == goal:
            return start, step
    return None, None


class TestHANA05(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.generator = torch.Generator().manual_seed(11)

    def test_graph_and_query_shape_match_across_depths(self):
        for split, facts in (("fresh", 96), ("relation", 96), ("composition", 96), ("length", 144)):
            episode = make_episode(20, split, self.generator)
            self.assertEqual(episode.query.edges.size(1), facts)
            self.assertTrue(episode.goals.sort(1).values.diff(dim=1).gt(0).all())
            for depth in (1, 2, 3):
                query, targets = episode.at_depth(depth)
                self.assertEqual(query.relations.shape, (20, 3))
                for reference, actual in zip(episode.query[:-1], query[:-1]):
                    self.assertIs(reference, actual)
                for example, target in zip(zip(*(field.tolist() for field in query)), targets.tolist()):
                    self.assertEqual(solve(*example), (target, depth))
                    if split == "relation":
                        self.assertIn(3, example[4])
                    if split == "composition":
                        self.assertNotIn(tuple(example[4]), patterns("train"))

    def test_counterfactual_preserves_paired_marginals_and_depth(self):
        episode = make_episode(32, "fresh", self.generator)
        for depth in (1, 2, 3):
            original, original_target = episode.at_depth(depth)
            query, targets = counterfactual(episode, depth)
            self.assertTrue(targets.ne(original_target).all())
            self.assertTrue(query.edges.ne(original.edges).any(-1).sum(1).eq(2).all())
            torch.testing.assert_close(query.edges[:, :, :2], original.edges[:, :, :2])
            torch.testing.assert_close(query.edges[:, :, 2].sort(1).values,
                                       original.edges[:, :, 2].sort(1).values)
            for reference, actual in zip(original[1:], query[1:]):
                torch.testing.assert_close(reference, actual)
            for example, target in zip(zip(*(field.tolist() for field in query)), targets.tolist()):
                self.assertEqual(solve(*example), (target, depth))

    def test_policies_share_retrieval_and_stop_at_expected_cycles(self):
        model = HANA05(width=8).eval()
        episode = make_episode(5, "fresh", self.generator)
        query, _ = episode.at_depth(2)
        calls = []
        hook = model.core.register_forward_hook(lambda _, args, __: calls.append(args[0].size(0)))
        try:
            with torch.inference_mode():
                direct, cycles, certificate = model.infer(query, "deterministic")
                self.assertEqual(calls, [5, 5])
                self.assertTrue(cycles.eq(2).all())
                self.assertTrue(certificate.all())
                fixed, counts, _ = model.infer(query, "fixed", budgets=torch.full_like(cycles, 2))
                torch.testing.assert_close(direct, fixed)
                torch.testing.assert_close(cycles, counts)
                budgets = torch.tensor([1, 2, 3, 2, 1])
                _, counts, _ = model.infer(query, "random", budgets=budgets)
                torch.testing.assert_close(budgets, counts)
        finally:
            hook.remove()

    def test_controller_receives_training_gradients(self):
        model = HANA05(width=8)
        episode = make_episode(12, "train", self.generator)
        query, target = episode.at_depth(2)
        loss = model.training_loss(query, target)
        loss.backward()
        self.assertGreater(model.halt[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.core.weight_ih.grad.abs().sum().item(), 0)

import unittest

import torch

from hikari.hana06 import HANA06, baseline, memory, read
from hikari.hana06_data import UNKNOWN, make_batch, oracle
from hikari.hana05_data import make_episode


class TestHANA06(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_missing_edges_break_only_unverified_answers(self):
        episode = make_episode(24, "fresh", torch.Generator().manual_seed(19))
        query, expected = episode.at_depth(1)
        visible = torch.ones_like(query.edges[:, :, 0], dtype=torch.bool)
        first_source = query.edges[:, :, 0].eq(query.start[:, None])
        first_bank = query.edges[:, :, 1].eq(query.relations[:, 0, None])
        later_bank = query.edges[:, :, 1].eq(query.relations[:, 1, None])
        target, depth = oracle(query, visible & ~later_bank)
        torch.testing.assert_close(target, expected)
        self.assertTrue(depth.eq(1).all())
        target, depth = oracle(query, visible & ~(first_source & first_bank))
        self.assertTrue(target.eq(UNKNOWN).all())
        self.assertTrue(depth.eq(0).all())

    def test_paired_noise_keeps_structure_and_labels(self):
        low, y, depth = make_batch(64, "fresh", 0.10, torch.Generator().manual_seed(8))
        high, other_y, other_depth = make_batch(64, "fresh", 0.55, torch.Generator().manual_seed(8))
        for field, other in zip(low._replace(votes=None), high._replace(votes=None)):
            if field is not None:
                torch.testing.assert_close(field, other)
        torch.testing.assert_close(y, other_y)
        torch.testing.assert_close(depth, other_depth)
        self.assertFalse(torch.equal(low.votes, high.votes))
        self.assertTrue(high.votes.gt(0).sum(2).gt(1).any())

    def test_counterfactual_keeps_observations_except_edges(self):
        a, y, _, b, z, _ = make_batch(96, "fresh", 0.3, torch.Generator().manual_seed(4), paired=True)
        for left, right in zip(a[1:], b[1:]):
            self.assertIs(left, right)
        self.assertTrue(y.ne(z).any())
        self.assertTrue(y.eq(z).any())
        self.assertTrue(a.edges[~a.visible].eq(0).all())

    def test_read_cannot_traverse_missing_edge_and_baselines_abstain(self):
        obs, _, _ = make_batch(16, "fresh", 0.2, torch.Generator().manual_seed(11))
        obs = obs._replace(visible=torch.zeros_like(obs.visible), edges=torch.zeros_like(obs.edges))
        table, labels = memory(obs)
        node, votes = read(table, labels, obs.start, obs.relations[:, 0])
        self.assertTrue(node.eq(-1).all())
        self.assertTrue(votes.eq(0).all())
        for mode in ("deterministic", "probabilistic"):
            answer, confidence, cycles = baseline(obs, mode)
            self.assertTrue(answer.eq(UNKNOWN).all())
            self.assertTrue(confidence.eq(1).all())
            self.assertTrue(cycles.eq(1).all())

    def test_controller_and_confidence_receive_learning_signal(self):
        x, y, _ = make_batch(12, "train", 0.3, torch.Generator().manual_seed(6))
        model = HANA06(width=8)
        loss = model.loss(x, y)
        loss.backward()
        for layer in (model.stop, model.slots, model.recurrent):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in layer.parameters()), 0)
        model.eval()
        with torch.inference_mode():
            pred, conf, cycles = model.infer(x, "adaptive")
            self.assertTrue(pred.ge(0).all() and pred.le(UNKNOWN).all())
            self.assertTrue(conf.ge(0).all() and conf.le(1).all())
            self.assertTrue(cycles.ge(1).all() and cycles.le(3).all())

import unittest

import torch

from hikari.hana03 import HANA03
from hikari.hana03_experiment import task_batch


class TestHANA03(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_three_routes_and_training_inference_agreement(self):
        model = HANA03(5, width=4)
        tokens = torch.tensor([[0], [1], [2]])
        with torch.no_grad():
            model.embedding.weight.zero_()
            model.embedding.weight[:3, :2] = torch.tensor([[-1., -1.], [1., -1.], [1., 1.]])
            model.continue_gate.weight.zero_()
            model.continue_gate.weight[0, 0] = 1
            model.continue_gate.weight[1, 1] = 1
            model.continue_gate.bias.zero_()
            train = model(tokens)
            model.eval()
            batched = model(tokens)
            self.assertEqual(batched[2].flatten().tolist(), [1, 2, 3])
            torch.testing.assert_close(train[2], batched[2])
            torch.testing.assert_close(train[0], batched[0])
            for row in range(3):
                single = model(tokens[row:row + 1])
                torch.testing.assert_close(single[0], batched[0][row:row + 1])
                torch.testing.assert_close(single[2], batched[2][row:row + 1])

    def test_cost_gradients_penalize_both_continuation_stages(self):
        model = HANA03(5, width=4)
        with torch.no_grad():
            model.continue_gate.weight.zero_()
            model.continue_gate.bias.zero_()
        tokens = torch.tensor([[0, 1], [2, 3]])
        _, _, _, cost = model(tokens, return_cost=True)
        self.assertAlmostEqual(cost.item(), 0.75)
        cost.backward()
        torch.testing.assert_close(model.continue_gate.bias.grad, torch.tensor([0.375, 0.125]))

    def test_inference_skips_unselected_core_calls(self):
        model = HANA03(5, width=4).eval()
        with torch.inference_mode():
            model.continue_gate.weight.zero_()
            calls = []
            handle = model.core.register_forward_hook(lambda *args: calls.append(1))
            try:
                for bias in ((-5., 5.), (5., -5.), (5., 5.)):
                    model.continue_gate.bias.copy_(torch.tensor(bias))
                    calls.clear()
                    _, _, cycles = model(torch.tensor([[0, 1, 2]]))
                    self.assertEqual(len(calls), cycles.sum().item())
            finally:
                handle.remove()

    def test_long_task_queries_and_targets(self):
        tokens, targets, positions = task_batch("recall", 80, 4)
        self.assertEqual(tokens.shape, (4, 105))
        pairs = tokens[:, :16].reshape(4, 8, 2)
        matches = tokens[:, positions, None].eq(pairs[:, None, :, 1])
        self.assertTrue(matches.sum(-1).eq(1).all())
        torch.testing.assert_close(targets, pairs[:, :, 0].gather(1, matches.long().argmax(-1)))

import unittest

import torch

from hikari.hana import HANA0Nano, recall, remember
from hikari.hana_experiment import experiment


class TestHANA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)
        self.model = HANA0Nano(7, width=8, memory_slots=2)
        self.tokens = torch.tensor([[1, 2, 1, 3], [4, 2, 3, 5]])

    def test_forward_and_gradient(self):
        logits, state, counts = self.model(self.tokens)
        self.assertEqual(logits.shape, (2, 4, 7))
        self.assertEqual(state.hidden.shape, (2, 8))
        self.assertEqual(state.working.shape, (2, 8))
        self.assertEqual(counts.shape, self.tokens.shape)
        logits.square().mean().backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(name=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_memory_latest_value_and_eviction(self):
        memory = self.model.initial_state(self.tokens).memory
        query = torch.tensor([1, 2])
        first = torch.ones(2, 8)
        second = 2 * first
        self.assertEqual(recall(memory, query).count_nonzero().item(), 0)
        memory = remember(memory, query, first)
        memory = remember(memory, query, second)
        torch.testing.assert_close(recall(memory, query), second)
        memory = remember(memory, query + 1, first)
        torch.testing.assert_close(recall(memory, query), second)
        memory = remember(memory, query + 2, first)
        self.assertEqual(recall(memory, query).count_nonzero().item(), 0)

    def test_states_continue_across_chunks(self):
        with torch.inference_mode():
            self.model.eval()
            entire, _, counts = self.model(self.tokens)
            first, state, first_counts = self.model(self.tokens[:, :2])
            second, _, second_counts = self.model(self.tokens[:, 2:], state)
        torch.testing.assert_close(torch.cat((first, second), dim=1), entire)
        torch.testing.assert_close(torch.cat((first_counts, second_counts), dim=1), counts)

    def test_routes_different_inputs_in_same_batch(self):
        with torch.no_grad():
            self.model.embedding.weight.zero_()
            self.model.embedding.weight[1, 0] = -1
            self.model.embedding.weight[4, 0] = 1
            self.model.continue_gate.weight.zero_()
            self.model.continue_gate.weight[0, 0] = 1
            self.model.continue_gate.bias.zero_()
            trained, _, train_counts = self.model(self.tokens[:, :1])
            self.model.eval()
            result, _, counts = self.model(self.tokens[:, :1])
        self.assertEqual(counts.tolist(), [[1], [3]])
        torch.testing.assert_close(train_counts, counts)
        torch.testing.assert_close(trained, result)

    def test_synthetic_learning_and_reproducible_weights(self):
        first, metrics = experiment(steps=80, seed=7, vocab=5, length=6, batch_size=16)
        second, repeated = experiment(steps=80, seed=7, vocab=5, length=6, batch_size=16)
        self.assertGreater(metrics["evaluation_accuracy_after_first_token"], 0.85)
        self.assertEqual(metrics["evaluation_accuracy_after_first_token"],
                         repeated["evaluation_accuracy_after_first_token"])
        for name, parameter in first.state_dict().items():
            torch.testing.assert_close(parameter, second.state_dict()[name], atol=0, rtol=0)
        self.assertGreater(metrics["parameters"], 0)
        self.assertGreater(metrics["process_peak_rss_kib_linux"], 0)
        self.assertGreater(metrics["inference_ms_per_sequence"], 0)

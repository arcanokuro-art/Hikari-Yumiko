import unittest

import torch
from torch.nn import functional as F

from hikari.hana02_experiment import TASKS, make_model, make_task


class TestHANA02(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_batch_one_matches_original_and_indexed_routing(self):
        original = make_model("nano").eval()
        model = make_model("adaptive").eval()
        for bias, expected in ((-10, 1), (10, 3)):
            with torch.inference_mode():
                original.continue_gate.bias.fill_(bias)
                model.load_state_dict(original.state_dict())
                tokens, _, _ = make_task("recall-short", 1)
                model.fast_inference = True
                direct = model(tokens)
                model.fast_inference = False
                indexed = model(tokens)
                reference = original(tokens)
                self.assertTrue(direct[2].eq(expected).all())
                for logits, state, counts in (indexed, reference):
                    torch.testing.assert_close(direct[0], logits)
                    torch.testing.assert_close(direct[1], state)
                    torch.testing.assert_close(direct[2], counts)

    def test_fixed_routes_and_memory_ablation(self):
        tokens, _, _ = make_task("recall-short", 2)
        for cycles in (1, 2, 3):
            model = make_model(f"fixed-{cycles}")
            self.assertFalse(hasattr(model, "continue_gate"))
            _, state, counts = model(tokens)
            self.assertTrue(counts.eq(cycles).all())
            self.assertTrue(state.memory.keys.ge(0).any())
        model = make_model("no-memory")
        state = model.initial_state(tokens)
        _, next_state, _ = model(tokens, state)
        self.assertIs(next_state.memory, state.memory)

    def test_prediction_loss_updates_controller(self):
        model = make_model("adaptive")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        weight = model.continue_gate.weight.detach().clone()
        for task in TASKS:
            tokens, targets, positions = make_task(task, 3)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(tokens)[0][:, positions].transpose(1, 2), targets)
            loss.backward()
            optimizer.step()
        self.assertFalse(torch.equal(weight, model.continue_gate.weight))

    def test_recall_targets_align_with_shuffled_queries(self):
        for task, pairs in (("recall-short", 4), ("recall-long", 16)):
            tokens, values, query_slice = make_task(task, 4)
            examples = tokens[:, :2 * pairs].reshape(4, pairs, 2)
            matches = tokens[:, query_slice, None].eq(examples[:, None, :, 1])
            self.assertTrue(matches.sum(-1).eq(1).all())
            expected = examples[:, :, 0].gather(1, matches.long().argmax(-1))
            torch.testing.assert_close(values, expected)

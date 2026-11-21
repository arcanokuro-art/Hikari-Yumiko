import unittest
import torch
from hikari.model import ZNC0
from hikari.train import make_batch


class TestZNC0(unittest.TestCase):
    def test_shapes_and_gradients(self):
        torch.manual_seed(3)
        model = ZNC0(17, width=12)
        data = torch.randint(0, 17, (3, 5))
        logits, state = model(data)
        self.assertEqual(tuple(logits.shape), (3, 5, 17))
        self.assertEqual(tuple(state.shape), (3, 12))
        logits.sum().backward()
        self.assertIsNotNone(model.gate.weight.grad)

    def test_state_chunking(self):
        torch.manual_seed(3)
        model = ZNC0(17, width=12)
        tokens = torch.randint(0, 17, (3, 7))
        entire, _ = model(tokens)
        left, state = model(tokens[:, :3])
        right, _ = model(tokens[:, 3:], state)
        self.assertTrue(torch.allclose(entire, torch.cat([left, right], dim=1), atol=1e-6))

    def test_batch(self):
        x, y = make_batch(3, 5, 17)
        self.assertEqual(tuple(x.shape), (3, 5))
        self.assertEqual(tuple(y.shape), (3, 5))


if __name__ == "__main__":
    unittest.main()

"""Run matched CPU experiments in separate processes for comparable peak RSS."""
import argparse
import json
import resource
import time

import torch
from torch import nn
from torch.nn import functional as F

from .hana import HANA0Nano
from .model import ZNC0
from .train import make_batch


class GRUBaseline(nn.Module):
    def __init__(self, vocab, width):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.core = nn.GRU(width, width, batch_first=True)
        self.head = nn.Linear(width, vocab)

    def forward(self, tokens):
        output, state = self.core(self.embedding(tokens))
        return self.head(output), state


def experiment(model_name="hana", steps=250, seed=7, vocab=17, length=12, batch_size=32):
    factories = {"hana": (HANA0Nano, 32), "znc": (ZNC0, 60), "gru": (GRUBaseline, 49)}
    factory, width = factories[model_name]
    # Separate RNG streams keep train/evaluation examples identical across models.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = factory(vocab, width)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        torch.manual_seed(seed + 1)
        for _ in range(steps):
            tokens, targets = make_batch(batch_size, length, vocab)
            loss = F.cross_entropy(model(tokens)[0][:, 1:].transpose(1, 2), targets[:, 1:])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        torch.manual_seed(seed + 2)
        tokens, targets = make_batch(256, length, vocab)

    model.eval()
    with torch.inference_mode():
        result = model(tokens)
        accuracy = (result[0][:, 1:].argmax(dim=-1) == targets[:, 1:]).float().mean().item()
        sample = tokens[:1]
        for _ in range(5):
            model(sample)
        start = time.perf_counter()
        for _ in range(30):
            model(sample)
        latency = (time.perf_counter() - start) * 1000 / 30
        model_cycles = result[2].float().mean().item() if model_name == "hana" else 1.0

    return model, {
        "model": model_name,
        "seed": seed,
        "steps": steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "evaluation_accuracy_after_first_token": accuracy,
        "inference_ms_per_sequence": latency,
        "inference_sequence_length": length,
        "process_peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mean_cycles_per_token": model_cycles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("hana", "znc", "gru"), default="hana")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save")
    args = parser.parse_args()
    if args.steps < 1 or args.seed < 0:
        parser.error("positive steps and nonnegative seed required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, metrics = experiment(args.model, args.steps, args.seed)
    if args.save:
        torch.save({"model": model.state_dict(), "metrics": metrics}, args.save)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

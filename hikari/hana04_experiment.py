"""CPU-only comparison on fresh relational graphs and held-out splits."""
import argparse
import json
import resource
import statistics
import time

import torch
from torch.nn import functional as F

from .hana03 import HANA03
from .hana04 import ENTITIES, RELATIONS, HANA04
from .hana04_data import QueryBatch, make_queries
from .hana_experiment import GRUBaseline
from .model import ZNC0


MODELS = ("adaptive", "no-cost", "no-memory", "fixed-1", "fixed-2", "fixed-3", "hana03", "znc", "gru")
SPLITS = ("fresh", "relation", "composition", "length")
VOCAB = ENTITIES + RELATIONS + 1


def make_model(name):
    if name == "hana03":
        return HANA03(VOCAB, width=29, memory_slots=128)
    if name == "znc":
        return ZNC0(VOCAB, width=53)
    if name == "gru":
        return GRUBaseline(VOCAB, width=44)
    cycles = int(name[-1]) if name.startswith("fixed-") else None
    return HANA04(fixed_cycles=cycles, use_memory=name != "no-memory")


def predict(model, batch):
    if isinstance(model, HANA04):
        return model(batch.facts, batch.start, batch.relations)
    outputs = model(batch.serialized())
    tokens = outputs[0][:, -1, :ENTITIES]
    counts = outputs[2].sum(1) if len(outputs) == 3 else torch.full_like(batch.start, outputs[0].size(1))
    return tokens, counts


def evaluate(model, split, depth, generator):
    batch = make_queries(192, depth, split, generator=generator)
    predictions, cycles = predict(model, batch)
    sample = QueryBatch(*(item[:1] for item in batch))
    for _ in range(3):
        predict(model, sample)
    timings = []
    for _ in range(7):
        start = time.perf_counter()
        predict(model, sample)
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "accuracy": predictions.argmax(1).eq(batch.target).float().mean().item(),
        "latency_ms": statistics.median(timings),
        "mean_reasoning_cycles": cycles.float().mean().item() if isinstance(model, HANA04) else None,
        "cycle_histogram": cycles.bincount(minlength=4)[1:4].tolist() if isinstance(model, HANA04) else None,
        "mean_state_updates_per_query": cycles.float().mean().item(),
        "fact_count": batch.facts.size(1),
    }


def run(name, seed, steps):
    torch.manual_seed(seed)
    model = make_model(name)
    batches = torch.Generator().manual_seed(seed + 1000)
    evaluation = torch.Generator().manual_seed(seed + 2000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008)
    routed = name in ("adaptive", "no-cost", "no-memory")
    base_cost = 0.15 if name in ("adaptive", "no-memory") else 0.0
    cycle_costs = torch.arange(3, dtype=torch.float32)
    start = time.perf_counter()
    for step in range(steps):
        batch = make_queries(16, 1 + step % 3, generator=batches)
        logits, route = predict(model, batch)
        if routed:
            losses = F.cross_entropy(logits.transpose(1, 2),
                                     batch.target[:, None].expand(-1, 3), reduction="none")
            # Warm up the shared predictor, then fit the policy to each
            # exit's observed error plus the explicit price of computation.
            loss = losses.mean()
            if step >= steps // 3:
                utilities = losses.detach() + base_cost * cycle_costs
                preferred = (-utilities / 0.08).softmax(1)
                loss = loss - (preferred * route).sum(1).mean()
        else:
            loss = F.cross_entropy(logits, batch.target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    training_s = time.perf_counter() - start
    model.eval()
    with torch.inference_mode():
        metrics = {f"{split}-{depth}": evaluate(model, split, depth, evaluation)
                   for split in SPLITS for depth in (1, 2, 3)
                   if split != "composition" or depth > 1}
    return {"name": name, "seed": seed, "steps": steps,
            "parameters": sum(p.numel() for p in model.parameters()), "training_s": training_s,
            "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cycle_cost": base_cost, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="adaptive")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=240)
    args = parser.parse_args()
    if args.steps < 1 or args.seed < 0:
        parser.error("positive steps and nonnegative seed required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    print(json.dumps(run(args.model, args.seed, args.steps), indent=2))


if __name__ == "__main__":
    main()

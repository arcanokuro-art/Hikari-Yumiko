"""Matched synthetic training and length generalization on a single CPU thread."""
import argparse
import json
import resource
import time

import torch
from torch import nn
from torch.nn import functional as F

from .hana02 import HANA02
from .hana02_experiment import VOCAB, make_model, median_latency
from .hana03 import HANA03
from .train import make_batch


MODELS = ("penalized", "unpenalized", "memory-128", "adaptive", "fixed-1", "fixed-1-128", "znc", "gru")
TRAIN = (("cyclic", 0), ("recall", 8), ("recall", 40))
EVAL = (*TRAIN, ("recall", 52), ("recall", 80))


def task_batch(task, gap, batch):
    if task == "cyclic":
        tokens, targets = make_batch(batch, 12, 7)
        return tokens, targets[:, 1:], slice(1, None)
    pairs = 8
    keys = torch.rand(batch, 17).argsort(dim=1)[:, :pairs]
    values = torch.randint(17, 34, (batch, pairs))
    order = torch.rand(batch, pairs).argsort(dim=1)
    examples = torch.stack((values, keys), dim=-1).flatten(1)
    distractors = torch.full((batch, gap + 1), 34, dtype=torch.long)
    tokens = torch.cat((examples, distractors, keys.gather(1, order)), dim=1)
    return tokens, values.gather(1, order), slice(2 * pairs + gap + 1, None)


def inspect_task(model, task, gap, seed):
    torch.manual_seed(seed)
    tokens, targets, positions = task_batch(task, gap, 96)
    prediction = model(tokens)
    cycles = prediction[2] if len(prediction) == 3 else torch.ones_like(tokens)
    scored = cycles[:, positions]
    return {
        "tokens": tokens.size(1),
        "accuracy": prediction[0][:, positions].argmax(-1).eq(targets).float().mean().item(),
        "mean_cycles_all_tokens": cycles.float().mean().item(),
        "mean_cycles_scored": scored.float().mean().item(),
        "scored_cycle_counts": torch.bincount(scored.flatten(), minlength=4)[1:].tolist(),
        "latency_ms": median_latency(model, tokens[:1]),
    }


def run(name, seed, steps, batch):
    torch.manual_seed(seed)
    experimental = name in MODELS[:3]
    if experimental:
        model = HANA03(VOCAB, memory_slots=128 if name == "memory-128" else 64)
    elif name == "fixed-1-128":
        model = HANA02(VOCAB, memory_slots=128, fixed_cycles=1)
    else:
        model = make_model(name)
    coefficient = 0.035 if name in ("penalized", "memory-128") else 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    torch.manual_seed(seed + 1000)
    start = time.perf_counter()
    for step in range(steps):
        task, gap = TRAIN[step % len(TRAIN)]
        tokens, targets, positions = task_batch(task, gap, batch)
        if experimental:
            logits, _, _, cost = model(tokens, return_cost=True)
        else:
            logits, *_ = model(tokens)
            cost = 0.0
        loss = F.cross_entropy(logits[:, positions].transpose(1, 2), targets) + coefficient * cost
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.inference_mode():
        tasks = {f"{task}-{gap}": inspect_task(model, task, gap, seed + 2000 + i)
                 for i, (task, gap) in enumerate(EVAL)}
    return {"model": name, "seed": seed, "steps": steps, "batch": batch,
            "cycle_cost": coefficient, "parameters": sum(p.numel() for p in model.parameters()),
            "training_s": elapsed, "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "tasks": tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="penalized")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--batch", type=int, default=12)
    args = parser.parse_args()
    if min(args.steps, args.batch) <= 0 or args.seed < 0:
        parser.error("positive steps/batch and nonnegative seed required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    print(json.dumps(run(args.model, args.seed, args.steps, args.batch), indent=2))


if __name__ == "__main__":
    main()

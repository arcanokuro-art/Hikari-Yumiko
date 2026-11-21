"""Multi-task CPU benchmark. Run each model and seed in a fresh process."""
import argparse
import json
import resource
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

from .hana import HANA0Nano
from .hana02 import HANA02
from .hana_experiment import GRUBaseline
from .model import ZNC0
from .train import make_batch


VOCAB = 35
TASKS = ("cyclic", "recall-short", "recall-long")
MODELS = ("nano", "adaptive", "no-memory", "fixed-1", "fixed-2", "fixed-3", "znc", "gru")


def make_model(name):
    if name == "nano":
        return HANA0Nano(VOCAB, memory_slots=64)
    if name == "znc":
        return ZNC0(VOCAB, 59)
    if name == "gru":
        return GRUBaseline(VOCAB, 49)
    if name.startswith("fixed-"):
        return HANA02(VOCAB, fixed_cycles=int(name[-1]))
    return HANA02(VOCAB, memory_slots=0 if name == "no-memory" else 64)


def make_task(task, batch):
    if task == "cyclic":
        tokens, targets = make_batch(batch, 12, VOCAB)
        return tokens, targets[:, 1:], slice(1, None)
    pairs = {"recall-short": 4, "recall-long": 16}[task]
    keys = torch.rand(batch, 17).argsort(dim=1)[:, :pairs]
    values = torch.randint(17, 34, (batch, pairs))
    order = torch.rand(batch, pairs).argsort(dim=1)
    queries = keys.gather(1, order)
    examples = torch.stack((values, keys), dim=-1).flatten(1)
    marker = torch.full((batch, 1), 34, dtype=torch.long)
    tokens = torch.cat((examples, marker, queries), dim=1)
    return tokens, values.gather(1, order), slice(2 * pairs + 1, None)


def median_latency(model, tokens):
    for _ in range(4):
        model(tokens)
    samples = []
    for _ in range(15):
        start = time.perf_counter()
        model(tokens)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def metrics_for(model, task, seed):
    torch.manual_seed(seed)
    tokens, targets, positions = make_task(task, 96)
    result = model(tokens)
    scored = result[0][:, positions]
    accuracy = scored.argmax(-1).eq(targets).float().mean().item()
    cycles = result[2] if len(result) == 3 else torch.ones_like(tokens)
    histogram = torch.bincount(cycles[:, positions].flatten(), minlength=4)[1:].tolist()
    return {
        "accuracy": accuracy,
        "sequence_length": tokens.size(1),
        "latency_ms": median_latency(model, tokens[:1]),
        "scored_cycle_counts": histogram,
        "mean_scored_cycles": cycles[:, positions].float().mean().item(),
    }


def run(name, seed, steps, batch):
    torch.manual_seed(seed)
    model = make_model(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    torch.manual_seed(seed + 1000)
    start = time.perf_counter()
    for step in range(steps):
        tokens, targets, positions = make_task(TASKS[step % len(TASKS)], batch)
        logits = model(tokens)[0][:, positions]
        loss = F.cross_entropy(logits.transpose(1, 2), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    training = time.perf_counter() - start

    model.eval()
    with torch.inference_mode():
        tasks = {task: metrics_for(model, task, seed + 2000 + index)
                 for index, task in enumerate(TASKS)}
        if name == "adaptive":
            tokens, _, _ = make_task("recall-long", 1)
            paired = {True: [], False: []}
            for fast in (True, False, False, True):
                model.fast_inference = fast
                paired[fast].append(median_latency(model, tokens))
            model.fast_inference = True
            fast_path = {"direct_ms": statistics.mean(paired[True]),
                         "indexed_ms": statistics.mean(paired[False])}
        else:
            fast_path = None
    return {
        "model": name, "seed": seed, "steps": steps, "batch": batch,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_s": training,
        "peak_process_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "tasks": tasks, "paired_batch_one_routing": fast_path,
    }


def profile():
    torch.manual_seed(7)
    model = make_model("nano").eval()
    tokens, _, _ = make_task("recall-long", 1)
    with torch.inference_mode():
        model(tokens)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            for _ in range(5):
                model(tokens)
    rows = sorted(trace.key_averages(), key=lambda row: row.self_cpu_time_total, reverse=True)
    return [{"op": row.key, "self_ms": row.self_cpu_time_total / 1000, "calls": row.count}
            for row in rows[:15]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="adaptive")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch) < 1 or args.seed < 0:
        parser.error("positive steps/batch and nonnegative seed required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    print(json.dumps(profile() if args.profile else run(args.model, args.seed, args.steps, args.batch),
                     indent=2))


if __name__ == "__main__":
    main()

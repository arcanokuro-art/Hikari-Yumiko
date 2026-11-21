"""Train shared backbones, then compare halting policies on matched queries."""
import argparse
import json
import resource
import statistics
import time

import torch

from .hana05 import HANA05
from .hana05_data import counterfactual, make_episode


POLICIES = ("neural", "deterministic", "fixed-1", "fixed-2", "fixed-3", "random")
SPLITS = ("fresh", "relation", "composition", "length", "counterfactual")


def train(seed, steps, use_memory):
    torch.manual_seed(seed)
    model = HANA05()
    generator = torch.Generator().manual_seed(seed + 1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006)
    start = time.perf_counter()
    for step in range(steps):
        episode = make_episode(24, "train", generator)
        query, target = episode.at_depth(1 + step % 3)
        loss = model.training_loss(query, target, use_memory)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model.eval(), time.perf_counter() - start


def evaluate(model, query, target, depth, policy, budgets, use_memory):
    route = "fixed" if policy.startswith("fixed") else policy
    if route == "fixed":
        budgets = torch.full_like(target, int(policy[-1]))
    options = {"budgets": budgets, "use_memory": use_memory}
    logits, cycles, certified = model.infer(query, route, **options)
    sample = query.select(slice(0, 1))
    options["budgets"] = budgets[:1]
    for _ in range(3):
        model.infer(sample, route, **options)
    timings = []
    for _ in range(11):
        start = time.perf_counter()
        model.infer(sample, route, **options)
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "accuracy": logits.argmax(1).eq(target).float().mean().item(),
        "mean_cycles": cycles.float().mean().item(),
        "early_count": cycles.lt(depth).sum().item(),
        "exact_count": cycles.eq(depth).sum().item(),
        "late_count": cycles.gt(depth).sum().item(),
        "terminal_certificate_count": certified.sum().item(),
        "cycle_histogram": cycles.bincount(minlength=4)[1:4].tolist(),
        "latency_ms": statistics.median(timings),
        "queries": target.numel(),
    }


def run(seed, steps, use_memory):
    model, elapsed = train(seed, steps, use_memory)
    generator = torch.Generator().manual_seed(seed + 2000)
    policy_rng = torch.Generator().manual_seed(seed + 3000)
    results = {policy: {} for policy in (POLICIES if use_memory else ("fixed-3",))}
    with torch.inference_mode():
        for split in SPLITS:
            episode = make_episode(192, "fresh" if split == "counterfactual" else split, generator)
            for depth in (1, 2, 3):
                if split == "counterfactual":
                    query, target = counterfactual(episode, depth)
                else:
                    query, target = episode.at_depth(depth)
                budgets = torch.randint(1, 4, target.shape, generator=policy_rng)
                for policy, metrics in results.items():
                    metrics[f"{split}-{depth}"] = evaluate(
                        model, query, target, depth, policy, budgets, use_memory,
                    )
    return {
        "seed": seed, "steps": steps, "memory": use_memory,
        "training_s": elapsed, "parameters": sum(p.numel() for p in model.parameters()),
        "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "policies": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--no-memory", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.seed < 0:
        parser.error("positive steps and nonnegative seed required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    print(json.dumps(run(args.seed, args.steps, not args.no_memory), indent=2))


if __name__ == "__main__":
    main()

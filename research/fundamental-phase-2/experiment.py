"""CPU comparisons under identical environment, actions and posterior state."""
import argparse
from collections import Counter
from dataclasses import replace
import json
import platform
import resource
import statistics
import time

import numpy as np

from controller import Controller, examples
from environment import (ACTIONS, PRICE, Memory, Profile, initial, legal, planner, profile,
                         realized_reward, terminal_values, world)

POLICIES = ("neural", "myopic", "deterministic", "random", "fixed", "oracle",
            "no-cost", "one-step", "no-memory")


def policy(name, controller, rng):
    def choose(s, p):
        stop = int(np.argmax(terminal_values(s, p)))
        feasible = legal(s, p)
        if name in ("neural", "no-cost", "one-step"):
            return controller.select(s, p)
        if name in ("myopic", "oracle"):
            return int(np.argmax(planner(p)(s, 1 if name == "myopic" else 3)))
        if name == "random":
            # Each terminal decision shares the same Bayes readout.
            return int(rng.choice([stop, *[i for i in range(3, 6) if feasible[i]]]))
        if name == "no-memory":
            return stop
        if name == "deterministic" and max(terminal_values(s, p)) >= .78:
            return stop
        order = (3, 4, 5) if name == "fixed" else (5, 3, 4)
        return next((action for action in order if feasible[action]), stop)
    return choose


def rollout(p, reports, choose):
    s = initial(p)
    memory = Memory(reports)
    for _ in range(4):
        action = choose(s, p)
        if not legal(s, p)[action]:
            raise AssertionError("policy selected an illegal operation")
        if action < 3:
            return action, memory.reads, p.budget - s.remaining
        s = memory.acquire(s, action, p)
    raise AssertionError("episode did not terminate")


def evaluate(cases, choose):
    actions = Counter()
    correct, reward, costs, abstentions, reads = [], [], [], [], []
    latencies = []
    for j, (p, label, reports) in enumerate(cases):
        start = time.perf_counter_ns()
        answer, trace, cost = rollout(p, reports, choose)
        elapsed = (time.perf_counter_ns() - start) / 1000
        if j >= 16:
            latencies.append(elapsed)
        actions.update(trace)
        actions.update([answer])
        correct.append(answer == label)
        reward.append(realized_reward(answer, label) - PRICE * cost)
        costs.append(cost)
        abstentions.append(answer == 2)
        reads.append(len(trace))
    return {"accuracy": statistics.mean(correct), "utility": statistics.mean(reward),
            "cost_units": statistics.mean(costs), "abstention": statistics.mean(abstentions),
            "accesses": statistics.mean(reads), "action_counts": dict(zip(
                ACTIONS, (actions[i] for i in range(6)))),
            "median_us": statistics.median(latencies),
            "p95_us": float(np.quantile(latencies, .95)), "n": len(cases)}


def counterfactual(cases, choose):
    flips, correct, eligible = 0, 0, 0
    for p, _, _ in cases:
        alternative = replace(p, costs=(4, p.costs[1], p.costs[2])) if p.costs[0] != 4 else replace(p, costs=(1, *p.costs[1:]))
        before, after = choose(initial(p), p), choose(initial(alternative), alternative)
        oracle_before = int(np.argmax(planner(p)(initial(p), 3)))
        oracle_after = int(np.argmax(planner(alternative)(initial(alternative), 3)))
        flips += before != after
        eligible += oracle_before != oracle_after
        correct += oracle_before != oracle_after and before == oracle_before and after == oracle_after
    return {"action_change_rate": flips / len(cases), "eligible_pairs": eligible,
            "oracle_pair_agreement": correct / eligible if eligible else None}


def operation_timings():
    p = Profile((.4, .4, .2), (.7, .9), (.1, .1), (1, 1, 2), 5)
    memory = Memory((0, 1))
    root = initial(p)
    indexed = memory.acquire(root, 4, p)
    timing = {}
    for action, s in ((3, root), (4, root), (5, indexed)):
        blocks = []
        for _ in range(10):
            start = time.perf_counter_ns()
            for _ in range(256):
                memory.acquire(s, action, p)
            blocks.append((time.perf_counter_ns() - start) / 256000)
            memory.reads.clear()
        timing[ACTIONS[action]] = statistics.median(blocks)
    return timing


def run(seed, steps, count):
    train_rng = np.random.default_rng(seed + 10000)
    val_rng = np.random.default_rng(seed + 20000)
    train = [profile(train_rng, "train") for _ in range(1200)]
    val = [profile(val_rng, "train") for _ in range(200)]
    fitted, losses, train_seconds = {}, {}, {}
    for name, horizon in (("neural", 3), ("no-cost", 3), ("one-step", 1)):
        rng = np.random.default_rng(seed)
        fitted[name] = Controller(rng, no_cost=name == "no-cost")
        timer = time.perf_counter()
        data = examples(train, horizon, name == "no-cost")
        validation = examples(val, horizon, name == "no-cost")
        losses[name] = fitted[name].fit(data, rng, steps, validation)
        train_seconds[name] = time.perf_counter() - timer
    results, pairs = {}, {}
    for i, split in enumerate(("id", "ood", "unhelpful")):
        rng = np.random.default_rng(seed + 30000 + 1000 * i)
        cases = []
        for _ in range(count):
            p = profile(rng, split)
            if split == "unhelpful":
                p = replace(p, fidelity=(.5, .5))
            label, reports = world(p, rng.random(), rng.random(2))
            cases.append((p, label, reports))
        # Rotate ordering by seed to reduce systematic timing order effects.
        ordered = POLICIES[seed % len(POLICIES):] + POLICIES[:seed % len(POLICIES)]
        results[split] = {}
        for name in ordered:
            choose = policy(name, fitted.get(name), np.random.default_rng(seed + 40000 + i))
            results[split][name] = evaluate(cases, choose)
            if split == "id" and name != "random":
                pairs[name] = counterfactual(cases[:128], choose)
    return {"seed": seed, "steps": steps, "validation_mse": losses, "training_s": train_seconds,
            "learned_parameters": sum(x.size for x in fitted["neural"].params),
            "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "operator_median_us": operation_timings(), "neural_macs_per_decision": 20 * 32 + 32 * 6,
            "machine": platform.machine(), "numpy": np.__version__, "test": results,
            "counterfactual": pairs, "energy_joules": None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--count", type=int, default=1024)
    args = parser.parse_args()
    if args.steps < 1 or args.count <= 16 or args.seed < 0:
        parser.error("positive steps, at least 17 episodes and nonnegative seed required")
    print(json.dumps(run(args.seed, args.steps, args.count), indent=2))

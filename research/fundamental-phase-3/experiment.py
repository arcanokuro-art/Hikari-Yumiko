"""Matched budget sweep, latency, decision audit and counterfactuals."""
import argparse
from collections import Counter
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np

from residual import PHASE2, Controller, Residual, examples, gate
from controller import examples as absolute_examples
from environment import ACTIONS, PRICE, Memory, initial, legal, planner, profile, world

spec = importlib.util.spec_from_file_location("phase2_benchmark", PHASE2 / "experiment.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)

SEEDS = (7, 19, 43, 71, 101)
BUDGETS = (3, 5, 7)
SCENARIOS = ("id", "ood", "noise", "incomplete", "cost-shift", "unhelpful")
POLICIES = ("residual", "p2", "myopic", "fixed", "deterministic", "random", "oracle",
            "always-residual", "no-cost-residual", "two-step")


def audit(cases, choose, mark_gate=False):
    counts = Counter()
    regret = 0.
    for p, _, reports in cases:
        s, mem, plan = initial(p), Memory(reports), planner(p)
        while True:
            a = choose(s, p)
            q = plan(s, 3)
            stop = max(q[:3])
            best = max(q)
            difference = best - q[a]
            regret += difference
            counts["decisions"] += 1
            counts["positive_regret"] += difference > 1e-9
            if mark_gate:
                counts["gated_network_calls"] += gate(s, p)
            if a < 3:
                counts["premature_stop"] += best - stop > 1e-9
                break
            counts["acquisitions"] += 1
            counts["unnecessary_" + ACTIONS[a]] += stop - q[a] > 1e-9
            counts["suboptimal_" + ACTIONS[a]] += difference > 1e-9
            s = mem.acquire(s, a, p)
    return {**counts, "regret_per_decision": regret / counts["decisions"]}


def environment_cases(seed, scenario, budget, count):
    # Except for joint OOD, recreate the same worlds and random draws for
    # each intervention. The budget and prices cannot affect reports.
    rng = np.random.default_rng(seed + 50000)
    cases = []
    for _ in range(count):
        p = profile(rng, "ood" if scenario == "ood" else "id")
        p = replace(p, budget=budget)
        if scenario == "noise":
            p = replace(p, fidelity=(.53, .55))
        elif scenario == "incomplete":
            p = replace(p, erasure=(.10, .95))
        elif scenario == "cost-shift":
            p = replace(p, costs=(p.costs[0] + 2, p.costs[1], p.costs[2] + 1))
        elif scenario == "unhelpful":
            p = replace(p, fidelity=(.5, .5))
        label, reports = world(p, rng.random(), rng.random(2))
        cases.append((p, label, reports))
    return cases


def native_cases(seed, scenario, count):
    i = ("id", "ood", "unhelpful").index(scenario)
    rng = np.random.default_rng(seed + 30000 + 1000 * i)
    cases = []
    for _ in range(count):
        p = profile(rng, scenario)
        if scenario == "unhelpful":
            p = replace(p, fidelity=(.5, .5))
        label, reports = world(p, rng.random(), rng.random(2))
        cases.append((p, label, reports))
    return cases


def policy(name, models, seed):
    if name in ("residual", "always-residual", "no-cost-residual"):
        model = models["no-cost-residual" if name == "no-cost-residual" else "residual"]
        return lambda s, p: model.choose(s, p, gated=name != "always-residual")
    if name == "two-step":
        return lambda s, p: int(np.argmax(planner(p)(s, 2)))
    return previous.policy("neural" if name == "p2" else name, models.get("p2"),
                           np.random.default_rng(seed))


def run(seed, steps, count):
    training = np.random.default_rng(seed + 10000)
    validation = np.random.default_rng(seed + 20000)
    train = [profile(training, "train") for _ in range(1200)]
    val = [profile(validation, "train") for _ in range(200)]
    models, fit = {}, {}
    for name in ("p2", "residual", "no-cost-residual"):
        rng = np.random.default_rng(seed)
        start = time.perf_counter()
        if name == "p2":
            model = Controller(rng)
            a, b = absolute_examples(train, 3, False), absolute_examples(val, 3, False)
        else:
            stripped = name == "no-cost-residual"
            model = Residual(rng, stripped)
            a, b = examples(train, stripped), examples(val, stripped)
        loss = model.fit(a, rng, steps, b)
        fit[name] = {"validation_mse": loss, "seconds": time.perf_counter() - start,
                     "parameters": sum(x.size for x in model.params),
                     "macs_per_network_call": sum(x.size for x in model.params[::2]),
                     "train_states": len(a[0]), "validation_states": len(b[0])}
        models[name] = model

    comparisons, diagnostics, counterfactuals = {}, {}, {}
    native = {}
    for scenario in ("id", "ood", "unhelpful"):
        cases = native_cases(seed, scenario, count)
        native[scenario] = {name: audit(cases, policy(name, models, seed))
                            for name in ("p2", "myopic", "residual")}
    for index, scenario in enumerate(SCENARIOS):
        comparisons[scenario], diagnostics[scenario] = {}, {}
        for budget in BUDGETS:
            cases = environment_cases(seed, scenario, budget, count)
            rows, decisions = {}, {}
            ordered = POLICIES[seed % len(POLICIES):] + POLICIES[:seed % len(POLICIES)]
            for name in ordered:
                rng_seed = seed + 60000 + index
                choose = policy(name, models, rng_seed)
                rows[name] = previous.evaluate(cases, choose)
                decisions[name] = audit(cases, policy(name, models, rng_seed),
                                        name in ("residual", "no-cost-residual"))
                if scenario == "id" and budget == 5 and name != "random":
                    counterfactuals[name] = previous.counterfactual(cases[:128], choose)
            comparisons[scenario][str(budget)] = rows
            diagnostics[scenario][str(budget)] = decisions
    return {"seed": seed, "steps": steps, "episodes": count, "fit": fit,
            "test": comparisons, "audit": diagnostics, "native_audit": native,
            "counterfactual": counterfactuals, "machine": platform.machine(),
            "numpy": np.__version__, "rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "energy_joules": None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--count", type=int, default=1024)
    args = parser.parse_args()
    if args.steps < 1 or args.count <= 16 or args.seed < 0:
        parser.error("positive steps, at least 17 episodes and nonnegative seed required")
    print(json.dumps(run(args.seed, args.steps, args.count), indent=2, allow_nan=False))

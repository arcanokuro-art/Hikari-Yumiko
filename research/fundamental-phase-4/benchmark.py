"""Matched experiments with a learned advantage over the option to stop."""
import argparse
from collections import Counter
from dataclasses import replace
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

import numpy as np

P3 = Path(__file__).resolve().parents[1] / "fundamental-phase-3"
sys.path.append(str(P3))
spec = importlib.util.spec_from_file_location("phase3_benchmark", P3 / "experiment.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
from controller import features
from environment import (ACTIONS, PRICE, Memory, belief, initial, legal, outcomes,
                         planner, profile, terminal_values)
from residual import Residual, examples as residual_examples

SEEDS = prior.SEEDS
SCENARIOS = (*prior.SCENARIOS, "two-required")
BUDGETS = prior.BUDGETS
POLICIES = ("net", "no-margin", "absolute", "residual", "myopic", "two-step",
            "oracle", "fixed", "deterministic", "stop", "local-once", "random")
MARGINS = (0., .005, .01, .02, .035, .05, .075)
EPS = 1e-9


class NetPolicy(Residual):
    """Predict signed gain directly, with an exact shared terminal readout."""

    def __init__(self, rng, centered=True):
        super().__init__(rng)
        self.centered = centered

    def decide(self, s, p, margin=0.):
        mask = legal(s, p)
        acquired = mask[3:]
        if not any(acquired):
            return max(range(3), key=terminal_values(s, p).__getitem__)
        x = features(s, p)
        a, b, unknown = x[:3]
        stop_values = (a, b, unknown + .25 * (a + b))
        stop = max(range(3), key=stop_values.__getitem__)
        scores = self.forward(np.asarray(x))[1]
        if not self.centered:
            scores -= stop_values[stop]
        best = max((i for i in range(3) if acquired[i]), key=scores.__getitem__)
        return best + 3 if scores[best] > margin else stop


def target_data(profiles):
    x, gains, values, masks = [], [], [], []
    for p in profiles:
        plan = planner(p)
        pending, seen = [initial(p)], set()
        while pending:
            s = pending.pop()
            if s in seen:
                continue
            seen.add(s)
            legal_ops = legal(s, p)[3:]
            if not any(legal_ops):
                continue
            q = plan(s, 3)
            stop = max(q[:3])
            x.append(features(s, p))
            gains.append([q[i] - stop if legal_ops[i - 3] else 0. for i in range(3, 6)])
            values.append([q[i] if legal_ops[i - 3] else 0. for i in range(3, 6)])
            masks.append(legal_ops)
            for i, allowed in enumerate(legal_ops, 3):
                if allowed:
                    pending.extend(child for prob, child in outcomes(s, i, p) if prob > 0)
    return tuple(map(np.asarray, (x, gains, values, masks)))


def policy_value(p, choose):
    @lru_cache(None)
    def value(s):
        action = choose(s, p)
        if action < 3:
            return terminal_values(s, p)[action]
        return sum(prob * value(child) for prob, child in outcomes(s, action, p)
                   if prob > 0) - PRICE * p.costs[action - 3]

    return value(initial(p))


def fit_models(seed, steps):
    train_rng, val_rng = (np.random.default_rng(seed + n) for n in (10000, 20000))
    train = [profile(train_rng, "train") for _ in range(1200)]
    val = [profile(val_rng, "train") for _ in range(200)]
    start = time.perf_counter()
    training, validation = target_data(train), target_data(val)
    common_label_seconds = time.perf_counter() - start
    models, metrics = {}, {}
    for name, centered in (("net", True), ("absolute", False)):
        start = time.perf_counter()
        rng = np.random.default_rng(seed)
        model = NetPolicy(rng, centered)
        target = 1 if centered else 2
        fit = lambda data: (data[0], data[target], data[3])
        mse = model.fit(fit(training), rng, steps, fit(validation))
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        scores = {margin: statistics.mean(policy_value(p, lambda s, p: model.decide(s, p, margin))
                                          for p in val) for margin in MARGINS}
        threshold = max(MARGINS, key=scores.__getitem__)
        calibrate_seconds = time.perf_counter() - start
        models[name] = model, threshold
        metrics[name] = {"validation_mse": mse, "fit_seconds": fit_seconds,
                         "label_seconds_shared": common_label_seconds,
                         "calibration_seconds": calibrate_seconds,
                         "validation_margin_utilities": scores, "margin": threshold,
                         "parameters": sum(w.size for w in model.params),
                         "macs_per_call": sum(w.size for w in model.params[::2]),
                         "train_states": len(training[0]), "validation_states": len(validation[0])}
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    residual = Residual(rng)
    mse = residual.fit(residual_examples(train), rng, steps, residual_examples(val))
    metrics["residual"] = {"validation_mse": mse, "fit_and_labels_seconds": time.perf_counter() - start,
                           "parameters": sum(w.size for w in residual.params),
                           "macs_per_call": sum(w.size for w in residual.params[::2])}
    models["residual"] = residual
    return models, metrics


def cases_for(seed, scenario, budget, count):
    if scenario != "two-required":
        return prior.environment_cases(seed, scenario, budget, count)
    # Report random draws are shared with other budgets for this generator.
    rng = np.random.default_rng(seed + 70000)
    return [(p, *prior.world(p, rng.random(), rng.random(2)))
            for p in (replace(profile(rng, "id"), prior=(.46, .46, .08),
                              fidelity=(.50, .99), erasure=(.10, .03),
                              costs=(2, 1, 1), budget=budget) for _ in range(count))]


def best_stop(s, p):
    return max(range(3), key=terminal_values(s, p).__getitem__)


def get_policy(name, models, seed):
    if name in ("net", "absolute", "no-margin"):
        model, margin = models["absolute" if name == "absolute" else "net"]
        return lambda s, p: model.decide(s, p, 0. if name == "no-margin" else margin)
    if name == "stop":
        return best_stop
    if name == "local-once":
        return lambda s, p: 3 if legal(s, p)[3] else best_stop(s, p)
    return prior.policy(name, models, seed)


def diagnose(cases, choose):
    counts = Counter()
    regret = 0.
    for p, _, reports in cases:
        state, memory, plan = initial(p), Memory(reports), planner(p)
        root = plan(state, 3)
        immediate = max(root[:3])
        difficulty = ("stop" if max(root) <= immediate + EPS else
                      "one" if max(plan(state, 1)) >= max(root) - EPS else "multiple")
        counts["tasks_" + difficulty] += 1
        while True:
            action = choose(state, p)
            q = plan(state, 3)
            stop = max(q[:3])
            regret += max(q) - q[action]
            counts["decisions"] += 1
            if action < 3:
                counts["premature"] += max(q) > stop + EPS
                counts["immediate_stop"] += not memory.reads
                break
            gain = q[action] - stop
            category = "useful" if gain > EPS else "harmful" if gain < -EPS else "redundant"
            op = ACTIONS[action]
            counts[f"{category}_{op}"] += 1
            counts["suboptimal"] += max(q) - q[action] > EPS
            price = p.costs[action - 3]
            counts[f"units_{op}"] += price
            counts[f"high_price_{op}"] += price >= 3
            state = memory.acquire(state, action, p)
    return {**counts, "regret_per_decision": regret / counts["decisions"]}


def selector_timing(cases, choose):
    """Instrument separately so decision clocks cannot inflate rollout latency."""
    elapsed, calls = 0, 0
    for p, _, reports in cases[16:]:
        state, mem = initial(p), Memory(reports)
        while True:
            start = time.perf_counter_ns()
            action = choose(state, p)
            elapsed += time.perf_counter_ns() - start
            calls += 1
            if action < 3:
                break
            state = mem.acquire(state, action, p)
    return {"selector_mean_us_per_episode": elapsed / (len(cases) - 16) / 1000,
            "selector_mean_us_per_call": elapsed / calls / 1000}


def run(seed, steps, count):
    models, fit = fit_models(seed, steps)
    test, audit, cf = {}, {}, {}
    for i, scenario in enumerate(SCENARIOS):
        test[scenario], audit[scenario] = {}, {}
        for budget in BUDGETS:
            cases = cases_for(seed, scenario, budget, count)
            rows, diagnostics = {}, {}
            offset = seed % len(POLICIES)
            for name in POLICIES[offset:] + POLICIES[:offset]:
                chooser = lambda: get_policy(name, models, seed + 60000 + i)
                rows[name] = prior.previous.evaluate(cases, chooser())
                rows[name]["memory_payload_reads"] = (
                    rows[name]["action_counts"]["local"] + rows[name]["action_counts"]["verify"]) / count
                rows[name].update(selector_timing(cases, chooser()))
                diagnostics[name] = diagnose(cases, chooser())
                if scenario in ("id", "ood") and budget == 5 and name != "random":
                    cf.setdefault(scenario, {})[name] = prior.previous.counterfactual(cases[:128], chooser())
            test[scenario][budget], audit[scenario][budget] = rows, diagnostics
    return {"seed": seed, "steps": steps, "episodes": count, "test": test, "audit": audit,
            "fit": fit, "counterfactual": cf, "machine": platform.machine(), "numpy": np.__version__,
            "python": platform.python_version(),
            "rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "energy_joules": None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--count", type=int, default=1024)
    args = parser.parse_args()
    if args.seed < 0 or args.steps < 1 or args.count < 17:
        parser.error("nonnegative seed, positive steps and at least 17 episodes required")
    print(json.dumps(run(args.seed, args.steps, args.count), allow_nan=False, separators=(",", ":")))

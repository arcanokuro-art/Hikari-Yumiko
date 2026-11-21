"""Independent training, validation and test streams for uncertain reasoning."""
import argparse
import json
import resource
import statistics
import time

import torch
from torch.nn import functional as F

from .hana06 import HANA06, baseline
from .hana06_data import UNKNOWN, make_batch


VARIANTS = ("shared", "no-confidence", "no-memory", "recurrent")
SCENARIOS = (
    ("low", "fresh", 0.10, False), ("high", "fresh", 0.30, False),
    ("extreme", "fresh", 0.55, False), ("correlated", "fresh", 0.30, True),
    ("relation", "relation", 0.30, False), ("composition", "composition", 0.30, False),
    ("length", "length", 0.30, False),
)


def dispatch(model, observation, policy, threshold, variant):
    if policy in ("deterministic", "probabilistic"):
        return baseline(observation, policy, threshold)
    return model.infer(observation, policy, threshold, use_memory=variant != "no-memory",
                       use_confidence=variant != "no-confidence")


def train(seed, variant, steps):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1000)
    model = HANA06()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    start = time.perf_counter()
    for step in range(steps):
        x, y, _ = make_batch(32, "train", (0.10, 0.30)[step % 2], generator)
        loss = model.loss(x, y, use_memory=variant != "no-memory",
                          use_confidence=variant != "no-confidence",
                          adaptive=variant in ("shared", "no-confidence"))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    return model.eval(), time.perf_counter() - start


def validate(model, policy, batches, variant):
    options = {"adaptive": (0.30, 0.45, 0.55, 0.65, 0.8),
               "probabilistic": (0.55, 0.65, 0.75, 0.85, 0.95)}.get(policy, (0.5,))
    best = None
    for threshold in options:
        outputs = [dispatch(model, x, policy, threshold, variant) for x, _, _ in batches]
        correct = torch.cat([pred.eq(y) for (pred, _, _), (_, y, _) in zip(outputs, batches)]).float()
        cycles = torch.cat([steps for _, _, steps in outputs]).float()
        confidence = torch.cat([score for _, score, _ in outputs])
        utility = (correct - 0.02 * (cycles - 1)).mean().item()
        if best is None or utility > best[0]:
            best = utility, threshold, correct, confidence
    _, threshold, correct, confidence = best
    logits = confidence.clamp(1e-4, 1 - 1e-4).logit()
    temperature = min((0.5, 0.75, 1., 1.25, 1.5, 2., 3., 4.),
                      key=lambda t: F.binary_cross_entropy_with_logits(logits / t, correct).item())
    return threshold, temperature


def fraction(values, mask):
    return values[mask].float().mean().item() if mask.any() else None


def metrics(pred, confidence, steps, y, depth, temperature):
    correct = pred.eq(y)
    confidence = (confidence.clamp(1e-4, 1 - 1e-4).logit() / temperature).sigmoid()
    ece = 0.
    bins = (confidence * 10).long().clamp_max(9)
    for i in range(10):
        bucket = bins.eq(i)
        if bucket.any():
            ece += bucket.float().mean().item() * abs(
                confidence[bucket].mean().item() - correct[bucket].float().mean().item())
    answered = pred.ne(UNKNOWN)
    return {
        "accuracy": correct.float().mean().item(),
        "accuracy_by_depth_0_to_3": [fraction(correct, depth == d) for d in range(4)],
        "cycles_by_depth_0_to_3": [fraction(steps, depth == d) for d in range(4)],
        "count_by_depth_0_to_3": depth.bincount(minlength=4).tolist(),
        "coverage": answered.float().mean().item(),
        "selective_accuracy": fraction(correct, answered),
        "unknown_recall": fraction(pred.eq(UNKNOWN), y.eq(UNKNOWN)),
        "false_abstention": fraction(pred.eq(UNKNOWN), y.ne(UNKNOWN)),
        "unknown_precision": fraction(y.eq(UNKNOWN), pred.eq(UNKNOWN)),
        "ece": ece,
        "brier_top_label": (confidence - correct.float()).square().mean().item(),
        "cycles": steps.float().mean().item(),
        "cycle_histogram": steps.bincount(minlength=4)[1:].tolist(),
    }


def run(seed, variant, steps):
    model, elapsed = train(seed, variant, steps)
    validation = torch.Generator().manual_seed(seed + 2000)
    val = [make_batch(384, "fresh", noise, validation) for noise in (0.10, 0.30)]
    policies = {
        "shared": ("adaptive", "deterministic", "probabilistic", "fixed-1", "fixed-2", "fixed-3"),
        "no-confidence": ("adaptive",), "no-memory": ("fixed-3",), "recurrent": ("fixed-3",),
    }[variant]
    results = {}
    with torch.inference_mode():
        tuned = {p: validate(model, p, val, variant) for p in policies}
        low_cycles = {}
        for scenario, (name, split, noise, correlated) in enumerate(SCENARIOS):
            # Compare corruption changes on the same graphs, queries, masks
            # and corruption draws. Structural OOD uses separate streams.
            test = torch.Generator().manual_seed(seed + 3000 + (scenario if scenario >= 4 else 0))
            x, y, depth = make_batch(512, split, noise, test, correlated=correlated)
            results[name] = {}
            for policy, (threshold, temperature) in tuned.items():
                pred, confidence, cycles = dispatch(model, x, policy, threshold, variant)
                result = metrics(pred, confidence, cycles, y, depth, temperature)
                if name == "low":
                    low_cycles[policy] = cycles
                elif scenario < 4:
                    result["paired_more_cycles_than_low"] = cycles.gt(low_cycles[policy]).float().mean().item()
                    result["paired_fewer_cycles_than_low"] = cycles.lt(low_cycles[policy]).float().mean().item()
                for _ in range(3):
                    dispatch(model, x.select(slice(0, 1)), policy, threshold, variant)
                timings = []
                for i in range(15):
                    sample = x.select(slice(i, i + 1))
                    start = time.perf_counter()
                    dispatch(model, sample, policy, threshold, variant)
                    timings.append((time.perf_counter() - start) * 1000)
                result["latency_ms"] = statistics.median(timings)
                results[name][policy] = result

        cf_generator = torch.Generator().manual_seed(seed + 3100)
        x, y, _, altered, altered_y, _ = make_batch(512, "fresh", 0.30, cf_generator, paired=True)
        changed = y.ne(altered_y)
        pairs = {}
        for policy, (threshold, _) in tuned.items():
            before = dispatch(model, x, policy, threshold, variant)[0]
            after = dispatch(model, altered, policy, threshold, variant)[0]
            pairs[policy] = {
                "changed_targets": changed.sum().item(),
                "paired_accuracy": fraction(before.eq(y) & after.eq(altered_y), changed),
                "prediction_changed": fraction(before.ne(after), changed),
                "unchanged_target_stability": fraction(before.eq(after), ~changed),
            }
    return {"seed": seed, "variant": variant, "steps": steps, "training_s": elapsed,
            "parameters": sum(p.numel() for p in model.parameters()),
            "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "validation_threshold_temperature": tuned, "test": results, "counterfactual": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--variant", choices=VARIANTS, default="shared")
    parser.add_argument("--steps", type=int, default=650)
    args = parser.parse_args()
    if args.steps < 1 or args.seed < 0:
        parser.error("positive steps and nonnegative seed required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    print(json.dumps(run(args.seed, args.variant, args.steps), indent=2))


if __name__ == "__main__":
    main()

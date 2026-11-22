"""Score the shared posterior before and after learned acquisition."""
import argparse
from collections import Counter
import json
from statistics import mean

from audit import SCENARIOS, cases_for, fit_broad, p4
from environment import Memory, belief, initial


def calibration(cases, choose):
    bins = [Counter() for _ in range(10)]
    losses = []
    for p, label, reports in cases:
        s, memory = initial(p), Memory(reports)
        if choose is not None:
            while (action := choose(s, p)) >= 3:
                s = memory.acquire(s, action, p)
        posterior = belief(s, p)
        predicted = max(range(3), key=posterior.__getitem__)
        confidence = posterior[predicted]
        losses.append(sum((p_i - (i == label)) ** 2 for i, p_i in enumerate(posterior)))
        cell = bins[min(9, int(10 * confidence))]
        cell['n'] += 1
        cell['confidence'] += confidence
        cell['correct'] += int(predicted == label)
    gap = sum(abs(b['confidence'] - b['correct']) for b in bins) / len(cases)
    return {'brier': float(mean(losses)), 'ece': float(gap), 'bins': bins}


def run(seed):
    net = p4.fit_models(seed, 1800)[0]['net']
    broad = fit_broad(seed, 1800)[0]
    results = {}
    for scenario in SCENARIOS:
        cases = cases_for(seed, scenario, 5, 1024)
        results[scenario] = {'initial': calibration(cases, None)}
        for name, (model, margin) in (('net', net), ('broad', broad)):
            results[scenario][name] = calibration(cases, lambda s,p: model.decide(s,p,margin))
    return {'seed':seed, 'budget':5, 'episodes':1024, 'test':results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    print(json.dumps(run(parser.parse_args().seed), default=float, allow_nan=False))

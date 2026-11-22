"""Binary knowledge queries with a third, unanswerable outcome."""
from dataclasses import dataclass
from functools import lru_cache
from math import inf

import numpy as np

ACTIONS = ("answer-0", "answer-1", "abstain", "local", "index", "verify")
TERMINAL = 3
PRICE = 0.035


@dataclass(frozen=True)
class Profile:
    prior: tuple
    fidelity: tuple
    erasure: tuple
    costs: tuple
    budget: int


@dataclass(frozen=True)
class State:
    reports: tuple = (-1, -1)
    indexed: bool = False
    remaining: int = 0


def initial(profile):
    return State(remaining=profile.budget)


def legal(s, p):
    local, index, verify = p.costs
    return (True, True, True, s.reports[0] == -1 and local <= s.remaining,
            not s.indexed and index + verify <= s.remaining,
            s.indexed and s.reports[1] == -1 and verify <= s.remaining)


def likelihood(p, channel, report, label):
    e = p.erasure[channel]
    if report == 2:
        return e
    if label == 2:
        return (1 - e) * 0.5
    f = p.fidelity[channel]
    return (1 - e) * (f if label == report else 1 - f)


def belief(s, p):
    scores = list(p.prior)
    for channel, report in enumerate(s.reports):
        if report >= 0:
            scores = [w * likelihood(p, channel, report, label)
                      for label, w in enumerate(scores)]
    norm = sum(scores)
    return tuple(w / norm for w in scores)


def transition(s, action, p, result=None):
    if action < TERMINAL or not legal(s, p)[action]:
        raise ValueError("unavailable transition")
    remaining = s.remaining - p.costs[action - TERMINAL]
    if action == 4:
        return State(s.reports, True, remaining)
    channel = int(action == 5)
    if result not in (0, 1, 2):
        raise ValueError("missing acquired report")
    reports = list(s.reports)
    reports[channel] = result
    return State(tuple(reports), s.indexed, remaining)


def outcomes(s, action, p):
    if action == 4:
        yield 1., transition(s, action, p)
        return
    weights = belief(s, p)
    channel = int(action == 5)
    for report in range(3):
        probability = sum(w * likelihood(p, channel, report, y) for y, w in enumerate(weights))
        yield probability, transition(s, action, p, report)


def terminal_values(s, p):
    a, b, unknown = belief(s, p)
    return a, b, unknown + 0.25 * (a + b)


def values(s, p, horizon, cache):
    result = list(terminal_values(s, p))
    for action in range(3, 6):
        if not horizon or not legal(s, p)[action]:
            result.append(-inf)
            continue
        score = sum(prob * max(cache(child, horizon - 1))
                    for prob, child in outcomes(s, action, p))
        result.append(score - PRICE * p.costs[action - 3])
    return tuple(result)


def planner(p):
    @lru_cache(None)
    def q(s, horizon):
        return values(s, p, horizon, q)
    return q


def realized_reward(answer, label):
    return 1. if answer == label else 0.25 if answer == 2 else 0.


class Memory:
    """Episode-private channel; a controller receives only State and Profile."""

    def __init__(self, reports):
        self._reports = tuple(reports)
        self.reads = []

    def acquire(self, s, action, p):
        # Validation happens before the channel is addressed.
        if action < TERMINAL or not legal(s, p)[action]:
            raise ValueError("cannot acquire unavailable evidence")
        report = None if action == 4 else self._reports[int(action == 5)]
        next_state = transition(s, action, p, report)
        self.reads.append(action)
        return next_state


def profile(rng, split):
    unknown = rng.uniform(.08, .48)
    prior = rng.uniform(.22, .78)
    local, verify = rng.uniform(.53, .87), rng.uniform(.76, .98)
    costs = tuple(int(x) for x in rng.integers(1, 4, 3))
    erasure = tuple(float(x) for x in rng.uniform(.03, .22, 2))
    if split == "ood":
        local, verify = rng.uniform(.76, .93), rng.uniform(.53, .74)
        erasure = tuple(float(x) for x in rng.uniform(.25, .45, 2))
        costs = tuple(int(x) for x in rng.integers(2, 5, 3))
    return Profile(((1 - unknown) * prior, (1 - unknown) * (1 - prior), unknown),
                   (local, verify), erasure, costs, int(rng.integers(3, 8)))


def world(p, label_draw, channel_draws):
    label = int(np.searchsorted(np.cumsum(p.prior), label_draw))
    reports = tuple(int(np.searchsorted(
        np.cumsum([likelihood(p, k, r, label) for r in range(3)]), draw))
        for k, draw in enumerate(channel_draws))
    return label, reports

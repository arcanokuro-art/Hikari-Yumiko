"""Public-state marginal gain relative to stopping and one-step planning."""
from pathlib import Path
import sys

import numpy as np

PHASE2 = Path(__file__).resolve().parents[1] / "fundamental-phase-2"
sys.path.append(str(PHASE2))
from controller import Controller, features
from environment import initial, legal, outcomes, planner


def gate(s, p):
    mask = legal(s, p)
    # Index + verification, or two still unread independent channels.
    return mask[4] or (mask[3] and mask[5] and p.costs[0] + p.costs[2] <= s.remaining)


def examples(profiles, no_cost=False):
    x, y, masks = [], [], []
    for p in profiles:
        plan = planner(p)
        queue, seen = [initial(p)], set()
        while queue:
            s = queue.pop()
            if s in seen:
                continue
            seen.add(s)
            mask = legal(s, p)[3:]
            one, three = plan(s, 1), plan(s, 3)
            x.append(features(s, p, no_cost))
            y.append([three[a] - one[a] if mask[a - 3] else 0. for a in range(3, 6)])
            masks.append(mask)
            for a in range(3, 6):
                if mask[a - 3]:
                    queue.extend(child for _, child in outcomes(s, a, p))
    return np.asarray(x), np.asarray(y), np.asarray(masks)


class Residual(Controller):
    def __init__(self, rng, no_cost=False):
        super().__init__(rng, no_cost=no_cost)
        self.params[2] = self.params[2][:, :3].copy()
        self.params[3] = self.params[3][:3].copy()

    def choose(self, s, p, gated=True):
        base = np.asarray(planner(p)(s, 1))
        if any(legal(s, p)[3:]) and (not gated or gate(s, p)):
            x = np.asarray(features(s, p, self.no_cost))
            delta = self.forward(x)[1]
            # Under nested horizons, the additional continuation value
            # cannot be negative. Infeasible actions remain at -infinity.
            base[3:] += np.maximum(delta, 0.)
        return int(base.argmax())

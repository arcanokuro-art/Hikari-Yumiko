"""A small action-value network fitted to model-based targets."""
import numpy as np

from environment import belief, initial, legal, outcomes, planner, terminal_values


def features(s, p, no_cost=False):
    return (*belief(s, p), *p.fidelity, *p.erasure,
            *((0., 0., 0., 0.) if no_cost else (*[c / 7 for c in p.costs], s.remaining / 7)),
            *(float(s.reports[0] == r) for r in (-1, 0, 1, 2)),
            *(float(s.reports[1] == r) for r in (-1, 0, 1, 2)), float(s.indexed))


def examples(profiles, horizon, no_cost):
    x, y, masks = [], [], []
    for p in profiles:
        q = planner(p)
        seen, queue = set(), [initial(p)]
        while queue:
            s = queue.pop()
            if s in seen:
                continue
            seen.add(s)
            mask = np.array(legal(s, p))
            target = np.array(q(s, horizon))
            x.append(features(s, p, no_cost))
            y.append(np.where(mask, target, 0.))
            masks.append(mask)
            for action in range(3, 6):
                if mask[action]:
                    queue.extend(child for _, child in outcomes(s, action, p))
    return np.asarray(x), np.asarray(y), np.asarray(masks)


class Controller:
    def __init__(self, rng, width=32, no_cost=False):
        self.no_cost = no_cost
        self.params = [rng.normal(0, .12, (20, width)), np.zeros(width),
                       rng.normal(0, .12, (width, 6)), np.zeros(6)]

    def forward(self, x):
        w, b, v, c = self.params
        h = np.tanh(x @ w + b)
        return h, h @ v + c

    def select(self, s, p):
        q = self.forward(np.asarray(features(s, p, self.no_cost)))[1]
        q[:3] = terminal_values(s, p)
        return int(np.argmax(np.where(legal(s, p), q, -np.inf)))

    def fit(self, data, rng, steps, validation):
        x, y, mask = data
        vx, vy, vm = validation
        means = [np.zeros_like(p) for p in self.params]
        variances = [np.zeros_like(p) for p in self.params]
        best, best_loss = None, np.inf
        for t in range(1, steps + 1):
            ids = rng.integers(len(x), size=192)
            batch = x[ids]
            h, prediction = self.forward(batch)
            dy = (prediction - y[ids]) * mask[ids] / len(ids)
            dh = (dy @ self.params[2].T) * (1 - h ** 2)
            grads = (batch.T @ dh, dh.sum(0), h.T @ dy, dy.sum(0))
            for param, mean, variance, grad in zip(self.params, means, variances, grads):
                mean[:] = .9 * mean + .1 * grad
                variance[:] = .999 * variance + .001 * grad ** 2
                param -= .004 * (mean / (1 - .9 ** t)) / (np.sqrt(variance / (1 - .999 ** t)) + 1e-8)
            if t % 100 == 0 or t == steps:
                loss = np.square((self.forward(vx)[1] - vy) * vm).sum() / vm.sum()
                if loss < best_loss:
                    best = [p.copy() for p in self.params]
                    best_loss = float(loss)
        self.params = best
        return best_loss

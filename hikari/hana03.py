"""HANA-0.3 conditional halting with a differentiable compute cost."""
import torch
from torch import nn

from .hana import State, recall, remember
from .hana02 import HANA02


class HANA03(HANA02):
    def __init__(self, vocab, width=32, memory_slots=64):
        super().__init__(vocab, width, memory_slots)
        self.continue_gate = nn.Linear(2 * width, 2)
        nn.init.constant_(self.continue_gate.bias, 0.25)

    def continuation(self, embedding, hidden, stage):
        context = torch.cat((embedding, hidden), dim=-1)
        return self.continue_gate(context)[:, stage].sigmoid()

    def train_cycles(self, embedding, context, first):
        second = self.core(context, first)
        third = self.core(context, second)
        p2 = self.continuation(embedding, first, 0)
        p3 = self.continuation(embedding, second, 1)
        reach_second = p2 >= 0.5
        reach_third = reach_second & (p3 >= 0.5)
        soft = torch.stack((p2, p2 * p3), dim=1)
        hard = torch.stack((reach_second, reach_third), dim=1)
        weights = hard.to(soft.dtype) + (soft - soft.detach())
        hidden = first + weights[:, :1] * (second - first) + weights[:, 1:] * (third - second)
        return hidden, 1 + hard.sum(1), soft.sum(1)

    def infer_cycles(self, embedding, context, hidden, rows):
        counts = torch.ones_like(rows)
        if hidden.size(0) == 1:
            for stage in range(2):
                if not (self.continuation(embedding, hidden, stage) >= 0.5).item():
                    break
                hidden = self.core(context, hidden)
                counts += 1
            return hidden, counts
        for stage in range(2):
            continuation = self.continuation(embedding[rows], hidden[rows], stage)
            rows = rows[continuation >= 0.5]
            if not rows.numel():
                break
            hidden = hidden.index_copy(0, rows, self.core(context[rows], hidden[rows]))
            counts[rows] += 1
        return hidden, counts

    def forward(self, tokens, state=None, *, return_cost=False):
        if tokens.ndim != 2 or not tokens.numel():
            raise ValueError("tokens must have nonempty shape [batch, sequence]")
        if return_cost and not self.training:
            raise ValueError("differentiable cycle cost is available in training mode")
        hidden, working, memory = self.initial_state(tokens) if state is None else state
        features, counts, costs = [], [], []
        rows = torch.arange(tokens.size(0), device=tokens.device)
        for token, embedding in zip(tokens.unbind(1), self.embedding(tokens).unbind(1)):
            retrieved = recall(memory, token) if self.use_memory else torch.zeros_like(hidden)
            context = torch.cat((embedding, working, retrieved), dim=-1)
            hidden = self.core(context, hidden)
            if self.training:
                hidden, cycles, cost = self.train_cycles(embedding, context, hidden)
                if return_cost:
                    costs.append(cost)
            else:
                hidden, cycles = self.infer_cycles(embedding, context, hidden, rows)
            gate = self.working_gate(torch.cat((working, hidden), dim=-1)).sigmoid()
            working = torch.lerp(working, hidden, gate)
            if self.use_memory:
                memory = remember(memory, token, hidden)
            features.append(torch.cat((hidden, working), dim=-1))
            counts.append(cycles)
        result = self.head(torch.stack(features, dim=1)), State(hidden, working, memory), torch.stack(counts, dim=1)
        if return_cost:
            return (*result, torch.stack(costs, dim=1).mean())
        return result

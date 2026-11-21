"""HANA-0 Nano: recurrent token prediction with bounded adaptive computation."""
from typing import NamedTuple

import torch
from torch import nn


class Memory(NamedTuple):
    keys: torch.Tensor
    values: torch.Tensor


def recall(memory: Memory, query: torch.Tensor) -> torch.Tensor:
    """Recall the latest value for each matching token; a miss returns zero."""
    matches = memory.keys.eq(query[:, None])
    slots = torch.arange(memory.keys.size(1), device=query.device)
    index = torch.where(matches, slots, -1).amax(dim=1)
    rows = torch.arange(query.size(0), device=query.device)
    return memory.values[rows, index.clamp_min(0)] * index.ge(0)[:, None]


def remember(memory: Memory, key: torch.Tensor, value: torch.Tensor) -> Memory:
    return Memory(
        torch.cat((memory.keys[:, 1:], key[:, None]), dim=1),
        torch.cat((memory.values[:, 1:], value[:, None]), dim=1),
    )


class State(NamedTuple):
    hidden: torch.Tensor
    working: torch.Tensor
    memory: Memory


class HANA0Nano(nn.Module):
    def __init__(self, vocab: int, width: int = 32, memory_slots: int = 8,
                 max_cycles: int = 3):
        super().__init__()
        if min(vocab, width, memory_slots) < 1 or max_cycles < 2:
            raise ValueError("positive sizes and at least two cycles required")
        self.width = width
        self.memory_slots = memory_slots
        self.max_cycles = max_cycles
        self.embedding = nn.Embedding(vocab, width)
        self.core = nn.GRUCell(3 * width, width)
        self.working_gate = nn.Linear(2 * width, width)
        self.continue_gate = nn.Linear(2 * width, 1)
        self.head = nn.Linear(2 * width, vocab)

    def initial_state(self, tokens: torch.Tensor) -> State:
        batch = tokens.size(0)
        zeros = self.embedding.weight.new_zeros
        return State(
            zeros(batch, self.width), zeros(batch, self.width),
            Memory(tokens.new_full((batch, self.memory_slots), -1),
                   zeros(batch, self.memory_slots, self.width)),
        )

    def forward(self, tokens: torch.Tensor, state: State | None = None):
        if tokens.ndim != 2 or not tokens.numel():
            raise ValueError("tokens must have nonempty shape [batch, sequence]")
        hidden, working, memory = self.initial_state(tokens) if state is None else state
        outputs, counts = [], []
        for token, embedding in zip(tokens.unbind(dim=1),
                                    self.embedding(tokens).unbind(dim=1)):
            context = torch.cat((embedding, working, recall(memory, token)), dim=-1)
            hidden = self.core(context, hidden)
            cycles = torch.ones_like(token)
            active = torch.arange(token.size(0), device=token.device)
            for _ in range(1, self.max_cycles):
                current = hidden[active]
                probability = self.continue_gate(
                    torch.cat((embedding[active], current), dim=-1)
                ).sigmoid()
                proceed = probability.squeeze(-1) >= 0.5

                # Hard routing in the forward pass; a straight-through residual
                # gives the controller gradients during training.
                if self.training:
                    candidate = self.core(context[active], current)
                    weight = proceed[:, None].to(probability.dtype)
                    weight = weight + (probability - probability.detach())
                    hidden = hidden.index_copy(0, active, current + weight * (candidate - current))
                active = active[proceed]
                if not active.numel():
                    break
                if not self.training:
                    hidden = hidden.index_copy(0, active, self.core(context[active], hidden[active]))
                cycles[active] += 1

            gate = self.working_gate(torch.cat((working, hidden), dim=-1)).sigmoid()
            working = torch.lerp(working, hidden, gate)
            memory = remember(memory, token, hidden)
            outputs.append(self.head(torch.cat((hidden, working), dim=-1)))
            counts.append(cycles)
        return torch.stack(outputs, dim=1), State(hidden, working, memory), torch.stack(counts, dim=1)

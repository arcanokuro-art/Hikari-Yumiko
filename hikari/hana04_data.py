"""Random bijective relation graphs and held-out query distributions."""
from itertools import permutations as ordered_choices
from typing import NamedTuple

import torch

from .hana04 import ENTITIES, RELATIONS, STOP


class QueryBatch(NamedTuple):
    facts: torch.Tensor
    start: torch.Tensor
    relations: torch.Tensor
    target: torch.Tensor

    def serialized(self):
        source, relation, destination = self.facts.unbind(-1)
        edges = torch.stack((source, relation + ENTITIES, destination), dim=-1).flatten(1)
        return torch.cat((edges, self.start[:, None], self.relations + ENTITIES), dim=1)


def query_patterns(depth, split):
    if split == "relation":
        return [p for p in ordered_choices(range(RELATIONS - 1), depth) if 2 in p]
    patterns = list(ordered_choices(range(RELATIONS - 1), depth))
    held_out = [p for p in patterns if depth > 1 and p[0] == 0 and p[-1] == 1]
    if split == "composition":
        if not held_out:
            raise ValueError("held-out compositions require at least two hops")
        return held_out
    return [p for p in patterns if p not in held_out]


def make_queries(batch, depth, split="train", *, generator):
    if depth not in (1, 2, 3) or split not in ("train", "fresh", "relation", "composition", "length"):
        raise ValueError("unsupported depth or split")
    size = 18 if split == "length" else 10
    banks = RELATIONS - 1
    patterns = torch.tensor(query_patterns(depth, split))
    actions = patterns[torch.randint(len(patterns), (batch,), generator=generator)]
    choices = torch.rand(batch, ENTITIES, generator=generator).argsort(-1)[:, :size]
    permutations = torch.rand(batch, banks, size, generator=generator).argsort(-1)
    source = choices[:, None, :].expand(-1, banks, -1)
    target_ids = source.gather(2, permutations)
    bank_ids = torch.arange(banks)[None, :, None].expand_as(source)
    facts = torch.stack((source, bank_ids, target_ids), dim=-1).flatten(1, 2)
    order = torch.rand(batch, banks * size, generator=generator).argsort(-1)
    facts = facts.gather(1, order[:, :, None].expand_as(facts))

    rows = torch.arange(batch)
    current = torch.randint(size, (batch,), generator=generator)
    start = choices[rows, current]
    for hop in actions.unbind(1):
        current = permutations[rows, hop, current]
    if split == "relation":
        # Rename one bank while keeping graph size and vocabulary shape fixed.
        facts[:, :, 1] = torch.where(facts[:, :, 1] == 2, 3, facts[:, :, 1])
        actions = torch.where(actions == 2, 3, actions)
    relations = torch.cat((actions, torch.full((batch, 3 - depth), STOP)), dim=1)
    return QueryBatch(facts, start, relations, choices[rows, current])

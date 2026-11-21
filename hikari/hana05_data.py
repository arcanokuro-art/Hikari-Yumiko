"""Matched graph queries whose stopping depth is a property of traversal."""
from itertools import permutations
from typing import NamedTuple

import torch


ENTITIES = 48
TAGS = 4
BANKS = 4


class Query(NamedTuple):
    edges: torch.Tensor
    nodes: torch.Tensor
    tags: torch.Tensor
    start: torch.Tensor
    relations: torch.Tensor
    goal: torch.Tensor

    def select(self, index):
        return Query(*(field[index] for field in self))


class Episode(NamedTuple):
    query: Query
    goals: torch.Tensor
    targets: torch.Tensor
    phases: torch.Tensor

    def at_depth(self, depth):
        return self.query._replace(goal=self.goals[:, depth - 1]), self.targets[:, depth - 1]


def patterns(split):
    choices = list(permutations(range(3)))
    reserved = (0, 2, 1)
    return [reserved] if split == "composition" else [p for p in choices if p != reserved]


def make_episode(batch, split, generator):
    if split not in ("train", "fresh", "relation", "composition", "length"):
        raise ValueError("unknown split")
    width = 12 if split == "length" else 8
    nodes = torch.rand(batch, ENTITIES, generator=generator).argsort(1)[:, :4 * width]
    nodes = nodes.reshape(batch, 4, width)
    order = torch.rand(batch, 3, 4, width, generator=generator).argsort(-1)
    successors = nodes.roll(-1, dims=1)[:, None].expand_as(order).gather(3, order)
    sources = nodes[:, None].expand_as(order)
    banks = torch.arange(3)[None, :, None, None].expand_as(order)
    options = torch.tensor(patterns(split))
    relations = options[torch.randint(len(options), (batch,), generator=generator)]
    rows = torch.arange(batch)
    phase = torch.randint(4, (batch,), generator=generator)
    offset = torch.randint(width, (batch,), generator=generator)
    start = nodes[rows, phase, offset]

    # A single graph supports all three depths. Assign three distinct,
    # exchangeable tags along the queried path, preserving each phase's
    # multiset of tags by swapping with a uniformly selected slot.
    goals = torch.rand(batch, TAGS, generator=generator).argsort(1)[:, :3]
    tag_order = torch.rand(batch, 4, width, generator=generator).argsort(-1)
    tags = tag_order.remainder(TAGS)
    targets = []
    for step in range(3):
        offset = order[rows, relations[:, step], phase, offset]
        phase = (phase + 1) % 4
        targets.append(nodes[rows, phase, offset])
        current = tags[rows, phase, offset].clone()
        choices = tags[rows, phase].eq(goals[:, step, None])
        scores = torch.rand(batch, width, generator=generator).masked_fill(~choices, -1)
        partner = scores.argmax(1)
        tags[rows, phase, partner] = current
        tags[rows, phase, offset] = goals[:, step]

    if split == "relation":
        banks = torch.where(banks == 2, 3, banks)
        relations = torch.where(relations == 2, 3, relations)
    edges = torch.stack((sources, banks, successors), dim=-1).flatten(1, 3)
    edge_order = torch.rand(batch, edges.size(1), generator=generator).argsort(1)
    edges = edges.gather(1, edge_order[:, :, None].expand_as(edges))
    flat_nodes, flat_tags = nodes.flatten(1), tags.flatten(1)
    node_order = torch.rand(batch, flat_nodes.size(1), generator=generator).argsort(1)
    query = Query(edges, flat_nodes.gather(1, node_order), flat_tags.gather(1, node_order),
                  start, relations, goals[:, 0])
    phases = torch.zeros(batch, ENTITIES, dtype=torch.long)
    phases.scatter_(1, flat_nodes, torch.arange(4).repeat_interleave(width).expand(batch, -1))
    return Episode(query, goals, torch.stack(targets, dim=1), phases)


def counterfactual(episode, depth):
    """Swap destinations in the decisive bank while preserving marginals."""
    query, target = episode.at_depth(depth)
    predecessor = query.start if depth == 1 else episode.targets[:, depth - 2]
    source, bank, destination = query.edges.unbind(-1)
    decisive_bank = bank.eq(query.relations[:, depth - 1, None])
    decisive = decisive_bank & source.eq(predecessor[:, None])
    tag_map = torch.zeros(query.start.size(0), ENTITIES, dtype=torch.long)
    tag_map.scatter_(1, query.nodes, query.tags)
    phase = episode.phases.gather(1, target[:, None])
    same_phase = episode.phases.gather(1, destination).eq(phase)
    alternatives = (decisive_bank & ~decisive & same_phase
                    & tag_map.gather(1, destination).eq(query.goal[:, None]))
    partner = alternatives.long().argmax(1)
    original = decisive.long().argmax(1)
    rows = torch.arange(query.start.size(0))
    new_target = destination[rows, partner]
    edges = query.edges.clone()
    edges[rows, original, 2] = new_target
    edges[rows, partner, 2] = target
    return query._replace(edges=edges), new_target

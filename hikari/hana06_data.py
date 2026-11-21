"""Uncertain variants of HANA-0.5 episodes with hidden clean labels."""
from typing import NamedTuple

import torch
from torch.nn import functional as F

from .hana05_data import BANKS, ENTITIES, TAGS, counterfactual, make_episode

UNKNOWN = ENTITIES
DEPTHS = 3
REPORTS = 5


class Observation(NamedTuple):
    edges: torch.Tensor
    nodes: torch.Tensor
    votes: torch.Tensor
    visible: torch.Tensor
    start: torch.Tensor
    relations: torch.Tensor
    goal: torch.Tensor

    def select(self, index):
        return Observation(*(field[index] for field in self))


def oracle(query, visible):
    """First clean matching reachable node; a missing edge breaks verification."""
    rows = torch.arange(query.start.size(0))
    source, bank, dest = query.edges.unbind(-1)
    table = query.start.new_full((rows.numel(), BANKS * ENTITIES), -1)
    table.scatter_(1, bank * ENTITIES + source, dest.masked_fill(~visible, -1))
    tags = query.start.new_zeros(rows.numel(), ENTITIES).scatter_(1, query.nodes, query.tags)
    target = torch.full_like(query.start, UNKNOWN)
    depth = torch.zeros_like(query.start)
    current = query.start
    for step, relation in enumerate(query.relations.unbind(1), 1):
        next_node = table[rows, relation * ENTITIES + current.clamp_min(0)]
        current = torch.where(current.ge(0), next_node, -1)
        match = current.ge(0) & tags[rows, current.clamp_min(0)].eq(query.goal) & depth.eq(0)
        target = torch.where(match, current, target)
        depth = torch.where(match, step, depth)
    return target, depth


def make_batch(batch, split, noise, generator, *, paired=False, correlated=False):
    episode = make_episode(batch, split, generator)
    query = episode.query
    # A query requests one of the three traversed tags or the unused tag.
    # The order of the tags was randomized before corruption.
    unused = TAGS * (TAGS - 1) // 2 - episode.goals.sum(1)
    goals = torch.cat((episode.goals, unused[:, None]), dim=1)
    choice = torch.randint(4, (batch,), generator=generator)
    query = query._replace(goal=goals.gather(1, choice[:, None]).squeeze(1))
    order = torch.rand(query.edges.shape[:2], generator=generator).argsort(1)
    visible = order.ge(round(query.edges.size(1) * 0.15))
    shape = (*query.tags.shape, REPORTS)
    true_tags = query.tags.unsqueeze(-1).expand(shape)
    replacements = (true_tags + torch.randint(1, TAGS, shape, generator=generator)) % TAGS
    flipped = torch.rand(shape, generator=generator).lt(noise)
    if correlated:
        # Two witnesses share a common source in this reserved stress test.
        replacements[:, :, 1] = replacements[:, :, 0]
        flipped[:, :, 1] = flipped[:, :, 0]
    present = torch.rand(shape, generator=generator).ge(0.15)
    observed = torch.where(flipped, replacements, true_tags)
    votes = (F.one_hot(observed, TAGS) * present.unsqueeze(-1)).sum(2).float()
    observation = Observation(query.edges.masked_fill(~visible.unsqueeze(-1), 0),
                              query.nodes, votes, visible, query.start, query.relations, query.goal)
    target, depth = oracle(query, visible)
    if not paired:
        return observation, target, depth

    alternatives = [counterfactual(episode, d)[0].edges for d in (1, 2, 3)]
    swapped = torch.stack((*alternatives, query.edges), dim=1)[torch.arange(batch), choice]
    altered_target, altered_depth = oracle(query._replace(edges=swapped), visible)
    return (observation, target, depth,
            observation._replace(edges=swapped.masked_fill(~visible.unsqueeze(-1), 0)),
            altered_target, altered_depth)

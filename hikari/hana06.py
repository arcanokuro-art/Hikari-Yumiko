"""Shared observation memory, recurrent confidence and sequential policies."""
import torch
from torch import nn
from torch.nn import functional as F

from .hana05_data import BANKS, ENTITIES, TAGS
from .hana06_data import REPORTS, UNKNOWN


def memory(observation):
    source, bank, dest = observation.edges.unbind(-1)
    # Missing edges use an extra cell, outside the addressable range.
    key = (bank * ENTITIES + source).masked_fill(~observation.visible, BANKS * ENTITIES)
    table = source.new_full((source.size(0), BANKS * ENTITIES + 1), -1).scatter_(1, key, dest)
    votes = observation.votes
    labels = votes.new_zeros(source.size(0), ENTITIES, TAGS)
    labels.scatter_(1, observation.nodes.unsqueeze(-1).expand_as(votes), votes)
    return table, labels


def read(table, labels, current, relation):
    rows = torch.arange(current.size(0), device=current.device)
    next_node = table[rows, relation * ENTITIES + current.clamp_min(0)]
    current = torch.where(current.ge(0), next_node, -1)
    votes = labels[rows, current.clamp_min(0)] * current.ge(0).unsqueeze(1)
    return current, votes


def candidates(history, device):
    return torch.cat((history, torch.full((history.size(0), 1), UNKNOWN, device=device)), dim=1)


class HANA06(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.entity = nn.Embedding(ENTITIES + 1, width)
        self.relation = nn.Embedding(BANKS, width)
        self.edge = nn.Linear(3 * width, width)
        self.attribute = nn.Linear(width + TAGS, width)
        self.recurrent = nn.GRUCell(width + 2 * TAGS + 3, width)
        self.slots = nn.Linear(width + 3 * (TAGS + 3), 4)
        self.answer = nn.Linear(width, ENTITIES + 1)
        self.stop = nn.Linear(width + 2, 1)

    def initial(self, observation):
        src, bank, dst = observation.edges.unbind(-1)
        facts = self.edge(torch.cat((self.entity(src), self.relation(bank), self.entity(dst)), dim=-1))
        facts = (facts.tanh() * observation.visible.unsqueeze(-1)).mean(1)
        attributes = self.attribute(torch.cat((self.entity(observation.nodes),
                                               observation.votes / REPORTS), dim=-1)).tanh().mean(1)
        return (facts + attributes + self.entity(observation.start)).tanh()

    def advance(self, observation, hidden, table, labels, current, history, evidence, stage, use_memory):
        relation = observation.relations[:, stage]
        goal = F.one_hot(observation.goal, TAGS).float()
        if use_memory:
            current, votes = read(table, labels, current, relation)
        else:
            votes = evidence.new_zeros(current.size(0), TAGS)
        valid = current.ge(0)
        matches = (votes * goal).sum(1, keepdim=True)
        competing = (votes * (1 - goal)).amax(1, keepdim=True)
        features = torch.cat((votes / REPORTS, matches / REPORTS, competing / REPORTS,
                              valid[:, None]), dim=1)
        hidden = self.recurrent(torch.cat((self.relation(relation), features, goal), dim=1), hidden)
        scores = self.answer(hidden)
        if use_memory:
            history = torch.cat((history[:, :stage], current[:, None], history[:, stage + 1:]), dim=1)
            evidence = torch.cat((evidence[:, :stage], features[:, None], evidence[:, stage + 1:]), dim=1)
            indices = candidates(history, scores.device)
            logits = self.slots(torch.cat((hidden, evidence.flatten(1)), dim=1))
            logits = logits + scores.gather(1, indices.clamp_min(0))
            logits = logits.masked_fill(indices.lt(0), -1e4)
        else:
            indices = torch.arange(ENTITIES + 1, device=scores.device).expand(scores.size(0), -1)
            logits = scores
        return hidden, current, history, evidence, logits, indices

    def confidence_features(self, logits, use_confidence):
        prob = logits.softmax(1)
        confidence = torch.stack((prob.amax(1), prob[:, -1]), dim=1)
        return confidence if use_confidence else torch.zeros_like(confidence)

    def initialize(self, observation, use_memory):
        hidden = self.initial(observation)
        if use_memory:
            table, labels = memory(observation)
        else:
            table = labels = hidden.new_empty((hidden.size(0), 0))
        history = observation.start.new_full((hidden.size(0), 3), -1)
        evidence = hidden.new_zeros(hidden.size(0), 3, TAGS + 3)
        return hidden, table, labels, observation.start, history, evidence

    def loss(self, observation, target, *, use_memory=True, use_confidence=True, adaptive=True):
        hidden, table, labels, current, history, evidence = self.initialize(observation, use_memory)
        errors, decisions, accessible = [], [], []
        for stage in range(3):
            hidden, current, history, evidence, logits, indices = self.advance(
                observation, hidden, table, labels, current, history, evidence, stage, use_memory)
            matches = indices.eq(target[:, None])
            accessible.append(matches.any(1))
            positions = torch.where(matches.any(1), matches.long().argmax(1), indices.size(1) - 1)
            errors.append(F.cross_entropy(logits, positions, reduction="none"))
            confidence = self.confidence_features(logits, use_confidence).detach()
            decisions.append(self.stop(torch.cat((hidden.detach(), confidence), dim=1)).flatten())
        errors = torch.stack(errors, dim=1)
        if not adaptive:
            return errors[:, -1].mean()
        # At training only, use future supervised losses to estimate the
        # value of another read. No answer or depth is passed in inference.
        costs = errors.detach().masked_fill(~torch.stack(accessible, dim=1), 6.) + errors.new_tensor([0., 0.15, 0.30])
        stop_now = torch.stack((costs[:, 0] <= costs[:, 1:].amin(1),
                                costs[:, 1] <= costs[:, 2]), dim=1)
        policy = F.binary_cross_entropy_with_logits(torch.stack(decisions[:2], dim=1), stop_now.float())
        return errors.mean() + policy

    def infer(self, observation, mode, threshold=0.5, *, use_memory=True, use_confidence=True):
        hidden, table, labels, current, history, evidence = self.initialize(observation, use_memory)
        rows = torch.arange(current.size(0))
        answer = observation.start.new_full(current.shape, UNKNOWN)
        confidence = hidden.new_zeros(current.shape)
        counts = torch.zeros_like(current)
        for stage in range(3):
            hidden, current, history, evidence, logits, indices = self.advance(
                observation, hidden, table, labels, current, history, evidence, stage, use_memory)
            probability, slot = logits.softmax(1).max(1)
            answer[rows] = indices.gather(1, slot[:, None]).flatten()
            confidence[rows] = probability
            counts[rows] += 1
            if stage == 2 or mode == f"fixed-{stage + 1}":
                break
            if mode != "adaptive":
                continue
            features = self.confidence_features(logits, use_confidence)
            stop = self.stop(torch.cat((hidden, features), dim=1)).flatten().sigmoid()
            onward = (stop < threshold) & current.ge(0)
            rows = rows[onward]
            if not rows.numel():
                break
            observation = observation.select(onward)
            hidden, table, labels, current, history, evidence = (
                value[onward] for value in (hidden, table, labels, current, history, evidence))
        return answer, confidence, counts


def baseline(observation, mode, threshold=0.72):
    """Plurality rule or an independent symmetric-label noise model."""
    table, labels = memory(observation)
    current = observation.start
    rows = torch.arange(current.size(0))
    counts = torch.zeros_like(current)
    answer = torch.full_like(current, UNKNOWN)
    confidence = labels.new_zeros(current.shape)
    mass = torch.ones_like(confidence)
    best = torch.zeros_like(confidence)
    best_node = torch.full_like(current, UNKNOWN)
    for stage in range(3):
        current, votes = read(table, labels, current, observation.relations[:, stage])
        goal_votes = votes.gather(1, observation.goal[:, None]).flatten()
        if mode == "deterministic":
            known = (goal_votes >= 3) & current.ge(0)
            pred = torch.where(known, current, UNKNOWN)
            prob = torch.where(known, goal_votes / REPORTS, 1 - goal_votes / (REPORTS + 1))
            onward = ~known & current.ge(0)
        else:
            # Likelihood ratio for symmetric corruption with nominal p=0.20.
            goal_prob = (votes * 2.48490665).softmax(1).gather(1, observation.goal[:, None]).flatten()
            goal_prob = goal_prob * current.ge(0)
            contribution = mass * goal_prob
            better = contribution > best
            best = torch.maximum(best, contribution)
            best_node = torch.where(better, current, best_node)
            mass = mass * (1 - goal_prob)
            pred = torch.where(best > mass, best_node, UNKNOWN)
            prob = torch.maximum(best, mass)
            onward = (best < threshold) & current.ge(0)
        answer[rows], confidence[rows] = pred, prob
        counts[rows] += 1
        if stage == 2:
            break
        rows = rows[onward]
        if not rows.numel():
            break
        observation = observation.select(onward)
        current, table, labels, mass, best, best_node = (
            value[onward] for value in (current, table, labels, mass, best, best_node))
    return answer, confidence, counts

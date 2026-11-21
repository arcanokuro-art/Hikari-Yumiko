"""Recurrent graph traversal and interchangeable halting policies."""
import torch
from torch import nn
from torch.nn import functional as F

from .hana05_data import BANKS, ENTITIES, TAGS


class HANA05(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.entity = nn.Embedding(ENTITIES, width)
        self.tag = nn.Embedding(TAGS, width)
        self.relation = nn.Embedding(BANKS, width)
        self.edge_encoder = nn.Linear(3 * width, width)
        self.node_encoder = nn.Linear(2 * width, width)
        self.core = nn.GRUCell(4 * width, width)
        self.answer = nn.Linear(width + ENTITIES, ENTITIES)
        self.halt = nn.Sequential(nn.Linear(width + 2 * TAGS, width), nn.Tanh(), nn.Linear(width, 1))

    def prepare(self, query, use_memory):
        source, bank, destination = query.edges.unbind(-1)
        facts = torch.cat((self.entity(source), self.relation(bank), self.entity(destination)), dim=-1)
        attributes = torch.cat((self.entity(query.nodes), self.tag(query.tags)), dim=-1)
        hidden = (self.edge_encoder(facts).tanh().mean(1)
                  + self.node_encoder(attributes).tanh().mean(1) + self.entity(query.start))
        table = query.start.new_full((source.size(0), BANKS * ENTITIES), -1)
        labels = query.start.new_zeros((source.size(0), ENTITIES))
        if use_memory:
            table.scatter_(1, bank * ENTITIES + source, destination)
            labels.scatter_(1, query.nodes, query.tags)
        return hidden.tanh(), table, labels

    def step(self, hidden, current, relation, goal, table, labels, use_memory):
        if use_memory:
            key = relation * ENTITIES + current
            current = table.gather(1, key[:, None]).squeeze(1)
            retrieved_tag = labels.gather(1, current[:, None]).squeeze(1)
            pointer = F.one_hot(current, ENTITIES).to(hidden.dtype)
        else:
            retrieved_tag = torch.zeros_like(current)
            pointer = hidden.new_zeros(hidden.size(0), ENTITIES)
        hidden = self.core(torch.cat((self.entity(current), self.relation(relation),
                                      self.tag(retrieved_tag), self.tag(goal)), dim=-1), hidden)
        scores = self.answer(torch.cat((hidden, pointer), dim=-1))
        return hidden, current, retrieved_tag, scores

    def halt_logits(self, hidden, retrieved_tag, goal):
        # The network sees the two tags separately. The checked equality
        # is used as a training target and by the deterministic control.
        retrieved = F.one_hot(retrieved_tag, TAGS).to(hidden.dtype)
        requested = F.one_hot(goal, TAGS).to(hidden.dtype)
        return self.halt(torch.cat((hidden, retrieved, requested), dim=-1)).squeeze(1)

    def training_loss(self, query, target, use_memory=True):
        hidden, table, labels = self.prepare(query, use_memory)
        current = query.start
        losses, stop_losses = [], []
        eligible = torch.ones_like(current, dtype=torch.bool)
        for relation in query.relations.unbind(1):
            hidden, current, found, scores = self.step(
                hidden, current, relation, query.goal, table, labels, use_memory,
            )
            losses.append(F.cross_entropy(scores, current if use_memory else target))
            if use_memory:
                verified = found.eq(query.goal)
                decision = self.halt_logits(hidden, found, query.goal)
                cost = F.binary_cross_entropy_with_logits(decision, verified.to(decision.dtype),
                                                          reduction="none")
                stop_losses.append((cost * eligible).sum() / eligible.sum().clamp_min(1))
                eligible = eligible & ~verified
        loss = torch.stack(losses).mean()
        return loss + torch.stack(stop_losses).mean() if use_memory else loss

    def infer(self, query, policy, *, budgets=None, use_memory=True):
        if policy not in ("neural", "deterministic", "fixed", "random"):
            raise ValueError("unknown halting policy")
        if policy in ("fixed", "random") and budgets is None:
            raise ValueError("a fixed or presampled budget is required")
        hidden, table, labels = self.prepare(query, use_memory)
        current = query.start
        rows = torch.arange(current.size(0), device=current.device)
        counts = torch.zeros_like(current)
        certified = torch.zeros_like(current, dtype=torch.bool)
        logits = hidden.new_zeros(current.size(0), ENTITIES)
        for step in range(3):
            hidden, current, found, scores = self.step(
                hidden, current, query.relations[rows, step], query.goal[rows],
                table, labels, use_memory,
            )
            matched = found.eq(query.goal[rows]) if use_memory else torch.zeros_like(current, dtype=torch.bool)
            logits[rows] = scores
            counts[rows] += 1
            certified[rows] = matched
            if step == 2:
                break
            if policy == "neural":
                onward = self.halt_logits(hidden, found, query.goal[rows]) < 0
            elif policy == "deterministic":
                onward = ~matched
            else:
                onward = budgets[rows] > step + 1
            rows = rows[onward]
            if not rows.numel():
                break
            hidden, current = hidden[onward], current[onward]
            table, labels = table[onward], labels[onward]
        return logits, counts, certified

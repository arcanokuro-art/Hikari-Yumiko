"""Compact recurrent reasoning over an explicitly supplied relational table."""
import torch
from torch import nn
from torch.nn import functional as F


ENTITIES = 24
RELATIONS = 4
STOP = RELATIONS


class HANA04(nn.Module):
    def __init__(self, width=32, fixed_cycles=None, use_memory=True):
        super().__init__()
        if fixed_cycles not in (None, 1, 2, 3):
            raise ValueError("fixed_cycles must be 1, 2, 3, or None")
        self.fixed_cycles = fixed_cycles
        self.use_memory = use_memory
        self.entities = nn.Embedding(ENTITIES, width)
        self.edge_encoder = nn.Linear(2 * width + RELATIONS, width)
        self.query_encoder = nn.Linear(3 * (RELATIONS + 1), width)
        self.core = nn.GRUCell(2 * width, width)
        self.read_gate = nn.Linear(width, 1)
        nn.init.constant_(self.read_gate.bias, 2.0)
        self.head = nn.Linear(width + ENTITIES, ENTITIES)
        if fixed_cycles is None:
            self.continue_gate = nn.Linear(width + 5, 1)
            nn.init.constant_(self.continue_gate.bias, 0.5)

    def prepare(self, facts, start, relations):
        source, relation, destination = facts.unbind(-1)
        left, right = self.entities(source), self.entities(destination)
        labels = F.one_hot(relation, RELATIONS).to(left.dtype)
        embeddings = torch.cat((left, right, labels), dim=-1)
        query = F.one_hot(relations, RELATIONS + 1).flatten(1).to(left.dtype)
        hidden = (self.edge_encoder(embeddings).tanh().mean(dim=1)
                  + self.query_encoder(query) + self.entities(start)).tanh()
        pointer = F.one_hot(start, ENTITIES).to(hidden.dtype)

        # All variants see the same compressed summary. The ablation removes
        # random access while retaining the facts and all learned parameters.
        table = None
        if self.use_memory:
            flat = F.one_hot(destination, ENTITIES).to(hidden.dtype)
            flat_keys = (relation * ENTITIES + source).unsqueeze(-1).expand_as(flat)
            table = hidden.new_zeros(facts.size(0), RELATIONS * ENTITIES, ENTITIES)
            table.scatter_add_(1, flat_keys, flat)
            table = table.reshape(-1, RELATIONS, ENTITIES, ENTITIES)
        route = F.one_hot(relations.ne(STOP).sum(1), 4).to(hidden.dtype)
        return hidden, pointer, table, route

    def step(self, hidden, pointer, table, relation):
        if table is None:
            read = torch.zeros_like(pointer)
        else:
            rows = torch.arange(pointer.size(0), device=pointer.device)
            bank = table[rows, relation.clamp_max(RELATIONS - 1)]
            read = torch.bmm(pointer.unsqueeze(1), bank).squeeze(1)
        read = torch.where(relation[:, None].eq(STOP), pointer, read)
        embedding = self.entities.weight
        hidden = self.core(torch.cat((pointer @ embedding, read @ embedding), dim=-1), hidden)
        pointer = torch.lerp(pointer, read, self.read_gate(hidden).sigmoid())
        return hidden, pointer

    def continuation(self, hidden, route, stage):
        stage_feature = hidden.new_full((hidden.size(0), 1), stage)
        return self.continue_gate(torch.cat((hidden, route, stage_feature), dim=-1)).flatten()

    def forward(self, facts, start, relations):
        hidden, pointer, table, route = self.prepare(facts, start, relations)
        batch = start.size(0)
        if self.fixed_cycles is not None:
            for stage in range(self.fixed_cycles):
                hidden, pointer = self.step(hidden, pointer, table, relations[:, stage])
            logits = self.head(torch.cat((hidden, pointer), dim=-1))
            return logits, start.new_full((batch,), self.fixed_cycles)

        if self.training:
            predictions, continuation = [], []
            for stage in range(3):
                hidden, pointer = self.step(hidden, pointer, table, relations[:, stage])
                predictions.append(self.head(torch.cat((hidden, pointer), dim=-1)))
                if stage < 2:
                    continuation.append(self.continuation(hidden, route, stage))
            first, second = continuation
            log_first = F.logsigmoid(first)
            log_routes = torch.stack((F.logsigmoid(-first), log_first + F.logsigmoid(-second),
                                      log_first + F.logsigmoid(second)), dim=1)
            return torch.stack(predictions, dim=1), log_routes

        rows = torch.arange(batch, device=start.device)
        counts = torch.ones_like(start)
        for stage in range(3):
            if batch == 1:
                hidden, pointer = self.step(hidden, pointer, table, relations[:, stage])
                if stage == 2 or self.continuation(hidden, route, stage).item() < 0:
                    break
                counts += 1
                continue
            active_hidden, active_pointer = self.step(
                hidden[rows], pointer[rows], None if table is None else table[rows],
                relations[rows, stage],
            )
            hidden = hidden.index_copy(0, rows, active_hidden)
            pointer = pointer.index_copy(0, rows, active_pointer)
            if stage == 2:
                break
            gate = self.continuation(active_hidden, route[rows], stage)
            rows = rows[gate >= 0]
            if not rows.numel():
                break
            counts[rows] += 1
        return self.head(torch.cat((hidden, pointer), dim=-1)), counts

"""CPU experiments for HANA-0.2 with independently configurable ablations."""
import torch

from .hana import HANA0Nano, State, recall, remember


class HANA02(HANA0Nano):
    def __init__(self, vocab, width=32, memory_slots=64, fixed_cycles=None,
                 fast_inference=True):
        if memory_slots < 0 or fixed_cycles not in (None, 1, 2, 3):
            raise ValueError("invalid memory capacity or fixed cycle count")
        super().__init__(vocab, width, memory_slots=max(1, memory_slots))
        self.use_memory = memory_slots > 0
        self.fixed_cycles = fixed_cycles
        self.fast_inference = fast_inference
        if fixed_cycles is not None:
            del self.continue_gate

    def refine(self, embedding, context, hidden, active, cycles):
        if not self.training and self.fast_inference and hidden.size(0) == 1:
            for _ in range(1, self.max_cycles):
                proceed = self.continue_gate(torch.cat((embedding, hidden), dim=-1)).sigmoid()
                if not (proceed >= 0.5).item():
                    break
                hidden = self.core(context, hidden)
                cycles += 1
            return hidden, cycles

        for _ in range(1, self.max_cycles):
            current = hidden[active]
            probability = self.continue_gate(
                torch.cat((embedding[active], current), dim=-1)
            ).sigmoid()
            proceed = probability.squeeze(-1) >= 0.5
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
        return hidden, cycles

    def forward(self, tokens, state=None):
        if tokens.ndim != 2 or not tokens.numel():
            raise ValueError("tokens must have nonempty shape [batch, sequence]")
        hidden, working, memory = self.initial_state(tokens) if state is None else state
        features, counts = [], []
        active = torch.arange(tokens.size(0), device=tokens.device)
        empty_recall = torch.zeros_like(hidden)
        for token, embedding in zip(tokens.unbind(1), self.embedding(tokens).unbind(1)):
            retrieved = recall(memory, token) if self.use_memory else empty_recall
            context = torch.cat((embedding, working, retrieved), dim=-1)
            hidden = self.core(context, hidden)
            if self.fixed_cycles is None:
                hidden, cycles = self.refine(embedding, context, hidden, active, torch.ones_like(token))
            else:
                for _ in range(1, self.fixed_cycles):
                    hidden = self.core(context, hidden)
                cycles = torch.full_like(token, self.fixed_cycles)
            gate = self.working_gate(torch.cat((working, hidden), dim=-1)).sigmoid()
            working = torch.lerp(working, hidden, gate)
            if self.use_memory:
                memory = remember(memory, token, hidden)
            features.append(torch.cat((hidden, working), dim=-1))
            counts.append(cycles)
        return self.head(torch.stack(features, dim=1)), State(hidden, working, memory), torch.stack(counts, dim=1)

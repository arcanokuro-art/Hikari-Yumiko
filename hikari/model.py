"""ZNC-0: an experimental gated recurrent token predictor (NOT claimed novel)."""
import torch
from torch import nn


class ZNC0(nn.Module):
    """Simple recurrent state, gated updates, no self-attention or pretrained weights."""

    def __init__(self, vocab_size: int, width: int = 48):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.proposal = nn.Linear(width * 2, width)
        self.gate = nn.Linear(width * 2, width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, tokens: torch.Tensor, state: torch.Tensor | None = None):
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        batch, length = tokens.shape
        if length == 0:
            raise ValueError("sequence length must be greater than zero")
        if state is None:
            state = self.embedding.weight.new_zeros(batch, self.proposal.out_features)
        outputs = []
        for step in range(length):
            x = self.embedding(tokens[:, step])
            mix = torch.cat((x, state), dim=-1)
            proposal = torch.tanh(self.proposal(mix))
            gate = torch.sigmoid(self.gate(mix))
            state = (1 - gate) * state + gate * proposal
            outputs.append(self.head(state))
        return torch.stack(outputs, dim=1), state

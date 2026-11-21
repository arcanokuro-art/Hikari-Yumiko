"""CPU-friendly, deterministic synthetic next-token learning smoke experiment."""
import argparse
import json
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import ZNC0


def make_batch(batch_size: int, sequence_length: int, vocabulary: int, device="cpu"):
    # Repeating cyclic patterns: requires using sequence context, not just a single token.
    starts = torch.randint(0, vocabulary, (batch_size, 1), device=device)
    strides = torch.randint(1, vocabulary, (batch_size, 1), device=device)
    positions = torch.arange(sequence_length + 1, device=device).unsqueeze(0)
    sequence = (starts + strides * positions) % vocabulary
    return sequence[:, :-1], sequence[:, 1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--vocab", type=int, default=17)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.length, args.width) < 1 or args.vocab < 3:
        parser.error("positive sizes and vocab >= 3 required")
    random.seed(7)
    torch.manual_seed(7)
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    model = ZNC0(args.vocab, args.width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    first = last = None
    for step in range(args.steps):
        x, y = make_batch(args.batch_size, args.length, args.vocab)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, args.vocab), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0:
            first = float(loss.item())
        last = float(loss.item())
    # Held-out samples from same distribution; not a general reasoning benchmark.
    with torch.no_grad():
        x, y = make_batch(256, args.length, args.vocab)
        logits, _ = model(x)
        accuracy = float((logits.argmax(dim=-1) == y).float().mean().item())
    print(json.dumps({"model": "ZNC-0", "steps": args.steps, "first_loss": first,
                      "last_loss": last, "holdout_token_accuracy": accuracy}, indent=2))
    if args.save:
        path = Path("checkpoints/znc0.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "width": args.width,
                    "vocab": args.vocab, "seed": 7}, path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

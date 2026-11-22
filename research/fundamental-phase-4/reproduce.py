"""Compare a rerun of Phase 3 against its archived deterministic fields."""
import argparse
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "fundamental-phase-3" / "results"
REMOVED = {"median_us", "p95_us", "seconds", "rss_mib", "machine", "numpy"}


def compare(expected, actual, path=""):
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() - REMOVED != actual.keys() - REMOVED:
            raise ValueError(f"{path}: different keys")
        for key in sorted(expected.keys() - REMOVED):
            compare(expected[key], actual[key], f"{path}/{key}")
    elif path.endswith(("/validation_mse", "/regret_per_decision")):
        if not math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{path}: {expected} vs {actual}")
    elif actual != expected:
        raise ValueError(f"{path}: {expected} vs {actual}")


def verify(path):
    compare(json.loads((BASE / path.name).read_text()), json.loads(path.read_text()))
    print(f"{path.name}: Phase 3 reproduced")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    for run in parser.parse_args().runs:
        verify(run)

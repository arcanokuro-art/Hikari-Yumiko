"""Verify Phase 2 against its archived quantitative outcomes."""
import json
import math
from pathlib import Path
import sys

from residual import PHASE2


def deterministic_fields(p):
    d = json.loads(p.read_text())
    for key in ("training_s", "peak_rss_kib_linux", "operator_median_us", "machine", "numpy"):
        d.pop(key)
    for group in d["test"].values():
        for metrics in group.values():
            metrics.pop("median_us")
            metrics.pop("p95_us")
    return d


def compare(p):
    old = PHASE2 / "results" / p.name
    expected = deterministic_fields(old)
    actual = deterministic_fields(p)
    expected_mse = expected.pop("validation_mse")
    actual_mse = actual.pop("validation_mse")
    same_mse = expected_mse.keys() == actual_mse.keys() and all(
        math.isclose(expected_mse[k], actual_mse[k], rel_tol=1e-9, abs_tol=1e-12)
        for k in expected_mse
    )
    if actual != expected or not same_mse:
        raise SystemExit(f"{p.name}: mismatch in deterministic fields; inspect both JSON files")
    print(f"{p.name}: deterministic Phase 2 results match")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        compare(Path(name))

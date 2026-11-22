"""Verify Phase 2 against its archived quantitative outcomes."""
import argparse
import json
import math
from pathlib import Path

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


def first_difference(expected, actual, prefix=""):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            if key not in expected or key not in actual:
                return f"{prefix}/{key}: missing key"
            result = first_difference(expected[key], actual[key], f"{prefix}/{key}")
            if result:
                return result
        return None
    if expected != actual:
        return f"{prefix}: expected {expected!r}, observed {actual!r}"
    return None


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
    difference = first_difference(expected, actual)
    if difference:
        raise SystemExit(f"{p.name}: {difference}")
    if not same_mse:
        raise SystemExit(f"{p.name}: validation MSE beyond tolerance; expected {expected_mse}, observed {actual_mse}")
    print(f"{p.name}: deterministic Phase 2 results match")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    for run in parser.parse_args().runs:
        compare(run)

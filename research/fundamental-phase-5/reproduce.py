"""Verify deterministic Phase 4 results against the archived five seeds."""
import gzip
import json
import math
from pathlib import Path
import re
import sys

ARCHIVE = Path(__file__).resolve().parents[1] / 'fundamental-phase-4' / 'results'
TIMING = re.compile(r'(_us($|_)|seconds($|_))')
METADATA = {'machine', 'numpy', 'python', 'rss_mib'}


def compare(a, b, path=''):
    if isinstance(a, dict):
        keys = {k for k in a if k not in METADATA and not TIMING.search(k)}
        actual = {k for k in b if k not in METADATA and not TIMING.search(k)}
        if keys != actual:
            raise ValueError(f'{path}: different keys {keys ^ actual}')
        for key in sorted(keys):
            compare(a[key], b[key], f'{path}/{key}')
    elif isinstance(a, float):
        if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f'{path}: archived {a}, observed {b}')
    elif a != b:
        raise ValueError(f'{path}: archived {a}, observed {b}')


if __name__ == '__main__':
    for name in sys.argv[1:]:
        p = Path(name)
        compare(json.loads(gzip.decompress((ARCHIVE / f'{p.name}.gz').read_bytes())),
                json.loads(p.read_bytes()))
        print(f'{p.name}: verified')

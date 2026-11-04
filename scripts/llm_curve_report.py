"""Summarise a ROLLOUT_ONLY curve produced by scripts/r4_llm_lane.sh.

    python scripts/llm_curve_report.py <lane dir> [--json out.json]

Reads <lane>/curve/offline.jsonl and engine.jsonl, groups by checkpoint label, averages the seeds, and prints the
stillness (free / oracle) and speed per 500-step checkpoint.
"""
import argparse, json, re
from pathlib import Path
import numpy as np


def step_of(label):
    m = re.search(r'(\d+)$', label)
    return int(m.group(1)) if m else 10 ** 9


def load(path):
    out = {}
    if not Path(path).is_file():
        return out
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.setdefault(r['label'], []).append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lane'); ap.add_argument('--json', default=None)
    a = ap.parse_args()
    root = Path(a.lane)
    res = {}
    for mode, fn in (('offline', 'offline.jsonl'), ('engine', 'engine.jsonl')):
        for label, rows in load(root / 'curve' / fn).items():
            res.setdefault(label, {})[mode] = dict(
                seeds=len(rows), steps=step_of(label),
                free_still=float(np.mean([r['free']['still'] for r in rows])),
                free_speed=float(np.mean([r['free']['speed'] for r in rows])),
                oracle_still=float(np.mean([r['oracle']['still'] for r in rows])),
                oracle_speed=float(np.mean([r['oracle']['speed'] for r in rows])),
                free_copy=float(np.mean([r['free']['copy'] for r in rows])),
                oracle_copy=float(np.mean([r['oracle']['copy'] for r in rows])))
    order = sorted(res, key=lambda x: (step_of(x), x))
    print(f"{'checkpoint':14s} {'step':>5s} {'mode':>8s} {'seeds':>5s} {'free_still':>10s} {'free_speed':>10s} "
          f"{'oracle_still':>12s} {'oracle_speed':>12s} {'free_copy':>9s} {'oracle_copy':>11s}")
    for label in order:
        for mode in ('offline', 'engine'):
            if mode not in res[label]:
                continue
            r = res[label][mode]
            print(f"{label:14s} {r['steps']:5d} {mode:>8s} {r['seeds']:5d} {r['free_still']:10.4f} {r['free_speed']:10.4f} "
                  f"{r['oracle_still']:12.4f} {r['oracle_speed']:12.4f} {r['free_copy']:9.4f} {r['oracle_copy']:11.4f}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()

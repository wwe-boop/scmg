"""Clip-level bootstrap 95 % intervals of the main interruption metrics from eval_interrupt.py --per-item files (11-05).
    python scripts/interrupt/boot.py --human human_ref_v2_train_drop --jsonl a.jsonl b.jsonl [--n 1000]
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

ap = argparse.ArgumentParser()
ap.add_argument('--human', required=True); ap.add_argument('--jsonl', nargs='+', required=True); ap.add_argument('--n', type=int, default=1000)
a = ap.parse_args()
H = json.loads(Path(a.human + '.json').read_text()); P95 = H['peak_jerk']['p95']; W = 64 / 30.
hw = np.load(a.human + '_windows.npz'); import icommon as I
Hk = np.stack([I.kin_features(I.fk(w), 30) for w in hw['rr_ctx'].astype(np.float64)])[hw['retract'].astype(bool)]
rows = defaultdict(list)
for f in a.jsonl:
    for l in open(f): r = json.loads(l); rows[r['method']].append(r)


def stats(rs):
    need = [r for r in rs if r['need_R_sil']]
    y = np.array([r['yield_R_sil'] if r['yield_R_sil'] is not None else W for r in need])
    return dict(yield_s=y.mean(), yield_rate=np.mean([r['yield_R_sil'] is not None for r in need]),
                abrupt=np.mean([r['jerk'] > P95 for r in rs]), unsafe=np.mean([r['unsafe_any'] for r in rs]),
                rkd=I.kin_distance(np.array([r['kin'] for r in rs]), Hk)[0])


rng = np.random.default_rng(0)
for m, rs in rows.items():
    by = defaultdict(list)
    for r in rs: by[r['stem']].append(r)
    stems = sorted(by); pt = stats(rs); bs = defaultdict(list)
    for _ in range(a.n):
        smp = [x for s in rng.choice(stems, len(stems)) for x in by[s]]
        for k, v in stats(smp).items(): bs[k].append(v)
    ci = {k: (pt[k], *np.percentile(bs[k], [2.5, 97.5])) for k in pt}
    print(f'{m:14s} ' + ' | '.join(f'{k} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]' for k, v in ci.items()), flush=True)

"""Boundary jerk of fixed-action insertions (11-07 item 5): for every insertion edge (t0 and t1 of generator['lib_inserts']) the
peak upper-body jerk (waist + arm joints, 50 Hz q50, third difference x 50^3, L2 over joints) in +-0.3 s, divided by the same window
of the run WITHOUT insertion (same model, seed, clip). Reports the median ratio and the share of edges with ratio <= 1.5.
    python scripts/semantic/lib_boundary_jerk.py --base <dir> --out res.json label=<dir> ...
"""
import argparse, json
from pathlib import Path
import numpy as np, torch

UP = list(range(22, 39)); W = 0.3; FPS = 50


def jerk(q):
    j = np.zeros(len(q)); d = np.diff(q[:, UP], n=3, axis=0) * FPS ** 3
    j[2:2 + len(d)] = np.linalg.norm(d, axis=-1); return j


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('specs', nargs='+'); a = ap.parse_args(); res = {}
    for spec in a.specs:
        lab, d = spec.split('=', 1); r = []
        for p in sorted(Path(d).glob('*.pt')):
            g = torch.load(p, map_location='cpu', weights_only=False)['generator']
            if not g.get('lib_inserts'): continue
            qi = np.load(p.with_suffix('.q50.npy')); qb = np.load(Path(a.base) / f'{p.stem}.q50.npy'); n = min(len(qi), len(qb))
            ji, jb = jerk(qi[:n]), jerk(qb[:n])
            for ins in g['lib_inserts']:
                for te in (ins['t0'], ins['t1']):
                    lo, hi = max(0, int((te - W) * FPS)), min(n, int((te + W) * FPS) + 1)
                    if hi - lo < 4: continue
                    r.append(float(ji[lo:hi].max() / max(jb[lo:hi].max(), 1e-6)))
        r = np.asarray(r)
        res[lab] = dict(edges=int(len(r)), median_ratio=float(np.median(r)) if len(r) else None,
                        share_le_1_5=float((r <= 1.5).mean()) if len(r) else None, p90_ratio=float(np.percentile(r, 90)) if len(r) else None)
        print(lab, json.dumps(res[lab]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

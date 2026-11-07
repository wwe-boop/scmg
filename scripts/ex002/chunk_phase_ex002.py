"""11-08 EX002 chunk-stutter probe (EX002_SWEEP_1107.md addition 656c942; the G1 beat_phase.py speed-phase probe carried to EX002):
per frame the mean speed of both wrists (static_share.py: exact FK points of scmg/ex002_ctl, central difference at 50 Hz), binned by the
phase of the 16-frame (320 ms) chunk grid (frame index mod 16; bench outputs start on the grid), each run normalised by its own mean;
reports the 16 bin means and max / min. GT: the dev GT files gt_<ci>.npy next to a bench dir.
    python scripts/ex002/chunk_phase_ex002.py --out O.json label=<dir of <ci>.npy> ... [GT=<dir with gt_*.npy>]"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import ex002_ctl as E


def wrist_speed(q):
    g = E.points(np.asarray(q, np.float32))['wrist']                      # [T, 2, 3]
    v = np.zeros(g.shape[:2]); v[1:-1] = np.linalg.norm(g[2:] - g[:-2], axis=-1) * 25.; v[0], v[-1] = v[1], v[-2]
    return v.mean(1)


ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--period', type=int, default=16)
ap.add_argument('specs', nargs='+')
a = ap.parse_args(); res = {}
for spec in a.specs:
    lab, ds = spec.split('=', 1); s = np.zeros(a.period); c = np.zeros(a.period)
    for d in ds.split(','):
        fs = sorted(glob.glob(d + ('/gt_*.npy' if lab == 'GT' else '/[0-9]*.npy')))
        for f in fs:
            v = wrist_speed(np.load(f)); ph = np.arange(len(v)) % a.period
            np.add.at(s, ph, v); np.add.at(c, ph, 1)
    m = s / np.maximum(c, 1); m = m / m.mean()
    res[lab] = dict(bins=m.tolist(), max_over_min=float(m.max() / m.min()))
    print(f'{lab:24s} max/min {m.max() / m.min():.3f} | ' + ' '.join(f'{x:.3f}' for x in m), flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))

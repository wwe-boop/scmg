"""EX002 lane 4 (11-07; criteria docs/criteria/EX002_CTL.md "lane 4"): FGD v2 (ours, validated: the two frozen human_seed{0,1}
MotionAE encoders, 32-frame windows / stride 16, mean of the two) restricted to frames more than R s from every command of a plan.
Frames within +-R s of a command (round(t * 50) +- round(R * 50)) are masked; a 32-frame window is kept only if it holds no masked
frame (fgd-ex002-v2 model.windows mask semantics), for the generated clip and its GT alike (same mask, same length -> same windows).
Clips with no window left are skipped for both. R = 0 means the whole clip (must equal ex002_ctl_run.py eval's FGD_v2).
Does not touch ex002_ctl_run.evaluate.
    python scripts/ex002/ex002_fgd_outside.py --plan plan.json --radius 0,1.0 --out res.json label=<dir> ...   (gt_<ci>.npy in <dir>/..)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

FPS = 50
FGD_CODE = '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/fgd-ex002-v2-20261102/code'


def masks(events, n, r):
    m = np.zeros(n, bool)
    if r <= 0: return m
    R = int(round(r * FPS))
    for e in events:
        f = int(round(e['t'] * FPS)); m[max(0, f - R):min(n, f + R + 1)] = True
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', required=True); ap.add_argument('--radius', default='0,1.0')
    ap.add_argument('--out', required=True); ap.add_argument('specs', nargs='+')
    a = ap.parse_args()
    sys.path.insert(0, FGD_CODE)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from model import Embedder
    from scmg.evaluate import frechet                  # the Frechet of ex002_ctl_run.evaluate (R = 0 reproduces its FGD_v2)
    from config import WEIGHTS
    emb = [Embedder(WEIGHTS / f'human_seed{k}' / 'best.pt') for k in (0, 1)]
    P = json.loads(Path(a.plan).read_text()); by = {}
    for e in P['events']: by.setdefault(e['ci'], []).append(e)
    radii = [float(x) for x in a.radius.split(',')]; res = {}
    for spec in a.specs:
        lab, d = spec.split('=', 1); d = Path(d); r_ = {}
        clips = []
        for f in sorted(d.glob('*.npy'), key=lambda p: int(p.stem)):
            ci = int(f.stem); y = np.load(f); gt = np.load(d.parent / f'gt_{ci}.npy'); n = min(len(y), len(gt)); clips.append((ci, y[:n], gt[:n]))
        for r in radii:
            fg = [[], []]; ft = [[], []]; nw = 0; nfr = 0; nout = 0; skipped = 0
            for ci, y, gt in clips:
                m = masks(by.get(ci, []), len(y), r); nfr += len(m); nout += int((~m).sum())
                try:
                    for k, e_ in enumerate(emb): fg[k].append(e_.features(y, m if r > 0 else None)); ft[k].append(e_.features(gt, m if r > 0 else None))
                except ValueError: skipped += 1; continue
                nw += len(fg[0][-1])
            seeds = [float(frechet(np.concatenate(ft[k]), np.concatenate(fg[k]))) for k in (0, 1)]
            r_[f'R{r:g}'] = dict(FGD_v2=float(np.mean(seeds)), FGD_v2_seeds=seeds, windows=nw, frames_outside_frac=nout / max(nfr, 1),
                                 clips_skipped=skipped)
        res[lab] = r_; print(lab, json.dumps(r_), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

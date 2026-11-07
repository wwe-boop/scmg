"""EX002 module 5 (11-08, default off; criteria docs/criteria/EX002_CTL.md "Module 5"): command-local blend of two generations of the
same model / per-clip seed: y = (1 - w) y_none + w y_given, w(t) = max over the plan's commands of 1 for |t - t_cmd| <= inner and a
cosine ramp to 0 at outer; optional finger override (module 2) after the blend. Writes <out>/<ci>.npy and copies gt_<ci>.npy next to
<out> (evaluate reads the GT from the run dir's parent).
    python ex002_local_blend.py --none D0 --given D1 --plan plan.json --out OUT [--inner 0.6 --outer 0.9] [--finger-override HAND_SHAPES.json]"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
FPS = 50


def weight(T, ts, inner, outer):
    fr = np.arange(T) / FPS; w = np.zeros(T)
    for t in ts:
        d = np.abs(fr - t); x = np.clip((outer - d) / (outer - inner), 0, 1); w = np.maximum(w, 0.5 - 0.5 * np.cos(np.pi * x))
    return w


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--none', required=True); ap.add_argument('--given', required=True)
    ap.add_argument('--plan', required=True); ap.add_argument('--out', required=True); ap.add_argument('--inner', type=float, default=.6)
    ap.add_argument('--outer', type=float, default=.9); ap.add_argument('--finger-override'); ap.add_argument('--mode', default='given')
    a = ap.parse_args()
    from ex002_ctl_run import finger_override, tokens_for
    P = json.loads(Path(a.plan).read_text()); by = {}
    for e in P['events']: by.setdefault(e['ci'], []).append(e)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); n0, n1 = Path(a.none), Path(a.given)
    shapes = json.loads(Path(a.finger_override).read_text())['shapes'] if a.finger_override else None
    for f in sorted(n1.glob('*.npy'), key=lambda p: int(p.stem)):
        ci = int(f.stem); y1 = np.load(f); y0 = np.load(n0 / f.name); T = min(len(y0), len(y1)); y0, y1 = y0[:T], y1[:T]
        w = weight(T, [e['t'] for e in by.get(ci, [])], a.inner, a.outer)[:, None]
        y = ((1 - w) * y0 + w * y1).astype(np.float32)
        if shapes is not None: y = finger_override(y, tokens_for(by.get(ci, []), a.mode), shapes)
        np.save(out / f.name, y)
        g = n1.parent / f'gt_{ci}.npy'
        if g.exists() and not (out.parent / g.name).exists(): shutil.copy(g, out.parent / g.name)
    print('BLEND_DONE', out)


if __name__ == '__main__':
    main()

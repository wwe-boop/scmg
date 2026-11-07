"""EX002 lane 4 (11-07; criteria docs/criteria/EX002_CTL.md "lane 4"): executability after the CBF-QP (scmg/safety.py, EX002
hardware limits limits34 with the 0.03 rad margin) and the boundary jerk of the finger-override ramps.
  executable clip : after CBFFilter, no frame violates position / velocity / acceleration (violation_rate; jerk is best-effort
                    in the CBF and reported separately as a frame rate)
  ramp jerk       : |third difference| x 50^3 of the 16 finger joints on the ramp frames of the plan's finger events
                    (hold 0.4 s < |t - t_cmd| <= 0.55 s, the override's cosine ramps), mean and fraction above the 1500 rad/s^3
                    finger limit; the same frames in every run (comparable with / without the override)
    SCMG_J=34 python scripts/ex002/ex002_exec.py --view V --plan plan.json --out res.json label=<dir> ...
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
FPS = 50


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--plan', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('specs', nargs='+')
    ap.add_argument('--hold', type=float, default=0.4); ap.add_argument('--ramp', type=float, default=0.15)
    a = ap.parse_args()
    from scmg.safety import limits34, CBFFilter, violation_rate
    meta = json.loads((Path(a.view) / 'META.json').read_text())['ex002']
    lim = limits34(meta['lower'], meta['upper'])
    P = json.loads(Path(a.plan).read_text()); by = {}
    for e in P['events']:
        if e['kind'] != 'aim': by.setdefault(e['ci'], []).append(e['t'])
    res = {}
    for spec in a.specs:
        lab, d = spec.split('=', 1); d = Path(d)
        n_clip = n_exec = 0; raw_any = []; post = {k: [] for k in ('pos', 'vel', 'acc', 'jerk')}; dmean = []; rj = []; rj_over = []
        for f in sorted(d.glob('*.npy'), key=lambda p: int(p.stem)):
            ci = int(f.stem); y = np.load(f).astype(np.float64); n_clip += 1
            raw_any.append(violation_rate(y, lim)['any'])
            cb = CBFFilter(lim, y[0]); q = np.stack([cb.step(x) for x in y])
            v = violation_rate(q, lim)
            for k in post: post[k].append(v[k])
            n_exec += int(v['pos'] == 0 and v['vel'] == 0 and v['acc'] == 0); dmean.append(float(np.abs(q - y).mean()))
            j = np.abs(np.diff(y[:, 18:34], n=3, axis=0)) * FPS ** 3          # frame i -> jerk around i + 1.5
            fr = (np.arange(len(j)) + 1.5) / FPS
            for t in by.get(ci, []):
                z = (np.abs(fr - t) > a.hold) & (np.abs(fr - t) <= a.hold + a.ramp)
                if z.any(): rj.append(j[z].mean(1)); rj_over.append((j[z] > lim['j'][18:34]).any(1))
        rj = np.concatenate(rj) if rj else np.zeros(0); rj_over = np.concatenate(rj_over) if rj_over else np.zeros(0)
        r = dict(clips=n_clip, exec_after_cbf=n_exec / max(n_clip, 1), raw_violation_any=float(np.mean(raw_any)),
                 **{f'post_{k}': float(np.mean(x)) for k, x in post.items()}, cbf_mean_abs_rad=float(np.mean(dmean)),
                 ramp_finger_jerk_mean=float(rj.mean()) if len(rj) else None,
                 ramp_frames_over_finger_jerk_limit=float(rj_over.mean()) if len(rj_over) else None, ramp_frames=int(len(rj)))
        res[lab] = r; print(lab, json.dumps(r), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

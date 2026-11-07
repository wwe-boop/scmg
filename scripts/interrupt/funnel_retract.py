"""Convergence-funnel retraction (11-07, zero training, inference only).

Problem (dev, final recipe czm2kws, 900 interrupts): the model's own causal continuation after an interruption is
natural (RKD 0.239, human half-split floor 0.213) but rarely yields (20 %); R0causal (fixed-weight cross-fade into a
critically damped retraction) yields 99.5 % but RKD 0.723. Pinning to a deadline (outpainting) made v3 abruptness 41 %.

Funnel: keep the continuation c inside a time-shrinking tube around the critically damped retraction r (B2 from the
executed frame f_i - 1, velocity matched), on the waist + arm joints:
    y[t] = r[t] + S(c[t] - r[t]; e[t]),   e[t] = e0 * phi(t - t_i) + eps,
    phi(tau) = (1 + w tau) exp(-w tau),  w = 4.744 / T_f   (smooth, phi' = 0 at tau = 0, 95 % shrunk at T_f)
    S(d; e) = d                                       if |d| <= a e
            = sign(d) (a e + (1 - a) e tanh((|d| - a e) / ((1 - a) e)))   otherwise   (C1, |S| < e)
e0 per joint = max(m0, |c[f_i] - r[f_i]| / a) so the first frame is in the identity region (no jump). The output
converges to r (hence to the rest pose) by construction: |y - r| < e -> eps; the continuation is untouched while it
stays inside the tube (its texture survives), and large departures are compressed smoothly instead of pinned.
Non-upper channels (base, legs): the R0causal cross-fade (w0 0.6, T_b 0.5) into the frozen B2 values.
Strictly causal: uses c up to t and the executed frames before t_i only.
    python funnel_retract.py --in-dir C --prefix-from U --out-dir O --interrupts I.json --human H.json --tf 1.2 --m0 0.3
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--icommon', default='/home/xr-25b7/projects/gesture/scmg-wt-floor/scripts/interrupt')
ap.add_argument('--in-dir', required=True); ap.add_argument('--out-dir', required=True); ap.add_argument('--interrupts', required=True)
ap.add_argument('--human', required=True); ap.add_argument('--rest', default='R_sil'); ap.add_argument('--prefix-from', required=True)
ap.add_argument('--tr', type=float, default=1.4, help='T_r of the damped retraction r (s)')
ap.add_argument('--tf', type=float, default=1.2, help='funnel shrink time (95 %%, s)')
ap.add_argument('--m0', type=float, default=0.3, help='minimum initial tube half-width (rad)')
ap.add_argument('--eps', type=float, default=0.02, help='final tube half-width (rad)')
ap.add_argument('--alpha', type=float, default=0.6, help='identity share of the tube')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--smooth', type=float, default=0., help='causal EMA time constant (s) on the residual c - r before the tube (0 = off)')
a = ap.parse_args()
sys.path.insert(0, a.icommon)
import icommon as I


def soft(d, e, al):
    ad = np.abs(d); k = al * e; out = d.copy(); m = ad > k
    out[m] = np.sign(d[m]) * (k[m] + (1 - al) * e[m] * np.tanh((ad[m] - k[m]) / ((1 - al) * e[m])))
    return out


rest = np.array(json.loads(Path(a.human).read_text())['rest'][a.rest]['joints'])
rows = json.loads(Path(a.interrupts).read_text())['interrupts']
keep = set(sorted({r['stem'] for r in rows})[::a.every])
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); n = 0; sat = []
CH = 10 + I.UPPER; w = 4.744 / a.tf
for r in rows:
    if r['stem'] not in keep: continue
    src = Path(a.in_dir) / f"{r['stem']}@{r['frac']:.1f}.pt"
    if not src.exists(): continue
    x = torch.load(src, map_location='cpu', weights_only=False); c = np.asarray(x['robot_repr'], np.float64).copy(); f = r['f_i']
    u = np.asarray(torch.load(Path(a.prefix_from) / f"{r['stem']}.pt", map_location='cpu', weights_only=False)['robot_repr'], np.float64)
    c[:f] = u[:f]
    rr = I.retract_after(c, f, rest, a.tr)
    if len(rr) > len(c): c = np.concatenate([c, np.repeat(c[-1:], len(rr) - len(c), 0)])
    else: rr = rr[:len(c)]
    if len(rr) < len(c): c = c[:len(rr)]
    L = len(c) - f; tau = (np.arange(L) / I.FPS)[:, None]
    # non-upper channels: R0causal cross-fade into the frozen values
    ub = np.clip(tau / 0.5, 0, 1); b = 0.6 + 0.4 * ub * ub * (3 - 2 * ub)
    y = c.copy(); y[f:] = (1 - b) * c[f:] + b * rr[f:]
    d = c[f:, CH] - rr[f:, CH]
    if a.smooth:                                # causal low-pass of the residual (removes the continuation's jerk at the audio cut)
        al = 1 - np.exp(-1 / (I.FPS * a.smooth)); sm = d[0].copy()
        for k in range(len(d)): sm = sm + al * (d[k] - sm); d[k] = sm
    e0 = np.maximum(a.m0, np.abs(d[0]) / a.alpha)
    e = e0[None] * ((1 + w * tau) * np.exp(-w * tau)) + a.eps
    y[f:, CH] = rr[f:, CH] + soft(d, e, a.alpha)
    sat.append(float((np.abs(d) > a.alpha * e).mean()))
    x['robot_repr'] = torch.from_numpy(y.astype(np.float32))
    x.setdefault('generator', {})['funnel_retract'] = dict(source=str(a.in_dir), rest=a.rest, tr=a.tr, tf=a.tf, m0=a.m0, eps=a.eps, alpha=a.alpha, smooth=a.smooth)
    torch.save(x, out / src.name); n += 1
print(f'{n} interrupts -> {out}; share of frames outside the identity region {np.mean(sat):.3f}')

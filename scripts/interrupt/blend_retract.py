"""Joint-space cross-fade retraction (11-05, zero training): the model's own post-interruption continuation blended into
the fixed retraction.

For each interrupt (30 fps, frame f_i): c = the streaming model's continuation after the interruption (an infer_interrupt /
infer_retract output <stem>@<frac>.pt; frames < f_i are the uninterrupted output), r = icommon.retract_after(c, f_i, rest,
T_r) (B2 from the same executed frames: waist + arms critically damped to the rest pose, velocity matched; legs / base held),
    out[t] = (1 - b(t)) c[t] + b(t) r[t]   for t >= f_i,   b(t) = w0 + (1 - w0) smoothstep((t - f_i) / (T_b * 30))
(+ optional --tex g: g b(t) (c - EMA_tau(c)) on waist + arms = the model's own motion texture on top of the retraction)
(all 39 channels; both signals start from the executed motion at f_i - 1, so there is no jump at t_i for any w0).
Frames < f_i are copied unchanged.
    python scripts/interrupt/blend_retract.py --in-dir C --out-dir O --interrupts I.json --human human_ref.json
        [--rest R_sil] [--tr 1.8] [--tb 1.0] [--w0 0.3] [--every n]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import icommon as I

ap = argparse.ArgumentParser()
ap.add_argument('--in-dir', required=True); ap.add_argument('--out-dir', required=True); ap.add_argument('--interrupts', required=True)
ap.add_argument('--human', required=True); ap.add_argument('--rest', default='R_sil'); ap.add_argument('--tr', type=float, default=1.8)
ap.add_argument('--tb', type=float, default=1.0); ap.add_argument('--w0', type=float, default=0.3); ap.add_argument('--every', type=int, default=1)
ap.add_argument('--tex', type=float, default=0., help='add g * b(t) * (c - causal EMA of c) on the waist + arm joints (motion texture)')
ap.add_argument('--tex-tau', type=float, default=0.25)
ap.add_argument('--tex-smooth', type=float, default=0., help='causal EMA (s) applied to the texture (band-pass: less jerk)')
ap.add_argument('--prefix-from', default=None, help='dir of the uninterrupted outputs <stem>.pt: frames < f_i are taken from them '
                'bit for bit (strict causality at 30 fps; the continuation file may differ there by the to30 resampling / decoder look-ahead)')
ap.add_argument('--full-clip', action='store_true', help='inputs are uninterrupted full-clip outputs <stem>.pt (with --w0 1 --tex 0: B2 files)')
ap.add_argument('--tr-json', default=None, help='11-07 end head: per-interrupt T_r (end_head.py apply output; overrides --tr)')
a = ap.parse_args()
trj = json.loads(Path(a.tr_json).read_text())['tr'] if a.tr_json else None
rest = np.array(json.loads(Path(a.human).read_text())['rest'][a.rest]['joints'])
rows = json.loads(Path(a.interrupts).read_text())['interrupts']
keep = set(sorted({r['stem'] for r in rows})[::a.every])
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); n = 0
for r in rows:
    if r['stem'] not in keep: continue
    src = Path(a.in_dir) / f"{r['stem']}@{r['frac']:.1f}.pt"
    if a.full_clip: src = Path(a.in_dir) / f"{r['stem']}.pt"
    if not src.exists(): continue
    x = torch.load(src, map_location='cpu', weights_only=False); c = np.asarray(x['robot_repr'], np.float64); f = r['f_i']
    if a.prefix_from:
        u = np.asarray(torch.load(Path(a.prefix_from) / f"{r['stem']}.pt", map_location='cpu', weights_only=False)['robot_repr'], np.float64)
        c = c.copy(); c[:f] = u[:f]
    tr = trj[f"{r['stem']}@{r['frac']:.1f}"]['tr'] if trj is not None else a.tr
    rr = I.retract_after(c, f, rest, tr)                  # f_i + 64 frames (the clip end is padded by holding)
    if len(rr) > len(c): c = np.concatenate([c, np.repeat(c[-1:], len(rr) - len(c), 0)])   # full clips ending < f_i + 64
    else: rr = rr[:len(c)]
    if len(rr) < len(c): c = c[:len(rr)]
    u = np.clip((np.arange(len(c) - f) / I.FPS) / a.tb, 0, 1)[:, None]
    b = a.w0 + (1 - a.w0) * u * u * (3 - 2 * u)
    y = c.copy(); y[f:] = (1 - b) * c[f:] + b * rr[f:]
    if a.tex:                                   # causal high-pass of the model continuation (EMA from frame f_i on)
        al = 1 - np.exp(-1 / (I.FPS * a.tex_tau)); ema = c[f - 1].copy(); hp = np.zeros_like(c[f:])
        for k in range(len(c) - f): ema = ema + al * (c[f + k] - ema); hp[k] = c[f + k] - ema
        if a.tex_smooth:
            a2 = 1 - np.exp(-1 / (I.FPS * a.tex_smooth)); sm = np.zeros_like(hp[0])
            for k in range(len(hp)): sm = sm + a2 * (hp[k] - sm); hp[k] = sm
        ch = 10 + I.UPPER; y[f:, ch] += a.tex * b[:, :1] * hp[:, ch]
    x['robot_repr'] = torch.from_numpy(y.astype(np.float32))
    x.setdefault('generator', {})['blend_retract'] = dict(source=str(a.in_dir), rest=a.rest, tr=tr, tr_json=a.tr_json, tb=a.tb, w0=a.w0, prefix_from=a.prefix_from, tex=a.tex, tex_tau=a.tex_tau, tex_smooth=a.tex_smooth)
    torch.save(x, out / f"{r['stem']}@{r['frac']:.1f}.pt"); n += 1
print('BLEND_DONE', n)

"""11-08 EX002 ports of the G1 post-hoc interruption rows (zero training, inference only), same code path:
  r0causal : scripts/interrupt/blend_retract.py (out = (1 - b) c + b r, b(t) = w0 + (1 - w0) smoothstep((t - f_i) / T_b), + texture
             g b (c - EMA_tau(c)) [EMA-smoothed] on the upper-body channels); frozen G1 recipe --tr 1.4 --tb 0.5 --w0 0.6 --tex 1.0
             --tex-smooth 0.1 --prefix-from <uninterrupted>
  funnel   : scripts/interrupt/funnel_retract.py (tube around r on the upper-body channels, R0causal cross-fade elsewhere); frozen
             Funnel_p recipe --tr 1.4 --tf 2.2 --m0 1.0 --smooth 0.1 --eps 0.02 --alpha 0.6
The math lives in r0causal_core / funnel_core (representation-agnostic, line for line the G1 scripts); test_retract_post_ex002.py feeds
G1 robot_repr through them and checks bitwise equality with the G1 outputs. EX002 adapter: q34 at 50 Hz (.npy), r =
icommon_ex002.retract_after (waist + arms critically damped to R_sil of the EX002 human tails, head / fingers held), upper channels =
icommon_ex002.UPPER, frames < f_i from the uninterrupted output (executed prefix).
    python scripts/ex002/retract_post_ex002.py r0causal|funnel --in-dir C --prefix-from U --out-dir O --interrupts I.json --human H [--tr 1.4]"""
import argparse, json, sys
from pathlib import Path
import numpy as np


def r0causal_core(c, f, rr, fps, ch, tb=0.5, w0=0.6, tex=1.0, tex_tau=0.25, tex_smooth=0.1):
    """= blend_retract.py from 'u = np.clip(...)' on (c, rr already aligned, c[:f] = executed prefix)."""
    u = np.clip((np.arange(len(c) - f) / fps) / tb, 0, 1)[:, None]
    b = w0 + (1 - w0) * u * u * (3 - 2 * u)
    y = c.copy(); y[f:] = (1 - b) * c[f:] + b * rr[f:]
    if tex:
        al = 1 - np.exp(-1 / (fps * tex_tau)); ema = c[f - 1].copy(); hp = np.zeros_like(c[f:])
        for k in range(len(c) - f): ema = ema + al * (c[f + k] - ema); hp[k] = c[f + k] - ema
        if tex_smooth:
            a2 = 1 - np.exp(-1 / (fps * tex_smooth)); sm = np.zeros_like(hp[0])
            for k in range(len(hp)): sm = sm + a2 * (hp[k] - sm); hp[k] = sm
        y[f:, ch] += tex * b[:, :1] * hp[:, ch]
    return y


def _soft(d, e, al):
    ad = np.abs(d); k = al * e; out = d.copy(); m = ad > k
    out[m] = np.sign(d[m]) * (k[m] + (1 - al) * e[m] * np.tanh((ad[m] - k[m]) / ((1 - al) * e[m])))
    return out


def funnel_core(c, f, rr, fps, ch, tf=2.2, m0=1.0, eps=0.02, alpha=0.6, smooth=0.1):
    """= funnel_retract.py from 'L = len(c) - f' on."""
    w = 4.744 / tf
    L = len(c) - f; tau = (np.arange(L) / fps)[:, None]
    ub = np.clip(tau / 0.5, 0, 1); b = 0.6 + 0.4 * ub * ub * (3 - 2 * ub)
    y = c.copy(); y[f:] = (1 - b) * c[f:] + b * rr[f:]
    d = c[f:, ch] - rr[f:, ch]
    if smooth:
        al = 1 - np.exp(-1 / (fps * smooth)); sm = d[0].copy()
        for k in range(len(d)): sm = sm + al * (d[k] - sm); d[k] = sm
    e0 = np.maximum(m0, np.abs(d[0]) / alpha)
    e = e0[None] * ((1 + w * tau) * np.exp(-w * tau)) + eps
    y[f:, ch] = rr[f:, ch] + _soft(d, e, alpha)
    return y, float((np.abs(d) > alpha * e).mean())


def align(c, rr):
    """= the length handling of both G1 scripts."""
    if len(rr) > len(c): c = np.concatenate([c, np.repeat(c[-1:], len(rr) - len(c), 0)])
    else: rr = rr[:len(c)]
    if len(rr) < len(c): c = c[:len(rr)]
    return c, rr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['r0causal', 'funnel'])
    ap.add_argument('--in-dir', required=True); ap.add_argument('--prefix-from', required=True); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--interrupts', required=True); ap.add_argument('--human', required=True, help='human_ref_ex002 prefix (.json)')
    ap.add_argument('--tr', type=float, default=1.4, help='T_r of the damped retraction inside the row (frozen G1 recipes: 1.4)')
    ap.add_argument('--tf', type=float, default=2.2); ap.add_argument('--m0', type=float, default=1.0)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import icommon_ex002 as I
    rest = np.array(json.loads(Path(a.human + '.json').read_text())['rest']['R_sil']['joints'])
    rows = json.loads(Path(a.interrupts).read_text())['interrupts']; out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    n = 0; sat = []
    for r in rows:
        name = f"{r['ci']}@{r['item']}@{r['frac']:.1f}.npy"; src = Path(a.in_dir) / name
        if not src.exists(): continue
        c = np.load(src).astype(np.float64).copy(); f = int(r['f_i'])
        u = np.load(Path(a.prefix_from) / f"{r['ci']}.npy").astype(np.float64); c[:f] = u[:f]
        rr = I.retract_after(c, f, rest, a.tr); c, rr = align(c, rr)
        if a.mode == 'r0causal': y = r0causal_core(c, f, rr, I.FPS, I.UPPER)
        else: y, s = funnel_core(c, f, rr, I.FPS, I.UPPER, tf=a.tf, m0=a.m0); sat.append(s)
        np.save(out / name, y.astype(np.float32)); n += 1
    (out / 'PROTOCOL.json').write_text(json.dumps(dict(mode=a.mode, src=a.in_dir, prefix_from=a.prefix_from, human=a.human, tr=a.tr,
                                                       tf=a.tf if a.mode == 'funnel' else None, m0=a.m0 if a.mode == 'funnel' else None, n=n), indent=1))
    print(f'{a.mode.upper()}_DONE', n, f'sat {np.mean(sat):.3f}' if sat else '')


if __name__ == '__main__':
    main()

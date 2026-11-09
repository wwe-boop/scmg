"""Summarize energy_probe.py outputs: does generated motion energy follow GT energy per clip / per speaker, what drives
BA, and how much of GT energy is predictable from audio prosody alone (an upper bound for an audio-driven calibrator).
    python scripts/echog_g1/energy_probe_summary.py label=probe.jsonl [label=probe.jsonl ...]
"""
import json, sys
import numpy as np


def ols(x, y):
    A = np.c_[x, np.ones_like(x)]; k, b = np.linalg.lstsq(A, y, rcond=None)[0]; return k, b


def r2_audio(R, target):
    X = np.array([[r['a_db_mean'], r['a_db_std'], r['a_onset_rate']] for r in R]); y = np.array([r[target] for r in R])
    X = np.c_[(X - X.mean(0)) / (X.std(0) + 1e-9), np.ones(len(X))]
    # 5-fold cross-validated R^2 (no in-sample optimism)
    idx = np.arange(len(y)); rng = np.random.default_rng(0); rng.shuffle(idx); pred = np.zeros_like(y)
    for f in range(5):
        te = idx[f::5]; tr = np.setdiff1d(idx, te)
        w = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; pred[te] = X[te] @ w
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


for spec in sys.argv[1:]:
    lab, f = spec.split('=', 1)
    R = [json.loads(l) for l in open(f)]
    R = [r for r in R if np.isfinite(r['gen_ba']) and np.isfinite(r['gt_ba'])]
    w = np.array([r['T'] for r in R], float)
    g = lambda k: np.array([r[k] for r in R], float)
    print(f'== {lab}: {len(R)} clips, {len({r["spk"] for r in R})} speakers')
    for m in ('div', 'ba', 'beats_per_s', 'frac_bodies_beats', 'norm_speed'):
        x, y = g('gt_' + m), g('gen_' + m)
        k, b = ols(x, y)
        print(f'  {m:18s} weighted mean gen {np.average(y, weights=w):.3f} gt {np.average(x, weights=w):.3f} | per-clip corr {np.corrcoef(x, y)[0,1]:+.2f} slope {k:.2f}')
    spk = sorted({r['spk'] for r in R})
    if len(spk) >= 3:
        for m in ('div', 'ba'):
            sg = np.array([np.average(g('gen_' + m)[[r['spk'] == s for r in R]], weights=w[[r['spk'] == s for r in R]]) for s in spk])
            st = np.array([np.average(g('gt_' + m)[[r['spk'] == s for r in R]], weights=w[[r['spk'] == s for r in R]]) for s in spk])
            k, b = ols(st, sg)
            print(f'  per-speaker {m}: corr {np.corrcoef(st, sg)[0,1]:+.2f} slope {k:.2f} | mean |gen-gt| {np.mean(np.abs(sg - st)):.3f} | range gt {st.min():.3f}-{st.max():.3f} gen {sg.min():.3f}-{sg.max():.3f}')
    else:
        for s in spk:
            m_ = [r['spk'] == s for r in R]
            print(f'  {s}: div gen {np.average(g("gen_div")[m_], weights=w[m_]):.3f} gt {np.average(g("gt_div")[m_], weights=w[m_]):.3f} | ba gen {np.average(g("gen_ba")[m_], weights=w[m_]):.3f} gt {np.average(g("gt_ba")[m_], weights=w[m_]):.3f}')
    print(f'  BA vs ingredients (GT clips): corr(ba, beats/s) {np.corrcoef(g("gt_ba"), g("gt_beats_per_s"))[0,1]:+.2f}  corr(ba, frac bodies) {np.corrcoef(g("gt_ba"), g("gt_frac_bodies_beats"))[0,1]:+.2f}  corr(div, norm speed) {np.corrcoef(g("gt_div"), g("gt_norm_speed"))[0,1]:+.2f}')
    print(f'  audio prosody -> GT div: 5-fold R2 {r2_audio(R, "gt_div"):.2f} | -> GT norm speed {r2_audio(R, "gt_norm_speed"):.2f} | -> GT beats/s {r2_audio(R, "gt_beats_per_s"):.2f}')

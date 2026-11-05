"""Control-token metrics on G1 dev predictions (11-05). Definitions: scmg/ctl.py (geometry, Score_a / Score_b / STG of
arXiv 2610.03105) and g1_ctl_apex.py (frozen apex detector, action-speed threshold).
    python scripts/semantic/g1_ctl_eval.py --ctl <dir> --view <hos view> --out res.json [--pair-none <dir>] label=<pred dir>|- ...
<pred dir>: infer_test.py output with --save50 (<stem>.q50.npy) and, for runs WITH tokens, generator['ctl'] (the tokens given);
'-' = the ground-truth motion. Each run is scored against
  given : the tokens it received (none-mode runs: none -> only 'gt' is reported)
  gt    : the dev ground-truth tokens of CTL.json (same time / span; for shuffle / amp runs this is NOT what was asked)
Per run (bootstrap 95 % CI over tokens, 2000 resamples, seed 0):
  point_*   (POINT tokens with a target, against 'given' targets): Score_a, Score_b, STG, ray error (deg) at a_hat
  amp_hit   amplitude bin of the generated span == the token's bin (annotation tokens with a bin); amp_mean (m)
  act_03    P(peak wrist speed (max over hands) within [t - 0.3, t + 0.3] s > action_speed) at every annotation token time
  dq_span   (with --pair-none) mean |q - q_none| over the arm joints inside the token spans (rad): is the token used?
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from scmg import ctl
from g1_ctl_apex import Detector, window, speed, gt_geo, FPS

ARMS = list(range(15, 29))     # within the 29 joints


def ci(x, n=2000):
    x = np.asarray(x, np.float64)
    if len(x) == 0: return dict(mean=None, lo=None, hi=None, n=0)
    r = np.random.default_rng(0); b = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), n=int(len(x)))


def score(geo, toks, det, edges, thr):
    T = len(geo['wrist']); spm = np.maximum(speed(geo['wrist'][:, 0]), speed(geo['wrist'][:, 1]))
    out = dict(sa=[], sb=[], stg=[], ray=[], hit=[], amp=[], act=[])
    for t in toks:
        if ctl.group_of(t['type']) != 0: continue
        fa, fb = int(round(t['a'] * FPS)), min(T, int(round(t['b'] * FPS)))
        if fb - fa < 2: continue
        tf = int(round(t['t'] * FPS)); w = spm[max(0, tf - 15):min(T, tf + 16)]
        if len(w): out['act'].append(float(w.max() > thr))
        cls = ctl.TYPE_NAMES[t['type']]
        if t.get('amp_bin') is not None:
            m = ctl.measure(geo, fa, fb, hand=t.get('hand') if t['type'] == ctl.POINT else None)
            amp = m['amp_hand'] if t['type'] == ctl.POINT and t.get('hand') is not None else m['amp_max']
            out['amp'].append(amp); out['hit'].append(float(ctl.amp_bin(amp, edges[cls]) == int(t['amp_bin'])))
        if t['type'] == ctl.POINT and t.get('target') is not None and t.get('hand') is not None:
            f0, f1 = window(t, T); ah = det.apex(geo, int(t['hand']), f0, f1)
            sa = ctl.score_a(ah / FPS, t['apex'])
            wr = geo['wrist'][ah, int(t['hand'])]; sb, th = ctl.score_b(wr, ctl.ray_at(geo, ah, int(t['hand'])), t['target'])
            out['sa'].append(sa); out['sb'].append(sb); out['stg'].append(sa * sb); out['ray'].append(th)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ctl', required=True); ap.add_argument('--view', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--pair-none', default=None); ap.add_argument('specs', nargs='+')
    a = ap.parse_args(); torch.set_num_threads(4)
    raw = json.loads((Path(a.ctl) / 'CTL.json').read_text()); prep = json.loads((Path(a.ctl) / 'PREP.json').read_text())
    det_d = json.loads((Path(a.ctl) / 'APEX.json').read_text()); det = Detector(det_d); thr = det_d['action_speed']
    edges = prep['amp_edges']
    dev = sorted(s for s, ts in raw.items() if ts and ts[0]['split'] == 'dev')
    gtg = gt_geo(a.view, dev)
    res = {}
    for spec in a.specs:
        lab, d = spec.split('=', 1)
        agg = {k: {m: [] for m in ('sa', 'sb', 'stg', 'ray', 'hit', 'amp', 'act')} for k in ('given', 'gt')}
        dq = []; n = 0
        for s in dev:
            if d == '-':
                geo = gtg[s]; given = raw[s]
            else:
                f = Path(d) / f'{s}.q50.npy'
                if not f.exists(): continue
                q = np.load(f)[:, 10:39]; geo = ctl.geometry_np(q)
                gen = torch.load(Path(d) / f'{s}.pt', map_location='cpu', weights_only=False)['generator']
                given = gen.get('ctl') if 'ctl_mode' in gen else None
                if a.pair_none and given and 'stream' not in gen:     # |dq| only against the offline no-token run
                    qn = np.load(Path(a.pair_none) / f'{s}.q50.npy')[:, 10:39]; L = min(len(q), len(qn))
                    for t in given:
                        if ctl.group_of(t['type']) == 0:
                            fa, fb = int(round(t['a'] * FPS)), min(L, int(round(t['b'] * FPS)))
                            if fb > fa: dq.append(float(np.abs(q[fa:fb, ARMS] - qn[fa:fb, ARMS]).mean()))
            n += 1
            for k, toks in (('given', given), ('gt', raw[s])):
                if not toks: continue
                o = score(geo, toks, det, edges, thr)
                for m in o: agg[k][m] += o[m]
        r = dict(clips=n)
        for k in agg:
            if not agg[k]['act']: continue
            r[k] = dict(point_score_a=ci(agg[k]['sa']), point_score_b=ci(agg[k]['sb']), point_stg=ci(agg[k]['stg']),
                        point_ray_deg=ci(agg[k]['ray']), amp_hit=ci(agg[k]['hit']), amp_mean=ci(agg[k]['amp']), act_03=ci(agg[k]['act']))
        if dq: r['dq_span_rad'] = ci(dq)
        res[lab] = r
        short = {k: {m: (round(v['mean'], 3) if isinstance(v, dict) and v.get('mean') is not None else None) for m, v in r[k].items()}
                 for k in ('given', 'gt') if k in r}
        print(lab, n, json.dumps(short), json.dumps(r.get('dq_span_rad')), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

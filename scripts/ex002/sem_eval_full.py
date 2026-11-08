"""P2 sem_cf full-run metric (11-09; exp/text-path docs/criteria/P2_SEMCF_1109.md "Full run + controls"): descriptor error E at the
oracle events, pooled over seeds, with recording-clustered bootstrap CIs, gates G1 / G2, realisation ratios.
Runs: label=dir_s0,dir_s1,dir_s2 (one dir per seed; GT gt_<ci>.npy from --gt). Labels 'none', 'given', 'swapped' are gated
(G1: E(given) <= 0.7 E(none); G2: swapped follow); any other label is reported (E vs oracle; 'swapped*' also vs swapped d;
'retrieval*' also vs its own retrieved descriptors at token time + 0.2 + lag).
    SCMG_J=34 python scripts/ex002/sem_eval_full.py --sem SEM --plans DIR --gt GTDIR --out O.json none=a,b,c given=... [--plan-file PLAN_oracle.json]"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import sem_desc as SD


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sem', required=True); ap.add_argument('--plans', required=True); ap.add_argument('--gt', required=True)
    ap.add_argument('--plan-file', default='PLAN_oracle.json'); ap.add_argument('--out', required=True); ap.add_argument('specs', nargs='+'); a = ap.parse_args()
    st = json.loads((Path(a.sem) / 'SEM_BANK.json').read_text()); mu, sd = np.asarray(st['mu']), np.asarray(st['sd'])
    Bk = np.load(Path(a.sem) / 'SEM_BANK.npz'); lag = float(np.median(Bk['t'] - (Bk['t_tok'] + 0.2)))
    P = json.loads((Path(a.plans) / a.plan_file).read_text())['events']
    RP = Path(a.plans) / 'PLAN_retrieval.json'; R = json.loads(RP.read_text())['events'] if RP.exists() else []
    std = lambda raw: np.concatenate([(raw[:SD.NS] - mu) / sd, raw[SD.NS:]]); raw_of = lambda D: np.concatenate([np.asarray(D[:SD.NS]) * sd + mu, D[SD.NS:]])
    E = lambda D, d: float(np.sum((D - np.asarray(d)) ** 2) / SD.ND)
    def Dof(y, t, h):
        c = int(round(t * SD.FPS))
        if c - SD.NW // 2 < 0 or c + SD.NW // 2 >= len(y): return None, None
        raw = SD.np_desc(SD.window(y, t), h); return std(raw), raw
    runs = {k: v.split(',') for k, v in (s.split('=', 1) for s in a.specs)}; rng = np.random.default_rng(0)
    rows = {}
    for lab, dirs in runs.items():
        out = []
        for si, d in enumerate(dirs):
            cache = {}
            for e in P:
                if e['ci'] not in cache:
                    f = Path(d) / f"{e['ci']}.npy"; cache[e['ci']] = np.load(f) if f.exists() else None
                y = cache[e['ci']]
                if y is None: continue
                D0, r0 = Dof(y, e['ev_t'], e['hand'])
                if D0 is None: continue
                row = dict(rec=e['clip'], seed=si, E=E(D0, e['desc']), amp_ratio=r0[0] / max(raw_of(e['desc'])[0], 1e-6), vpk_ratio=r0[3] / max(raw_of(e['desc'])[3], 1e-6))
                if lab.startswith('swapped'):
                    D1, r1 = Dof(y, e['ev_t'], e['shuffle']['hand']); row['E_sw'] = E(D1, e['shuffle']['desc'])
                    row['amp_ratio_sw'] = r1[0] / max(raw_of(e['shuffle']['desc'])[0], 1e-6); row['vpk_ratio_sw'] = r1[3] / max(raw_of(e['shuffle']['desc'])[3], 1e-6)
                out.append(row)
            if lab.startswith('retrieval'):
                for e in R:
                    y = cache.get(e['ci'])
                    if y is None:
                        f = Path(d) / f"{e['ci']}.npy"; y = np.load(f) if f.exists() else None
                    if y is None: continue
                    D0, _ = Dof(y, e['t'] + 0.2 + lag, e['hand'])
                    if D0 is not None: out.append(dict(rec=e['clip'], seed=si, E_own=E(D0, e['desc']), own=True))
        rows[lab] = out
    def arr(lab, key, own=False): return [r for r in rows[lab] if (r.get('own', False) == own) and key in r]
    res = dict(n_events=len(P), lag=lag, seeds=len(next(iter(runs.values()))))
    for lab in rows:
        x = arr(lab, 'E'); r = dict(E=float(np.mean([z['E'] for z in x])), n=len(x),
                                     realisation=dict(amp=[float(np.percentile([z['amp_ratio'] for z in x], q)) for q in (25, 50, 75)],
                                                      vpk=[float(np.percentile([z['vpk_ratio'] for z in x], q)) for q in (25, 50, 75)]))
        if lab.startswith('swapped'):
            r['E_sw'] = float(np.mean([z['E_sw'] for z in x]))
            r['realisation_sw'] = dict(amp=float(np.median([z['amp_ratio_sw'] for z in x])), vpk=float(np.median([z['vpk_ratio_sw'] for z in x])))
        o = arr(lab, 'E_own', own=True)
        if o: r['E_own_retrieved'] = float(np.mean([z['E_own'] for z in o])); r['n_own'] = len(o)
        res[lab] = r
    recs = sorted({z['rec'] for v in rows.values() for z in v})
    def boot(fn):
        bs = []
        for _ in range(2000):
            pick = rng.choice(recs, len(recs)); bs.append(fn(pick))
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    def mean_on(lab, key, pick):
        by = {}
        for z in arr(lab, key): by.setdefault(z['rec'], []).append(z[key])
        v = [x for p in pick for x in by.get(p, [])]; return np.mean(v) if v else np.nan
    if 'none' in rows and 'given' in rows:
        red = 1 - res['given']['E'] / res['none']['E']; ci = boot(lambda pk: 1 - mean_on('given', 'E', pk) / mean_on('none', 'E', pk))
        res['G1'] = dict(rule='E(given) <= 0.70 E(none), CI of the reduction above 0', reduction=float(red), ci95=ci, pass_=bool(red >= 0.30 and ci[0] > 0))
    if 'swapped' in rows:
        dd = res['swapped']['E_sw'] - res['swapped']['E']; ci = boot(lambda pk: mean_on('swapped', 'E_sw', pk) - mean_on('swapped', 'E', pk))
        res['G2'] = dict(rule='E(swapped run, swapped d) < E(swapped run, oracle d), CI below 0', diff=float(dd), ci95=ci, pass_=bool(dd < 0 and ci[1] < 0))
    for lab in rows:                                   # every other given-like row: reduction vs none with CI
        if lab in ('none', 'given', 'swapped') or not lab.startswith(('c1', 'c2', 'c3', 'psync', 'tts', 'retrieval')): continue
        base = 'psync_none' if lab.startswith(('psync', 'tts')) and 'psync_none' in rows else 'none'
        if base not in rows or not arr(lab, 'E'): continue
        res[f'reduction_{lab}_vs_{base}'] = dict(value=float(1 - res[lab]['E'] / res[base]['E']), ci95=boot(lambda pk: 1 - mean_on(lab, 'E', pk) / mean_on(base, 'E', pk)))
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

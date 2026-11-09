"""P2 sem_cf quick-lane metric (11-09; exp/text-path docs/criteria/P2_SEMCF_1109.md §1, §5): descriptor realisation error E at the
oracle events of PLAN_oracle.json (window at the event's GT time ev_t, the commanded hand).
  none / given : E(D(run, hand), oracle d)                 swapped : E(D(run, swapped hand), swapped d) vs E(D(run, hand), oracle d)
  retrieval    : E(D(run, hand), oracle d) at the oracle events, and E vs its own retrieved descriptors at (token time + 0.2 s +
                 the bank's median lag from keyword onset to segment midpoint)
  GT sanity    : E(D(GT), oracle d) (= V1, 0) and E(D(GT), swapped d) (mismatched)
Kill: K1 E(given) > 0.7 E(none); K2 E(swapped run, swapped d) >= E(swapped run, oracle d).
    SCMG_J=34 python scripts/ex002/sem_eval.py --sem SEM_DIR --plans DIR --gt GTDIR --out O.json none=D given=D swapped=D retrieval=D [ttsync=D psync=D]"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import sem_desc as SD


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sem', required=True); ap.add_argument('--plans', required=True); ap.add_argument('--gt', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('specs', nargs='+'); a = ap.parse_args()
    st = json.loads((Path(a.sem) / 'SEM_BANK.json').read_text()); mu, sd = np.asarray(st['mu']), np.asarray(st['sd'])
    Bk = np.load(Path(a.sem) / 'SEM_BANK.npz'); lag = float(np.median(Bk['t'] - (Bk['t_tok'] + 0.2)))
    P = json.loads((Path(a.plans) / 'PLAN_oracle.json').read_text())['events']
    R = json.loads((Path(a.plans) / 'PLAN_retrieval.json').read_text())['events']
    std = lambda raw: np.concatenate([(raw[:SD.NS] - mu) / sd, raw[SD.NS:]])
    E = lambda D, d: float(np.sum((D - np.asarray(d)) ** 2) / SD.ND)
    def Dof(y, t, h):
        c = int(round(t * SD.FPS))
        if c - SD.NW // 2 < 0 or c + SD.NW // 2 >= len(y): return None
        return std(SD.np_desc(SD.window(y, t), h))
    load = lambda d, ci: np.load(Path(d) / f'{ci}.npy') if (Path(d) / f'{ci}.npy').exists() else None
    res = dict(n_events=len(P), lag_kw_to_mid=lag); runs = dict(s.split('=', 1) for s in a.specs)
    gts = {ci: np.load(Path(a.gt) / f'gt_{ci}.npy') for ci in {e['ci'] for e in P}}
    g_own, g_mis = [], []
    for e in P:
        D0 = Dof(gts[e['ci']], e['ev_t'], e['hand']); D1 = Dof(gts[e['ci']], e['ev_t'], e['shuffle']['hand'])
        if D0 is not None: g_own.append(E(D0, e['desc'])); g_mis.append(E(D1, e['shuffle']['desc']))
    res['GT'] = dict(own=float(np.mean(g_own)), mismatched=float(np.mean(g_mis)), n=len(g_own))
    for lab, d in runs.items():
        o, s_sw = [], []
        for e in P:
            y = load(d, e['ci'])
            if y is None: continue
            D0 = Dof(y, e['ev_t'], e['hand'])
            if D0 is None: continue
            o.append(E(D0, e['desc']))
            if lab.startswith('swapped'): s_sw.append(E(Dof(y, e['ev_t'], e['shuffle']['hand']), e['shuffle']['desc']))
        r = dict(E_oracle=float(np.mean(o)) if o else None, n=len(o))
        if s_sw: r['E_swapped_desc'] = float(np.mean(s_sw))
        if lab.startswith('retrieval'):
            own = []
            for e in R:
                y = load(d, e['ci'])
                if y is None: continue
                D0 = Dof(y, e['t'] + 0.2 + lag, e['hand'])
                if D0 is not None: own.append(E(D0, e['desc']))
            r['E_own_retrieved'] = float(np.mean(own)) if own else None; r['n_own'] = len(own)
        res[lab] = r
    if 'none' in res and 'given' in res:
        en, eg = res['none']['E_oracle'], res['given']['E_oracle']
        res['K1'] = dict(rule='E(given) <= 0.7 E(none)', none=en, given=eg, reduction=(en - eg) / en, kill=eg > 0.7 * en)
    if 'swapped' in res:
        res['K2'] = dict(rule='E(swapped run, swapped d) < E(swapped run, oracle d)', swapped_d=res['swapped']['E_swapped_desc'],
                         oracle_d=res['swapped']['E_oracle'], kill=res['swapped']['E_swapped_desc'] >= res['swapped']['E_oracle'])
    if 'K1' in res and 'K2' in res: res['VERDICT'] = 'KILL' if (res['K1']['kill'] or res['K2']['kill']) else 'PASS'
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

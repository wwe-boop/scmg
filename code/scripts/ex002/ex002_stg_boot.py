"""EX002 STG gain with a recording-clustered bootstrap (11-08; criteria docs/criteria/EX002_CTL.md "Module 5b"). Per aim event the
official forearm-ray STG and our index-ray STG are recomputed exactly as ex002_ctl_run.evaluate does (apex detector APEX.json on
+-1 s, score_a against the command time, score_b against the given command's target). Pairs given / none runs of the same seed.
Gain = event-weighted mean STG(given) - STG(none), pooled over the seeds; 95 % CI by resampling the dev recordings (clips) with
replacement (all seeds of a clip move together). The evaluate means are reproduced and printed as a check.
    python ex002_stg_boot.py --view V --plan plan.json --ctl tokens --out res.json --pair seed0:<given dir>:<none dir> ... [--n 10000]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
FPS = 50


def per_event(d, by, det, E, ctl, shuffled=False):
    out = {}
    for f in sorted(Path(d).glob('*.npy'), key=lambda p: int(p.stem)):
        ci = int(f.stem); y = np.load(f); gt = np.load(Path(d).parent / f'gt_{ci}.npy'); n = min(len(y), len(gt)); y = y[:n]
        geo = E.points(y); rows = []
        for e in by.get(ci, []):
            fe = int(round(e['t'] * FPS))
            if fe >= n or e['kind'] != 'aim': continue
            v = dict(e, **e['shuffle']) if shuffled else e
            h = v['hand']; f0, f1 = max(0, fe - FPS), min(n, fe + FPS); ah = det.apex(geo, h, f0, f1)
            sa = ctl.score_a(ah / FPS, e['t']); rr = E.rays(geo, ah, h)
            sb, _ = ctl.score_b(rr['forearm'][0], rr['forearm'][1], v['target']); sbi, _ = ctl.score_b(rr['index'][0], rr['index'][1], v['target'])
            rows.append((sa * sb, sa * sbi))
        out[ci] = np.array(rows).reshape(-1, 2)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--plan', required=True)
    ap.add_argument('--ctl', required=True); ap.add_argument('--out', required=True); ap.add_argument('--pair', action='append', required=True)
    ap.add_argument('--n', type=int, default=10000); ap.add_argument('--seed', type=int, default=20261108)
    a = ap.parse_args()
    from scmg import ex002_ctl as E, ctl
    from ex002_apex import Detector
    det = Detector(json.loads((Path(a.ctl) / 'APEX.json').read_text()))
    P = json.loads(Path(a.plan).read_text()); by = {}
    for e in P['events']: by.setdefault(e['ci'], []).append(e)
    G, N, res = [], [], dict(pairs={})
    for p in a.pair:
        lab, dg, dn = p.split(':'); g = per_event(dg, by, det, E, ctl); nn = per_event(dn, by, det, E, ctl)
        clips = sorted(set(g) & set(nn)); G.append({c: g[c] for c in clips}); N.append({c: nn[c] for c in clips})
        sg = np.concatenate([g[c] for c in clips]); sn = np.concatenate([nn[c] for c in clips])
        res['pairs'][lab] = dict(stg_given=float(sg[:, 0].mean()), stg_none=float(sn[:, 0].mean()), gain=float(sg[:, 0].mean() - sn[:, 0].mean()),
                                 stgi_given=float(sg[:, 1].mean()), stgi_none=float(sn[:, 1].mean()), gain_index=float(sg[:, 1].mean() - sn[:, 1].mean()),
                                 events=len(sg))
        print(lab, json.dumps({k: round(v, 4) for k, v in res['pairs'][lab].items()}), flush=True)
    clips = sorted(set.intersection(*[set(x) for x in G]))
    # per clip sums over seeds: diff sum and event count (event-weighted gain = sum diff / sum events)
    D = np.array([[sum(float((G[s][c][:, k] - N[s][c][:, k]).sum()) for s in range(len(G))) for k in (0, 1)] for c in clips])
    C = np.array([sum(len(G[s][c]) for s in range(len(G))) for c in clips], float)
    keep = C > 0; D, C = D[keep], C[keep]
    point = D.sum(0) / C.sum(); rng = np.random.default_rng(a.seed); bs = []
    for _ in range(a.n):
        i = rng.integers(0, len(C), len(C)); bs.append(D[i].sum(0) / C[i].sum())
    bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    res.update(clips=int(keep.sum()), events_pooled=int(C.sum()), gain_forearm=float(point[0]), ci95_forearm=[float(lo[0]), float(hi[0])],
               gain_index=float(point[1]), ci95_index=[float(lo[1]), float(hi[1])], p_gain_forearm_below_0_10=float((bs[:, 0] < .10).mean()),
               bootstrap=dict(n=a.n, seed=a.seed, unit='dev recording (clip), all seeds together'))
    print('BOOT', json.dumps({k: res[k] for k in ('clips', 'events_pooled', 'gain_forearm', 'ci95_forearm', 'gain_index', 'ci95_index', 'p_gain_forearm_below_0_10')}))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

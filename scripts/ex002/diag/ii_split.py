"""11-08 read-only (coordinator): where the (ii) cost of the token path sits on the sparse plan (enc-s0, plans 1112, seeds 0-2).
FGD v2 windows (32 frames, stride 16) of each clip: inside = overlaps +-1 s of a command, outside = no overlap (= the (i) set).
Per run: FGD whole (= (ii)), outside, inside-only, and 'swap' = whole-clip FGD with the inside windows taken from the none run of the
same seed (excess attributable to the command windows = whole - swap); per-window Mahalanobis d^2 to the GT window Gaussian."""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/fgd-ex002-v2-20261102/code'); sys.path.insert(0, sys.argv[2])
from model import Embedder, windows
from config import WEIGHTS
from scmg.evaluate import frechet
X = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002'); M, C5 = X / 'm9b-enc', X / 'm5c-enc'
P = json.loads((X / 'PLAN_dev_m9b_sparse.json').read_text()); by = {}
for e in P['events']: by.setdefault(e['ci'], []).append(int(round(e['t'] * 50)))
emb = [Embedder(WEIGHTS / f'human_seed{k}' / 'best.pt') for k in (0, 1)]


def feats(d, ci, n):
    y = np.load(d / f'{ci}.npy')[:n]; return [e.features(y) for e in emb]


R = {}
for pr in ('offline', 'psync'):
    G = M / pr / 'gen'
    runs = {'none': lambda s: G / f'none_s{s}', 'tok5cplain': lambda s: C5 / pr / 'tok5cplain' / f's_given_s{s}', 'tok5c': lambda s: C5 / pr / 'tok5c' / f's_given_s{s}'}
    for s in range(3):
        cis = sorted(int(f.stem) for f in (G / f'none_s{s}').glob('[0-9]*.npy'))
        GT, INS, F = [[], []], [], {r: [[], []] for r in runs}
        for ci in cis:
            gt = np.load(G.parent / 'gen' / f'gt_{ci}.npy'); y0 = np.load(G / f'none_s{s}' / f'{ci}.npy'); n = min(len(gt), len(y0))
            st = np.arange(0, n - 32 + 1, 16); ins = np.array([any(abs(f - (a + 16)) <= 50 + 16 for f in by.get(ci, [])) for a in st])
            g = [e.features(gt[:n]) for e in emb]
            for k in (0, 1): GT[k].append(g[k])
            INS.append(ins)
            for r, fn in runs.items():
                f = feats(fn(s), ci, n)
                for k in (0, 1): F[r][k].append(f[k])
        ins = np.concatenate(INS)
        gt = [np.concatenate(GT[k]) for k in (0, 1)]
        for r in runs:
            ff = [np.concatenate(F[r][k]) for k in (0, 1)]; f0 = [np.concatenate(F['none'][k]) for k in (0, 1)]
            fr = lambda A, B: float(np.mean([frechet(A[k], B[k]) for k in (0, 1)]))
            sw = [np.where(ins[:, None], f0[k], ff[k]) for k in (0, 1)]
            mu = [gt[k].mean(0) for k in (0, 1)]; ci_ = [np.linalg.pinv(np.cov(gt[k].T)) for k in (0, 1)]
            d2 = np.mean([np.einsum('ij,jk,ik->i', ff[k] - mu[k], ci_[k], ff[k] - mu[k]) for k in (0, 1)], 0)
            row = dict(whole=fr(gt, ff), outside=fr([g[~ins] for g in gt], [x[~ins] for x in ff]), inside=fr([g[ins] for g in gt], [x[ins] for x in ff]),
                       swap=fr(gt, sw), d2_inside=float(d2[ins].mean()), d2_outside=float(d2[~ins].mean()), inside_share=float(ins.mean()))
            R.setdefault(pr, {}).setdefault(r, []).append(row)
        print(pr, s, {r: {k: round(v, 2) for k, v in R[pr][r][-1].items()} for r in runs}, flush=True)
out = {pr: {r: {k: float(np.mean([x[k] for x in v])) for k in v[0]} for r, v in d.items()} for pr, d in R.items()}
print(json.dumps(out, indent=1)); json.dump(dict(mean=out, per_seed=R), open(sys.argv[1], 'w'), indent=1)

"""Insertion fidelity of the semantic pipeline (11-07): at every inserted semantic event (generator['ctl'] tokens of a --lib run:
deictic / iconic / metaphoric with time t) the semantic recogniser (the R-half recognisers; the library is K-half) classifies the
2.4 s window centred on t of the generated motion. Reported per intended class:
  acc3     argmax over {deictic, iconic, metaphoric} == intended class (chance 1/3; empirical chance = the same windows of --base)
  win      P(p_intended(insert run) > p_intended(base run)) at the same window (0.5 = the insertion adds nothing)
Sanity modes (run before any report, user request 11-07): label=gt scores the GROUND-TRUTH motion at the dev annotation events with
their true class (-> the recogniser's ceiling); label=gt_shuf the same windows with the classes permuted (-> chance).
    python scripts/semantic/lib_fidelity.py --data <data-sem-hos-R> --recs <rec dirs...> --base <base run> --ctl <ctl dir> \
        --out res.json label=<run dir> ... [gt=- gt_shuf=-]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_recognizer as R
from scmg import ctl

SEM = ['deictic', 'iconic', 'metaphoric']


def load_q(d, stem, n_gt):
    x = torch.load(Path(d) / f'{stem}.pt', map_location='cpu', weights_only=False)
    q = np.asarray(x['robot_repr'], np.float32)[:int(x.get('real_num_frames', len(x['robot_repr'])))]
    q = q[:n_gt] if len(q) >= n_gt else np.concatenate([q, np.repeat(q[-1:], n_gt - len(q), 0)])
    return q[:, R.CH] - q[:, R.CH].mean(0, keepdims=True), x.get('generator', {})


def probs(models, Q, t):
    """Q [T, C] one clip, t [n] s -> mean over recognisers of softmax [n, classes]."""
    Qt = torch.tensor(Q, device='cuda'); off = np.zeros(len(t), np.int64); L = np.full(len(t), len(Q)); ci = np.zeros(len(t), np.int64)
    out = 0
    for m, f in models:
        out = out + R.predict(m, f, Qt, off[:1].repeat(len(t)), L[:1].repeat(len(t)), ci, np.asarray(t))
    return out / len(models)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--recs', nargs='+', required=True)
    ap.add_argument('--base', required=True); ap.add_argument('--ctl', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('specs', nargs='+'); a = ap.parse_args()
    D = R.load_events(a.data); R.configure(D['meta']); classes = D['meta']['classes']; cid = {c: i for i, c in enumerate(classes)}
    models = [R.load_model(str(Path(r) / 'model.pt'))[:2] for r in a.recs]
    q, stems, off, L, idx = R.clip_table(a.data)
    raw = json.loads((Path(a.ctl) / 'CTL.json').read_text())
    dev = sorted(s for s, ts in raw.items() if ts and ts[0]['split'] == 'dev' and s in idx)
    sem_idx = [cid[c] for c in SEM]; res = {}
    for spec in a.specs:
        lab, d = spec.split('=', 1); acc = {c: [] for c in SEM}; win = {c: [] for c in SEM}; base_acc = {c: [] for c in SEM}
        rng = np.random.default_rng(0)
        for s in dev:
            n_gt = int(L[idx[s]]); Qg = q[off[idx[s]]:off[idx[s]] + n_gt]
            if d == '-':
                ev = [(t['t'], ctl.TYPE_NAMES[t['type']]) for t in raw[s] if ctl.TYPE_NAMES[t['type']] in SEM]
                Qi, Qb = Qg, None
            else:
                if not (Path(d) / f'{s}.pt').exists(): continue
                Qi, g = load_q(d, s, n_gt); Qb, _ = load_q(a.base, s, n_gt)
                ev = []
                for t in g.get('ctl', []) or []:
                    c = t.get('cls') if t.get('cls') in SEM else ctl.TYPE_NAMES[int(t['type'])] if int(t.get('type', 0)) < len(ctl.TYPE_NAMES) else None
                    if c in SEM: ev.append((t['t'], c))
            ev = [(t, c) for t, c in ev if R.HALF <= t <= n_gt / R.FPS - R.HALF]
            if not ev: continue
            tt = np.array([t for t, _ in ev]); cc = [c for _, c in ev]
            if lab.endswith('shuf'): cc = list(rng.permutation(cc)) if len(set(cc)) > 1 else [SEM[(SEM.index(c) + 1) % 3] for c in cc]
            P = probs(models, Qi, tt)[:, sem_idx]
            Pb = probs(models, Qb, tt)[:, sem_idx] if Qb is not None else None
            for k, c in enumerate(cc):
                j = SEM.index(c); acc[c].append(float(P[k].argmax() == j))
                if Pb is not None: win[c].append(float(P[k, j] > Pb[k, j])); base_acc[c].append(float(Pb[k].argmax() == j))
        r = {}
        for c in SEM:
            r[c] = dict(n=len(acc[c]), acc3=float(np.mean(acc[c])) if acc[c] else None,
                        base_acc3=float(np.mean(base_acc[c])) if base_acc[c] else None, win=float(np.mean(win[c])) if win[c] else None)
        allv = sum(acc.values(), []); allb = sum(base_acc.values(), []); allw = sum(win.values(), [])
        r['all'] = dict(n=len(allv), acc3=float(np.mean(allv)) if allv else None, base_acc3=float(np.mean(allb)) if allb else None,
                        win=float(np.mean(allw)) if allw else None)
        res[lab] = r; print(lab, json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in r.items()}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

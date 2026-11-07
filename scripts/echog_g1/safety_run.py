"""Apply the C3 safety layers (scmg/safety_proj.py) to a directory of G1 robot_repr predictions, streaming in chunks.

    python scripts/echog_g1/safety_run.py --pred-dir P --stems S --out-root O [--layers none,clip,...] [--K 10] [--workers 12]

Writes O/<layer>/<stem>.pt (input dict, robot_repr[:, 10:39] replaced by the safe joints, root 0:10 untouched, plus
'safety_layer' metadata) and O/<layer>/stats.jsonl (one line per clip: chunks, fallback frames / chunks, per-chunk solve
seconds, layer counters, correction RMSE). Every clip is an independent stream (layer.reset() per clip); chunks are fed
one at a time (no look-ahead). CPU only by default (--device cuda puts the perceptual encoder on the visible GPU).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from scmg import safety_proj as S

_W = {}


def _init(layers, K, device, kw, rg_pairs=False):
    torch.set_num_threads(1)
    lim = S.Limits(); coll = S.G1Collision()
    if rg_pairs:                     # 11-07: + RoboGesture's collision pairs (penetration) as constraints (scmg/safety_rg.py)
        from scmg.safety_rg import RGCollision, UnionCollision
        coll = UnionCollision(coll, RGCollision(S.G1_JOINTS))
    enc = S.FGDEncoder(device=device) if any(n.startswith('proj_perc') for n in layers) else None
    lkw = lambda n: {} if not n.startswith('proj') else (kw if n == 'proj_perc_s' else {k: v for k, v in kw.items() if k != 'smooth'})
    _W.update(lim=lim, coll=coll, K=K, layers={n: S.make_layer(n, lim, coll, enc, K=K, **lkw(n)) for n in layers})


def _one(job):
    pred_dir, stem = job
    x = torch.load(f'{pred_dir}/{stem}.pt', map_location='cpu', weights_only=False)
    rr = (x['robot_repr'] if isinstance(x, dict) else x).double()
    q = rr[:, 10:].numpy()
    res = {}
    for name, L in _W['layers'].items():
        t = time.perf_counter()
        y, st = S.run_stream(L, q, _W['K'])
        wall = time.perf_counter() - t
        out = rr.clone(); out[:, 10:] = torch.from_numpy(y)
        d = dict(x) if isinstance(x, dict) else {}
        d['robot_repr'] = out.to((x['robot_repr'] if isinstance(x, dict) else x).dtype)
        d['safety_layer'] = dict(name=name, K=_W['K'], source_pred=os.path.abspath(f'{pred_dir}/{stem}.pt'))
        rec = {k: v for k, v in st.items() if k not in ('feat_l2', 'feat_perc')}
        rec.update(stem=stem, T=int(q.shape[0]), wall=wall,
                   rmse=float(np.sqrt(((y - q) ** 2).mean())), rmse_upper=float(np.sqrt(((y[:, 12:] - q[:, 12:]) ** 2).mean())),
                   max_abs=float(np.abs(y - q).max()))
        if st.get('feat_l2'):
            rec.update(feat_l2=float(np.mean(st['feat_l2'])), feat_perc=float(np.mean(st['feat_perc'])), n_perc=len(st['feat_l2']))
        res[name] = (d, rec)
    return stem, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', required=True); ap.add_argument('--stems', required=True); ap.add_argument('--out-root', required=True)
    ap.add_argument('--layers', default=','.join(S.LAYERS)); ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--lam', type=float, default=None); ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--B', type=int, default=None); ap.add_argument('--smooth', type=float, default=None)
    ap.add_argument('--rg-pairs', action='store_true', help="11-07: constrain RoboGesture's collision pairs too (penetration, "
                    "scmg/safety_rg.py; default off)")
    a = ap.parse_args()
    layers = a.layers.split(',')
    kw = {k: v for k, v in dict(lam=a.lam, steps=a.steps, B=a.B, smooth=a.smooth).items() if v is not None}
    stems = [s.strip() for s in open(a.stems) if s.strip()][: a.limit or None]
    for n in layers: os.makedirs(f'{a.out_root}/{n}', exist_ok=True)
    logs = {n: open(f'{a.out_root}/{n}/stats.jsonl', 'w') for n in layers}
    t0 = time.time(); done = 0
    jobs = [(a.pred_dir, s) for s in stems]
    if a.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn' if a.device != 'cpu' else 'fork')
        pool = ctx.Pool(a.workers, initializer=_init, initargs=(layers, a.K, a.device, kw, a.rg_pairs))
        it = pool.imap_unordered(_one, jobs, chunksize=1)
    else:
        _init(layers, a.K, a.device, kw, a.rg_pairs); it = map(_one, jobs)
    for stem, res in it:
        for n, (d, rec) in res.items():
            torch.save(d, f'{a.out_root}/{n}/{stem}.pt')
            logs[n].write(json.dumps(rec) + '\n'); logs[n].flush()
        done += 1
        if done % 50 == 0 or done == len(stems):
            print(f'{done}/{len(stems)} {time.time() - t0:.0f}s', flush=True)
    meta = dict(pred_dir=os.path.abspath(a.pred_dir), stems=os.path.abspath(a.stems), layers=layers, K=a.K, kw=kw,
                device=a.device, workers=a.workers, limits=S.Limits().source, mjcf=S.MJCF, wall=time.time() - t0,
                **({'rg_pairs': True} if a.rg_pairs else {}))
    json.dump(meta, open(f'{a.out_root}/RUN.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

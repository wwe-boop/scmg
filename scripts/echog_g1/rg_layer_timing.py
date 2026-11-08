"""11-08 RG-pairs CBF-QP: fast (capped distances) vs reference implementation -- outputs bitwise, per-chunk time (single process).
    python scripts/echog_g1/rg_layer_timing.py --pred-dir P --stems S --n N --out O.json"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from scmg import safety_proj as S
from scmg.safety_rg import RGCollision, UnionCollision
ap = argparse.ArgumentParser(); ap.add_argument('--pred-dir', required=True); ap.add_argument('--stems', required=True)
ap.add_argument('--n', type=int, default=40); ap.add_argument('--out', required=True); ap.add_argument('--skip', type=int, default=0)
a = ap.parse_args(); torch.set_num_threads(1)
stems = [l.strip() for l in open(a.stems) if l.strip()][a.skip:a.skip + a.n]
lim = S.Limits()
layers = dict(cbf_qp=S.CBFQPLayer(lim, S.G1Collision()),
              rg_ref=S.CBFQPLayer(lim, UnionCollision(S.G1Collision(), RGCollision(S.G1_JOINTS, fast=False))),
              rg_fast=S.CBFQPLayer(lim, UnionCollision(S.G1Collision(), RGCollision(S.G1_JOINTS, fast=True))))
T = {k: [] for k in layers}; same = 0; diff = []
for s in stems:
    q = torch.load(f'{a.pred_dir}/{s}.pt', map_location='cpu', weights_only=False)['robot_repr'].double().numpy()[:, 10:]
    out = {}
    for k, L in layers.items():
        y, st = S.run_stream(L, q, 10); out[k] = y; T[k] += list(st['times'])
    same += int(np.array_equal(out['rg_ref'], out['rg_fast'])); diff.append(float(np.abs(out['rg_ref'] - out['rg_fast']).max()))
res = dict(clips=len(stems), bitwise_equal=same, max_abs_diff=max(diff), load=os.getloadavg())
for k, t in T.items():
    t = np.array(t) * 1e3; res[k] = dict(chunks=len(t), p50=float(np.percentile(t, 50)), p95=float(np.percentile(t, 95)), p99=float(np.percentile(t, 99)), max=float(t.max()))
print(json.dumps(res, indent=1)); json.dump(res, open(a.out, 'w'), indent=1)

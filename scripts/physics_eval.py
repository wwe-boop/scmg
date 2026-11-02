"""Physics evaluation of saved trajectories (any model) in the EX002+RM003 MuJoCo simulation.
    python scripts/physics_eval.py <out.json> <label=eval_dir> [<label=eval_dir> ...]
eval_dir holds gen_Hinf_<ci>.npy and gt_<ci>.npy (34 joints, rad, 50 Hz) for the same clips. Each trajectory is passed
through the CBF-QP projection (as deployed), executed from its ground-truth start pose, and scored with physics_score
(tracking, torque saturation, self-penetration, SELF-COLLISION RATE, energy, acceleration p99, joint-limit
violation, + CBF modification). The raw (pre-CBF) command violation rate is reported too. Ground truth is scored the
same way as a reference row."""
import json, sys
from pathlib import Path
import numpy as np
from scmg.physim import EX002Sim, physics_score
from scmg.safety import limits34, CBFFilter, violation_rate

meta = json.load(open('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1/META.json'))
lim = limits34(meta['lower'], meta['upper']); sim = EX002Sim()
cols = [a.split('=', 1) for a in sys.argv[2:]]


def run(q):
    sim.nominalize(); sim.reset(np.clip(q[0], sim.lo + 1e-3, sim.hi - 1e-3))
    f = CBFFilter(lim, q[0]); cmd = np.array([f.step(x) for x in q])
    sc = physics_score(sim.rollout(cmd), cmd, lim)
    sc['cbf_mod_deg'] = float(np.abs(cmd - q)[:, 4:18].mean() * 57.3); sc['raw_violation'] = violation_rate(q, lim)['any']
    return sc


res = {}
gt_dir = Path(cols[0][1]); cis = sorted(int(p.stem.split('_')[1]) for p in gt_dir.glob('gt_*.npy'))
rows = {'真人': [run(np.load(gt_dir / f'gt_{ci}.npy').astype(float)) for ci in cis]}
for label, d in cols:
    rows[label] = [run(np.load(Path(d) / f'gen_Hinf_{ci}.npy').astype(float)) for ci in cis if (Path(d) / f'gen_Hinf_{ci}.npy').exists()]
    print(label, len(rows[label]), flush=True)
for k, rs in rows.items():
    res[k] = {m: float(np.mean([r[m] for r in rs])) for m in rs[0]}; res[k]['clips'] = len(rs)
Path(sys.argv[1]).write_text(json.dumps(res, ensure_ascii=False, indent=1))
keys = ('total', 'collision_rate', 'pen_mm', 'limit_viol', 'acc_p99', 'track_rms', 'sat', 'cbf_mod_deg', 'raw_violation')
print('label'.ljust(14), ' '.join(k[:12].rjust(12) for k in keys))
for k, r in res.items(): print(k.ljust(14), ' '.join(f'{r[m]:12.4f}' for m in keys))

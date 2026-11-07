"""(c)1 G5 render prep (run with the retargeter python from the release07 dir): for the seeded sample of newly accepted items,
write <out>/<id>/robot.npz (Q37 = q_full, fps 50) and human.npz (joints [T,55,3] of the retargeter's own human side at 50 Hz,
near [T,2], repair_cm [T,2]).   python regate_render_prep.py <snapshot.jsonl> <NEW_ACCEPT.json> <out> <n> <seed>"""
import json, os, sys
sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/retarget-release07-headroom-20261026')
import numpy as np
from scipy.ndimage import gaussian_filter1d
from retarget_adapter import load
from validate_motion import C
from smplx2ex002_3f.source_policy import options_for

snap = {json.loads(l)['id']: json.loads(l) for l in open(sys.argv[1]) if l.strip()}
new = sorted(json.load(open(sys.argv[2]))['new']); out = sys.argv[3]; n = int(sys.argv[4])
pick = [new[i] for i in sorted(np.random.default_rng(int(sys.argv[5])).choice(len(new), n, replace=False))]
for i in pick:
    row = snap[i]; d = os.path.dirname(row['result']); o_ = os.path.join(out, i); os.makedirs(o_, exist_ok=True)
    q = np.load(d + '/q_full.npy'); np.savez(o_ + '/robot.npz', Q37=q, fps=50.0)
    seq, fps = load(row['source']); o = dict(C.DEFAULTS); o.update(options_for(row['source'], {})); hs = C.human_side(seq, fps, o)
    near = np.stack([C._smoothstep((o['near_off'] - gaussian_filter1d(hs['prox'][s]['dist'], 2, mode='nearest')) / (o['near_off'] - o['near_on'])) > .5
                     for s in ('left', 'right')], 1)
    np.savez(o_ + '/human.npz', joints=hs['joints'], near=near, repair_cm=np.load(d + '/repair_cm.npy'))
    print(i, q.shape, hs['joints'].shape, flush=True)
json.dump(pick, open(os.path.join(out, 'PICK.json'), 'w'))

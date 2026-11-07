"""G6 data build step 1 (retargeter python, release07 dir): select the (c)1 newly accepted Seamless parents that are Seamless-train
and have no whole-trajectory safety back-off; write SELECT.jsonl (snapshot rows) and near/<id>.npy (per-frame near-body mask [T,2],
the retargeter's own wnear > 0.5, as regate_seamless.py R4) for the subwindow re-gate.
    python select_near.py <regate dir> <snapshot.jsonl> <out dir> <worker i> <workers n>"""
import glob, json, os, sys
sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/retarget-release07-headroom-20261026')
import numpy as np
from scipy.ndimage import gaussian_filter1d
from retarget_adapter import load
from validate_motion import C
from smplx2ex002_3f.source_policy import options_for
rg, snapf, out, i, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
rows = [json.loads(l) for f in sorted(glob.glob(rg + '/regate_[0-9].jsonl')) for l in open(f)]
snap = {json.loads(l)['id']: json.loads(l) for l in open(snapf)}
ok = lambda r: (not r['R0'] and r['R1'] and r['R2'] and r['R3'] and r['R4'] and (r.get('backoff') or 1) >= 1
                and snap[r['id']]['source']['split'] == 'train')
sel = sorted(r['id'] for r in rows if ok(r))
os.makedirs(out + '/near', exist_ok=True)
if i == 0: open(out + '/SELECT.jsonl', 'w').write(''.join(json.dumps(snap[k]) + '\n' for k in sel))
for k in sel[i::n]:
    p = f'{out}/near/{k}.npy'
    if os.path.exists(p): continue
    row = snap[k]; seq, fps = load(row['source']); o = dict(C.DEFAULTS); o.update(options_for(row['source'], {})); hs = C.human_side(seq, fps, o)
    near = np.stack([C._smoothstep((o['near_off'] - gaussian_filter1d(hs['prox'][s]['dist'], 2, mode='nearest')) / (o['near_off'] - o['near_on'])) > .5
                     for s in ('left', 'right')], 1)
    np.save(p, near)
print('SELECT_DONE', i, len(sel))

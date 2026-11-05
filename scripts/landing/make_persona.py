"""Persona = one fixed reference motion (+ the speaker embedding of the same recording) used for EVERY sentence
(landing L3, Claude 11-05). The model already has the reference-motion condition (style encoder, ref dropped 50 % in
training) but deployment sends a null reference and a null speaker.

    python scripts/landing/make_persona.py <out dir> <name> --energy-pct 50 [--sources beat2_en,beat2_zh,beat2_en_additional]

Candidates: 128-frame windows (stride 64) of TRAIN clips of the clean sources whose frames are all valid (no
pose-audit ARM_MASK frame, no retargeting glitch). Energy = z(log mean arm-joint std) + z(log mean arm speed) over the
candidates; the persona is the candidate at the requested energy percentile (ties: the one whose mean arm pose is
closest to the clean-human median pose). Writes <name>_ref.npy ([128, 34], NORMALISED with STATS_v36_1 -- the form
WindowDataset.sample gives the model), <name>_spk.npy ([192]) and <name>.json (provenance + style descriptor).
"""
import argparse, json, os
import numpy as np
from scmg.common import P, CPFS
from scmg.data import View
from scmg.landing import style_descriptor, arm_speed, ARMS14

VIEW = CPFS + '/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1'
STATS = CPFS + '/dataset/zehan/gesture/scmg/STATS_v36_1.json'
MASK = CPFS + '/USERS/zehan/gesture/pose-audit-1104/full/ARM_MASK.npy'
R = 128

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('name')
ap.add_argument('--energy-pct', type=float, default=50.); ap.add_argument('--sources', default='beat2_en,beat2_zh,beat2_en_additional')
ap.add_argument('--stride', type=int, default=64)
a = ap.parse_args()
st = json.load(open(P(STATS))); v = View(VIEW, 'train', None, st, mmap=True)
mask = np.load(P(MASK), mmap_mode='r')
if len(mask) != len(v.q): raise SystemExit('ARM_MASK rows != view rows')
srcs = set(a.sources.split(','))
cand = []
for ci, c in enumerate(v.clips):
    if c['source'] not in srcs or c['frames'] < R: continue
    for s in range(0, c['frames'] - R + 1, a.stride):
        r0 = c['start'] + s
        if np.asarray(mask[r0:r0 + R]).any(): continue
        if v.glitch is not None and np.asarray(v.glitch[r0:r0 + R]).any(): continue
        q = np.asarray(v.q[r0:r0 + R], np.float64)
        cand.append((ci, s, q[:, ARMS14].std(0).mean(), arm_speed(q).mean(), q[:, ARMS14].mean(0)))
if len(cand) < 100: raise SystemExit(f'only {len(cand)} candidate windows')
amp = np.log(np.array([x[2] for x in cand]) + 1e-6); spd = np.log(np.array([x[3] for x in cand]) + 1e-6)
z = lambda x: (x - x.mean()) / x.std()
energy = z(amp) + z(spd)
target = np.percentile(energy, a.energy_pct)
near = np.flatnonzero(np.abs(energy - target) <= np.percentile(np.abs(energy - target), 1))   # closest 1 % in energy
pose_med = np.median(np.stack([x[4] for x in cand]), 0)
k = int(near[np.argmin([np.linalg.norm(cand[i][4] - pose_med) for i in near])])
ci, s = cand[k][0], cand[k][1]; c = v.clips[ci]; r0 = c['start'] + s
q = np.asarray(v.q[r0:r0 + R], np.float32)
ref = (q - v.qm) / v.qs
it = next(it for it in c['items'] if it['offset'] <= r0 < it['offset'] + it['frames'])
row = v.cond.get(str(it['id']))
if not row or not row[1]: raise SystemExit(f'item {it["id"]} has no valid speaker embedding')
spk = v.spk_emb[row[0]].astype(np.float32)
os.makedirs(a.out, exist_ok=True)
np.save(f'{a.out}/{a.name}_ref.npy', ref.astype(np.float32)); np.save(f'{a.out}/{a.name}_spk.npy', spk)
d = style_descriptor(q)
info = dict(name=a.name, energy_pct=a.energy_pct, energy=float(energy[k]), candidates=len(cand), sources=sorted(srcs),
            clip=c['clip'], item=it['id'], speaker=c['speaker'], source=c['source'], start_frame=int(s), view_row=int(r0),
            text=it.get('text'), arm_std_mean=float(cand[k][2]), arm_speed_mean=float(cand[k][3]),
            cand_arm_std_pct=float((np.array([x[2] for x in cand]) < cand[k][2]).mean() * 100),
            cand_speed_pct=float((np.array([x[3] for x in cand]) < cand[k][3]).mean() * 100),
            descriptor=d.round(5).tolist(), stats=STATS, view=VIEW, normalised=True, frames=R)
json.dump(info, open(f'{a.out}/{a.name}.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k_: info[k_] for k_ in ('name', 'clip', 'speaker', 'start_frame', 'arm_std_mean', 'arm_speed_mean', 'cand_arm_std_pct', 'cand_speed_pct', 'candidates')}, ensure_ascii=False))

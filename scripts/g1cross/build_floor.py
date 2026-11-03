"""Retargeting noise floor (G1 -> EX002), 11-03 Claude.
Human G1 motion of the 43 dev recordings (ECHO-G dataset v0.2.0 motion_39d_30fps; the dev recordings are in ECHO-G TRAIN)
is stitched onto each dev clip's timeline (dev frame -> BEAT2 recording time via the view items' clip_start_s; ECHO-G utterance
-> recording time via annotations/words source.start_sec/end_sec) and passed through EXACTLY the baseline adapter
(echog-official/code/export_dev.py: robot_repr -> q60 by joint name -> RoboGesture retarget_g1_ex002.Retargeter, PCHIP 30->50 Hz,
torso-relative arm IK, 0.2 s GT transition, fingers open).  Output: <out>/gen_Hinf_<ci>.npy, gt_<ci>.npy, meta_<ci>.json
    python build_floor.py <echog dataset root (motion_39d_30fps/, annotations/words/)> <out dir>"""
import json, math, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/code')
sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/official/src')
from echo_g.evaluation.robot_repr import G1_JOINT_NAMES
from retarget_g1_ex002 import Retargeter
DS, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
RG = Path('/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/robogesture/audio')
VIEW = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1')
rows = json.loads((RG / 'MANIFEST.json').read_text())
dev = {d['ci']: d for d in json.loads(Path('/mnt/cpfs/zbl-cpfs-new/USERS/zehan/fgd-ex002-v2-20261102/DEV_CLIPS.json').read_text())}
items = json.loads((VIEW / 'train.json').read_text()); by_off = {it['offset']: it for it in items}
offs = np.array(sorted(by_off))
utts = {}
for p in sorted((DS / 'annotations' / 'words').glob('*.json')):
    a = json.loads(p.read_text()); s = a['source']; utts.setdefault(s['recording'], []).append((s['start_sec'], s['end_sec'], a['clip_id']))
rt = Retargeter(); only = set(int(x) for x in sys.argv[3].split(',')) if len(sys.argv) > 3 else None
for row in rows:
    ci = row['ci']
    if only and ci not in only: continue
    if (OUT / f'meta_{ci}.json').exists(): continue
    t0 = time.perf_counter(); d = dev[ci]; T = row['T']; N = math.ceil(T / 50 * 30)
    rec = d['clip'].split('-', 2)[2]                                     # e.g. 13_lu_2_14_14
    # dev frame f (50 Hz) -> recording seconds, through the items covering [start, start+T)
    rec_t = np.full(T, np.nan)
    for o in offs[(offs >= d['start']) & (offs < d['start'] + T)]:
        it = by_off[o]; f0 = o - d['start']; n = min(it['frames'], T - f0)
        rec_t[f0:f0 + n] = it['clip_start_s'] + np.arange(n) / 50
    assert np.isfinite(rec_t).all(), (ci, 'dev frames without an item')
    tq = np.interp(np.arange(N) / 30, np.arange(T) / 50, rec_t)        # 30 fps clip timeline -> recording seconds
    native = np.full((N, 39), np.nan, np.float32); src = []
    for (s0, s1, cid) in utts.get(rec, []):
        m = (tq >= s0) & (tq < s1)
        if not m.any(): continue
        z = torch.load(DS / 'motion_39d_30fps' / f'{cid}.pt', map_location='cpu', weights_only=False)['robot_repr'].numpy()
        k = np.clip(np.round((tq[m] - s0) * 30).astype(int), 0, len(z) - 1); native[m] = z[k]; src.append(cid)
    cov = float(np.isfinite(native[:, 0]).mean())
    # uncovered frames (silence / utterances dropped by ECHO-G): hold the nearest covered frame
    ok = np.isfinite(native[:, 0]); idx = np.where(ok)[0]
    if len(idx) == 0: print('NO COVERAGE', ci, flush=True); continue
    near = idx[np.clip(np.searchsorted(idx, np.arange(N)), 0, len(idx) - 1)]
    prev = idx[np.clip(np.searchsorted(idx, np.arange(N)) - 1, 0, len(idx) - 1)]
    pick = np.where(np.abs(near - np.arange(N)) <= np.abs(prev - np.arange(N)), near, prev); native[~ok] = native[pick[~ok]]
    q60 = np.zeros((N, 60), np.float32)
    for j, name in enumerate(G1_JOINT_NAMES):
        q60[:, 7 + int(rt.g.joint(name + '_joint').qposadr[0])] = native[:, 10 + j]
    gt = np.load(RG / f'gt_{ci}.npy'); assert gt.shape == (T, 34)
    y, quality = rt.retarget(q60, T, gt)
    np.save(OUT / f'gen_Hinf_{ci}.npy', y); np.save(OUT / f'gt_{ci}.npy', gt)
    np.savez_compressed(OUT / f'g1_{ci}.npz', robot_repr=native, fps=np.float32(30), covered=ok)
    (OUT / f'meta_{ci}.json').write_text(json.dumps(dict(ci=ci, clip=d['clip'], recording=rec, frames=T, g1_coverage=cov, utterances=src,
        retarget_quality=quality, seconds=time.perf_counter() - t0, method='ECHO-G human G1 (train split) stitched by recording time; '
        'uncovered frames hold nearest; export_dev.py adapter + RoboGesture Retargeter unchanged'), indent=1))
    print('FLOOR', ci, 'coverage %.3f' % cov, 'utts', len(src), '%.0fs' % (time.perf_counter() - t0), flush=True)

"""Landing metrics for the arms (user 11-04 L1: limited upper-arm amplitude, no wrist / palm eversion, not rigid).

Scores generated EX002 exports (gen_*.npy, [T,34] rad, 50 Hz) against the ground truth of the same clips (gt_*.npy) with
the pose-audit detectors (scene.xml FK + MuJoCo collision + reference distributions fitted on human training data only):

  frame rates  shoulder_extreme (arm_joint3 within 5% of an end), palm_outlier, elbow_flip, elbow_torso, hand/forearm
               contacts, temporal_flip (wrist roll >120 deg without hand motion), distribution (1 s outlier), near_limit
  amplitude    per clip, std over time of upper-arm joints 1-3 and wrist joints 5-7, gen / GT (1 = like the human;
               >1 = too large; <<1 = rigid), and the p95 deviation from the clip median.

    python scripts/landing/arm_audit.py <out.jsonl> label=<eval dir with gen_*.npy + gt_*.npy> ...
Run with the pose-audit python (zehan-server: /mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python), CPU only.
"""
import sys, json, glob, os, importlib.util
import numpy as np
AUD = os.environ.get('POSE_AUDIT', '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture/pose-audit-1104')
spec = importlib.util.spec_from_file_location('pose_audit', AUD + '/scripts/pose_audit.py'); pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)
meta = pa.loadj(pa.VIEW / 'META.json'); ref = pa.loadj(AUD + '/full/REFERENCE.json'); kernel = pa.Kernel(meta)
J = meta['joints']; UP = [J.index(f'{s}_arm_joint{k}') for s in ('left', 'right') for k in (1, 2, 3)]
WR = [J.index(f'{s}_arm_joint{k}') for s in ('left', 'right') for k in (5, 6, 7)]
CATS = ['shoulder_extreme', 'palm_outlier', 'elbow_flip', 'elbow_torso', 'hand_torso', 'hand_head', 'hand_hand', 'forearm_torso',
        'temporal_flip', 'distribution', 'near_limit']


def score(q):
    g = kernel.run(q); _, h, _ = pa.detect(q, g, meta, ref)
    return {k: int(h[k].sum()) for k in CATS}


out = open(sys.argv[1], 'a')
for spec_ in sys.argv[2:]:
    lab, d = spec_.split('=', 1)
    gens = sorted(glob.glob(d + '/gen_*.npy'), key=lambda p: int(p.rsplit('_', 1)[1][:-4]))
    if not gens: print(lab, 'no gen_*.npy in', d); continue
    acc = {w: dict(frames=0, **{k: 0 for k in CATS}) for w in ('gen', 'gt')}; amp = dict(up=[], wr=[], up95=[], wr95=[])
    for p in gens:
        i = p.rsplit('_', 1)[1][:-4]; qg = np.load(p).astype(np.float64); qt = np.load(f'{d}/gt_{i}.npy').astype(np.float64)
        n = min(len(qg), len(qt)); qg, qt = qg[:n], qt[:n]
        for w, q in (('gen', qg), ('gt', qt)):
            s = score(q); acc[w]['frames'] += n
            for k in CATS: acc[w][k] += s[k]
        sd = lambda q, c: q[:, c].std(0); dev = lambda q, c: np.percentile(np.abs(q[:, c] - np.median(q[:, c], 0)), 95, axis=0)
        amp['up'].append(float(np.mean(sd(qg, UP) / np.maximum(sd(qt, UP), 1e-3)))); amp['wr'].append(float(np.mean(sd(qg, WR) / np.maximum(sd(qt, WR), 1e-3))))
        amp['up95'].append(float(np.mean(dev(qg, UP) / np.maximum(dev(qt, UP), 1e-3)))); amp['wr95'].append(float(np.mean(dev(qg, WR) / np.maximum(dev(qt, WR), 1e-3))))
    r = dict(label=lab, dir=d, clips=len(gens),
             rate={w: {k: round(acc[w][k] / max(1, acc[w]['frames']), 5) for k in CATS} for w in acc},
             amp_ratio=dict(upper_std=round(float(np.median(amp['up'])), 3), wrist_std=round(float(np.median(amp['wr'])), 3),
                            upper_p95dev=round(float(np.median(amp['up95'])), 3), wrist_p95dev=round(float(np.median(amp['wr95'])), 3)))
    print(json.dumps(r), flush=True); out.write(json.dumps(r) + '\n'); out.flush()

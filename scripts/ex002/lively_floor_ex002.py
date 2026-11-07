"""11-08 EX002 liveliness guard on the MOVING upper body (EX002_SWEEP_1107.md correction bf8290e; supersedes the withdrawn floor):
points (exact FK, scmg/kinematics.py, sphere-proxy bodies) shoulder / elbow / wrist (L, R) + fingertip chains 1_3 / 2_3 / 3_2 (L, R) +
head (head pitch body): 13 points; pedestal / lift / base excluded. Per-point speed normaliser = mean speed over the dev GT (the GT spec,
computed first). 50 Hz, central differences.
  still      : share of frames with mean normalised speed (13 points) < 0.15                    gate <= 2 x GT
  wrist      : mean wrist speed (m/s) / wrist amplitude (m, per-axis std)                        gates >= 0.75 x GT
  held clips : longest run inside the items' speech spans with mean normalised speed < 0.15 > 2.0 s   gate share <= 10 %
  stutter    : wrist speed per phase of the 16-frame chunk grid, max / min                        gate <= 1.10 x GT
Clip index ci = View(data-v36_1, 'val').clips order (= bench file names).
    python lively_floor_ex002.py --out O.json GT=<dir with gt_*.npy> label=<dir>[,<dir>...] ..."""
import argparse, glob, json, sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
VIEW = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1')
NPZ = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/sphere_proxy_vox4e.npz'
JOINTS = dict(shoulder=(4, 11), elbow=(7, 14), wrist=(10, 17), tip1=(20, 28), tip2=(23, 31), tip3=(25, 33), head=(3, 3))
_K = {}


def pts(q):
    if not _K:
        from scmg.kinematics import Kinematics
        from scmg.common import P
        z = np.load(P(NPZ)); bj = np.asarray(z['body_joint']); b = lambda j: int(np.nonzero(bj == j)[0][0])
        idx = [b(JOINTS[k][0]) for k in JOINTS] + [b(JOINTS[k][1]) for k in JOINTS if k != 'head']
        _K.update(k=Kinematics(P(NPZ), 'cpu'), idx=idx)
    with torch.no_grad():
        _, p = _K['k'].fk(torch.as_tensor(np.asarray(q, np.float32)))
    return p[:, _K['idx']].double().numpy()            # [T, 13, 3]: L of the 7 groups (head once), then R of the 6 bilateral ones


WRIST = (2, 9)                                          # wrist L (group 2) / wrist R (7 + 2)


def speech_masks():
    val = set(json.loads((VIEW / 'SPLIT.json').read_text())['val_ids']); cl = OrderedDict()
    for it in json.loads((VIEW / 'train.json').read_text()):
        if it['id'] in val: cl.setdefault(it['clip'], []).append(it)
    out = []
    for c, xs in cl.items():
        xs.sort(key=lambda x: x['offset']); st = xs[0]['offset']; L = sum(x['frames'] for x in xs); m = np.zeros(L, bool)
        for it in xs:
            if not it.get('speech'): continue
            i0 = it['offset'] - st; s0, s1 = (int(round(float(x) * 50)) for x in it['speech']); m[max(0, i0 + s0):min(L, i0 + s1)] = True
        out.append(m)
    return out


def longest_run(b):
    best = run = 0
    for x in b:
        run = run + 1 if x else 0; best = max(best, run)
    return best


def speed(P):
    v = np.zeros(P.shape[:2]); v[1:-1] = np.linalg.norm(P[2:] - P[:-2], axis=-1) * 25.; v[0], v[-1] = v[1], v[-2]; return v


ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--thr', type=float, default=.15)
ap.add_argument('--hold-s', type=float, default=2.0); ap.add_argument('specs', nargs='+')
a = ap.parse_args()
SM = speech_masks(); specs = [s.split('=', 1) for s in a.specs]; gt = [d for l, d in specs if l == 'GT']
files = lambda lab, ds: [f for d in ds.split(',') for f in sorted(glob.glob(d + ('/gt_*.npy' if lab == 'GT' else '/[0-9]*.npy')))]
norm = np.concatenate([speed(pts(np.load(f))) for f in files('GT', gt[0])]).mean(0)
res = dict(_meta=dict(norm=norm.tolist(), points=list(JOINTS), thr=a.thr, hold_s=a.hold_s))
for lab, ds in specs:
    st = fr = held = nclip = 0; ws, wa = [], []; ph_s = np.zeros(16); ph_c = np.zeros(16)
    for f in files(lab, ds):
        ci = int(Path(f).stem.split('_')[-1]); P = pts(np.load(f)); sp = speed(P); ns = (sp / norm).mean(1)
        still = ns < a.thr; st += int(still.sum()); fr += len(P)
        m = SM[ci] if ci < len(SM) else np.zeros(0, bool); m = np.pad(m[:len(P)], (0, max(0, len(P) - len(m))))
        held += int(longest_run(still & m) > a.hold_s * 50); nclip += 1
        w = sp[:, list(WRIST)]; ws.append(float(w.mean())); wa.append(float(P[:, list(WRIST)].std(0).mean()))
        p = np.arange(len(w)) % 16; np.add.at(ph_s, p, w.mean(1)); np.add.at(ph_c, p, 1)
    ph = ph_s / np.maximum(ph_c, 1); ph = ph / ph.mean()
    res[lab] = dict(clips=nclip, still=st / max(fr, 1), held_share=held / max(nclip, 1), stutter=float(ph.max() / ph.min()),
                    wrist_speed=float(np.mean(ws)), wrist_amp=float(np.mean(wa)))
    print(lab, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[lab].items()}), flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))

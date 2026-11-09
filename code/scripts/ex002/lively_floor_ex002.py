"""11-08 EX002 liveliness guard on the MOVING upper body (EX002_SWEEP_1107.md correction bf8290e; supersedes the withdrawn floor):
points (exact FK, scmg/kinematics.py, sphere-proxy bodies) shoulder / elbow / wrist (L, R) + fingertip chains 1_3 / 2_3 / 3_2 (L, R) +
head (head pitch body): 13 points; pedestal / lift / base excluded. Per-point speed normaliser = mean speed over the dev GT (the GT spec,
computed first). 50 Hz, central differences.
  still      : share of frames with mean normalised speed (13 points) < 0.15                    gate <= 2 x GT
  wrist      : mean wrist speed (m/s) / wrist amplitude (m, per-axis std)                        gates >= 0.75 x GT
  held clips : longest run inside the items' speech spans with mean normalised speed < 0.15 > 2.0 s   gate share <= 10 %
  stutter    : wrist speed per phase of the 16-frame chunk grid, max / min                        gate <= 1.10 x GT
Clip index ci = View(data-v36_1, 'val').clips order (= bench file names).
    python lively_floor_ex002.py --out O.json GT=<dir with gt_*.npy> label=<dir>[,<dir>...] ...
Importable (11-08): guard(specs) / verdict(); bench_ex002.py eval prints it for every lane by default."""
import argparse, glob, json, sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
VIEW = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1')
import os as _os
if _os.environ.get('SCMG_EXPL_VIEW'): VIEW = Path(_os.environ['SCMG_EXPL_VIEW'])   # 11-09 (default unset): test-only view, split test_ids
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
    val = set(json.loads((VIEW / 'SPLIT.json').read_text())['test_ids' if _os.environ.get('SCMG_EXPL_VIEW') else 'val_ids']); cl = OrderedDict()
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


GATES = dict(still=('<=', 2.0), wrist_speed=('>=', 0.75), wrist_amp=('>=', 0.75), held_share=('<=', None), stutter=('<=', 1.10))
HELD_MAX = 0.10                                         # absolute gate on the held-clip share


def files(lab, ds):
    return [f for d in ds.split(',') for f in sorted(glob.glob(d + ('/gt_*.npy' if lab == 'GT' else '/[0-9]*.npy')))]


def lane_stats(fs, norm, thr=.15, hold_s=2.0, SM=None):
    SM = speech_masks() if SM is None else SM
    st = fr = held = nclip = 0; ws, wa = [], []; ph_s = np.zeros(16); ph_c = np.zeros(16)
    for f in fs:
        ci = int(Path(f).stem.split('_')[-1]); P = pts(np.load(f)); sp = speed(P); ns = (sp / norm).mean(1)
        still = ns < thr; st += int(still.sum()); fr += len(P)
        m = SM[ci] if ci < len(SM) else np.zeros(0, bool); m = np.pad(m[:len(P)], (0, max(0, len(P) - len(m))))
        held += int(longest_run(still & m) > hold_s * 50); nclip += 1
        w = sp[:, list(WRIST)]; ws.append(float(w.mean())); wa.append(float(P[:, list(WRIST)].std(0).mean()))
        p = np.arange(len(w)) % 16; np.add.at(ph_s, p, w.mean(1)); np.add.at(ph_c, p, 1)
    ph = ph_s / np.maximum(ph_c, 1); ph = ph / ph.mean()
    return dict(clips=nclip, still=st / max(fr, 1), held_share=held / max(nclip, 1), stutter=float(ph.max() / ph.min()),
                wrist_speed=float(np.mean(ws)), wrist_amp=float(np.mean(wa)))


def verdict(r, g):
    """Gates of one lane r against the GT row g (same clips): dict of per-gate booleans + 'pass'."""
    v = {}
    for k, (op, x) in GATES.items():
        lim = HELD_MAX if k == 'held_share' else x * g[k]
        v[k] = bool(r[k] <= lim) if op == '<=' else bool(r[k] >= lim)
    v['pass'] = all(v.values()); return v


def guard(specs, thr=.15, hold_s=2.0):
    """specs: [(label, dirs)] with one label 'GT' (dir holding gt_*.npy). Returns {label: stats (+ 'gates' for lanes)}."""
    SM = speech_masks(); gt = [d for l, d in specs if l == 'GT']
    if not gt: raise ValueError('the liveliness guard needs a GT=<dir> spec (per-point speed normaliser and gate references)')
    norm = np.concatenate([speed(pts(np.load(f))) for f in files('GT', gt[0])]).mean(0)
    res = dict(_meta=dict(norm=norm.tolist(), points=list(JOINTS), thr=thr, hold_s=hold_s))
    for lab, ds in specs: res[lab] = lane_stats(files(lab, ds), norm, thr, hold_s, SM)
    for lab, _ in specs:
        if lab != 'GT': res[lab]['gates'] = verdict(res[lab], res['GT'])
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--thr', type=float, default=.15)
    ap.add_argument('--hold-s', type=float, default=2.0); ap.add_argument('specs', nargs='+')
    a = ap.parse_args()
    res = guard([s.split('=', 1) for s in a.specs], a.thr, a.hold_s)
    for lab, r in res.items():
        if lab != '_meta': print(lab, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))

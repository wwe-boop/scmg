"""EX002 executable-gesture contract (11-05): the SAME final gates the data retargeter applies before a trajectory may be
published (smplx2ex002_3f core.retarget, release06-beat-nearoff), applied to RAW model output (before any safety layer):
  C1 position   : R.lo <= q <= R.hi (hard limits incl. the conservative torso / head envelope)
  C2 dynamics   : |v|, |a|, |jerk| per joint <= SafetyGovernor tables (robot.Limits; finite differences at 50 Hz)
  C3 collision  : every runtime WBC collision pair separated (R.clearance)
  C4 continuity : max |q[t+1] - q[t]| <= 0.06 rad
A clip "passes" iff it violates no clause on any frame (= would be accepted by the retargeter's final gate as is).
Inputs: evaluate.py export dirs (gen_H<h>_<ci>.npy / gt_<ci>.npy: [T, 34] rad, 50 Hz, META joint order == CONTROLLED).
    python scripts/landing/ex002_contract.py --hz 2 --out res.json GT=<dir>:gt label=<dir> ...
Run with a python that has the retargeter deps (e.g. echog python + robogesture-py310 site-packages, MUJOCO_GL=disable).
"""
import argparse, glob, json, os, re, sys
import numpy as np

PKG = '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture-dlc-20260928/smplx2ex002-r2-20261022/candidates/codex-dlc-release06-beat-nearoff'
sys.path.insert(0, PKG)
from smplx2ex002_3f import core                                   # noqa: E402
from smplx2ex002_3f.robot import CONTROLLED                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--hz', default='2', help='horizon tag of gen_H<hz>_<ci>.npy'); ap.add_argument('--out', default=None)
ap.add_argument('--collision-stride', type=int, default=2); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--joints', default='all', choices=['all', 'body', 'fingers'], help='joints for C1/C2/C4: body = torso/head + arms (18), fingers = 16; C3 always full')
ap.add_argument('panels', nargs='+', help='label=dir (generated) or label=dir:gt (ground truth files)')
a = ap.parse_args()

R = core.robot(core.DEFAULTS['lift_offset']) if hasattr(core, 'DEFAULTS') else core.robot(.32)
L = R.limits
assert len(CONTROLLED) == 34 and len(L.v) == 34
JS = {'all': slice(0, 34), 'body': slice(0, 18), 'fingers': slice(18, 34)}[a.joints]
assert CONTROLLED[17] == 'right_arm_joint7' and CONTROLLED[18] == 'left_1_1_joint'


def check(Q):
    T = len(Q)
    Qc, lo, hi, lv, la, lj = Q[:, JS], R.lo[JS], R.hi[JS], L.v[JS], L.a[JS], L.j[JS]
    pos = ((Qc < lo) | (Qc > hi)).any(1)
    dt = core.DT
    v = np.abs(np.diff(Qc, axis=0)) / dt; acc = np.abs(np.diff(Qc, 2, axis=0)) / dt ** 2; jk = np.abs(np.diff(Qc, 3, axis=0)) / dt ** 3
    dyn = np.zeros(T, bool)
    dyn[1:] |= (v > lv).any(1); dyn[2:] |= (acc > la).any(1); dyn[3:] |= (jk > lj).any(1)
    jump = np.zeros(T, bool); jump[1:] = np.abs(np.diff(Qc, axis=0)).max(1) > .06
    col = np.zeros(T, bool); pairs = {}
    for t in range(0, T, a.collision_stride):
        ok, _, names = R.clearance(Q[t])
        if not ok:
            col[t: t + a.collision_stride] = True
            k = '|'.join(str(x) for x in names[:2]) if names else '?'; pairs[k] = pairs.get(k, 0) + 1
    worst = dict(v=float((v / lv).max()), a=float((acc / la).max()), j=float((jk / lj).max()))
    return dict(T=T, pos=int(pos.sum()), dyn=int(dyn.sum()), col=int(col.sum()), jump=int(jump.sum()),
                any=int((pos | dyn | col | jump).sum()), passed=not (pos | dyn | col | jump).any(), worst=worst, pairs=pairs)


res = {}
for spec in a.panels:
    lab, d = spec.split('=', 1); gt = d.endswith(':gt'); d = d[:-3] if gt else d
    hz = a.hz
    if '@' in d: d, hz = d.rsplit('@', 1)                       # label=dir@inf (offline exports are gen_Hinf_<ci>.npy)
    pat = re.compile(r'gt_(\d+)\.npy$' if gt else rf'gen_H{hz}_(\d+)\.npy$')
    files = sorted((int(pat.search(f).group(1)), f) for f in glob.glob(os.path.join(d, '*.npy')) if pat.search(f))
    if a.limit: files = files[: a.limit]
    if not files: raise SystemExit(f'{lab}: no {pat.pattern} files in {d}')
    per = {ci: check(np.load(f).astype(np.float64)) for ci, f in files}
    F = sum(r['T'] for r in per.values()); pairs = {}
    for r in per.values():
        for k, c in r['pairs'].items(): pairs[k] = pairs.get(k, 0) + c
    s = dict(clips=len(per), pass_rate=sum(r['passed'] for r in per.values()) / max(len(per), 1),
             frames_any=sum(r['any'] for r in per.values()) / F, pos=sum(r['pos'] for r in per.values()) / F,
             dyn=sum(r['dyn'] for r in per.values()) / F, col=sum(r['col'] for r in per.values()) / F,
             jump=sum(r['jump'] for r in per.values()) / F,
             worst_v=max(r['worst']['v'] for r in per.values()), worst_a=max(r['worst']['a'] for r in per.values()),
             worst_j=max(r['worst']['j'] for r in per.values()), top_pairs=sorted(pairs.items(), key=lambda kv: -kv[1])[:3])
    res[lab] = dict(summary=s, per_clip=per)
    print(f'{lab:22s} clips {s["clips"]:2d} | pass {100*s["pass_rate"]:5.1f}% | frames violating: any {100*s["frames_any"]:.2f}% '
          f'pos {100*s["pos"]:.2f}% dyn {100*s["dyn"]:.2f}% col {100*s["col"]:.2f}% jump {100*s["jump"]:.2f}% | '
          f'worst v/a/j x{s["worst_v"]:.2f}/{s["worst_a"]:.2f}/{s["worst_j"]:.2f} | {s["top_pairs"]}', flush=True)
if a.out: json.dump(res, open(a.out, 'w'), indent=1, default=float)

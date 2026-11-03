"""EX002 -> G1 mapping for the cross-space evaluation (11-03, Claude).  Inverse of RoboGesture's retarget_g1_ex002.Retargeter:
  arms   per frame, EX002 elbow / wrist directions and palm frame in the EX002 torso frame (waist_pitch_link) are scaled to the G1
         segment lengths and solved with a 7-dof bounded least-squares IK per G1 arm (same residual weights as the forward map:
         2 x elbow, 3 x wrist position, 0.12 x palm rotation, 0.008 x smoothness), warm-started from the previous frame
  waist  G1 waist_yaw / waist_pitch = EX002 waist yaw / pitch, waist_roll = 0;  fingers dropped (G1 eval has none)
  root + legs from the stitched human G1 of the same clip (build_floor.py g1_<ci>.npz), so only the upper body differs
Input [T, 34] rad 50 Hz -> 39-D G1 robot_repr at 30 fps (PCHIP resample, as the forward map).
    python ex002_to_g1.py <floor dir with g1_<ci>.npz> <out dir> label=<dir with gen_Hinf_<ci>.npy | gt> ... [--ci 0,1,..]"""
import json, math, sys
from pathlib import Path
import numpy as np, torch, mujoco
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from timing import check_duration
sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/code')
sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/official/src')
from echo_g.evaluation.robot_repr import G1_JOINT_NAMES
from retarget_g1_ex002 import Retargeter, unit
ARM = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw']


class Inverse:
    def __init__(self):
        rt = self.rt = Retargeter(); g = rt.g
        self.gd = mujoco.MjData(g); self.gd.qpos[:] = 0; mujoco.mj_kinematics(g, self.gd)
        self.tor = g.body('torso_link').id
        self.sides = []
        for s in rt.sides:
            names = [f"{s['side']}_{a}_joint" for a in ARM]
            qa = np.array([g.joint(n).qposadr[0] for n in names]); da = np.array([g.joint(n).dofadr[0] for n in names])
            lo = np.array([g.jnt_range[g.joint(n).id][0] for n in names]); hi = np.array([g.jnt_range[g.joint(n).id][1] for n in names])
            up = float(np.linalg.norm(self.gd.xpos[s['ge']] - self.gd.xpos[s['gs']])); fo = float(np.linalg.norm(self.gd.xpos[s['gw']] - self.gd.xpos[s['ge']]))
            self.sides.append(dict(s=s, qa=qa, da=da, lo=lo, hi=hi, up=up, fo=fo, j=[G1_JOINT_NAMES.index(f"{s['side']}_{a}") for a in ARM]))
        self.wy, self.wr, self.wp = (g.joint(n).qposadr[0] for n in ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'))
        self.jp = np.zeros((3, g.nv)); self.jr = np.zeros((3, g.nv))

    def ex_targets(self, q):
        rt = self.rt; rt.set_ex(q); Re = rt.d.xmat[rt.et].reshape(3, 3); out = []
        for S in self.sides:
            s = S['s']
            a = Re.T @ (rt.d.xpos[s['ee']] - rt.d.xpos[s['es']]); b = Re.T @ (rt.d.xpos[s['ew']] - rt.d.xpos[s['ee']])
            out.append((a, b, Re.T @ rt.d.xmat[s['ep']].reshape(3, 3) @ s['B_e']))
        return out

    def solve(self, q34, prev):
        g, gd = self.rt.g, self.gd
        gd.qpos[:] = 0; gd.qpos[self.wy] = q34[0]; gd.qpos[self.wp] = q34[1]
        for S in self.sides: gd.qpos[S['qa']] = prev[S['qa']]
        mujoco.mj_kinematics(g, gd); Rg = gd.xmat[self.tor].reshape(3, 3).copy()
        for S, (a, b, Rp) in zip(self.sides, self.ex_targets(q34)):
            s = S['s']; sh = gd.xpos[s['gs']].copy()
            te = sh + Rg @ unit(a) * S['up']; tw = te + Rg @ unit(b) * S['fo']; tR = Rg @ Rp
            x0 = np.clip(prev[S['qa']], S['lo'] + 1e-8, S['hi'] - 1e-8); pv = prev[S['qa']].copy(); cache = {}
            def calc(x):
                if 'x' in cache and np.array_equal(cache['x'], x): return
                gd.qpos[S['qa']] = x; mujoco.mj_kinematics(g, gd); mujoco.mj_comPos(g, gd)
                pe, pw = gd.xpos[s['ge']].copy(), gd.xpos[s['gw']].copy(); R = gd.xmat[s['gw']].reshape(3, 3) @ s['B_g']
                mujoco.mj_jacBody(g, gd, self.jp, self.jr, s['ge']); je = self.jp[:, S['da']].copy()
                mujoco.mj_jacBody(g, gd, self.jp, self.jr, s['gw']); jw = self.jp[:, S['da']].copy(); jr = self.jr[:, S['da']].copy()
                jR = np.stack([np.cross(jr[:, k, None].T, R.T).T.ravel() for k in range(7)], 1)
                cache.update(x=x.copy(), r=np.r_[2 * (pe - te), 3 * (pw - tw), .12 * (R - tR).ravel(), .008 * (x - pv)],
                             J=np.vstack([2 * je, 3 * jw, .12 * jR, .008 * np.eye(7)]))
            res = least_squares(lambda x: (calc(x), cache['r'])[1], x0, jac=lambda x: (calc(x), cache['J'])[1],
                                bounds=(S['lo'], S['hi']), max_nfev=35, ftol=2e-5, xtol=2e-5, gtol=2e-5)
            gd.qpos[S['qa']] = res.x
        return gd.qpos.copy()

    def map_clip(self, q50, base39):
        T = len(q50); N = len(base39)
        check_duration(T, 50, N, 30)
        t = np.minimum(np.arange(N) / 30, (T - 1) / 50)
        q30 = PchipInterpolator(np.arange(T) / 50, q50, axis=0)(t)
        out = base39.copy(); prev = np.zeros(self.rt.g.nq)
        for i in range(N):
            qp = self.solve(q30[i], prev); prev = qp
            out[i, 10 + G1_JOINT_NAMES.index('waist_yaw')] = q30[i, 0]; out[i, 10 + G1_JOINT_NAMES.index('waist_pitch')] = q30[i, 1]
            out[i, 10 + G1_JOINT_NAMES.index('waist_roll')] = 0.
            for S in self.sides: out[i, [10 + j for j in S['j']]] = qp[S['qa']]
        return out.astype(np.float32)


if __name__ == '__main__':
    floor, out = Path(sys.argv[1]), Path(sys.argv[2])
    args = [x for x in sys.argv[3:] if not x.startswith('--')]; cis = None
    if '--ci' in sys.argv: cis = [int(c) for c in sys.argv[sys.argv.index('--ci') + 1].split(',')]; args = [x for x in args if not x[0].isdigit()]
    inv = Inverse()
    for spec in args:
        lab, src = spec.split('=', 1); od = out / lab; od.mkdir(parents=True, exist_ok=True)
        for f in sorted(floor.glob('g1_*.npz'), key=lambda p: int(p.stem.split('_')[1])):
            ci = int(f.stem.split('_')[1])
            if cis is not None and ci not in cis: continue
            base = np.load(f)['robot_repr']
            q = np.load(floor / f'gt_{ci}.npy') if src == 'gt' else np.load(Path(src) / f'gen_Hinf_{ci}.npy')
            check_duration(len(q), 50, len(base), 30)
            if (od / f'dev_{ci}.pt').exists(): continue
            g1 = inv.map_clip(q.astype(np.float64), base)
            torch.save(dict(robot_repr=torch.from_numpy(g1), fps=30.0, real_num_frames=len(g1), source=f'{lab}:{ci}'), od / f'dev_{ci}.pt')
        print('MAPPED', lab, flush=True)

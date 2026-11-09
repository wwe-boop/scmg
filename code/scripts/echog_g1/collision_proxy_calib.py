"""Calibrate a differentiable self-collision proxy for G1 against MuJoCo mesh contacts (11-05, safety module).

For sampled frames of GT and of generated outputs: MuJoCo contacts (g1_mocap_29dof.xml with collisions on, the same
check as executability.py) grouped into (a) left-arm vs right-arm links (hands / wrists crossing) and (b) arm links vs
torso; and the proxy distances computed from the official FK body origins (G1_BFS_BODY_NAMES order):
  d_lr    = min distance between {elbow, wrist_roll, wrist_pitch, wrist_yaw} of the left and of the right arm
  d_torso = min distance between {elbow, wrist_*} of either arm and the torso segment (torso_link origin -> +0.35 m
            along the torso z axis)
Prints, per group, the proxy distance quantiles on contact frames vs non-contact frames and the GT 0.5 / 1 / 2 %
quantiles, so that a hinge threshold can be chosen that (i) lies above the contact-frame distances and (ii) below what
humans do (GT) -- i.e. it penalises what MuJoCo calls a collision but not normal human motion.
    python scripts/echog_g1/collision_proxy_calib.py --mjcf M --stems S --limit N label=dir ...  (first label = GT)
"""
import argparse, os, sys
import numpy as np
import torch
import mujoco

OFF = os.environ.get('ECHOG_OFFICIAL_SRC', '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/official/src')
sys.path.insert(0, OFF)
from echo_g.evaluation import g1_kinematics as G1K
from echo_g.evaluation.robot_repr import G1_JOINT_NAMES

ap = argparse.ArgumentParser()
ap.add_argument('--mjcf', required=True); ap.add_argument('--stems', required=True); ap.add_argument('--limit', type=int, default=60)
ap.add_argument('--stride', type=int, default=3); ap.add_argument('panels', nargs='+')
a = ap.parse_args()

BN = list(G1K.G1_BFS_BODY_NAMES)
def bi(n): return BN.index(n)
ARM = lambda s: [bi(f'{s}_elbow_link'), bi(f'{s}_wrist_roll_link'), bi(f'{s}_wrist_pitch_link'), bi(f'{s}_wrist_yaw_link')]
L, R, TORSO = ARM('left'), ARM('right'), bi('torso_link')

m = mujoco.MjModel.from_xml_path(a.mjcf)
for g in range(m.ngeom):
    on = int(m.geom_bodyid[g] != 0); m.geom_contype[g] = on; m.geom_conaffinity[g] = on
d = mujoco.MjData(m)
jn = [x if x.endswith('_joint') else x + '_joint' for x in G1_JOINT_NAMES]
qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jn])
free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
mjname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
side = lambda n: 'L' if n.startswith('left_') else 'R' if n.startswith('right_') else 'C'


def mj_groups(q):
    d.qpos[:] = 0
    for j in free: d.qpos[m.jnt_qposadr[j] + 3] = 1.0
    d.qpos[qadr] = q; mujoco.mj_forward(m, d)
    lr = tor = False
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist >= 0: continue
        n1, n2 = mjname(m.geom_bodyid[c.geom1]), mjname(m.geom_bodyid[c.geom2])
        s = {side(n1), side(n2)}
        if s == {'L', 'R'}: lr = True
        elif 'C' in s and ('torso' in n1 or 'torso' in n2): tor = True
    return lr, tor


def proxies(q):
    """q [T, 29] -> d_lr [T], d_torso [T] from the official FK (identity base, origin)"""
    j = torch.as_tensor(q, dtype=torch.float64)[None]
    T = j.shape[1]
    pos, rot = G1K.g1_forward_kinematics_with_rotations(j, torch.eye(3, dtype=torch.float64).expand(1, T, 3, 3), torch.zeros(1, T, 3, dtype=torch.float64))
    pos, rot = pos[0], rot[0]
    dl = torch.cdist(pos[:, L], pos[:, R]).flatten(1).min(1).values
    a0 = pos[:, TORSO]; a1 = a0 + rot[:, TORSO, :, 2] * 0.35                      # torso segment along its z axis
    P = pos[:, L + R]                                                             # [T, 8, 3]
    ab = (a1 - a0)[:, None]; t = (((P - a0[:, None]) * ab).sum(-1) / (ab * ab).sum(-1)).clamp(0, 1)
    dt = (P - (a0[:, None] + t[..., None] * ab)).norm(dim=-1).min(1).values
    return dl.numpy(), dt.numpy()


stems = [s.strip() for s in open(a.stems) if s.strip()][: a.limit]
cols = [tuple(p.split('=', 1)) for p in a.panels]
for lab, dd in cols:
    DL, DT_, CL, CT = [], [], [], []
    for s in stems:
        p = f'{dd}/{s}.pt'
        if not os.path.exists(p): continue
        x = torch.load(p, map_location='cpu', weights_only=False); rr = np.asarray(x['robot_repr'] if isinstance(x, dict) else x, np.float64)
        q = rr[::a.stride, 10:]
        dl, dt = proxies(q); DL.append(dl); DT_.append(dt)
        g = [mj_groups(qq) for qq in q]; CL.append(np.array([x[0] for x in g])); CT.append(np.array([x[1] for x in g]))
    DL, DT_, CL, CT = (np.concatenate(v) for v in (DL, DT_, CL, CT))
    q = lambda v, ps: ' '.join(f'{x:.3f}' for x in np.quantile(v, ps)) if len(v) else '-'
    print(f'== {lab}: {len(DL)} frames | contact rate L-R {100*CL.mean():.2f}% arm-torso {100*CT.mean():.2f}%')
    print(f'   d_lr    contact q50/q90/max: {q(DL[CL], [.5, .9, 1.])} | no-contact q0.5/1/2/5%: {q(DL[~CL], [.005, .01, .02, .05])}')
    print(f'   d_torso contact q50/q90/max: {q(DT_[CT], [.5, .9, 1.])} | no-contact q0.5/1/2/5%: {q(DT_[~CT], [.005, .01, .02, .05])}')

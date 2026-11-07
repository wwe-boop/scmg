"""EX002 (two arms + RM003 three-finger hands, fixed base) geometry for control tokens (11-07, EX002 semantic line).

Exact FK = scmg/kinematics.py Kinematics (sphere-proxy npz, 1.9e-15 vs MuJoCo); points are in the BASE frame (fixed base).
Per hand (0 = left, 1 = right), body origins of the child body of each joint:
  shoulder = arm_joint1, elbow = arm_joint4, wrist = arm_joint7, index base = x_2_1, index tip = x_2_3
Rays at a frame: forearm ray (OFFICIAL 2610.03105 Score_b: "a semi-infinite ray from the wrist along the elbow-to-wrist
direction of the annotated pointing hand ... without modelling finger articulation or gaze") and index-finger ray (OUR
extension: index base -> index tip, origin the index tip).
Finger extension (our metric, sanity-checked on GT): curl of a finger = sum of its flexion joints (thumb x_1_2 + x_1_3, index
x_2_2 + x_2_3, third x_3_1 + x_3_2) as a fraction of the sum of their upper limits; EXTENDED if the fraction < EXT_FRAC (0.15, the
lower valley of the train-GT histogram: fully straight fingers). Count of a hand = extended fingers (0..3).
"""
import numpy as np
import torch

NPZ = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/sphere_proxy_vox4e.npz'
FINGERS = {0: dict(thumb=[19, 20], index=[22, 23], third=[24, 25]), 1: dict(thumb=[27, 28], index=[30, 31], third=[32, 33])}
EXT_FRAC = 0.15
_K = {}


def kin(device='cpu'):
    if device not in _K:
        from .kinematics import Kinematics
        from .common import P
        _K[device] = Kinematics(P(NPZ), device)
    return _K[device]


def _bodies():
    z = np.load(__import__('scmg.common', fromlist=['P']).P(NPZ))
    bj = np.asarray(z['body_joint'])
    b = lambda j: int(np.nonzero(bj == j)[0][0])
    return dict(shoulder=(b(4), b(11)), elbow=(b(7), b(14)), wrist=(b(10), b(17)), index_base=(b(21), b(29)), index_tip=(b(23), b(31)))


BODIES = None


def points(q34, device='cpu', chunk=8192):
    """q34 [T, 34] rad -> dict of numpy [T, 2, 3] (base frame)."""
    global BODIES
    if BODIES is None: BODIES = _bodies()
    K = kin(device); out = {k: [] for k in BODIES}
    q = torch.as_tensor(np.asarray(q34, np.float32), device=device)
    with torch.no_grad():
        for i in range(0, len(q), chunk):
            _, p = K.fk(q[i:i + chunk])
            for k, (l, r) in BODIES.items(): out[k].append(torch.stack([p[:, l], p[:, r]], 1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def rays(geo, f, h):
    """-> (origin, unit dir) for 'forearm' (official) and 'index' (ours) at frame f, hand h."""
    def u(v): return v / max(np.linalg.norm(v), 1e-9)
    return dict(forearm=(geo['wrist'][f, h], u(geo['wrist'][f, h] - geo['elbow'][f, h])),
                index=(geo['index_tip'][f, h], u(geo['index_tip'][f, h] - geo['index_base'][f, h])))


def curl_frac(q34, upper):
    """q34 [T, 34], upper [34] -> [T, 2, 3] curl fraction of (thumb, index, third) per hand."""
    q = np.asarray(q34); up = np.asarray(upper)
    out = np.zeros((len(q), 2, 3), np.float32)
    for h in (0, 1):
        for i, f in enumerate(('thumb', 'index', 'third')):
            j = FINGERS[h][f]; out[:, h, i] = q[:, j].sum(1) / up[j].sum()
    return out


def finger_counts(q34, upper):
    """-> int [T, 2] extended fingers per hand."""
    return (curl_frac(q34, upper) < EXT_FRAC).sum(-1)


def soft_counts(q34_t, upper_t, beta=40.0):
    """torch [..., 34] -> soft extended-finger count per hand [..., 2] (sigmoid of the margin below EXT_FRAC)."""
    outs = []
    for h in (0, 1):
        s = 0
        for f in ('thumb', 'index', 'third'):
            j = FINGERS[h][f]; fr = q34_t[..., j].sum(-1) / upper_t[j].sum()
            s = s + torch.sigmoid(beta * (EXT_FRAC - fr))
        outs.append(s)
    return torch.stack(outs, -1)


def points_t(q34):
    """differentiable: torch [N, 34] -> dict wrist / elbow / index_base / index_tip [N, 2, 3] (base frame)."""
    global BODIES
    if BODIES is None: BODIES = _bodies()
    _, p = kin(str(q34.device)).fk(q34)
    return {k: torch.stack([p[:, l], p[:, r]], 1) for k, (l, r) in BODIES.items()}

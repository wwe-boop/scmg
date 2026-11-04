"""G1 foot-contact consistency (11-05, SCMG module for the ECHO-G benchmark's foot error / contact sliding).

Batched, differentiable version of the official evaluator's foot geometry (echo_g.evaluation.g1_motion_cls
fk_grounded_world / _sole_points; the FK itself is imported from the official code so the geometry is exactly the one
that is scored): root path integrated from the per-frame yaw change and local root velocity, FK of the 29 joints,
4 sole spheres per foot, grounded so that the lowest sole bottom of frame 0 sits at z = 0.

foot_contact_loss(pred, gt, w) -- on frame pairs where the GROUND TRUTH foot is in contact (official contact rule: some
sole bottom within [-0.02, 0.12] m on both frames), L1 between predicted and ground-truth sole xy velocity (sliding) and
between their lowest sole-bottom heights (floating / penetration). Frames outside contact are untouched, so steps and
weight shifts of the human are not suppressed.

Inputs are PHYSICAL robot_repr frames [B, T, 39] at `fps` (the G1 view is 50 Hz: yaw change in rad/frame at 50 Hz,
velocities in m/s -- the same units the evaluator integrates at 30 fps).
"""
import os
import sys
import numpy as np
import torch

OFFICIAL_SRC = os.environ.get('ECHOG_OFFICIAL_SRC', '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/official/src')
CONTACT_HEIGHT = 0.12          # evaluator defaults (FootMetrics.__init__)
CONTACT_PENETRATION = 0.02
_M = {}


def _mods():
    if not _M:
        if OFFICIAL_SRC not in sys.path: sys.path.insert(0, OFFICIAL_SRC)
        from echo_g.evaluation import g1_kinematics as G1K
        from echo_g.evaluation.rotation import extract_yaw_from_rotation, rotation_6d_to_matrix, rotation_matrix_from_yaw
        _M.update(G1K=G1K, yaw_of=extract_yaw_from_rotation, r6=rotation_6d_to_matrix, ryaw=rotation_matrix_from_yaw)
    return _M


def sole_points(rr, fps):
    """rr [B, T, 39] physical -> sole sphere centres [B, T, 8, 3] (grounded at frame 0), radii [8], foot ids [8]."""
    M = _mods(); G1K = M['G1K']
    base_rot = M['r6'](rr[..., :6])                                                   # [B, T, 3, 3]
    yaw_delta, vel, joints = rr[..., 6], rr[..., 7:10], rr[..., 10:]
    yaw0 = M['yaw_of'](base_rot[:, 0])                                                # [B]
    yaw = yaw0[:, None] + torch.cat([torch.zeros_like(yaw_delta[:, :1]), torch.cumsum(yaw_delta[:, 1:], 1)], 1)
    gvel = torch.einsum('btij,btj->bti', M['ryaw'](yaw), vel)
    trans = torch.cat([torch.zeros_like(gvel[:, :1]), torch.cumsum(gvel[:, 1:] / float(fps), 1)], 1)
    pos, rot = G1K.g1_forward_kinematics_with_rotations(joints, base_rot, trans)       # [B, T, 30, 3], [B, T, 30, 3, 3]
    pts, fids = [], []
    for foot, bfs in G1K.G1_ANKLE_BFS.items():
        for off in G1K.G1_SOLE_OFFSETS:
            o = torch.tensor(off, device=rr.device, dtype=rr.dtype)
            pts.append(pos[:, :, bfs] + torch.einsum('btij,j->bti', rot[:, :, bfs], o)); fids.append(foot)
    sole = torch.stack(pts, 2)                                                        # [B, T, 8, 3]
    r = float(G1K.G1_SOLE_RADIUS)
    lift = -(sole[:, 0, :, 2] - r).min(-1).values                                     # [B] ground the frame-0 lowest sole bottom
    sole = sole + torch.stack([torch.zeros_like(lift), torch.zeros_like(lift), lift], -1)[:, None, None]
    return sole, np.full(len(fids), r), np.asarray(fids)


def foot_contact_loss(pred, gt, w, fps):
    """pred, gt [B, T, 39] physical; w [B, T] frame weights -> dict(foot_slide [m/s], foot_height [m], foot_contact_frac)."""
    sp, r, fid = sole_points(pred.float(), fps)
    with torch.no_grad():
        sg, _, _ = sole_points(gt.float(), fps)
    bot_p, bot_g = sp[..., 2] - r[0], sg[..., 2] - r[0]
    slide, height, cnt, cnth, frac = 0., 0., 0., 0., []
    for f in np.unique(fid):
        m = torch.as_tensor(fid == f, device=pred.device)
        with torch.no_grad():
            c = ((bot_g[..., m] >= -CONTACT_PENETRATION) & (bot_g[..., m] <= CONTACT_HEIGHT)).any(-1)       # [B, T]
            pair = (c[:, 1:] & c[:, :-1]).float() * w[:, 1:] * w[:, :-1]
            hw = c.float() * w
            frac.append(c.float().mean())
        vp = torch.diff(sp[..., m, :2], dim=1) * fps; vg = torch.diff(sg[..., m, :2], dim=1) * fps        # [B, T-1, 4, 2]
        slide = slide + ((vp - vg).abs().sum(-1).mean(-1) * pair).sum(); cnt = cnt + pair.sum()
        lp, lg = bot_p[..., m].min(-1).values, bot_g[..., m].min(-1).values                             # [B, T]
        height = height + ((lp - lg).abs() * hw).sum(); cnth = cnth + hw.sum()
    return dict(foot_slide=slide / max(float(cnt), 1.), foot_height=height / max(float(cnth), 1.),
                foot_contact_frac=torch.stack(frac).mean())

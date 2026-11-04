"""Stance-consistent root odometry for G1 (11-05, SCMG module for the benchmark's contact sliding / foot metrics).

A standing co-speech robot cannot slide its feet: horizontal base motion has to come from the legs. Generated root
velocities are ~10x the human ones (0.19 vs 0.018 m/s) and are not consistent with the generated leg motion, so the
feet slide. This module recomputes ONLY the horizontal root velocity (robot_repr channels 7, 8) so that, between frames
t-1 and t, the stance foot (lower sole of frame t-1) keeps its horizontal position. Joint angles, base orientation, yaw
change and vertical velocity are untouched -> joint-space metrics (FGD, diversity, beat alignment) cannot change.
Causal: frame t uses frames t-1 and t only, so it runs inside the streaming loop.

The geometry is the official evaluator's (scmg/g1_contact.py imports its FK): root path integrated as
yaw[t] = yaw0 + sum yaw_delta[1..t], trans[t] = trans[t-1] + R_yaw(yaw[t]) v_local[t] / fps.
"""
import torch
from .g1_contact import _mods


def stance_root_velocity(rr, fps):
    """rr [T, 39] physical robot_repr (torch, float64 recommended) -> copy with channels 7, 8 recomputed."""
    M = _mods(); G1K = M['G1K']
    rr = rr.clone()
    T = rr.shape[0]
    if T < 2: return rr
    base_rot = M['r6'](rr[:, :6])
    yaw0 = M['yaw_of'](base_rot[0])
    yaw = yaw0 + torch.cat([torch.zeros_like(rr[:1, 6]), torch.cumsum(rr[1:, 6], 0)])
    Ryaw = M['ryaw'](yaw)                                                               # [T, 3, 3]
    # pose FK with the full base orientation and zero translation: sole points relative to the root, in world axes
    pos, rot = G1K.g1_forward_kinematics_with_rotations(rr[None, :, 10:], base_rot[None], torch.zeros_like(rr[None, :, :3]))
    pos, rot = pos[0], rot[0]
    feet = []
    for foot, bfs in sorted(G1K.G1_ANKLE_BFS.items()):
        pts = torch.stack([pos[:, bfs] + torch.einsum('tij,j->ti', rot[:, bfs], torch.tensor(o, dtype=rr.dtype, device=rr.device))
                           for o in G1K.G1_SOLE_OFFSETS], 1)                            # [T, 4, 3]
        feet.append(pts)
    feet = torch.stack(feet, 1)                                                        # [T, 2, 4, 3]
    low = feet[..., 2].min(-1).values                                                  # [T, 2] lowest sole point per foot
    centre = feet.mean(2)                                                              # [T, 2, 3]
    stance = low[:-1].argmin(-1)                                                       # stance foot of frame t-1, [T-1]
    idx = torch.arange(T - 1, device=rr.device)
    # world translation step so that the stance foot centre keeps its xy position: trans[t] - trans[t-1] = c[t-1] - c[t]
    dxy = centre[idx, stance, :2] - centre[idx + 1, stance, :2]                       # [T-1, 2]
    g = torch.zeros(T - 1, 3, dtype=rr.dtype, device=rr.device); g[:, :2] = dxy * fps
    g[:, 2] = torch.einsum('tij,tj->ti', Ryaw[1:], rr[1:, 7:10])[:, 2]                # keep the generated vertical motion
    v_local = torch.einsum('tji,tj->ti', Ryaw[1:], g)                                  # R^T g
    rr[1:, 7:9] = v_local[:, :2]
    rr[0, 7:9] = rr[1, 7:9]                                                            # frame 0 is never integrated
    return rr

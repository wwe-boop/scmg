"""G1 two-foot lock: contact-consistent kinematic projection of the legs (11-05, SCMG module for contact sliding).

Under the official contact rule both G1 feet are in contact essentially all the time while speaking (100% of frames for
GT, ECHO-G and ours), so contact sliding is the motion of the feet relative to each other (ours 0.175 m/s vs GT 0.054) plus
the motion of the feet relative to the ground. This module
  1. adjusts ONLY the 12 leg joints, per frame, so that the left foot keeps its frame-0 pose relative to the right foot
     (damped Gauss-Newton on [relative position, relative rotation] with a pull towards the generated leg angles), and
  2. recomputes the horizontal root velocity from the stance foot (scmg/g1_stance.py), so the feet stay put in the world.
Arms, waist, base orientation, yaw change and vertical root velocity are unchanged. Frame t only needs frame 0 and
frames t-1, t: it can run inside the streaming loop. Geometry = the official evaluator's FK (scmg/g1_contact.py).
"""
import torch
from .g1_contact import _mods
from .g1_stance import stance_root_velocity

LEG = slice(0, 12)          # robot_repr joints 10:22 = canonical joints 0:12 (both legs)


def _feet(joints, base_rot):
    """joints [N, T, 29], base_rot [T, 3, 3] -> ankle positions [N, T, 2, 3], rotations [N, T, 2, 3, 3] (root at origin)."""
    M = _mods(); G1K = M['G1K']
    N, T, _ = joints.shape
    pos, rot = G1K.g1_forward_kinematics_with_rotations(joints, base_rot[None].expand(N, -1, -1, -1), torch.zeros(N, T, 3, dtype=joints.dtype))
    ids = [G1K.G1_ANKLE_BFS[k] for k in sorted(G1K.G1_ANKLE_BFS)]
    return pos[:, :, ids], rot[:, :, ids]


def _vee_log(R):
    """rotation matrices [..., 3, 3] -> axis-angle vectors [..., 3] (small-angle safe)."""
    c = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7); th = torch.acos(c)
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    k = torch.where(th > 1e-6, th / (2 * torch.sin(th)), torch.full_like(th, 0.5))
    return w * k[..., None]


def _residual(feet_pos, feet_rot, p0, R0, w_rot):
    """left foot pose in the right-foot frame vs the frame-0 relation -> [N, T, 6]"""
    Rr, Rl = feet_rot[..., 1, :, :], feet_rot[..., 0, :, :]
    prel = torch.einsum('...ji,...j->...i', Rr, feet_pos[..., 0, :] - feet_pos[..., 1, :])
    Rrel = Rr.transpose(-1, -2) @ Rl
    return torch.cat([prel - p0, w_rot * _vee_log(R0.transpose(-1, -2) @ Rrel)], -1)


def foot_lock(rr, fps, iters=20, lam=1e-4, w_rot=0.1, eps=1e-4, max_step=0.2):
    """rr [T, 39] physical robot_repr (float64) -> projected copy (legs + horizontal root velocity changed)."""
    M = _mods()
    rr = rr.clone(); T = rr.shape[0]
    if T < 2: return rr
    base_rot = M['r6'](rr[:, :6]); q_gen = rr[:, 10:].clone(); q = q_gen.clone()
    fp, fr = _feet(q[None], base_rot)
    Rr0, Rl0 = fr[0, 0, 1], fr[0, 0, 0]
    p0 = Rr0.T @ (fp[0, 0, 0] - fp[0, 0, 1]); R0 = Rr0.T @ Rl0
    n = 12; eye = torch.eye(n, dtype=rr.dtype)
    for _ in range(iters):
        # finite-difference Jacobian over the 12 leg joints, all frames at once: N = 1 + 12 FK evaluations
        Q = q[None].repeat(n + 1, 1, 1); Q[1:, :, LEG] += eps * eye[:, None, :]
        fp, fr = _feet(Q, base_rot)
        r = _residual(fp, fr, p0, R0, w_rot)                                          # [13, T, 6]
        J = ((r[1:] - r[:1]) / eps).permute(1, 2, 0)                                  # [T, 6, 12]
        r0 = r[0]                                                                     # [T, 6]
        dq_reg = q[:, LEG] - q_gen[:, LEG]
        A = J.transpose(1, 2) @ J + lam * eye                                         # damped normal equations
        g = J.transpose(1, 2) @ r0[..., None] + lam * dq_reg[..., None]
        step = -torch.linalg.solve(A, g)[..., 0]
        step = step.clamp(-max_step, max_step)
        q[:, LEG] = q[:, LEG] + step
    rr[:, 10:] = q
    return stance_root_velocity(rr, fps)

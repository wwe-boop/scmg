"""Differentiable EX002+RM003 kinematics and self-collision proxy (torch), built from scripts/build_sphere_proxy.py.

fk(q [N, 34] rad) -> body rotations [N, nb, 3, 3], positions [N, nb, 3] (exact: matches MuJoCo to 1e-15 in the builder).
penetration(q, pair_idx=None) -> [N, P] = relu(d_safe - (|c_i - c_j| - r_i - r_j + offset_bp)) over sphere pairs: voxel-shell
spheres per collision body, per-body-pair offsets calibrated against MuJoCo contacts (inflation of the shells removed).
Used by the stage-1 physical regularisation loss (losses.py, StageCfg.w_phys) and the stage-2 differentiable variants."""
import numpy as np
import torch

from .common import require_ex002


def _quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Kinematics(torch.nn.Module):
    def __init__(self, npz, device='cpu'):
        super().__init__()
        require_ex002('Kinematics (EX002+RM003 FK / sphere proxy)')
        z = np.load(npz)
        self.nb = len(z['parent'])
        self.parent = [int(x) for x in z['parent']]
        self.jidx = [int(x) for x in z['body_joint']]; self.jtype = [int(x) for x in z['joint_type']]
        self.has_joint = [bool(np.any(z['joint_axis'][b])) for b in range(self.nb)]
        reg = lambda n, x: self.register_buffer(n, torch.as_tensor(np.asarray(x), dtype=torch.float32))
        reg('Rloc', np.stack([_quat_mat(q) for q in z['body_quat']])); reg('ploc', z['body_pos'])
        a = z['joint_axis'] / np.maximum(np.linalg.norm(z['joint_axis'], axis=1, keepdims=True), 1e-9)
        reg('axis', a); reg('anchor', z['joint_anchor']); reg('jconst', z['joint_const'])
        reg('sc', z['sphere_center']); reg('sr', z['sphere_radius'])
        self.register_buffer('sb', torch.as_tensor(z['sphere_body'], dtype=torch.long))
        self.register_buffer('pairs', torch.as_tensor(z['pairs'], dtype=torch.long))
        off = z['pair_offset'] if 'pair_offset' in z.files else np.zeros(len(z['pairs']))
        reg('poff', off)
        self.to(device)

    @staticmethod
    def _axis_rot(a, th):                                          # a [3], th [N] -> [N, 3, 3]
        K = torch.zeros(3, 3, device=a.device, dtype=th.dtype)
        K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -a[2], a[1], a[2], -a[0], -a[1], a[0]
        s, c = torch.sin(th)[:, None, None], torch.cos(th)[:, None, None]
        return torch.eye(3, device=a.device, dtype=th.dtype) + s * K + (1 - c) * (K @ K)

    def fk(self, q):
        N = q.shape[0]
        R = [None] * self.nb; p = [None] * self.nb
        R[0] = torch.eye(3, device=q.device, dtype=q.dtype).expand(N, 3, 3); p[0] = q.new_zeros(N, 3)
        for b in range(1, self.nb):
            pa = self.parent[b]
            Rb = R[pa] @ self.Rloc[b]; pb = p[pa] + (R[pa] @ self.ploc[b][:, None])[..., 0]
            if self.has_joint[b]:
                th = q[:, self.jidx[b]] if self.jidx[b] >= 0 else self.jconst[b].expand(N)
                if self.jtype[b] == 0:
                    Rj = self._axis_rot(self.axis[b], th)
                    pb = pb + (Rb @ (self.anchor[b][:, None] - Rj @ self.anchor[b][:, None]))[..., 0]
                    Rb = Rb @ Rj
                else:
                    pb = pb + (Rb @ (self.axis[b][:, None] * th[:, None, None]))[..., 0]
            R[b], p[b] = Rb, pb
        return torch.stack(R, 1), torch.stack(p, 1)

    def penetration(self, q, pair_idx=None, d_safe=0.0):
        R, p = self.fk(q)
        w = (R[:, self.sb] @ self.sc[None, :, :, None])[..., 0] + p[:, self.sb]          # [N, S, 3]
        pr = self.pairs if pair_idx is None else self.pairs[pair_idx]
        off = self.poff if pair_idx is None else self.poff[pair_idx]
        d = (w[:, pr[:, 0]] - w[:, pr[:, 1]]).norm(dim=-1) - self.sr[pr[:, 0]] - self.sr[pr[:, 1]] + off
        return torch.relu(d_safe - d)

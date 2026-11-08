"""Causal safety layers for G1 joint trajectories (design v6 C3, 11-05): perceptual projection + L2 projection + baselines.

Every layer has the same streaming interface and is strictly causal at chunk level:

    layer.reset();  for chunk in chunks:  safe = layer.step(chunk)      # chunk [k, 29] rad, k <= K
    (or run_stream(layer, q[T, 29], K))

`step` sees only the already executed output (layer history), the raw chunks seen so far and the current raw chunk;
nothing from later chunks. Only the 29 joint angles (robot_repr[:, 10:39], echo_g G1_JOINT_NAMES order) are filtered; the
base (robot_repr[:, 0:10]) is fixed and passed through by the callers.

Constraint set (identical definitions to the independent checker scripts/echog_g1/executability.py where it has them):
  position  lo <= q <= hi                       Unitree URDF g1_custom_collision_29dof.urdf <limit lower upper>
  velocity  |q_t - q_{t-1}| * fps <= vmax       URDF <limit velocity>; a backward-difference bound implies the checker's
                                                central difference (np.gradient) bound
  accel     |q_t - 2 q_{t-1} + q_{t-2}| fps^2 <= amax     } not in the URDF: per-joint 99.99th percentile of the training
  jerk      |q_t - 3 q_{t-1} + 3 q_{t-2} - q_{t-3}| fps^3 <= jmax } GT (scripts/echog_g1/g1_accjerk_limits.json, CLAUDE-SET)
  self-collision  no MuJoCo contact with dist < 0 between robot bodies, g1_mocap_29dof.xml with collisions switched on
                  for every robot geom, identity base at the origin (exactly the checker's contacts()), minus the
                  structural-overlap table g1_exclusion_train.json (empty for G1).
Free start: the first chunk of a stream has no history, so the first frames are constrained by position / collision only
and the derivative constraints apply from the first frame that has enough predecessors (the clip is a window of a
continuous stream; the robot state before it is unknown).

Layers
  NoFilter            identity
  ClipLayer           position clamp + velocity / acceleration limiting per frame (the SafetyGovernor recipe without
                      its braking planner); collision_fallback=True additionally holds the previous output frame
                      whenever the clipped frame is in contact ("clip + collision fallback")
  CBFQPLayer          per-frame discrete-time CBF-QP (no horizon): min ||q_t - r_t||^2 s.t. discrete CBF on the position
                      limits and on linearised pair distances, velocity hard, acceleration / jerk relaxed when infeasible;
                      collision rows re-linearised until the exact check passes, else hold the previous frame
  RuckigLayer         Ruckig online trajectory generation per joint (no synchronisation), target = clamped raw frame +
                      causal finite-difference target velocity, limits vmax / amax / jmax; collisions are not handled
  ProjectionLayer     the method. Per chunk: decision variables = the K chunk frames + B braking frames; constraints
                      above on all of them (history-coupled differences) + terminal braking (zero velocity and
                      acceleration at the end of the braking tail) + linearised self-collision rows
                      d(q_lin) + grad d . (q - q_lin) >= d_safe for every near body pair (dist < act). Solve the QP
                      (OSQP), check the solution with the exact contact test, add rows linearised at the colliding frames
                      of the current solution, repeat (<= max_iter). Execute the chunk, keep the verified braking tail as
                      the backup plan. If no verified solution: fallback = execute the previous verified braking tail
                      ("brake to the last safe pose"), then hold; counted per chunk / frame. Because the previous tail
                      followed by a hold is always feasible, the true problem is recursively feasible; only the
                      linearisation can make the QP miss it, and that case is exactly the counted fallback.
                      objective='l2': weighted L2 to the raw chunk.
                      objective='perceptual': start from the verified L2 solution, a few Adam steps on
                        ||E(window(executed past + z)) - E(window(raw past + raw chunk))||^2 / ||E_ref||^2
                        + lam * weighted L2 + mu * quadratic penalty of the linear constraint rows,
                      E = the ECHO-G official FGD encoder (pure2, weights/evaluation/g1_fgd_encoder.pt) on the last
                      `window` frames; then project the result back with the same QP (L2 to the refined chunk) and the
                      same verify / re-linearise loop; if that projection does not verify, the L2 solution is used
                      (counted as perc_reject).
"""
from __future__ import annotations

import json
import math
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import scipy.sparse as sp

FPS = 30.0
DT = 1.0 / FPS
J = 29
G1_JOINTS = [
    'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
    'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
    'waist_yaw', 'waist_roll', 'waist_pitch',
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow',
    'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow',
    'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw']
LEG = list(range(12))
ASSETS = os.environ.get('SCMG_G1_ASSETS', '/home/xr-25b7/bulk/g1-assets/unitree_g1')
URDF = os.path.join(ASSETS, 'g1_custom_collision_29dof.urdf')
MJCF = os.path.join(ASSETS, 'g1_mocap_29dof.xml')
_HERE = os.path.dirname(os.path.abspath(__file__))
ACCJERK = os.path.join(_HERE, '..', 'scripts', 'echog_g1', 'g1_accjerk_limits.json')
EXCLUSION = os.path.join(_HERE, '..', 'scripts', 'echog_g1', 'g1_exclusion_train.json')
FGD_CKPT = os.environ.get('ECHOG_FGD_CKPT', '/mnt/bulk/zehan/echog-dataset/weights-ro/evaluation/g1_fgd_encoder.pt')
COEF = {1: np.array([1., -1.]), 2: np.array([1., -2., 1.]), 3: np.array([1., -3., 3., -1.])}
INF = 1e20
SOLVER = os.environ.get('SCMG_SAFETY_QP', 'clarabel')     # 'clarabel' (default) or 'osqp'


def check_joint_order():
    """the hard-coded order must be echo_g's G1_JOINT_NAMES (checked when echo_g is importable)."""
    try:
        from echo_g.evaluation.robot_repr import G1_JOINT_NAMES
    except Exception:
        return None
    names = [n[:-6] if n.endswith('_joint') else n for n in G1_JOINT_NAMES]
    if names != G1_JOINTS:
        raise ValueError(f'G1 joint order mismatch: {names}')
    return True


# ----------------------------------------------------------------------------------------------------------- limits
class Limits:
    def __init__(self, urdf=URDF, accjerk=ACCJERK, names=G1_JOINTS):
        lim = {j.get('name'): j.find('limit').attrib for j in ET.parse(urdf).getroot().findall('joint')
               if j.get('type') == 'revolute'}
        n = [x + '_joint' for x in names]
        self.lo = np.array([float(lim[x]['lower']) for x in n])
        self.hi = np.array([float(lim[x]['upper']) for x in n])
        self.vmax = np.array([float(lim[x]['velocity']) for x in n])
        aj = json.load(open(accjerk))
        if aj['joints'] != [x if x.endswith('_joint') else x for x in aj['joints']] or \
                [x[:-6] if x.endswith('_joint') else x for x in aj['joints']] != list(names):
            raise ValueError('acc/jerk table joint order differs')
        self.amax = np.array(aj['acc_max'], float)
        self.jmax = np.array(aj['jerk_max'], float)
        self.rate = {1: self.vmax, 2: self.amax, 3: self.jmax}
        self.n, self.dt = len(names), DT
        self.source = dict(urdf=os.path.abspath(urdf), accjerk=os.path.abspath(accjerk))

    def step_bound(self, k, shrink=1.0):
        """bound on the k-th backward difference (rad) at the frame rate 1 / dt."""
        return self.rate[k] * self.dt ** k * shrink


# -------------------------------------------------------------------------------------------------------- collision
class G1Collision:
    """Exact contact test (= executability.contacts) and near-pair linearisation on the same MJCF."""
    midpoints = False           # the G1 checker tests frames only (EX002's validator also tests midpoints)

    def __init__(self, mjcf=MJCF, margin=0.03, exclusion=EXCLUSION, names=G1_JOINTS):
        import mujoco
        self.mj = mujoco
        self.m0 = self._model(mjcf, 0.0)          # checker model (geom margin as in the file = 0)
        self.m1 = self._model(mjcf, margin)       # near-contact model
        self.d0 = mujoco.MjData(self.m0)
        self.d1 = mujoco.MjData(self.m1)
        m = self.m0
        jn = [x + '_joint' for x in names]
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, x) for x in jn]
        if min(jid) < 0:
            raise ValueError('joint missing in MJCF')
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([m.jnt_dofadr[j] for j in jid])
        self.free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.excl = set()
        if exclusion and os.path.isfile(exclusion):
            for x, y, _ in json.load(open(exclusion))['excluded']:
                a, b = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s) for s in (x, y))
                self.excl.add(tuple(sorted((a, b))))
        self.margin = margin
        self.jp1 = np.zeros((3, m.nv)); self.jp2 = np.zeros((3, m.nv))
        self.n_exact = 0; self.n_lin = 0

    def _model(self, path, margin):
        m = self.mj.MjModel.from_xml_path(path)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != 0:
                m.geom_contype[g] = 1; m.geom_conaffinity[g] = 1
                if margin: m.geom_margin[g] = margin
            else:
                m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0
        return m

    def _set(self, m, d, q):
        d.qpos[:] = 0
        for j in self.free:
            d.qpos[m.jnt_qposadr[j] + 3] = 1.0
        d.qpos[self.qadr] = q

    def contacts(self, q):
        """set of (body, body) pairs with a contact dist < 0, exactly as the checker computes it (mj_forward)."""
        m, d = self.m0, self.d0
        self._set(m, d, q)
        self.mj.mj_forward(m, d)
        self.n_exact += 1
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist < 0:
                p = tuple(sorted((m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])))
                if p not in self.excl:
                    out.add(p)
        return out

    def collides(self, q):
        return bool(self.contacts(q))

    def near(self, q, act):
        """linearised pair distances at q: list of (grad [29], dist) for every contact (incl. separated ones within
        the margin model) with dist < act. d(q + dq) ~= dist + grad . dq."""
        mj, m, d = self.mj, self.m1, self.d1
        self._set(m, d, q)
        mj.mj_kinematics(m, d); mj.mj_comPos(m, d); mj.mj_collision(m, d)
        self.n_lin += 1
        rows = []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= act:
                continue
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if tuple(sorted((b1, b2))) in self.excl:
                continue
            n = np.array(c.frame[:3])
            p1 = c.pos - 0.5 * c.dist * n
            p2 = c.pos + 0.5 * c.dist * n
            mj.mj_jac(m, d, self.jp1, None, p1, b1)
            mj.mj_jac(m, d, self.jp2, None, p2, b2)
            g = n @ (self.jp2[:, self.dadr] - self.jp1[:, self.dadr])
            rows.append((g, float(c.dist)))
        return rows


def lin_rows(coll, q, q_safe, act, bisect=10):
    """collision rows for pose q. If q is in contact and a contact-free reference q_safe is given, linearise at the
    boundary point of the segment q_safe -> q (bisection on the exact test): distance ~0 there, so the half-space
    d(q_b) + grad . (x - q_b) >= d_safe is the local separating plane; deep-penetration normals at q are unreliable.
    Returns (rows, linearisation point)."""
    if q_safe is not None and coll.collides(q):
        lo, hi = 0.0, 1.0                                      # q_safe + s (q - q_safe): s=0 free, s=1 in contact
        for _ in range(bisect):
            mid = 0.5 * (lo + hi)
            if coll.collides(q_safe + mid * (q - q_safe)): hi = mid
            else: lo = mid
        qb = q_safe + hi * (q - q_safe)                        # just inside the boundary: the touching pair is listed
        return coll.near(qb, act), qb
    return coll.near(q, act), q


# --------------------------------------------------------------------------------------------------- exact checks
def diff_ok(H, X, lim, tol=1e-9):
    """derivative limits of the executed sequence [H; X] on every frame of X that has enough predecessors."""
    E = np.concatenate([H, X], 0) if len(H) else X
    h = len(H)
    for k in (1, 2, 3):
        if E.shape[0] <= k:
            continue
        D = np.diff(E, k, axis=0)                 # row i = difference ending at frame i + k
        D = D[max(h - k, 0):]                      # only differences that involve at least one frame of X
        if D.size and np.any(np.abs(D) > lim.step_bound(k) + tol):
            return False, f'd{k}'
    return True, ''


def pos_ok(X, lim, tol=1e-9):
    return bool(np.all(X >= lim.lo - tol) and np.all(X <= lim.hi + tol))


# -------------------------------------------------------------------------------------------------------- the QP
class ChunkQP:
    """linear constraints over N frames (frame-major x[t * J + j]) after history H [h <= 3, J]."""

    def __init__(self, lim, H, N, terminal=True, pos_margin=1e-4, shrink=0.995, deriv=(1, 2, 3)):
        J = self.J = lim.n
        h = len(H)
        blocks, L, U = [], [], []
        I = sp.identity(N * J, format='csr')
        blocks.append(I); L.append(np.tile(lim.lo + pos_margin, N)); U.append(np.tile(lim.hi - pos_margin, N))
        for k in deriv:
            F = h + N
            rows = [e for e in range(max(k, h), F)]     # difference ending at extended frame e (needs e - k >= 0)
            if not rows:
                continue
            D = np.zeros((len(rows), F))
            for r, e in enumerate(rows):
                D[r, e - k: e + 1] = COEF[k][::-1]
            Dh, Dx = D[:, :h], D[:, h:]
            const = (Dh @ H) if h else np.zeros((len(rows), J))          # [rows, J]
            blocks.append(sp.kron(sp.csr_matrix(Dx), sp.identity(J), format='csr'))
            b = lim.step_bound(k, shrink)
            L.append((-b[None] - const).reshape(-1)); U.append((b[None] - const).reshape(-1))
        if terminal and N >= 2:
            # end of the braking tail at rest: q_{N-1} = q_{N-2} (= q_{N-3}); missing frames come from H
            E = []
            Hn = H
            for (a, b_) in ((N - 1, N - 2), (N - 2, N - 3)):
                row = np.zeros(N); c = np.zeros(J)
                row[a] = 1.0
                if b_ >= 0: row[b_] -= 1.0
                elif h + b_ >= 0: c = -Hn[h + b_]
                else: continue
                E.append((row, c))
            for row, c in E:
                blocks.append(sp.kron(sp.csr_matrix(row[None]), sp.identity(J), format='csr'))
                L.append(-c); U.append(-c)
        self.A0 = sp.vstack(blocks, format='csc')
        self.l0 = np.concatenate(L); self.u0 = np.concatenate(U)
        self.N, self.H = N, H
        self.crow, self.cl = [], []                 # collision rows (accumulated cutting planes)

    def add_collision(self, t, rows, qlin, d_safe):
        J = self.J
        for g, dist in rows:
            r = np.zeros(self.N * J); r[t * J:(t + 1) * J] = g
            self.crow.append(r); self.cl.append(d_safe - dist + g @ qlin)

    def add_collision_mid(self, t, rows, qlin, d_safe):
        """rows linearised at qlin for the MIDPOINT of frames t-1 and t (frame -1 = last history frame)."""
        J = self.J
        for g, dist in rows:
            r = np.zeros(self.N * J); r[t * J:(t + 1) * J] = 0.5 * g; c = 0.0
            if t >= 1: r[(t - 1) * J:t * J] = 0.5 * g
            else: c = 0.5 * g @ self.H[-1]
            self.crow.append(r); self.cl.append(d_safe - dist + g @ qlin - c)

    def matrices(self):
        if self.crow:
            A = sp.vstack([self.A0, sp.csc_matrix(np.array(self.crow))], format='csc')
            l = np.concatenate([self.l0, np.array(self.cl)]); u = np.concatenate([self.u0, np.full(len(self.cl), INF)])
        else:
            A, l, u = self.A0, self.l0, self.u0
        return A, l, u

    def solve(self, target, w, tail_smooth=1e-6, reg=1e-9):
        """min sum_t w_t,j (x - target)^2 over frames with a target (rows of `target`, NaN = none) + tail_smooth *
        sum ||x_t - x_{t-1}||^2 over frames without a target + reg ||x||^2."""
        import osqp
        N, J = self.N, self.J
        tw = np.zeros((N, J)); tg = np.zeros((N, J))
        has = ~np.isnan(target[:, 0])
        tw[has] = w[has]; tg[has] = target[has]
        P = sp.diags((2 * tw + 2 * reg).reshape(-1))
        q = (-2 * tw * tg).reshape(-1)
        free = np.nonzero(~has)[0]
        if len(free) and tail_smooth:
            rows = []
            for t in free:
                if t >= 1:
                    r = np.zeros(N); r[t] = 1; r[t - 1] = -1; rows.append(r)
            if rows:
                Dm = sp.kron(sp.csr_matrix(np.array(rows)), sp.identity(J), format='csr')
                P = P + 2 * tail_smooth * (Dm.T @ Dm)
        A, l, u = self.matrices()
        P = sp.triu(sp.csc_matrix(P), format='csc')
        if SOLVER == 'clarabel':
            return self._clarabel(P, q, A, l, u)
        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                   polishing=True, max_iter=8000)
        res = prob.solve()
        st = str(res.info.status).lower()
        if res.x is None or 'infeasible' in st or not np.all(np.isfinite(res.x)) or 'solved' not in st:
            return None, st
        return res.x.reshape(N, J), st

    def _clarabel(self, P, q, A, l, u):
        """interior-point solve (robust on the tightly bounded jerk rows where OSQP can stall)."""
        import clarabel
        A = sp.csr_matrix(A)
        eq = np.abs(u - l) < 1e-15
        up = (u < INF / 2) & ~eq
        lo = (l > -INF / 2) & ~eq
        Ac = sp.vstack([A[eq], A[up], -A[lo]], format='csc')
        b = np.concatenate([u[eq], u[up], -l[lo]])
        cones = ([clarabel.ZeroConeT(int(eq.sum()))] if eq.any() else []) + [clarabel.NonnegativeConeT(int(up.sum() + lo.sum()))]
        st_ = clarabel.DefaultSettings(); st_.verbose = False; st_.max_iter = 200
        # 11-08 (EX002 3c-3b p99): KKT backend. Clarabel's 'auto' (default here, unchanged for the G1 layers) switches some chunks
        # (larger / denser collision sets) to multithreaded faer (32 threads on the 9950X): 550-600 ms instead of 55-70 ms with qdldl,
        # the backend of every other chunk. The EX002 runner pins SCMG_SAFETY_KKT=qdldl.
        st_.direct_solve_method = os.environ.get('SCMG_SAFETY_KKT', 'auto')
        sol = clarabel.DefaultSolver(P, q, Ac, b, cones, st_).solve()
        status = str(sol.status)
        if not status.endswith('Solved') or 'Almost' in status and False:
            return None, status
        x = np.array(sol.x)
        if not np.all(np.isfinite(x)):
            return None, status
        return x.reshape(self.N, self.J), status


# ------------------------------------------------------------------------------------------------------- encoder
class FGDEncoder:
    """differentiable ECHO-G official FGD encoder input (pure2 = sin/cos of the 29 joints, BFS order, root zero)."""

    def __init__(self, ckpt=FGD_CKPT, device='cpu'):
        import torch
        from echo_g.evaluation import g1_motion_cls as GM
        from echo_g.evaluation.g1_kinematics import canonical_to_bfs
        self.torch = torch
        self.net, self.cfg = GM.load_g1_ae(ckpt, device, FPS)
        if self.cfg['input_mode'] != 'pure2' or self.cfg['feature_mean'] is not None:
            raise ValueError(f"encoder input {self.cfg['input_mode']} not supported")
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.bfs = torch.as_tensor(canonical_to_bfs(np.arange(J)), device=device)
        self.device = device

    def __call__(self, q):
        """q [T, 29] canonical (torch, any dtype) -> latents [T / 16, 192]"""
        torch = self.torch
        q = q.to(self.device)
        qb = q.float()[:, self.bfs]
        x = torch.cat([torch.zeros(q.shape[0], 1, 2, device=q.device),
                       torch.stack([torch.sin(qb), torch.cos(qb)], -1)], 1)
        return self.net.map2latent(x.reshape(1, q.shape[0], 60))[0]


# --------------------------------------------------------------------------------------------------------- layers
class _Base:
    """start='free' (G1): no state before the first chunk; start='rest' (EX002 validator convention): the stream starts
    at rest at q0 = nearest safe pose to the first raw frame (3 history copies of q0, never emitted)."""
    name = 'base'
    start = 'free'

    def reset(self):
        self.hist = np.zeros((0, getattr(self, 'n', J)))
        self.q_start = None
        self.stats = dict(chunks=0, frames=0, fallback_chunks=0, fallback_frames=0, times=[])

    def _begin(self, r):
        if self.start == 'rest' and self.q_start is None:
            self.q_start = static_safe(np.asarray(r, float)[0], self.lim, self.coll)
            self.hist = np.repeat(self.q_start[None], 3, 0)
            if hasattr(self, 'raw'): self.raw = self.hist.copy()

    def _record(self, out, t0, fb_frames=0, fb_chunk=False):
        self.stats['chunks'] += 1; self.stats['frames'] += len(out)
        self.stats['fallback_frames'] += int(fb_frames); self.stats['fallback_chunks'] += int(fb_chunk)
        self.stats['times'].append(time.perf_counter() - t0)
        self.hist = np.concatenate([self.hist, out], 0)


class NoFilter(_Base):
    name = 'none'

    def __init__(self, *a, n=J, **k):
        self.n = n
        self.reset()

    def step(self, r):
        t0 = time.perf_counter(); out = np.array(r, float); self._record(out, t0); return out


def static_safe(q, lim, coll, d_safe=2e-3, act=0.01, max_iter=6, pos_margin=1e-4):
    """nearest single pose to q inside the position limits and contact-free (used when there is no history yet)."""
    qp = ChunkQP(lim, np.zeros((0, lim.n)), 1, terminal=False, pos_margin=pos_margin, deriv=())
    x0 = np.clip(q, lim.lo + pos_margin, lim.hi - pos_margin)
    if not coll.collides(x0):
        return x0
    lin = x0
    for _ in range(max_iter):
        qp.add_collision(0, coll.near(lin, act), lin, d_safe)
        x, _ = qp.solve(q[None], np.ones((1, lim.n)))
        if x is None:
            break
        if pos_ok(x, lim) and not coll.collides(x[0]):
            return x[0]
        lin = x[0]
    z = np.clip(getattr(lim, 'rest_pose', np.zeros(lim.n)), lim.lo + pos_margin, lim.hi - pos_margin)
    if coll.collides(z):
        raise RuntimeError('no contact-free start pose')
    return z


class ClipLayer(_Base):
    """per-frame: position clamp -> velocity clamp -> acceleration clamp -> velocity clamp -> position clamp;
    collision_fallback: hold the previous output frame when the result is in contact."""

    def __init__(self, lim, coll=None, collision_fallback=False, pos_margin=1e-4, shrink=0.995):
        self.lim, self.coll, self.cf, self.n = lim, coll, collision_fallback, lim.n
        self.lo, self.hi = lim.lo + pos_margin, lim.hi - pos_margin
        self.vb, self.ab = lim.step_bound(1, shrink), lim.step_bound(2, shrink)
        self.name = 'clip_coll' if collision_fallback else 'clip'
        self.reset()

    def step(self, r):
        self._begin(r)
        t0 = time.perf_counter(); out = []; fb = 0
        for x in np.asarray(r, float):
            H = self.hist if not out else np.concatenate([self.hist, np.array(out)], 0)
            q = np.clip(x, self.lo, self.hi)
            if len(H) >= 1:
                q = H[-1] + np.clip(q - H[-1], -self.vb, self.vb)
            if len(H) >= 2:
                v1 = H[-1] - H[-2]
                q = H[-1] + v1 + np.clip(q - H[-1] - v1, -self.ab, self.ab)
                q = H[-1] + np.clip(q - H[-1], -self.vb, self.vb)
            q = np.clip(q, self.lo, self.hi)
            if self.cf and (self.coll.collides(q) or (self.coll.midpoints and len(H) and self.coll.collides(0.5 * (q + H[-1])))):
                q = H[-1].copy() if len(H) else static_safe(q, self.lim, self.coll)
                fb += 1
            out.append(q)
        out = np.array(out); self._record(out, t0, fb, fb > 0); return out


class CBFQPLayer(_Base):
    """per-frame discrete-time CBF-QP, no horizon."""
    name = 'cbf_qp'

    def __init__(self, lim, coll, gamma=0.3, gamma_c=0.3, d_safe=2e-3, act=0.01, max_iter=4, pos_margin=1e-4,
                 shrink=0.995):
        self.lim, self.coll, self.n = lim, coll, lim.n
        self.g, self.gc, self.ds, self.act, self.it = gamma, gamma_c, d_safe, act, max_iter
        self.lo, self.hi = lim.lo + pos_margin, lim.hi - pos_margin
        self.b = {k: lim.step_bound(k, shrink) for k in (1, 2, 3)}
        self.reset()

    def reset(self):
        super().reset(); self.stats.update(relax_acc=0, relax_jerk=0, qp=0, qp_fail=0)

    def _box(self, H):
        lo, hi = self.lo.copy(), self.hi.copy()
        if len(H) >= 1:
            p = H[-1]
            lo = np.maximum(lo, self.lo + (1 - self.g) * np.maximum(p - self.lo, 0))
            hi = np.minimum(hi, self.hi - (1 - self.g) * np.maximum(self.hi - p, 0))
            lo = np.maximum(lo, p - self.b[1]); hi = np.minimum(hi, p + self.b[1])
        hard = (lo.copy(), hi.copy())
        for k in (2, 3):
            if len(H) < k:
                continue
            c = -sum(COEF[k][i] * H[-i] for i in range(1, k + 1))       # q_t + sum c_i q_{t-i} in [-b, b]
            l2, h2 = np.maximum(lo, c - self.b[k]), np.minimum(hi, c + self.b[k])
            bad = l2 > h2
            if bad.any():
                self.stats['relax_acc' if k == 2 else 'relax_jerk'] += int(bad.sum())
            lo, hi = np.where(bad, lo, l2), np.where(bad, hi, h2)
        return lo, hi, hard

    def _frame(self, x, H):
        lo, hi, hard = self._box(H)
        if np.any(lo > hi + 1e-12):            # cannot happen for hard part when H[-1] is in limits
            lo, hi = hard
        q = np.clip(x, lo, hi)
        if not self.coll.collides(q) and not (self.coll.midpoints and len(H) and self.coll.collides(0.5 * (q + H[-1]))):
            return q, False
        # collision rows: discrete CBF  d_t - ds >= (1 - gc) (d_{t-1} - ds), linearised, re-linearised at the solution
        qp = ChunkQP(self.lim, np.zeros((0, self.n)), 1, terminal=False, deriv=())
        qp.l0, qp.u0 = lo.copy(), hi.copy()
        lin = q
        for _ in range(self.it):
            rows, lin = lin_rows(self.coll, lin, H[-1] if len(H) else None, self.act)
            for g, dist in rows:
                dprev = dist + g @ (H[-1] - lin) if len(H) else dist
                rhs = max(self.ds + (1 - self.gc) * (dprev - self.ds), 0.25 * self.ds)
                qp.crow.append(np.array(g)); qp.cl.append(rhs - dist + g @ lin)
            self.stats['qp'] += 1
            y, _ = qp.solve(x[None], np.ones((1, self.n)))
            if y is None:
                self.stats['qp_fail'] += 1; break
            y = y[0]
            if np.all(y >= hard[0] - 1e-9) and np.all(y <= hard[1] + 1e-9) and not self.coll.collides(y) and \
                    not (self.coll.midpoints and len(H) and self.coll.collides(0.5 * (y + H[-1]))):
                return y, False
            lin = y
        return None, True

    def step(self, r):
        self._begin(r)
        t0 = time.perf_counter(); out = []; fb = 0
        for x in np.asarray(r, float):
            H = self.hist if not out else np.concatenate([self.hist, np.array(out)], 0)
            q, failed = self._frame(x, H[-3:])
            if failed:
                q = H[-1].copy() if len(H) else static_safe(x, self.lim, self.coll); fb += 1
            out.append(q)
        out = np.array(out); self._record(out, t0, fb, fb > 0); return out


class RuckigLayer(_Base):
    name = 'ruckig'

    def __init__(self, lim, coll=None, pos_margin=1e-4, shrink=0.995, start='free'):
        import ruckig
        self.rk, self.lim, self.coll, self.n, self.start = ruckig, lim, coll, lim.n, start
        self.lo, self.hi = lim.lo + pos_margin, lim.hi - pos_margin
        self.otg = ruckig.Ruckig(lim.n, lim.dt)
        self.inp = ruckig.InputParameter(lim.n); self.outp = ruckig.OutputParameter(lim.n)
        self.inp.max_velocity = lim.vmax * shrink
        self.inp.max_acceleration = lim.amax * shrink
        self.inp.max_jerk = lim.jmax * shrink
        self.inp.synchronization = ruckig.Synchronization.No
        self.reset()

    def reset(self):
        super().reset(); self.stats.update(errors=0); self.prev_raw = None; self.started = False

    def step(self, r):
        self._begin(r)
        t0 = time.perf_counter(); r = np.asarray(r, float); out = []
        lim = self.lim; J, DT = self.n, lim.dt
        if not self.started and self.start == 'rest':
            self.inp.current_position = self.q_start; self.inp.current_velocity = np.zeros(J)
            self.inp.current_acceleration = np.zeros(J); self.started = True; self.prev_raw = self.q_start
        for i, x in enumerate(r):
            tgt = np.clip(x, self.lo, self.hi)
            if not self.started:
                # free start: state estimated from the current chunk only (first frames of the stream)
                v0 = (r[1] - r[0]) / DT if len(r) > 1 else np.zeros(J)
                a0 = (r[2] - 2 * r[1] + r[0]) / DT ** 2 if len(r) > 2 else np.zeros(J)
                self.inp.current_position = tgt
                self.inp.current_velocity = np.clip(v0, -lim.vmax * .99, lim.vmax * .99)
                self.inp.current_acceleration = np.clip(a0, -lim.amax * .99, lim.amax * .99)
                self.started = True; self.prev_raw = x; out.append(tgt); continue
            tv = np.clip((x - self.prev_raw) / DT, -0.8 * lim.vmax, 0.8 * lim.vmax)
            tv[(tgt <= self.lo) | (tgt >= self.hi)] = 0.0
            self.prev_raw = x
            self.inp.target_position = tgt; self.inp.target_velocity = tv; self.inp.target_acceleration = np.zeros(J)
            res = self.otg.update(self.inp, self.outp)
            if res not in (self.rk.Result.Working, self.rk.Result.Finished):
                self.stats['errors'] += 1
                self.inp.target_velocity = np.zeros(J)
                res = self.otg.update(self.inp, self.outp)
            if res in (self.rk.Result.Working, self.rk.Result.Finished):
                q = np.array(self.outp.new_position)
                self.outp.pass_to_input(self.inp)
            else:
                q = np.array(self.inp.current_position)
            q = np.clip(q, self.lo, self.hi)
            out.append(q)
        out = np.array(out); self._record(out, t0); return out


class ProjectionLayer(_Base):
    def __init__(self, lim, coll, objective='l2', K=10, B=10, w=None, d_safe=2e-3, act=0.01, max_iter=4,
                 pos_margin=1e-5, shrink=0.995, encoder=None, window=64, steps=15, lr=None, lam=1.0, mu=1e3, smooth=0.0,
                 tail_smooth=1e-6):
        assert objective in ('l2', 'perceptual')
        self.lim, self.coll, self.obj, self.n = lim, coll, objective, lim.n
        J = lim.n
        self.K, self.B, self.ds, self.act, self.it = K, B, d_safe, act, max_iter
        self.pm, self.shrink = pos_margin, shrink
        self.w = np.ones(J) if w is None else np.asarray(w, float)
        self.enc, self.window, self.steps, self.lr, self.lam, self.mu = encoder, window, steps, lr, lam, mu
        self.smooth = smooth
        self.tail_smooth = tail_smooth            # 11-08: weight of the braking-tail velocity term (1e-6 = the original; EX002 1e-3)
        if objective == 'perceptual' and encoder is None:
            raise ValueError('perceptual objective needs an FGDEncoder')
        self.name = 'proj_l2' if objective == 'l2' else ('proj_perc_s' if smooth else 'proj_perc')
        import clarabel, osqp                    # 11-08: import the solvers at construction, not inside the first chunk's budget
        self.reset()

    def reset(self):
        super().reset()
        self.raw = np.zeros((0, self.n)); self.plan = None
        self.stats.update(qp_solves=0, coll_iters=0, qp_fail=0, verify_fail=0, perc_reject=0, perc_steps=0,
                          feat_l2=[], feat_perc=[])

    # one projection: chunk targets (k rows) + B free tail rows, verify + re-linearise
    def _project(self, H, target_k, lin_k, ref, qp=None):
        """ref [N, J]: contact-free reference poses per frame (the previous verified plan, then hold) for the
        boundary-point linearisation."""
        k = len(target_k); N = k + self.B
        if qp is None:
            qp = ChunkQP(self.lim, H, N, terminal=True, pos_margin=self.pm, shrink=self.shrink)
            lin = np.concatenate([lin_k, np.repeat(lin_k[-1:], self.B, 0)], 0)
            memo = {}                  # 11-08: identical (pose, reference) pairs (the held braking tail) are linearised once;
            for t in range(N):         # lin_rows is a deterministic function of its inputs, so the rows are bit-identical
                key = (lin[t].tobytes(), ref[t].tobytes())
                if key not in memo: memo[key] = lin_rows(self.coll, lin[t], ref[t], self.act)
                rows, ql = memo[key]
                if rows: qp.add_collision(t, rows, ql, self.ds)
        tgt = np.full((N, self.n), np.nan); tgt[:k] = target_k
        W = np.tile(self.w, (N, 1))
        for it in range(self.it):
            x, st = qp.solve(tgt, W, tail_smooth=self.tail_smooth)
            self.stats['qp_solves'] += 1
            if x is None:
                self.stats['qp_fail'] += 1; return None, qp
            Xh = np.concatenate([x, np.repeat(x[-1:], 3, 0)], 0)     # braking tail followed by a hold
            if not (pos_ok(x, self.lim) and diff_ok(H, Xh, self.lim)[0]):
                self.stats['verify_fail'] += 1; return None, qp
            bad = [t for t in range(N) if self.coll.collides(x[t])]
            badm = []
            if self.coll.midpoints and not bad:
                prev = np.concatenate([H[-1:], x[:-1]], 0) if len(H) else x[:-1]
                cur = x if len(H) else x[1:]
                off = 0 if len(H) else 1
                badm = [t + off for t in range(len(cur)) if self.coll.collides(0.5 * (cur[t] + prev[t]))]
            if not bad and not badm:
                return x, qp
            self.stats['coll_iters'] += 1
            for t in bad:
                rows, ql = lin_rows(self.coll, x[t], ref[t], self.act)
                if rows: qp.add_collision(t, rows, ql, self.ds)
            for t in badm:
                pm = 0.5 * (x[t] + (x[t - 1] if t >= 1 else H[-1]))
                rm = 0.5 * (ref[t] + (ref[t - 1] if t >= 1 else H[-1]))
                rows, ql = lin_rows(self.coll, pm, rm, self.act)
                if rows: qp.add_collision_mid(t, rows, ql, self.ds)
        return None, qp

    def _perceptual(self, H_all, raw_all, r, x_l2, qp):
        torch = self.torch
        k = len(r)
        Wn = min(self.window, len(H_all) + k)
        Wn = (Wn // 16) * 16
        if Wn < 32:
            return None
        past = Wn - k
        if np.abs(H_all[len(H_all) - past:] - raw_all[len(raw_all) - past:]).max(initial=0) < 1e-5 and \
                np.abs(x_l2[:k] - r).max() < 1e-5:
            return None                          # the projection changed nothing in this window: keep it exact
        hp = torch.as_tensor(H_all[len(H_all) - past:], dtype=torch.float32)
        rp = torch.as_tensor(np.concatenate([raw_all[len(raw_all) - past:], r], 0), dtype=torch.float32)
        with torch.no_grad():
            fref = self.enc(rp); den = (fref ** 2).mean() + 1e-8
        A, l, u = qp.matrices()
        A = A.tocsr()
        At = torch.sparse_csr_tensor(torch.as_tensor(A.indptr, dtype=torch.int64), torch.as_tensor(A.indices, dtype=torch.int64),
                                     torch.as_tensor(A.data, dtype=torch.float64), size=A.shape)
        lt = torch.as_tensor(np.maximum(l, -1e6)); ut = torch.as_tensor(np.minimum(u, 1e6))
        tail = torch.as_tensor(x_l2[k:].reshape(-1), dtype=torch.float64)
        rt = torch.as_tensor(r, dtype=torch.float64); wt = torch.as_tensor(self.w, dtype=torch.float64)
        z = torch.tensor(x_l2[:k], dtype=torch.float64, requires_grad=True)
        z0 = torch.as_tensor(x_l2[:k], dtype=torch.float64)
        jn = torch.as_tensor(self.lim.step_bound(3), dtype=torch.float64)

        def loss_terms(z):
            f = self.enc(torch.cat([hp, z.float()], 0))
            lf = (((f - fref) ** 2).mean() / den).double().cpu()
            l2 = (wt * (z - rt) ** 2).mean()
            if self.smooth:                      # jerk of the refinement z - x_l2 (zero before the chunk), / jerk bound
                dz = torch.cat([torch.zeros(3, z.shape[1], dtype=z.dtype), z - z0], 0)
                d3 = dz[3:] - 3 * dz[2:-1] + 3 * dz[1:-2] - dz[:-3]
                l2 = l2 + self.smooth * ((d3 / jn) ** 2).mean()
            Ax = At @ torch.cat([z.reshape(-1), tail])
            pen = (torch.relu(Ax - ut) ** 2).sum() + (torch.relu(lt - Ax) ** 2).sum()
            return lf, l2, pen
        with torch.no_grad():
            lf0 = float(loss_terms(z)[0])
        if lf0 < 1e-12:                          # executed window already equals the raw window: nothing to gain
            return None
        opt = torch.optim.LBFGS([z], lr=1.0, max_iter=self.steps, line_search_fn='strong_wolfe',
                                tolerance_grad=1e-10, tolerance_change=1e-12, history_size=10)

        def closure():
            opt.zero_grad()
            lf, l2, pen = loss_terms(z)
            loss = lf + self.lam * l2 + self.mu * pen
            loss.backward()
            return loss
        opt.step(closure)
        self.stats['perc_steps'] += self.steps
        with torch.no_grad():
            lf1 = float(loss_terms(z)[0])
        self.stats['feat_l2'].append(lf0); self.stats['feat_perc'].append(lf1)
        return z.detach().numpy()

    def step(self, r):
        self._begin(r)
        t0 = time.perf_counter()
        r = np.asarray(r, float); k = len(r)
        H = self.hist[-3:]
        lin_k = np.clip(r, self.lim.lo + self.pm, self.lim.hi - self.pm)
        N = k + self.B
        if self.plan is not None:
            ref = self.plan
        elif len(self.hist):
            ref = self.hist[-1:]
        else:
            ref = static_safe(r[0], self.lim, self.coll, self.ds, self.act)[None]
        ref = np.concatenate([ref, np.repeat(ref[-1:], max(N - len(ref), 0), 0)], 0)[:N]
        x, qp = self._project(H, r, lin_k, ref)
        if x is not None and self.obj == 'perceptual':
            if not hasattr(self, 'torch'):
                import torch; self.torch = torch
            z = self._perceptual(self.hist, self.raw, r, x, qp)
            if z is not None:
                x2, _ = self._project(H, z, z, ref, qp=qp)   # same constraint set (cutting planes kept), L2 to z
                if x2 is not None:
                    x = x2
                else:
                    self.stats['perc_reject'] += 1
        fb = x is None
        if not fb:
            out, self.plan = x[:k], x[k:]
        else:
            if self.plan is None:                         # no verified plan yet (first chunk): brake-free hold
                q0 = static_safe(r[0], self.lim, self.coll, self.ds, self.act) if not len(self.hist) else self.hist[-1]
                plan = np.repeat(q0[None], k, 0)
            else:
                plan = self.plan
            if len(plan) < k + self.B:
                plan = np.concatenate([plan, np.repeat(plan[-1:], k + self.B - len(plan), 0)], 0)
            out, self.plan = plan[:k], plan[k:]
        self.raw = np.concatenate([self.raw, r], 0)
        self._record(out, t0, k if fb else 0, fb)
        return out


def make_layer(name, lim=None, coll=None, encoder=None, K=10, **kw):
    lim = lim or Limits()
    if name == 'none': return NoFilter()
    coll = coll or G1Collision()
    if name == 'clip': return ClipLayer(lim, coll, collision_fallback=False)
    if name == 'clip_coll': return ClipLayer(lim, coll, collision_fallback=True)
    if name == 'cbf_qp': return CBFQPLayer(lim, coll)
    if name == 'ruckig': return RuckigLayer(lim, coll)
    if name == 'proj_l2': return ProjectionLayer(lim, coll, 'l2', K=K, **kw)
    if name == 'proj_perc': return ProjectionLayer(lim, coll, 'perceptual', K=K, encoder=encoder or FGDEncoder(), **kw)
    if name == 'proj_perc_s':
        kw = dict(kw); kw.setdefault('smooth', 1.0)
        return ProjectionLayer(lim, coll, 'perceptual', K=K, encoder=encoder or FGDEncoder(), **kw)
    raise ValueError(name)


LAYERS = ['none', 'clip', 'clip_coll', 'cbf_qp', 'ruckig', 'proj_l2', 'proj_perc']


def run_stream(layer, q, K=10):
    """q [T, 29] raw joints -> safe [T, 29], stats (chunks of K frames, last one shorter)."""
    layer.reset()
    out = [layer.step(q[s:s + K]) for s in range(0, len(q), K)]
    return np.concatenate(out, 0), layer.stats

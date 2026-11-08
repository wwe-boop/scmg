"""11-08 EX002 three-finger (PVT1 + RM003 v4, 34 controlled joints, 50 Hz) adapter for the chunk-causal safety layers
(docs/criteria/EX002_CBF_1108.md on exp/duplex). The gripper adapter (scmg/safety_proj_ex002.py) is reused unchanged for the collision
linearisation: EX002Collision works on any vendored-Embodiment subclass, and the release06 retargeter's EX002RM is one (RM003 model +
runtime-rm003 349 WBC rows, mimic 1_4 = 1.389831 * 1_3 + 0.072772 folded into the 1_3 column).
  embodiment  smplx2ex002_3f.robot.EX002().E -- the same instance type the interruption metric's WBC clearance uses
  position    EX002RM hard_lo / hard_hi
  v / a / j   smplx2ex002_3f.robot.Limits (D2.2 arms, D1.4 fingers, conservative waist / head)"""
from __future__ import annotations
import os, sys
import numpy as np

from . import safety_proj as S
from .safety_proj_ex002 import EX002Collision

RETARGETER = os.environ.get('SMPLX2EX002_3F', '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture-dlc-20260928/smplx2ex002-r2-20261022/candidates/codex-dlc-release06-beat-nearoff')
DT = 1.0 / 50
K_3F = 16


def _rb():
    os.environ.setdefault('MUJOCO_GL', 'disable')
    if RETARGETER not in sys.path: sys.path.insert(0, RETARGETER)
    from smplx2ex002_3f import robot as RB
    return RB


class EX002RMLimits:
    def __init__(self, X=None):
        RB = _rb(); self.X = X or RB.EX002(); self.E = self.X.E
        self.lo, self.hi = self.E.hard_lo.copy(), self.E.hard_hi.copy()
        L = RB.Limits(); self.vmax, self.amax, self.jmax = L.v.copy(), L.a.copy(), L.j.copy()
        self.rate = {1: self.vmax, 2: self.amax, 3: self.jmax}
        self.n, self.dt = len(self.lo), DT
        self.rest_pose = np.clip(self.E.anchor, self.lo, self.hi)
        self.source = dict(embodiment='smplx2ex002_3f EX002RM (release06)', runtime='runtime-rm003', table='robot.Limits (D2.2 / D1.4)')

    def step_bound(self, k, shrink=1.0):
        return self.rate[k] * self.dt ** k * shrink


def make_cbf_3f(lim=None):
    lim = lim or EX002RMLimits(); coll = EX002Collision(lim.E)
    L = S.CBFQPLayer(lim, coll, d_safe=1e-3); L.start = 'rest'; L.reset()
    return L, lim, coll


class SlackCBFLayer(S.CBFQPLayer):
    """11-08 module 3c-2 (exp/duplex docs/criteria/EX002_CBF_1108.md): the per-frame discrete CBF-QP of CBFQPLayer with an L1 slack
    s_i >= 0 on every collision row (cost rho * s_i) -> always feasible. Position / velocity bounds hard (the CBFQPLayer box; acc / jerk
    rows relaxed per joint when they conflict, as there). No hold fallback: after max_iter re-linearisations the last iterate is used and
    counted in stats['residual'] if the exact WBC check (frames + midpoints) still fails."""
    name = 'cbf_slack'

    def __init__(self, lim, coll, rho=1e4, **kw):
        super().__init__(lim, coll, **kw); self.rho = rho

    def reset(self):
        super().reset(); self.stats.update(residual=0, slack_used=0)

    def _solve(self, x, lo, hi, rows):
        import osqp, scipy.sparse as sp
        n = self.n; m = len(rows)
        P = sp.block_diag([sp.eye(n) * 2.0, sp.csc_matrix((m, m))], format='csc') if m else sp.eye(n, format='csc') * 2.0
        q = np.concatenate([-2.0 * x, np.full(m, self.rho)])
        if m:
            G = np.array([g for g, _ in rows]); r = np.array([c for _, c in rows])
            A = sp.vstack([sp.hstack([sp.eye(n), sp.csc_matrix((n, m))]), sp.hstack([sp.csc_matrix(G), sp.eye(m)]),
                           sp.hstack([sp.csc_matrix((m, n)), sp.eye(m)])], format='csc')
            l = np.concatenate([lo, r, np.zeros(m)]); u = np.concatenate([hi, np.full(m, np.inf), np.full(m, np.inf)])
        else:
            A = sp.eye(n, format='csc'); l, u = lo, hi
        sv = osqp.OSQP(); sv.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=20000, polishing=True)
        res = sv.solve(); z = np.asarray(res.x) if res.x is not None else None
        if z is None or not np.all(np.isfinite(z)): return np.clip(x, lo, hi), 0.
        return np.clip(z[:n], lo, hi), (float(z[n:].max()) if m else 0.)

    def _frame(self, x, H):
        lo, hi, hard = self._box(H)
        if np.any(lo > hi + 1e-12): lo, hi = hard
        q = np.clip(x, lo, hi)
        mid = lambda y: self.coll.midpoints and len(H) and self.coll.collides(0.5 * (y + H[-1]))
        if not self.coll.collides(q) and not mid(q): return q, False
        rows = []; lin = q; y = q
        for _ in range(self.it):
            rr, lin = S.lin_rows(self.coll, lin, H[-1] if len(H) else None, self.act)
            for g, dist in rr:
                dprev = dist + g @ (H[-1] - lin) if len(H) else dist
                rhs = max(self.ds + (1 - self.gc) * (dprev - self.ds), 0.25 * self.ds)
                rows.append((np.array(g), rhs - dist + g @ lin))
            self.stats['qp'] += 1
            y, smax = self._solve(x, lo, hi, rows)
            if smax > 1e-9: self.stats['slack_used'] += 1
            if not self.coll.collides(y) and not mid(y): return y, False
            lin = y
        self.stats['residual'] += 1
        return y, False


def make_slack_3f(lim=None, rho=1e4):
    lim = lim or EX002RMLimits(); coll = EX002Collision(lim.E)
    L = SlackCBFLayer(lim, coll, rho=rho, d_safe=1e-3); L.start = 'rest'; L.reset()
    return L, lim, coll

"""EX002 (PVT1, RM001-G gripper) adapter for the C3 safety layers in scmg/safety_proj.py (design v6 C3, 11-05).

Same chunk interface and the same layer classes as G1; only the embodiment changes:
  joints     the 16 controlled joints of edge_live_client.ex002.ex002_embodiment.Embodiment('gripper')
             (left/right arm joint1-7, left/right gripper), 50 Hz (DT = 0.02 s)
  position   Embodiment.hard_lo / hard_hi (installed WBC runtime bounds: 0.95 scaling + asymmetric joint1/3/4; gripper
             0.95 about centre)
  v / a / j  ex002_safety.Limits(scale=(1, 1, 1)) = the D2.2 table the independent validator judges against
             (shoulder 2.0 rad/s / 12 rad/s^2 / 300 rad/s^3, elbow 2.5 / 16 / 400, wrist 2.51 / 20 / 500, gripper 4 / 40 / 1000)
  collision  the 133 WBC pairs (runtime-gripper.json, iDist/sDist) on assets/ex002_collision.xml (convex hulls):
             exact test = Embodiment.min_clearance(q)[0] (sound GJK predicate: every pair separated by >= sDist), at
             every frame AND every midpoint of consecutive frames (as validate_sequence does).
             Linearisation: mj_geomDistance witness points (fromto) of the near pairs, n . (J_b(p_b) - J_a(p_a)) with
             the mimic finger joints folded into their source columns; row = (dist - sDist) + grad . dq >= d_safe.
  start      'rest': the stream starts at rest at the nearest safe pose to the first raw frame (validate_sequence with
             q_prev = that pose).
Independent check: ex002_safety.validate_sequence(E, qs, Limits(scale=1), q_prev) (frames + midpoints, mimic, v/a/j).
The reference deployed filter ex002_safety.SafetyGovernor (50 Hz jerk-limited tracker + certified brake plan + reach
certificate) is wrapped as layer 'governor'. The perceptual objective needs an EX002 motion encoder (the EX002 FGD v2
scorer lives on zehan-server) and is not wired here.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from . import safety_proj as S

EDGE = os.environ.get('EDGE_LIVE_CLIENT', '/home/xr-25b7/projects/gesture/edge-live/robot_client')
if EDGE not in sys.path:
    sys.path.insert(0, EDGE)
from edge_live_client.ex002.ex002_embodiment import Embodiment          # noqa: E402
from edge_live_client.ex002 import ex002_safety as ES                    # noqa: E402

K_EX, B_EX = 16, 25          # 16 frames = 320 ms chunk at 50 Hz; braking tail 0.5 s (stop from v_max within a/j limits)


class EX002Limits:
    def __init__(self, E=None, kind='gripper'):
        self.E = E or Embodiment(kind)
        self.L = ES.Limits(n=len(self.E.hard_lo), kind=kind, scale=(1., 1., 1.))
        self.lo, self.hi = self.E.hard_lo.copy(), self.E.hard_hi.copy()
        self.vmax, self.amax, self.jmax = self.L.v.copy(), self.L.a.copy(), self.L.j.copy()
        self.rate = {1: self.vmax, 2: self.amax, 3: self.jmax}
        self.n, self.dt = len(self.lo), ES.DT
        self.rest_pose = np.clip(self.E.anchor, self.lo, self.hi)
        self.source = dict(embodiment=str(self.E.spec['model']), runtime=str(self.E.spec['runtime']), table='ex002_safety.Limits D2.2')

    def step_bound(self, k, shrink=1.0):
        return self.rate[k] * self.dt ** k * shrink


class EX002Collision:
    midpoints = True

    def __init__(self, E):
        import mujoco
        self.mj, self.E = mujoco, E
        m = E.m
        n = len(E.controlled)
        self.Mmap = np.zeros((m.nv, n))                       # d qdof / d c  (controlled + mimic joints)
        for k, name in enumerate(E.controlled):
            self.Mmap[m.jnt_dofadr[m.joint(name).id], k] = 1.0
        for x in E.mimic:
            if x['source'] in E.controlled:
                self.Mmap[m.jnt_dofadr[m.joint(x['joint']).id], E.controlled.index(x['source'])] += x['multiplier']
        self.jp1 = np.zeros((3, m.nv)); self.jp2 = np.zeros((3, m.nv)); self.ft = np.zeros(6)
        self.n_exact = 0; self.n_lin = 0

    def collides(self, q):
        self.n_exact += 1
        return not self.E.min_clearance(np.asarray(q, float))[0]

    def near(self, q, act, carry=None, carry_act=None, ids=None):
        """[(grad [16], dist - sDist)] for every WBC pair with dist - sDist < act.
        11-08 (EX002 3c-5, default off): pairs in `carry` (pair indices) are kept up to dist - sDist < carry_act; `ids` (a list)
        receives the pair index of every returned row."""
        E, mj = self.E, self.mj
        m, d = E.m, E.d
        E.pose(np.asarray(q, float)); mj.mj_comPos(m, d)
        self.n_lin += 1
        lb = E._lb_centres()
        rows = []
        big = max(act, carry_act) if carry else act
        for i in np.nonzero(lb < E._mg + big)[0]:
            a, b, sd = int(E._a[i]), int(E._b[i]), float(E._mg[i])
            lim_i = carry_act if (carry and int(i) in carry) else act
            if lb[i] >= E._mg[i] + lim_i:
                continue
            dist = mj.mj_geomDistance(m, d, a, b, sd + lim_i + 0.01, self.ft)
            if dist >= sd + lim_i:
                continue
            p1, p2 = self.ft[:3].copy(), self.ft[3:].copy()
            v = p2 - p1; nv = np.linalg.norm(v)
            if nv < 1e-9:
                continue
            nrm = v / nv * (1.0 if dist >= 0 else -1.0)
            mj.mj_jac(m, d, self.jp1, None, p1, int(m.geom_bodyid[a]))
            mj.mj_jac(m, d, self.jp2, None, p2, int(m.geom_bodyid[b]))
            g = nrm @ (self.jp2 - self.jp1) @ self.Mmap
            rows.append((g, float(dist - sd)))
            if ids is not None: ids.append(int(i))
        return rows


class GovernorLayer(S._Base):
    """the deployed EX002 SafetyGovernor, one step per frame (its own DESIGN_SCALE margins and brake planner)."""
    name = 'governor'
    start = 'rest'

    def __init__(self, lim, coll):
        self.lim, self.coll, self.n = lim, coll, lim.n
        self.reset()

    def reset(self):
        super().reset(); self.gov = None

    def step(self, r):
        self._begin(r)
        if self.gov is None:
            self.gov = ES.SafetyGovernor(self.lim.E, q0=self.q_start)
        t0 = time.perf_counter(); out = []; fb = 0
        for x in np.asarray(r, float):
            q, info = self.gov.step(x)
            fb += info.get('status') in ('brake', 'hold')
            out.append(q)
        out = np.array(out); self._record(out, t0, fb, fb > 0)
        self.stats['governor'] = {k: v for k, v in self.gov.stats.items()}
        return out


LAYERS_EX = ['none', 'clip', 'clip_coll', 'cbf_qp', 'ruckig', 'proj_l2', 'governor']


def make_layer_ex002(name, lim=None, coll=None, K=K_EX, B=B_EX, **kw):
    lim = lim or EX002Limits()
    coll = coll or EX002Collision(lim.E)
    if name == 'none':
        L = S.NoFilter(n=lim.n)
    elif name == 'clip':
        L = S.ClipLayer(lim, coll, collision_fallback=False)
    elif name == 'clip_coll':
        L = S.ClipLayer(lim, coll, collision_fallback=True)
    elif name == 'cbf_qp':
        L = S.CBFQPLayer(lim, coll, d_safe=1e-3)
    elif name == 'ruckig':
        L = S.RuckigLayer(lim, coll, start='rest')
    elif name == 'proj_l2':
        L = S.ProjectionLayer(lim, coll, 'l2', K=K, B=B, d_safe=1e-3, **kw)
    elif name == 'governor':
        return GovernorLayer(lim, coll)
    else:
        raise ValueError(name)
    L.start = 'rest'
    L.reset()
    return L


def validate(lim, qs, q_prev, midpoints=True):
    """independent EX002 validator (ex002_safety.validate_sequence against the D2.2 table)."""
    return ES.validate_sequence(lim.E, np.asarray(qs, float), lim.L, q_prev=np.asarray(q_prev, float),
                                check_midpoints=midpoints)

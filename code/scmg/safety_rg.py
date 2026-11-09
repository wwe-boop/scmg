"""RoboGesture's collision pairs as constraints of the G1 chunk-causal safety layers (11-07; default off).

RGCollision exposes the collision interface the layers of scmg/safety_proj.py use (collides / near / midpoints) on
RoboGesture's own geometry and distance function, unmodified: robogesture.collision.CollisionFilter (G1 + BrainCo-hand
MJCF, its 55 geom pairs), _to_control (hand command -> rad, clip to its control limits), _distance (mj_geomDistance,
cap 0.10 m) and _jacobian (witness-point Jacobian, finite differences at touching pairs). The constraint is PENETRATION
(pair distance >= d_safe of the layer, 2 mm), not RoboGesture's 2-3 cm filter margins. Joint mapping by name (as
scripts/baselines_g1/rg_to_g1.py): legs -> the checker's fixed legs (no gradient), waist / arms -> its controls; the 12
BrainCo hand commands are fixed to HAND_CMD = RoboGesture's median command over its filtered generations
(baseline-repro-1104/robogesture/gen/filtered_seed0, first 60 clips, every 3rd frame; = rg_collision_check.hand_median).
UnionCollision(a, b): a frame collides if either model reports a contact; rows of both are linearised.
  safety_run.py --rg-pairs        (default off: the layers keep G1Collision only, bitwise as before)"""
import sys
import numpy as np

RG_OFFICIAL = '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/robogesture/official'
HAND_CMD = (376.0, 2.0, 6.0, 124.0, 192.0, 215.0, 376.0, 3.0, 112.0, 233.0, 253.0, 278.0)


class RGCollision:
    midpoints = False

    def __init__(self, names, hand_cmd=HAND_CMD, fast=False, cap_exact=1e-3):
        if RG_OFFICIAL not in sys.path: sys.path.insert(0, RG_OFFICIAL)
        import mujoco
        from robogesture import collision as C
        from robogesture.config import LEFT_HAND, RIGHT_HAND
        self.C = C; self.f = C.CollisionFilter(); m = self.f.model
        self.hands = list(LEFT_HAND + RIGHT_HAND); self.hand_cmd = np.asarray(hand_cmd, np.float64)
        jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
        first = [t[0][0] for t in C.CONTROL_TARGETS]
        self.leg, self.ctl = [], []                      # (index in names, checker leg qpos / control index)
        for k, nm in enumerate(names):
            mi = jn.index(nm + '_joint')
            if mi < C.LEG_DOF: self.leg.append((k, mi))
            else: self.ctl.append((k, first.index(mi)))
        assert len(self.leg) == 12 and len(self.ctl) == len(names) - 12
        self.n = len(names); self.n_exact = 0; self.n_lin = 0
        self.mj = mujoco; self.fast = fast; self.cap_exact = cap_exact; self._ft = np.empty(6)   # fast: capped distances (11-08)
        self._ga = np.array([a for a, _ in self.f.geom_pairs]); self._gb = np.array([b for _, b in self.f.geom_pairs])
        self._rb = m.geom_rbound[self._ga] + m.geom_rbound[self._gb]

    def _set(self, q):
        fr = np.zeros(30); fr[self.hands] = self.hand_cmd
        for k, i in self.ctl: fr[i] = q[k]
        c, _ = self.f._to_control(fr)
        legs = np.zeros(self.C.LEG_DOF)
        for k, mi in self.leg: legs[mi] = q[k]
        self.f.legs = legs
        return c

    def distances(self, q):
        return self.f._distance(self._set(np.asarray(q, np.float64)))

    def _capped(self, q, cap, com=False, first=False):
        """pair distances with mj_geomDistance capped at `cap` (same function and pairs as _distance; values below the cap are
        identical). Broad phase: a pair whose bounding spheres (geom_xpos, geom_rbound) are more than `cap` apart has a true
        distance > cap and gets the cap without a call. first=True: stop at the first penetrating pair (collides())."""
        mj = self.mj; c = self._set(np.asarray(q, np.float64)); self.f._kinematics(c, include_com=com)
        m, d = self.f.model, self.f.data; ft = self._ft
        x = d.geom_xpos; gap = np.linalg.norm(x[self._ga] - x[self._gb], axis=1) - self._rb
        out = np.full(len(self._ga), float(cap))
        for p in np.nonzero(gap <= cap)[0]:
            out[p] = mj.mj_geomDistance(m, d, int(self._ga[p]), int(self._gb[p]), cap, ft)
            if first and out[p] < 0: break
        return c, out

    def contacts(self, q):
        self.n_exact += 1
        if not self.fast: return {('rg', int(i)) for i in np.nonzero(self.distances(q) < 0)[0]}
        _, d = self._capped(q, self.cap_exact)
        return {('rg', int(i)) for i in np.nonzero(d < 0)[0]}

    def collides(self, q):
        if not self.fast: return bool(self.contacts(q))
        self.n_exact += 1
        return bool((self._capped(q, self.cap_exact, first=True)[1] < 0).any())

    def near(self, q, act):
        """(grad [n], dist) for every RoboGesture pair with distance < act; d(q + dq) ~= dist + grad . dq (legs: 0)."""
        self.n_lin += 1
        if not self.fast:
            J, d = self.f._jacobian(self._set(np.asarray(q, np.float64)))
            sel = np.nonzero(d < act)[0]
        else:                         # = _jacobian restricted to the pairs within act (its arithmetic, pair by pair)
            mj, C = self.mj, self.C; m, dd = self.f.model, self.f.data
            c, d = self._capped(q, min(act, C.DETECTION_DISTANCE), com=True)
            sel = np.nonzero(d < act)[0]; J = np.zeros((len(d), C.CONTROL_DOF)); ft = np.empty(6)
            ja, jb = np.empty((3, C.MODEL_DOF)), np.empty((3, C.MODEL_DOF))
            for p in sel:
                ga, gb = self.f.geom_pairs[p]
                dist = mj.mj_geomDistance(m, dd, ga, gb, C.DETECTION_DISTANCE, ft); d[p] = dist
                if abs(dist) <= 1e-10:
                    J[p] = self.f._finite_difference(c, ga, gb, self.f.active_controls[p]); continue
                normal = (ft[3:] - ft[:3]) / dist
                mj.mj_jac(m, dd, ja, None, ft[:3], int(m.geom_bodyid[ga])); mj.mj_jac(m, dd, jb, None, ft[3:], int(m.geom_bodyid[gb]))
                J[p] = (normal @ (jb - ja)) @ self.f.expand
        rows = []
        for p in sel:
            g = np.zeros(self.n)
            for k, i in self.ctl: g[k] = J[p, i]
            rows.append((g, float(d[p])))
        return rows


class UnionCollision:
    def __init__(self, *models):
        self.models = models; self.midpoints = any(m.midpoints for m in models)

    def contacts(self, q):
        out = set()
        for m in self.models: out |= m.contacts(q)
        return out

    def collides(self, q):
        return any(m.collides(q) for m in self.models)

    def near(self, q, act):
        return [r for m in self.models for r in m.near(q, act)]

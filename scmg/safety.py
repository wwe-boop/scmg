"""CBF-QP safety projection for the 34-joint EX002 command stream (50 Hz), placed between the generator and the
robot-side SafetyGovernor (edge-live-v2 ex002_safety.py: GJK self-collision clearance, brake-path certificate,
between-frame reach certificate, watchdog), which stays the last certified layer.

Per frame:  q_cmd = argmin ||q - q_ref||^2  subject to, for every joint (all constraints are per-joint intervals, so
the QP is separable and its exact solution is a clip of the next velocity v' = (q - q_t)/dt):
  velocity           |v'| <= v_max
  position CBF       2nd-order barrier: v' <= -A dt + sqrt((A dt)^2 + 2 A (hi - q_t)),  symmetric at lo
                     (A = brake_frac * a_max: the joint can always stop before the limit -> forward invariance)
  acceleration       |a'| <= a_max,        a' = (v' - v_t)/dt
  jerk               |a' - a_t| <= j_max dt
Priority when the intersection is empty: keep velocity + position barrier (hard), then acceleration, then jerk
(relaxed first). Unit test (saturating random walk + jumps beyond limits): position / velocity / acceleration exactly
held (max ratio 1.0); jerk is best-effort (relaxed when the hard layers require it, max 2.9x there) -- a hard jerk
guarantee needs a jerk-limited brake-path certificate, which the Governor provides. Self-collision is NOT handled here (no MuJoCo in the training image): it is the Governor's job.
Limits: Fable D2.2 table used by the Governor (arm shoulder 2.0/12/300, elbow 2.5/16/400, wrist 2.51/20/500,
RM003 fingers 4/60/1500); waist/head are not in the table -> conservative 1.0/8/200 (Claude's choice).
"""
import numpy as np

DT = 1 / 50.


def limits34(meta_lower, meta_upper, margin=0.03):
    v = np.r_[[1.0] * 4, ([2.0] * 3 + [2.5] + [2.51] * 3) * 2, [4.0] * 16]
    a = np.r_[[8.0] * 4, ([12.] * 3 + [16.] + [20.] * 3) * 2, [60.] * 16]
    j = np.r_[[200.] * 4, ([300.] * 3 + [400.] + [500.] * 3) * 2, [1500.] * 16]
    lo = np.asarray(meta_lower, float) + margin; hi = np.asarray(meta_upper, float) - margin
    return dict(lo=lo, hi=hi, v=v, a=a, j=j)


class CBFFilter:
    def __init__(self, lim, q0, brake_frac=0.5, kp=15.0):
        self.L = lim; self.A = brake_frac * lim['a']; self.kp = kp; self.q_ref_prev = None
        self.q = np.clip(np.asarray(q0, float), lim['lo'], lim['hi']); self.v = np.zeros_like(self.q); self.a = np.zeros_like(self.q)
        self.stats = dict(frames=0, modified=0, sum_abs=0., relaxed_jerk=0, relaxed_acc=0)

    def reset(self, q, v=None):
        self.q = np.asarray(q, float).copy(); self.v = np.zeros_like(self.q) if v is None else np.asarray(v, float).copy()
        self.a = np.zeros_like(self.q); self.q_ref_prev = None

    def step(self, q_ref):
        L, dt = self.L, DT
        q_ref = np.asarray(q_ref, float)
        # desired velocity = reference velocity (feed-forward) + position-error feedback. The first version asked for the
        # velocity that reaches q_ref in ONE frame: whenever the filter lagged, that hit the bounds without planning the
        # deceleration and overshot back and forth (on ground truth: every frame modified, speed x2, up to 114 deg off).
        v_ff = np.zeros_like(q_ref) if self.q_ref_prev is None else (q_ref - self.q_ref_prev) / dt
        self.q_ref_prev = q_ref.copy()
        v_ref = v_ff + self.kp * (q_ref - self.q)
        # hard: velocity + 2nd-order position barrier
        d_hi = np.maximum(L['hi'] - self.q, 0.); d_lo = np.maximum(self.q - L['lo'], 0.)
        Ad = self.A * dt
        b_hi = -Ad + np.sqrt(Ad ** 2 + 2 * self.A * d_hi); b_lo = -(-Ad + np.sqrt(Ad ** 2 + 2 * self.A * d_lo))
        lo_h, hi_h = np.maximum(-L['v'], b_lo), np.minimum(L['v'], b_hi)
        lo_h = np.minimum(lo_h, hi_h)                                   # numerically empty -> point interval
        lo, hi = lo_h, hi_h
        for key, (l, h) in (('acc', (self.v - L['a'] * dt, self.v + L['a'] * dt)),
                            ('jerk', (self.v + (self.a - L['j'] * dt) * dt, self.v + (self.a + L['j'] * dt) * dt))):
            nl, nh = np.maximum(lo, l), np.minimum(hi, h)
            bad = nl > nh
            if bad.any():                                               # relax this layer where infeasible:
                self.stats['relaxed_' + key] += int(bad.sum())          # stay as close to it as the higher layers allow
                c = np.clip(np.where(bad, (l + h) / 2, 0.), lo, hi)
                nl, nh = np.where(bad, c, nl), np.where(bad, c, nh)
            lo, hi = nl, nh
        v_new = np.clip(v_ref, lo, hi)
        q_new = self.q + v_new * dt
        a_new = (v_new - self.v) / dt
        dq = np.abs(q_new - q_ref)
        self.stats['frames'] += 1; self.stats['modified'] += int((dq > 1e-4).any()); self.stats['sum_abs'] += float(dq.mean())
        self.q, self.v, self.a = q_new, v_new, a_new
        return q_new


def violation_rate(q, lim):
    """Fraction of frames where the RAW command violates any position / velocity / acceleration / jerk limit."""
    q = np.asarray(q, float); tol = 1 + 1e-6                       # exactly-at-limit is not a violation
    lim = dict(lim, v=lim['v'] * tol, a=lim['a'] * tol, j=lim['j'] * tol)
    v = np.diff(q, axis=0) / DT; a = np.diff(v, axis=0) / DT; j = np.diff(a, axis=0) / DT
    pos = ((q < lim['lo'] - 0.03) | (q > lim['hi'] + 0.03)).any(1)
    out = dict(pos=float(pos.mean()), vel=float((np.abs(v) > lim['v']).any(1).mean()),
               acc=float((np.abs(a) > lim['a']).any(1).mean()), jerk=float((np.abs(j) > lim['j']).any(1).mean()))
    n = len(j); out['any'] = float((pos[:n] | (np.abs(v[:n]) > lim['v']).any(1) | (np.abs(a[:n]) > lim['a']).any(1) |
                                   (np.abs(j) > lim['j']).any(1)).mean()) if n else 0.
    return out

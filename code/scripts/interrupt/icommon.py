"""Shared definitions of the G1 interruption benchmark (11-05). Run with PYTHONPATH=<ECHO-G official>/src.

Representation: ECHO-G robot_repr [T, 39] physical, 30 fps (0:6 base rot6d, 6 yaw delta, 7:10 base velocity, 10: 29 joints
in G1_JOINT_NAMES order). Kinematic metrics use the official pose-only FK with identity base (fk_positions_ba): body
positions relative to the pelvis, so only the joint motion counts. Hands = bodies 28 / 29 (left / right wrist_yaw_link).
"""
import numpy as np
import torch
from echo_g.evaluation import g1_motion_cls as E

FPS = 30.0
HANDS = (28, 29)
UPPER = np.arange(12, 29)          # waist (3) + left arm (7) + right arm (7), canonical joint order (channels 10 + j)
WIN = 64                           # post-interrupt window, frames (2.13 s; the FGD encoder works on 32-frame multiples)
PRE = 3                            # frames before t_i included in the abruptness window (0.1 s)


def fk(rr):
    """rr [T, 39] (numpy / torch) -> body positions [T, 30, 3] (official fk_positions_ba)."""
    return E.fk_positions_ba(torch.as_tensor(np.asarray(rr), dtype=torch.float64))


def hand_speed(P):
    """P [T, 30, 3] -> per-hand speed [T, 2] (central differences, m/s)"""
    return E.body_speed_central(P[:, list(HANDS)], FPS)


def peak_jerk(P, a, b):
    """max over hands of |third difference| * fps^3 (m/s^3) for jerk samples whose 4 frames lie in [a, b)"""
    H = P[a:b, list(HANDS)]
    if len(H) < 4: return float('nan')
    j = H[3:] - 3 * H[2:-1] + 3 * H[1:-2] - H[:-3]
    return float(np.linalg.norm(j, axis=-1).max() * FPS ** 3)


def jerk_series(P, a, b):
    """per-sample hand jerk (max over the two hands, m/s^3) of the samples whose 4 frames lie in [a, b)"""
    H = P[max(0, a):b, list(HANDS)]
    if len(H) < 4: return np.zeros(0)
    j = H[3:] - 3 * H[2:-1] + 3 * H[1:-2] - H[:-3]
    return np.linalg.norm(j, axis=-1).max(-1) * FPS ** 3


IND_POST, IND_PRE = 15, WIN      # 11-06 induced-abruptness windows: [t_i, t_i + 0.5 s] vs the 2.13 s before t_i


def induced_abrupt(P, f, pre=IND_PRE, post=IND_POST):
    """11-06 main abruptness test: peak hand jerk of the samples ending in [t_i, t_i + 0.5 s] (frames f - 3 .. f + 15, so the
    sample across t_i counts) above the P95 of the SAME motion's per-sample jerk on the 2.13 s before t_i.
    Returns (peak post jerk, pre P95, abrupt bool); None if the pre window is too short."""
    js = jerk_series(P, f - pre, f)
    if len(js) < 10: return None
    thr = float(np.percentile(js, 95)); pk = float(jerk_series(P, f - PRE, f + post).max())
    return pk, thr, bool(pk > thr)


V3_TR = 1.8                       # 11-06 v3 retraction window (s): the human median retraction time (= the B2 T_r), for every method
V3_RATIO = 1.5                    # 11-06 v3 threshold, chosen a priori (criteria/metric_change_v3_1106.json)


def v3_ratio(P, PU, f, tr=V3_TR):
    """11-06 v3 main abruptness: peak hand jerk of the interrupted motion P over the retraction window [t_i, t_i + T_r] (jerk
    samples ending in [f, f + T_r*30), i.e. frames f - 3 ..) divided by the same peak of the counterfactual PU (the same method /
    clip / seed WITHOUT the interruption). Windows clipped to the frames both have. None if < 10 samples."""
    b = min(len(P), len(PU), f + int(round(tr * FPS)))
    a, u = jerk_series(P, f - PRE, b), jerk_series(PU, f - PRE, b)
    if len(a) < 10 or len(u) < 10: return None
    return float(a.max() / max(u.max(), 1e-9))


def peak_acc(P, a, b):
    H = P[a:b, list(HANDS)]
    if len(H) < 3: return float('nan')
    acc = H[2:] - 2 * H[1:-1] + H[:-2]
    return float(np.linalg.norm(acc, axis=-1).max() * FPS ** 2)


def yield_time(P, t0, z_rest, dz=0.05, horizon=WIN):
    """Frames after t0 until both hands are at rest height (z <= z_rest + dz, per hand) and STAY there to the end of the
    window [t0, t0 + horizon). Returns (seconds or None if not reached, needed: hands above rest height at t0)."""
    z = P[t0:t0 + horizon, list(HANDS), 2]
    low = (z <= np.asarray(z_rest)[None] + dz).all(1)
    needed = not bool(low[0])
    if not low[-1]: return None, needed
    k = len(low) - 1
    while k > 0 and low[k - 1]: k -= 1
    return k / FPS, needed


KIN_NAMES = ('log_energy_0_1s', 'log_energy_1_2s', 'hand_dz', 'still_frac', 'log_rel_jerk')


def kin_features(P, f):
    """speaker-normalised retraction descriptors of the window [f, f + WIN) relative to the second before f:
    log mean hand speed (first / second half of the window) over the pre-interrupt mean, hand height change (m),
    share of still frames (mean hand speed < 0.05 m/s), log peak jerk over the pre-interrupt peak jerk."""
    v = hand_speed(P).mean(1); a = max(0, f - 30); pre = v[a:f].mean() + 1e-3
    z = P[:, list(HANDS), 2].mean(1)
    return np.array([np.log((v[f:f + 32].mean() + 1e-3) / pre), np.log((v[f + 32:f + WIN].mean() + 1e-3) / pre),
                     z[f + WIN - 1] - z[f - 1], float((v[f:f + WIN] < 0.05).mean()),
                     np.log(peak_jerk(P, f - PRE, f + WIN) + 1) - np.log(peak_jerk(P, a, f) + 1)])


def kin_distance(X, Hf):
    """mean over descriptors of the 1-D Wasserstein distance to the human set, in units of the human std"""
    from scipy.stats import wasserstein_distance
    sd = Hf.std(0) + 1e-6
    per = [wasserstein_distance(X[:, k], Hf[:, k]) / sd[k] for k in range(Hf.shape[1])]
    return float(np.mean(per)), [float(x) for x in per]


def quintic(q0, v0, a0, q1, T, n):
    """per-channel quintic from (q0, v0, a0) to (q1, 0, 0) in T seconds, sampled at frames 1..n (30 fps); held after T.
    q*: [C]; returns [n, C]. With v0 = a0 = 0 this is the classic min-jerk profile."""
    t = np.minimum(np.arange(1, n + 1) / FPS, T)[:, None]
    d = q1 - q0 - v0 * T - 0.5 * a0 * T ** 2
    dv = -v0 - a0 * T; da = -a0
    c3 = (10 * d - 4 * dv * T + 0.5 * da * T ** 2) / T ** 3
    c4 = (-15 * d + 7 * dv * T - da * T ** 2) / T ** 4
    c5 = (6 * d - 3 * dv * T + 0.5 * da * T ** 2) / T ** 5
    return q0 + v0 * t + 0.5 * a0 * t ** 2 + c3 * t ** 3 + c4 * t ** 4 + c5 * t ** 5


def damped(q0, v0, q1, T, n):
    """critically damped 2nd-order approach from (q0, v0) to q1, 95 % settled at T s (omega = 4.744 / T), frames 1..n.
    Continuous position and velocity at the start (no velocity jump), no oscillation."""
    w = 4.744 / T; t = (np.arange(1, n + 1) / FPS)[:, None]; e = q0 - q1
    return q1 + (e + (v0 + w * e) * t) * np.exp(-w * t)


def freeze_after(rr, f):
    """B1: robot_repr frozen at frame f - 1 from frame f on (joints and base orientation held, base velocity / yaw 0)."""
    out = np.array(rr[:f + WIN], np.float64, copy=True)
    if len(out) < f + WIN: out = np.concatenate([out, np.repeat(out[-1:], f + WIN - len(out), 0)])
    out[f:] = out[f - 1]; out[f:, 6:10] = 0.
    return out


def retract_after(rr, f, rest, T_r, mode='damped'):
    """B2: B1 + the upper-body joints (waist + arms) move from frame f - 1 to `rest` [29]; legs and base stay frozen.
    mode 'damped'  : critically damped approach, velocity at f - 1 matched, 95 % settled at T_r   (the reported B2)
         'minjerk' : classic min-jerk (quintic from rest velocity) in T_r s (velocity jump at t_i)
         'quintic' : quintic with p, v, a at f - 1 matched, T_r s (overshoots for long T_r: v0 * T_r excursion)"""
    out = freeze_after(rr, f)
    q = np.asarray(rr, np.float64)[:, 10:]; n = len(out) - f; q1 = np.asarray(rest, np.float64)[UPPER]
    q0 = q[f - 1, UPPER]; v0 = (q[f - 1, UPPER] - q[f - 2, UPPER]) * FPS
    a0 = (q[f - 1, UPPER] - 2 * q[f - 2, UPPER] + q[f - 3, UPPER]) * FPS ** 2
    if mode == 'damped': out[f:, 10 + UPPER] = damped(q0, v0, q1, T_r, n)
    elif mode == 'minjerk': out[f:, 10 + UPPER] = quintic(q0, 0 * q0, 0 * q0, q1, T_r, n)
    elif mode == 'quintic': out[f:, 10 + UPPER] = quintic(q0, v0, a0, q1, T_r, n)
    else: raise ValueError(mode)
    return out


def load_rr(path):
    x = torch.load(path, map_location='cpu', weights_only=False)
    rr = x['robot_repr'] if isinstance(x, dict) else x
    return np.asarray(rr, np.float64)

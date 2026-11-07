"""EX002 port of scripts/interrupt/icommon.py (11-07, criteria docs/criteria/EX002_INTERRUPT_1107.md): the same metric definitions
on the EX002 34-joint, 50 Hz representation (rad). Positions = exact FK (scmg/kinematics.py, sphere-proxy npz; base frame,
fixed base). Hands = the two wrist bodies (children of arm_joint7: joints 10 / 17 of the 34-joint order). Window 2.13 s = 107
frames, 0.1 s = 5 frames, 1 s = 50 frames (the G1 30 fps constants scaled by time). Upper body for the retraction baselines =
waist (0, 1) + arms (4..17); head (2, 3) and the 16 finger joints are held."""
import numpy as np
import torch

FPS = 50.0
WIN = 107                          # 2.13 s
PRE = 5                            # 0.1 s
SEC = 50                           # 1 s
HALF = 53                          # first half of the window
UPPER = np.r_[0:2, 4:18]
NPZ = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/sphere_proxy_vox4e.npz'
_K = {}


def _kin():
    if 'k' not in _K:
        from scmg.kinematics import Kinematics
        from scmg.common import P
        z = np.load(P(NPZ)); bj = np.asarray(z['body_joint'])
        _K['k'] = Kinematics(P(NPZ), 'cpu'); _K['hands'] = (int(np.nonzero(bj == 10)[0][0]), int(np.nonzero(bj == 17)[0][0]))
    return _K['k']


def fk(q):
    """q [T, 34] rad -> wrist positions [T, 2, 3] (m, base frame)."""
    K = _kin(); out = []
    with torch.no_grad():
        qq = torch.as_tensor(np.asarray(q, np.float32))
        for i in range(0, len(qq), 8192):
            _, p = K.fk(qq[i:i + 8192]); out.append(p[:, list(_K['hands'])].double().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2, 3))


def hand_speed(P):
    v = np.zeros(P.shape[:2]); v[1:-1] = np.linalg.norm(P[2:] - P[:-2], axis=-1) * FPS / 2
    if len(P) > 1: v[0] = np.linalg.norm(P[1] - P[0], axis=-1) * FPS; v[-1] = np.linalg.norm(P[-1] - P[-2], axis=-1) * FPS
    return v


def jerk_series(P, a, b):
    H = P[max(0, a):b]
    if len(H) < 4: return np.zeros(0)
    j = H[3:] - 3 * H[2:-1] + 3 * H[1:-2] - H[:-3]
    return np.linalg.norm(j, axis=-1).max(-1) * FPS ** 3


def peak_jerk(P, a, b):
    js = jerk_series(P, a, b)
    return float(js.max()) if len(js) else float('nan')


V3_TR, V3_RATIO = 1.8, 1.5


def v3_ratio(P, PU, f, tr=V3_TR):
    b = min(len(P), len(PU), f + int(round(tr * FPS)))
    a, u = jerk_series(P, f - PRE, b), jerk_series(PU, f - PRE, b)
    if len(a) < 10 or len(u) < 10: return None
    return float(a.max() / max(u.max(), 1e-9))


def yield_time(P, t0, z_rest, dz=0.05, horizon=WIN):
    z = P[t0:t0 + horizon, :, 2]
    low = (z <= np.asarray(z_rest)[None] + dz).all(1)
    needed = not bool(low[0])
    if not low[-1]: return None, needed
    k = len(low) - 1
    while k > 0 and low[k - 1]: k -= 1
    return k / FPS, needed


KIN_NAMES = ('log_energy_0_1s', 'log_energy_1_2s', 'hand_dz', 'still_frac', 'log_rel_jerk')


def kin_features(P, f):
    v = hand_speed(P).mean(1); a = max(0, f - SEC); pre = v[a:f].mean() + 1e-3
    z = P[:, :, 2].mean(1)
    return np.array([np.log((v[f:f + HALF].mean() + 1e-3) / pre), np.log((v[f + HALF:f + WIN].mean() + 1e-3) / pre),
                     z[f + WIN - 1] - z[f - 1], float((v[f:f + WIN] < 0.05).mean()),
                     np.log(peak_jerk(P, f - PRE, f + WIN) + 1) - np.log(peak_jerk(P, a, f) + 1)])


def kin_distance(X, Hf):
    from scipy.stats import wasserstein_distance
    sd = Hf.std(0) + 1e-6
    per = [wasserstein_distance(X[:, k], Hf[:, k]) / sd[k] for k in range(Hf.shape[1])]
    return float(np.mean(per)), [float(x) for x in per]


def damped(q0, v0, q1, T, n):
    w = 4.744 / T; t = (np.arange(1, n + 1) / FPS)[:, None]; e = q0 - q1
    return q1 + (e + (v0 + w * e) * t) * np.exp(-w * t)


def freeze_after(q, f):
    out = np.array(q[:f + WIN], np.float64, copy=True)
    if len(out) < f + WIN: out = np.concatenate([out, np.repeat(out[-1:], f + WIN - len(out), 0)])
    out[f:] = out[f - 1]
    return out


def retract_after(q, f, rest, T_r):
    """B2 on EX002: frozen at f - 1, waist + arms critically damped to `rest` [34] (95 % at T_r, velocity matched)."""
    out = freeze_after(q, f); q = np.asarray(q, np.float64); n = len(out) - f; q1 = np.asarray(rest, np.float64)[UPPER]
    q0 = q[f - 1, UPPER]; v0 = (q[f - 1, UPPER] - q[f - 2, UPPER]) * FPS
    out[f:, UPPER] = damped(q0, v0, q1, T_r, n)
    return out

"""Unit tests for the EX002 safety layers (scmg/safety_proj_ex002.py, design v6 C3, 11-05). Synthetic 50 Hz trajectories
around the EX002 anchor pose (no dataset needed).

    PYTHONPATH=<repo>:<osqp/ruckig/mujoco deps> python scripts/tests/test_safety_proj_ex002.py

1 helpers: near() first-order distance prediction vs mj_geomDistance after a small random step; collides() == the
  validator's collision verdict.
2 safe clips (anchor + slow raised-cosine gestures that the independent validator passes with 0 violations, starting at
  rest) -> change ~0 (max |dq| per layer; asserted for clip / clip_coll / proj_l2 <= 1e-3 rad).
3 violating clips (arms driven across the body into WBC-pair contact, joint-limit overshoot, 0.3 rad steps): independent
  validator ex002_safety.validate_sequence (frames + midpoints, D2.2 v/a/j table, mimic) -> proj_l2 and governor have
  0 violations of every kind; position / collision for clip_coll and cbf_qp (their acc / jerk excess printed).
4 causality: replacing the input after chunk k leaves chunks 0..k bit-identical (k = 1, 3).
5 per-chunk solve time (16 frames = 320 ms).
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
from scmg import safety_proj as S
from scmg import safety_proj_ex002 as X

ok = True


def check(name, cond, info=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True); ok &= bool(cond)


lim = X.EX002Limits(); coll = X.EX002Collision(lim.E); E = lim.E
n, dt = lim.n, lim.dt
rng = np.random.default_rng(0)
times = {k: [] for k in X.LAYERS_EX}
print(f'EX002 {n} joints {E.controlled}; {len(E.pairs)} WBC pairs; dt {dt}')

# ---------------------------------------------------------------------------------------------- 1 helpers
errs = []; agree = 0; tot = 0
for _ in range(400):
    c = np.clip(E.anchor + rng.normal(size=n) * 0.5, lim.lo, lim.hi)
    tot += 1; agree += coll.collides(c) == (X.validate(lim, c[None], c, midpoints=False)['collision'] > 0)
    rows = coll.near(c, 0.01)
    if not rows or coll.collides(c): continue
    g, d0 = rows[0]
    dq = rng.normal(size=n) * 2e-3
    pred = d0 + g @ dq
    r2 = coll.near(c + dq, 0.05)
    if r2: errs.append(min(abs(d2 - pred) for _, d2 in r2) / max(abs(g @ dq), 1e-6))
    if len(errs) >= 30: break
check('collides() == validator collision verdict', agree == tot, f'{agree}/{tot}')
check('near(): first-order prediction (median rel err < 0.3)', len(errs) >= 5 and np.median(errs) < 0.3,
      f'n={len(errs)} median {np.median(errs) if errs else float("nan"):.3f}')


# ---------------------------------------------------------------------------------------------- data
def safe_clip(seed, T=300):
    r = np.random.default_rng(seed)
    t = np.arange(T) * dt
    q = np.repeat(E.anchor[None], T, 0).astype(float)
    for j in range(14):
        A = r.uniform(0.02, 0.12); f = r.uniform(0.3, 0.7); ph = r.uniform(0, 2 * np.pi)
        q[:, j] += A * (1 - np.cos(2 * np.pi * f * t)) * np.cos(ph)       # starts at rest
    return q


def bad_clip(seed, T=400):
    r = np.random.default_rng(100 + seed)
    q = safe_clip(seed, T)
    t = np.arange(T)
    # arms swept across the body (shoulder roll / yaw towards the midline) -> WBC pair contact
    w = np.zeros(T); w[60:200] = np.sin(np.linspace(0, np.pi, 140)) ** 2
    q[:, 1] -= 2.6 * w; q[:, 8] -= 2.6 * w
    q[:, 2] -= 1.2 * w; q[:, 9] += 1.2 * w
    q[:, 3] += 2.2 * np.exp(-((t - 250) / 15.0) ** 2)                      # left elbow above its upper bound (-0.123)
    q[300:, 4] += 0.3; q[320:, 11] -= 0.3                                  # wrist steps
    q[:, :14] += r.normal(size=(T, 14)) * 0.003
    return q


safe = []
for s in range(20):
    q = safe_clip(s)
    v = X.validate(lim, q, q[0])
    if v['total'] == 0: safe.append(q)
    if len(safe) >= 4: break
print(f'{len(safe)} synthetic safe clips (validator total 0)')
mx = {k: 0.0 for k in X.LAYERS_EX}
for q in safe:
    for name in X.LAYERS_EX:
        L = X.make_layer_ex002(name, lim, coll)
        y, st = S.run_stream(L, q, X.K_EX); times[name] += st['times']
        mx[name] = max(mx[name], float(np.abs(y - q).max()))
print('safe clips max |dq| (rad): ' + ', '.join(f'{k} {v:.2e}' for k, v in mx.items()))
for name in ('clip', 'clip_coll', 'proj_l2'):
    check(f'safe clips ~unchanged by {name}', mx[name] <= 1e-3, f'{mx[name]:.2e} rad')

viol = {k: {} for k in X.LAYERS_EX}; fb = {k: 0 for k in X.LAYERS_EX}
raw_v = {}
for s in range(3):
    q = bad_clip(s)
    rv = X.validate(lim, q, q[0])
    for k2 in ('position', 'collision', 'collision_midpoint', 'velocity', 'acceleration', 'jerk'): raw_v[k2] = raw_v.get(k2, 0) + rv[k2]
    for name in X.LAYERS_EX:
        L = X.make_layer_ex002(name, lim, coll)
        y, st = S.run_stream(L, q, X.K_EX); times[name] += st['times']; fb[name] += st['fallback_frames']
        v = X.validate(lim, y, L.q_start if L.q_start is not None else y[0])
        for k2 in ('position', 'collision', 'collision_midpoint', 'velocity', 'acceleration', 'jerk', 'mimic', 'total'):
            viol[name][k2] = viol[name].get(k2, 0) + v[k2]
check('violating clips do violate (raw)', raw_v['collision'] > 0 and raw_v['position'] > 0, str(raw_v))
for name in X.LAYERS_EX:
    print(f'  {name:10s} validator {viol[name]}  fallback frames {fb[name]}')
for name in ('proj_l2', 'governor'):
    check(f'{name}: independent validator 0 violations (all kinds)', viol[name]['total'] == 0, str(viol[name]))
for name in ('clip_coll', 'cbf_qp'):
    v = viol[name]
    check(f'{name}: 0 position / collision / midpoint violations', v['position'] + v['collision'] + v['collision_midpoint'] == 0, str(v))

# ---------------------------------------------------------------------------------------------- 4 causality
q = bad_clip(0); other = safe_clip(7, len(q))
K = X.K_EX
for name in X.LAYERS_EX:
    L = X.make_layer_ex002(name, lim, coll)
    y0, _ = S.run_stream(L, q, K); same = True
    for k in (1, 3):
        q2 = q.copy(); q2[(k + 1) * K:] = other[(k + 1) * K:]
        y2, _ = S.run_stream(L, q2, K)
        same &= np.array_equal(y0[:(k + 1) * K], y2[:(k + 1) * K]) and not np.array_equal(y0[(k + 1) * K:], y2[(k + 1) * K:])
    check(f'causality {name}', same)

print(f'per-chunk solve time (ms, CPU 1 thread, K={K} frames = {K * dt * 1e3:.0f} ms of motion):')
for k, t in times.items():
    t = np.array(t) * 1e3
    print(f'  {k:10s} n={len(t):5d} mean {t.mean():8.2f} p50 {np.percentile(t, 50):8.2f} p95 {np.percentile(t, 95):8.2f} '
          f'p99 {np.percentile(t, 99):8.2f} max {t.max():8.2f}')
print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)

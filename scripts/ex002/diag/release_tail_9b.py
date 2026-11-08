import json, glob, sys, numpy as np
sys.path.insert(0, '/home/xr-25b7/projects/gesture/scmg/scripts/ex002')
from ex002_keyframe import ARM, HOLD, RAMP, FPS
D = sys.argv[1]
acc = lambda y, jj: np.linalg.norm(np.diff(y[:, jj], n=2, axis=0), axis=1) * FPS ** 2
ev = [x for f in sorted(glob.glob(f'{D}/ins/KEYFRAMES*')) for x in json.load(open(f))['events']]
src = {ci: np.load(f'{D}/src/{ci}.npy') for ci in range(3)}
# generator spike frames (either arm, acc > 100; GT has 0.6 % such frames); report their spacing
sp = {}
for ci, y in src.items():
    a = np.maximum(acc(y, ARM[0]), acc(y, ARM[1])); s = np.where(a > 100)[0] + 1; sp[ci] = s
    d = np.diff(s); print('clip', ci, 'generator spike frames', len(s), 'spacing mode', np.bincount(d[d > 5]).argmax() if (d > 5).any() else None, 'share of spacings 74-76:', round(float(np.isin(d[d > 5], [74, 75, 76]).mean()), 2))
rows = []
for x in ev:
    ci = x['ci']; h = x['hand']; jj = ARM[h]; kind = 'aim' if x['kind'] == 'aim' else 'finger'
    y9 = np.load(f'{D}/ins/{ci}.npy'); yt = np.load(f'{D}/tok/{ci}.npy'); T = len(y9)
    tr = x.get('t_release', x['t'] + HOLD[kind][1] + RAMP); f0, f1 = max(1, int((tr - 0.1) * FPS)), min(T - 1, int((tr + 0.1) * FPS))
    a = acc(y9[f0 - 1:f1 + 1], jj); fp = f0 + int(a.argmax())
    near = np.abs(sp[ci] - fp).min() <= 2
    modified = np.abs(y9[fp - 1:fp + 2, jj] - yt[fp - 1:fp + 2, jj]).max() > 1e-6       # peak's 3-frame stencil touches inserted frames
    rows.append((a.max(), near, modified, 't_release' in x))
v = np.array([r[0] for r in rows]); near = np.array([r[1] for r in rows]); mod = np.array([r[2] for r in rows])
top = v >= np.percentile(v, 90)
print(f'9b release: median {np.median(v):.1f} p95 {np.percentile(v,95):.1f} max {v.max():.1f}')
print(f'top-10% events (n={top.sum()}): peak within 2 frames of a generator spike {near[top].mean():.2f}; peak stencil touches inserted frames {mod[top].mean():.2f}')
print(f'all events: peak at generator spike {near.mean():.2f}; without those events: median {np.median(v[~near]):.1f} p95 {np.percentile(v[~near],95):.1f} max {v[~near].max():.1f}')
print(f'events whose peak stencil touches inserted frames only: n {mod.sum()} median {np.median(v[mod]):.1f} p95 {np.percentile(v[mod],95):.1f} max {v[mod].max():.1f}')

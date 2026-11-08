"""EX002 module 9 (11-08; criteria docs/criteria/EX002_CTL.md "Module 9"): keyframe insertion with causal re-conditioning. Default off:
only used through this script and bench_ex002 --inpaint-dir.
  insert : for every command of a plan, an end-effector keyframe solved by damped least squares (Gauss-Newton + Levenberg damping) on the
           commanded arm's 7 joints only (exact FK scmg/kinematics, joint limits = data-v36_1 META bounds -/+ 0.03 rad, sphere-proxy
           self-penetration as a residual), seeded from the generated pose at the command time:
             point  : forearm ray (elbow -> wrist) toward the target (residual unit(w - e) - unit(target - w))
             present: (finger commands; the hand with the larger commanded count, ties -> right) wrist at shoulder + (0.30, 0.06 toward
                      the midline, -0.12) m and palm normal (cross(index base - wrist, third base - wrist)) along +-x (the sign of the seed)
           + regularisation sqrt(0.05) (q - q_seed); inserted in joint space with minimum-jerk ramps of 0.3 s, held on [t - 0.1, t + 0.6]
           (point) / [t - 0.4, t + 0.4] (present); the finger override (minimum-jerk 0.3 s) is applied after.  Writes <out>/<ci>.npy and
           <out>/<ci>.npz (inserted frames + spans, the input of the re-conditioning pass) and KEYFRAMES.<i>of<n>.json (solve residuals,
           self-penetration; a keyframe with sphere-proxy penetration > 0 after the solve is REJECTED: no insertion for that event, counted).
  release: release-continuity metric: per insertion, the peak arm-joint acceleration norm (rad/s^2, second difference x 50^2) in
           [t_release - 0.1, t_release + 0.1] (t_release = end of the ramp-out); reference = the p95 of the same quantity over all dev
           GT frames (gt_<ci>.npy); plus the share of ramp frames (in and out) whose arm jerk exceeds the limits34 arm jerk limits.
    SCMG_J=34 python ex002_keyframe.py insert --given D --plan P --out O [--mode given|shuffle]
    SCMG_J=34 python ex002_keyframe.py release --plan P --gt-dir G --out R.json label=<dir> ..."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
FPS = 50; ARM = {0: list(range(4, 11)), 1: list(range(11, 18))}
V = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1'
HS = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002/tokens/HAND_SHAPES.json'
A95 = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002/tokens/ARM_ACC_P95.json'
HOLD = dict(aim=(0.1, 0.6), finger=(0.4, 0.4)); RAMP = 0.3; LAM = 0.05
JLIM = np.array([300., 300., 300., 400., 500., 500., 500.])           # arm jerk limits (rad/s^3, scmg/safety.py limits34)


def bodies():
    from scmg import ex002_ctl as E
    from scmg.common import P
    z = np.load(P(E.NPZ)); bj = np.asarray(z['body_joint']); b = lambda j: int(np.nonzero(bj == j)[0][0])
    return dict(shoulder=(b(4), b(11)), elbow=(b(7), b(14)), wrist=(b(10), b(17)), index_base=(b(21), b(29)), third_base=(b(24), b(32)))


class IK:
    def __init__(self):
        from scmg import ex002_ctl as E
        meta = json.loads((Path(V) / 'META.json').read_text())['ex002']
        self.lo = torch.tensor(meta['lower'], dtype=torch.float64) + .03; self.hi = torch.tensor(meta['upper'], dtype=torch.float64) - .03
        self.K = E.kin('cpu'); self.B = bodies()

    def geo(self, q):
        _, p = self.K.fk(q[None].float()); p = p[0].double(); return {k: (p[l], p[r]) for k, (l, r) in self.B.items()}

    def pen(self, q):
        return self.K.penetration(q[None].float())[0].double()

    def solve(self, q0, h, kind, target=None, it=20, pen_w=10.0, present_off=None, wrist_pos=None, zmin=None):
        # 9c (default None = 9 / 9b unchanged): wrist_pos [3] adds w - wrist_pos (aim hold replay); zmin (m below the shoulder, e.g. 0.15)
        # adds a hard-weighted hinge on wrist z >= shoulder z - zmin (present)
        q0 = torch.as_tensor(q0, dtype=torch.float64); q = q0.clone(); idx = ARM[h]
        sgn = None
        def res(x):
            qq = q.clone(); qq[idx] = x; g = self.geo(qq); w, e, s = g['wrist'][h], g['elbow'][h], g['shoulder'][h]
            r = []
            if kind == 'aim':
                t = torch.as_tensor(target, dtype=torch.float64); r.append((w - e) / (w - e).norm() - (t - w) / (t - w).norm())
                if wrist_pos is not None: r.append(w - torch.as_tensor(wrist_pos, dtype=torch.float64))
            else:
                if present_off is None:                                  # m9: left hand toward -y (midline), right toward +y
                    mid = 0.06 if h == 0 else -0.06; off = torch.tensor([0.30, -mid, -0.12], dtype=torch.float64)
                else:                                                    # 9b: (forward, OUTWARD, dz); outward = +y left, -y right
                    fx, out, dz = present_off; off = torch.tensor([fx, out if h == 0 else -out, dz], dtype=torch.float64)
                pt = s + off; r.append(w - pt)
                n = torch.cross(g['index_base'][h] - w, g['third_base'][h] - w, dim=0); n = n / n.norm().clamp(min=1e-9)
                r.append(n - sgn * torch.tensor([1., 0., 0.], dtype=torch.float64))
                if zmin is not None: r.append(100. * torch.relu((s[2] - zmin) - w[2]).reshape(1))
            r.append(np.sqrt(LAM) * (x - q0[idx])); r.append(pen_w * self.pen(qq).square().sum().add(1e-12).sqrt().reshape(1))
            return torch.cat([x_.reshape(-1) for x_ in r])
        if kind != 'aim':
            g = self.geo(q); n = torch.cross(g['index_base'][h] - g['wrist'][h], g['third_base'][h] - g['wrist'][h], dim=0); sgn = 1. if float(n[0]) >= 0 else -1.
        x = q[idx].clone(); mu = 1e-2
        for _ in range(it):
            r = res(x); J = torch.autograd.functional.jacobian(res, x, vectorize=True, strategy='forward-mode')
            dx = torch.linalg.solve(J.T @ J + mu * torch.eye(len(x), dtype=torch.float64), -J.T @ r)
            xn = torch.clamp(x + dx, self.lo[idx], self.hi[idx]); rn = res(xn)
            if rn.norm() < r.norm(): x, mu = xn, max(mu / 3, 1e-6)
            else: mu = mu * 5
        qf = q.clone(); qf[idx] = x; pen = float(self.pen(qf).max()) if len(self.pen(qf)) else 0.
        info = dict(res=float(res(x).norm()), pen=pen)
        if zmin is not None or wrist_pos is not None:
            g = self.geo(qf); info['wz_minus_sz'] = float(g['wrist'][h][2] - g['shoulder'][h][2])
            if wrist_pos is not None: info['wpos_err'] = float((g['wrist'][h] - torch.as_tensor(wrist_pos, dtype=torch.float64)).norm())
        return qf.numpy().astype(np.float32), info


def mj(x): x = np.clip(x, 0, 1); return x ** 3 * (10 - 15 * x + 6 * x * x)


GRID9B = [(0.30, o, dz) for dz in (-0.12, -0.06, -0.18) for o in (0.05, 0.10, 0.00)]     # 9b present fallback grid (never inward)


def state(y, f):
    """position / velocity / acceleration (rad, rad/s, rad/s^2) of y at frame f (central differences, clamped)."""
    f = int(np.clip(f, 1, len(y) - 2)); return y[f], (y[f + 1] - y[f - 1]) * FPS / 2, (y[f + 1] - 2 * y[f] + y[f - 1]) * FPS ** 2


def quintic(p0, v0, a0, p1, v1, a1, T, n):
    """quintic Hermite samples at n frames spanning [0, T] (inclusive start, exclusive end), per joint."""
    t = np.arange(n)[:, None] / FPS; T = max(T, 1e-3)
    c0, c1, c2 = p0, v0, a0 / 2
    c3 = (20 * (p1 - p0) - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * T ** 2) / (2 * T ** 3)
    c4 = (30 * (p0 - p1) + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * T ** 2) / (2 * T ** 4)
    c5 = (12 * (p1 - p0) - 6 * (v1 + v0) * T - (a0 - a1) * T ** 2) / (2 * T ** 5)
    return c0 + c1 * t + c2 * t ** 2 + c3 * t ** 3 + c4 * t ** 4 + c5 * t ** 5


def insert9b(yi, qk, h, t, kind, a95):
    """adaptive quintic ramp-in / hold / quintic release to the generator's own state (9b)."""
    T = len(yi); jj = ARM[h]; a, b = HOLD[kind]; f_h0, f_h1 = int(round((t - a) * FPS)), int(round((t + b) * FPS))
    f_h0, f_h1 = max(1, f_h0), min(T - 2, f_h1)
    if f_h1 <= f_h0: return None
    a9 = np.asarray(a95)[[j - 4 for j in jj]]
    Tin = float(np.clip(np.sqrt(5.77 * np.max(np.abs(qk[jj] - yi[f_h0, jj]) / a9)), 0.3, 0.8))
    Tout = float(np.clip(np.sqrt(5.77 * np.max(np.abs(qk[jj] - yi[f_h1, jj]) / a9)), 0.3, 0.8))
    f_in = max(1, f_h0 - int(round(Tin * FPS))); f_out = min(T - 2, f_h1 + int(round(Tout * FPS)))
    p0, v0, a0 = state(yi[:, jj], f_in); p1, v1, a1 = state(yi[:, jj], f_out)
    z = np.zeros(len(jj))
    if f_h0 > f_in: yi[f_in:f_h0, jj] = quintic(p0, v0, a0, qk[jj], z, z, (f_h0 - f_in) / FPS, f_h0 - f_in)
    yi[f_h0:f_h1 + 1, jj] = qk[jj][None]
    if f_out > f_h1 + 1: yi[f_h1 + 1:f_out, jj] = quintic(qk[jj], z, z, p1, v1, a1, (f_out - f_h1 - 1) / FPS, f_out - f_h1 - 1)
    return dict(t_in0=f_in / FPS, t_hold=(f_h0 / FPS, f_h1 / FPS), t_release=f_out / FPS, Tin=Tin, Tout=Tout)


def insert9c(yi, Q, h, t, kind, a95):
    """9c: as insert9b, but the hold is a pose sequence Q [n, 34] (aim: replayed human hold; present: constant). Ramp-in ends in Q[0] with
    its velocity / acceleration, release starts from Q[-1] with its velocity / acceleration (finite differences of Q)."""
    T = len(yi); jj = ARM[h]; a, b = HOLD[kind]; f_h0 = max(1, int(round((t - a) * FPS))); f_h1 = min(T - 2, int(round((t + b) * FPS)))
    if f_h1 <= f_h0: return None
    n = f_h1 - f_h0 + 1; Q = np.asarray(Q, np.float64)[:, jj]
    Q = Q[:n] if len(Q) >= n else np.concatenate([Q, np.repeat(Q[-1:], n - len(Q), 0)])
    v = np.gradient(Q, axis=0) * FPS if n > 2 else np.zeros_like(Q); acc = np.gradient(v, axis=0) * FPS if n > 2 else np.zeros_like(Q)
    a9 = np.asarray(a95)[[j - 4 for j in jj]]
    Tin = float(np.clip(np.sqrt(5.77 * np.max(np.abs(Q[0] - yi[f_h0, jj]) / a9)), 0.3, 0.8))
    Tout = float(np.clip(np.sqrt(5.77 * np.max(np.abs(Q[-1] - yi[f_h1, jj]) / a9)), 0.3, 0.8))
    f_in = max(1, f_h0 - int(round(Tin * FPS))); f_out = min(T - 2, f_h1 + int(round(Tout * FPS)))
    p0, v0, a0 = state(yi[:, jj], f_in); p1, v1, a1 = state(yi[:, jj], f_out)
    if f_h0 > f_in: yi[f_in:f_h0, jj] = quintic(p0, v0, a0, Q[0], v[0], acc[0], (f_h0 - f_in) / FPS, f_h0 - f_in)
    yi[f_h0:f_h1 + 1, jj] = Q
    if f_out > f_h1 + 1:
        m = f_out - f_h1                                              # quintic from Q[-1] (frame f_h1) to the generator state (frame f_out)
        yi[f_h1 + 1:f_out, jj] = quintic(Q[-1], v[-1], acc[-1], p1, v1, a1, m / FPS, m)[1:]
    return dict(t_in0=f_in / FPS, t_hold=(f_h0 / FPS, f_h1 / FPS), t_release=f_out / FPS, Tin=Tin, Tout=Tout)


KEYS9C = 3                                                            # 9c: IK every 3rd hold frame (+ the last), cubic in between


def hold9c(ik, qk, h, target, dw, it=5):
    """per-key IK keeping the forearm ray on the target while the wrist follows wk + dw[tau] (dw [n, 3], human replay)."""
    from scipy.interpolate import CubicSpline
    g = ik.geo(torch.as_tensor(qk, dtype=torch.float64)); wk = g['wrist'][h].numpy(); n = len(dw)
    keys = sorted(set(list(range(0, n, KEYS9C)) + [n - 1])); Qk = []; q = np.asarray(qk, np.float32); bad = 0; perr = []
    for k in keys:
        if k == 0: Qk.append(q); perr.append(0.); continue
        qn, info = ik.solve(q, h, 'aim', target, it=it, pen_w=100.0, wrist_pos=wk + dw[k])
        if info['pen'] > 1e-6: bad += 1; Qk.append(Qk[-1]); perr.append(float('nan')); continue
        q = qn; Qk.append(qn); perr.append(info['wpos_err'])
    Qk = np.stack(Qk).astype(np.float64); Q = CubicSpline(np.asarray(keys, float), Qk, axis=0)(np.arange(n)) if len(keys) > 1 else np.repeat(Qk, n, 0)
    return Q.astype(np.float32), dict(hold_keys=len(keys), hold_bad_keys=bad, hold_wpos_err_median=float(np.nanmedian(perr)) if len(perr) else 0.)


def pool_index(clip, t, n):
    return int(hashlib.sha1(f'{clip}:{t:.3f}:9c'.encode()).hexdigest()[:12], 16) % n


def window(T, t, kind):
    a, b = HOLD[kind]; fr = np.arange(T) / FPS
    return mj((fr - (t - a - RAMP)) / RAMP) * mj(((t + b + RAMP) - fr) / RAMP), (t - a - RAMP, t + b + RAMP)


def insert(a):
    from ex002_ctl_run import finger_override, tokens_for
    P = json.loads(Path(a.plan).read_text()); by = {}
    for e in P['events']:
        if not e.get('present'): by.setdefault(e['ci'], []).append(e)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); ik = IK(); shapes = json.loads(Path(HS).read_text())['shapes']; log = []
    a95 = json.loads(Path(A95).read_text())['a95'] if a.version in ('9b', '9c') else None
    if a.version == '9c':
        PL = np.load(a.hold_pool); pdw, ph = PL[f'{a.pool}_dw'], PL[f'{a.pool}_hand']
    fs = sorted(Path(a.given).glob('*.npy'), key=lambda p: int(p.stem)); si, sn = (int(x) for x in a.shard.split('/'))
    for f in fs[si::sn]:
        ci = int(f.stem); y = np.load(f).astype(np.float32); T = len(y); yi = y.copy(); spans = []
        for e in by.get(ci, []):
            v = e if a.mode == 'given' else dict(e, **e['shuffle']); fe = min(int(round(e['t'] * FPS)), T - 1)
            if e['kind'] == 'aim': h, kind, tg = v['hand'], 'aim', v['target']
            else: h, kind, tg = (0 if v['fingers'][0] > v['fingers'][1] else 1), 'present', None
            if a.version == '9c':                                     # 9c: present with the wrist-height bound; aim keyframe as 9b
                if kind == 'aim': qk, info = ik.solve(yi[fe], h, kind, tg, pen_w=100.0); info['grid'] = 0
                else:
                    for gi, off in enumerate(GRID9B):
                        qk, info = ik.solve(yi[fe], h, kind, tg, pen_w=100.0, present_off=off, zmin=0.15); info['grid'] = gi
                        if info['pen'] <= 1e-6 and info['wz_minus_sz'] >= -0.15 - 1e-3: break
                    if info['wz_minus_sz'] < -0.15 - 1e-3: info['pen'] = max(info['pen'], 1.0); info['height_reject'] = True
            elif a.version == '9b':
                if kind == 'aim': qk, info = ik.solve(yi[fe], h, kind, tg, pen_w=100.0); info['grid'] = 0
                else:
                    for gi, off in enumerate(GRID9B):                 # first collision-free grid point (nominal first)
                        qk, info = ik.solve(yi[fe], h, kind, tg, pen_w=100.0, present_off=off); info['grid'] = gi
                        if info['pen'] <= 1e-6: break
            else: qk, info = ik.solve(yi[fe], h, kind, tg)
            rec = dict(ci=ci, t=e['t'], kind=kind, hand=int(h), **info); log.append(rec)
            if info['pen'] > 1e-6: continue                          # self-penetrating keyframe -> rejected, no insertion (counted)
            if a.version == '9c':
                kd = 'aim' if kind == 'aim' else 'finger'; nh = int(round((HOLD[kd][0] + HOLD[kd][1]) * FPS)) + 1
                if kind == 'aim':
                    i_ = pool_index(e['clip'], e['t'], len(pdw)); dw = np.array(pdw[i_][:nh], np.float64)
                    if int(ph[i_]) != h: dw[:, 1] = -dw[:, 1]                 # mirror the lateral axis for the other hand
                    Q, hi = hold9c(ik, qk, h, tg, dw); rec.update(hi, pool_i=i_)
                else: Q = np.repeat(np.asarray(qk)[None], nh, 0)
                tm = insert9c(yi, Q, h, e['t'], kd, a95)
                if tm: rec.update(tm); spans.append((tm['t_in0'], tm['t_release']))
                continue
            if a.version == '9b':
                tm = insert9b(yi, qk, h, e['t'], 'aim' if kind == 'aim' else 'finger', a95)
                if tm: rec.update(tm); spans.append((tm['t_in0'], tm['t_release']))
                continue
            w, sp = window(T, e['t'], 'aim' if kind == 'aim' else 'finger'); w = w[:, None]; jj = ARM[h]
            yi[:, jj] = (1 - w) * yi[:, jj] + w * qk[jj][None]; spans.append(sp)
        yi = finger_override(yi, tokens_for(by.get(ci, []), a.mode), shapes, ramp=0.3, profile='minjerk').astype(np.float32)
        np.save(out / f.name, yi); np.savez(out / f'{ci}.npz', frames=yi, spans=np.asarray(spans, np.float32).reshape(-1, 2))
        g = Path(a.given).parent / f'gt_{ci}.npy'
        if g.exists() and not (out.parent / g.name).exists(): __import__('shutil').copy(g, out.parent / g.name)
    pen = np.array([x['pen'] for x in log]) if log else np.zeros(0)
    (out / f'KEYFRAMES.{si}of{sn}.json').write_text(json.dumps(dict(n=len(log), collision_share=float((pen > 1e-6).mean()) if len(pen) else 0.,
                                                         res_median=float(np.median([x['res'] for x in log])) if log else None, events=log)))
    print('INSERT_DONE', out, len(log))


def make_inpaint(npz, m, vae, qm, qs, n_lat, seed, dev):
    """bench_ex002 --inpaint-dir: normalised latents of the inserted frames on the insertion spans, cosine-ramped weights (1 latent = 4 frames)."""
    z = np.load(npz); f = z['frames']; spans = z['spans']
    if not len(spans): return None
    Zd = m.lat_mean.shape[-1]; lat = torch.zeros(n_lat, Zd, device=dev); w = torch.zeros(n_lat, device=dev)
    T = len(f); Tp = -(-T // 4) * 4; ff = np.concatenate([f, np.repeat(f[-1:], Tp - T, 0)]) if Tp > T else f
    with torch.no_grad():
        mu, _ = vae.encode(((torch.as_tensor(ff, device=dev) - qm) / qs)[None])
    l = ((mu[0] - m.lat_mean.reshape(-1)) / m.lat_std.reshape(-1)).float(); n = min(len(l), n_lat); lat[:n] = l[:n]
    for t0, t1 in spans:
        i0, i1 = max(0, int(np.floor(t0 * FPS / 4))), min(n_lat, int(np.ceil(t1 * FPS / 4)))
        if i1 <= i0: continue
        k = i1 - i0; r = max(1, int(round(RAMP * FPS / 4))); ww = torch.ones(k, device=dev)
        if k > 2 * r:
            ramp = 0.5 - 0.5 * torch.cos(torch.pi * (torch.arange(r, device=dev) + 0.5) / r); ww[:r] = ramp; ww[k - r:] = ramp.flip(0)
        w[i0:i1] = torch.maximum(w[i0:i1], ww)
    return dict(lat=lat, w=w, seed=seed)


def release(a):
    P = json.loads(Path(a.plan).read_text()); by = {}
    for e in P['events']:
        if not e.get('present'): by.setdefault(e['ci'], []).append(e)
    acc = lambda y, jj: np.linalg.norm(np.diff(y[:, jj], n=2, axis=0), axis=1) * FPS ** 2
    allj = ARM[0] + ARM[1]; ref = np.concatenate([acc(np.load(f), allj) for f in sorted(Path(a.gt_dir).glob('gt_*.npy'))])
    p95 = float(np.percentile(ref, 95)); res = dict(_ref=dict(human_p95=p95, human_p50=float(np.median(ref)), frames=len(ref)))
    for spec in a.specs:
        lab, d = spec.split('=', 1); vals = []; jover = []
        TM = {(x['ci'], round(x['t'], 3)): x for f in Path(d).glob('KEYFRAMES.*.json') for x in json.loads(f.read_text())['events'] if 't_release' in x}
        for f in sorted(Path(d).glob('*.npy'), key=lambda p: int(p.stem)):
            ci = int(f.stem); y = np.load(f); T = len(y)
            for e in by.get(ci, []):
                kind = 'aim' if e['kind'] == 'aim' else 'finger'; tr = e['t'] + HOLD[kind][1] + RAMP; tm = TM.get((ci, round(e['t'], 3)))
                if tm: tr = tm['t_release']                          # 9b: adaptive release end
                h = e['hand'] if kind == 'aim' else (0 if e['fingers'][0] > e['fingers'][1] else 1)
                f0, f1 = max(1, int((tr - 0.1) * FPS)), min(T - 1, int((tr + 0.1) * FPS))
                if f1 - f0 < 3: continue
                vals.append(float(acc(y[f0 - 1:f1 + 1], ARM[h]).max()))
                a_, b_ = HOLD[kind]                                   # arm jerk over the limit on the ramp frames (in + out)
                rr = ((tm['t_in0'], tm['t_hold'][0]), (tm['t_hold'][1], tm['t_release'])) if tm else ((e['t'] - a_ - RAMP, e['t'] - a_), (e['t'] + b_, e['t'] + b_ + RAMP))
                for r0, r1 in rr:
                    g0, g1 = max(0, int(r0 * FPS)), min(T - 3, int(r1 * FPS))
                    if g1 <= g0: continue
                    jk = np.abs(np.diff(y[g0:g1 + 3, ARM[h]], n=3, axis=0)) * FPS ** 3; jover += list((jk > JLIM[None]).any(1))
        grp = lab.rsplit('_s', 1)[0]; res.setdefault(grp, dict(v=[], j=[])); res[grp]['v'] += vals; res[grp]['j'] += jover
    out = {'_ref': res.pop('_ref')}
    for k, vv in res.items():
        v = np.array(vv['v']); out[k] = dict(events=len(v), median=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(v.max()), share_above_human_p95=float((v > p95).mean()),
                                             arm_ramp_jerk_over_limit=float(np.mean(vv['j'])) if vv['j'] else None)
        print(k, json.dumps({kk: round(vv, 3) for kk, vv in out[k].items()}))
    print('REF', out['_ref']); Path(a.out).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    i = sub.add_parser('insert'); i.add_argument('--given', required=True); i.add_argument('--plan', required=True); i.add_argument('--out', required=True)
    i.add_argument('--mode', default='given', choices=['given', 'shuffle']); i.add_argument('--shard', default='0/1'); i.add_argument('--version', default='9', choices=['9', '9b', '9c'])
    i.add_argument('--hold-pool', default=None, help='9c: HOLD_POOL_TRAIN.npz'); i.add_argument('--pool', default='point', choices=['point', 'control'])
    r = sub.add_parser('release'); r.add_argument('--plan', required=True); r.add_argument('--gt-dir', required=True); r.add_argument('--out', required=True); r.add_argument('specs', nargs='+')
    a = ap.parse_args(); dict(insert=insert, release=release)[a.cmd](a)


if __name__ == '__main__':
    main()

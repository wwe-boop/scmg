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
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
FPS = 50; ARM = {0: list(range(4, 11)), 1: list(range(11, 18))}
V = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1'
HS = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002/tokens/HAND_SHAPES.json'
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

    def solve(self, q0, h, kind, target=None, it=20):
        q0 = torch.as_tensor(q0, dtype=torch.float64); q = q0.clone(); idx = ARM[h]
        sgn = None
        def res(x):
            qq = q.clone(); qq[idx] = x; g = self.geo(qq); w, e, s = g['wrist'][h], g['elbow'][h], g['shoulder'][h]
            r = []
            if kind == 'aim':
                t = torch.as_tensor(target, dtype=torch.float64); r.append((w - e) / (w - e).norm() - (t - w) / (t - w).norm())
            else:
                mid = 0.06 if h == 0 else -0.06                         # left hand toward -y (midline), right toward +y
                pt = s + torch.tensor([0.30, -mid, -0.12], dtype=torch.float64); r.append(w - pt)
                n = torch.cross(g['index_base'][h] - w, g['third_base'][h] - w, dim=0); n = n / n.norm().clamp(min=1e-9)
                r.append(n - sgn * torch.tensor([1., 0., 0.], dtype=torch.float64))
            r.append(np.sqrt(LAM) * (x - q0[idx])); r.append(10.0 * self.pen(qq).square().sum().add(1e-12).sqrt().reshape(1))
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
        return qf.numpy().astype(np.float32), dict(res=float(res(x).norm()), pen=pen)


def mj(x): x = np.clip(x, 0, 1); return x ** 3 * (10 - 15 * x + 6 * x * x)


def window(T, t, kind):
    a, b = HOLD[kind]; fr = np.arange(T) / FPS
    return mj((fr - (t - a - RAMP)) / RAMP) * mj(((t + b + RAMP) - fr) / RAMP), (t - a - RAMP, t + b + RAMP)


def insert(a):
    from ex002_ctl_run import finger_override, tokens_for
    P = json.loads(Path(a.plan).read_text()); by = {}
    for e in P['events']:
        if not e.get('present'): by.setdefault(e['ci'], []).append(e)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); ik = IK(); shapes = json.loads(Path(HS).read_text())['shapes']; log = []
    fs = sorted(Path(a.given).glob('*.npy'), key=lambda p: int(p.stem)); si, sn = (int(x) for x in a.shard.split('/'))
    for f in fs[si::sn]:
        ci = int(f.stem); y = np.load(f).astype(np.float32); T = len(y); yi = y.copy(); spans = []
        for e in by.get(ci, []):
            v = e if a.mode == 'given' else dict(e, **e['shuffle']); fe = min(int(round(e['t'] * FPS)), T - 1)
            if e['kind'] == 'aim': h, kind, tg = v['hand'], 'aim', v['target']
            else: h, kind, tg = (0 if v['fingers'][0] > v['fingers'][1] else 1), 'present', None
            qk, info = ik.solve(yi[fe], h, kind, tg)
            log.append(dict(ci=ci, t=e['t'], kind=kind, hand=int(h), **info))
            if info['pen'] > 1e-6: continue                          # self-penetrating keyframe -> rejected, no insertion (counted)
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
        for f in sorted(Path(d).glob('*.npy'), key=lambda p: int(p.stem)):
            ci = int(f.stem); y = np.load(f); T = len(y)
            for e in by.get(ci, []):
                kind = 'aim' if e['kind'] == 'aim' else 'finger'; tr = e['t'] + HOLD[kind][1] + RAMP
                h = e['hand'] if kind == 'aim' else (0 if e['fingers'][0] > e['fingers'][1] else 1)
                f0, f1 = max(1, int((tr - 0.1) * FPS)), min(T - 1, int((tr + 0.1) * FPS))
                if f1 - f0 < 3: continue
                vals.append(float(acc(y[f0 - 1:f1 + 1], ARM[h]).max()))
                a_, b_ = HOLD[kind]                                   # arm jerk over the limit on the ramp frames (in + out)
                for r0, r1 in ((e['t'] - a_ - RAMP, e['t'] - a_), (e['t'] + b_, e['t'] + b_ + RAMP)):
                    g0, g1 = max(0, int(r0 * FPS)), min(T - 3, int(r1 * FPS))
                    if g1 <= g0: continue
                    jk = np.abs(np.diff(y[g0:g1 + 3, ARM[h]], n=3, axis=0)) * FPS ** 3; jover += list((jk > JLIM[None]).any(1))
        grp = lab.rsplit('_s', 1)[0]; res.setdefault(grp, dict(v=[], j=[])); res[grp]['v'] += vals; res[grp]['j'] += jover
    out = {'_ref': res.pop('_ref')}
    for k, vv in res.items():
        v = np.array(vv['v']); out[k] = dict(events=len(v), median=float(np.median(v)), p95=float(np.percentile(v, 95)), share_above_human_p95=float((v > p95).mean()),
                                             arm_ramp_jerk_over_limit=float(np.mean(vv['j'])) if vv['j'] else None)
        print(k, json.dumps({kk: round(vv, 3) for kk, vv in out[k].items()}))
    print('REF', out['_ref']); Path(a.out).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    i = sub.add_parser('insert'); i.add_argument('--given', required=True); i.add_argument('--plan', required=True); i.add_argument('--out', required=True)
    i.add_argument('--mode', default='given', choices=['given', 'shuffle']); i.add_argument('--shard', default='0/1')
    r = sub.add_parser('release'); r.add_argument('--plan', required=True); r.add_argument('--gt-dir', required=True); r.add_argument('--out', required=True); r.add_argument('specs', nargs='+')
    a = ap.parse_args(); dict(insert=insert, release=release)[a.cmd](a)


if __name__ == '__main__':
    main()

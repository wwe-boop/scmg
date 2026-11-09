"""Unit tests for scmg/safety_proj.py (design v6 C3, 11-05).

    PYTHONPATH=<repo>:<osqp/ruckig/mujoco deps>:$OFF/src python scripts/tests/test_safety_proj.py <gt_dir> <stems> [n_safe=12]

1 helpers: hard-coded joint order == echo_g G1_JOINT_NAMES; differentiable encoder == official map2latent_feature;
  G1Collision.contacts on the margin model agrees with the checker model; near() gradient matches finite differences.
2 already-safe GT clips (D-spk GT that passes every constraint incl. the d_safe clearance) -> change ~0 (max |dq|, RMSE and
  frames changed by > 1e-3 rad printed per layer; asserted max <= 2e-4 rad for none / clip / clip_coll / proj_l2 / proj_perc;
  cbf_qp and ruckig change safe motion by design and are only printed).
3 constructed violating clips (GT + position-limit overshoot + an arm driven into a contacting pose + a 0.4 rad step):
  the INDEPENDENT checker scripts/echog_g1/executability.py (--collision-stride 1, fixed exclusion table) finds 0
  position / velocity / collision violations on clip_coll, cbf_qp, proj_l2, proj_perc; projection layers additionally have
  0 acceleration / jerk violations; any violation would have to lie in fallback frames (printed).
4 causality: replacing every input frame after chunk k leaves outputs of chunks 0..k bit-identical (all layers, k = 1, 4, 9).
5 per-chunk solve time (CPU, one thread) per layer: mean / p50 / p95 / p99 / max.
"""
import json, os, subprocess, sys, tempfile, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
from scmg import safety_proj as S

torch.set_num_threads(1)
ok = True


def check(name, cond, info=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True); ok &= bool(cond)


gt_dir, stems_f = sys.argv[1:3]; n_safe = int(sys.argv[3]) if len(sys.argv) > 3 else 12
stems = [s.strip() for s in open(stems_f) if s.strip()]
lim = S.Limits(); coll = S.G1Collision(); enc = S.FGDEncoder(device='cpu')
load = lambda s: torch.load(f'{gt_dir}/{s}.pt', map_location='cpu', weights_only=False)
joints = lambda s: load(s)['robot_repr'].double().numpy()[:, 10:]
times = {n: [] for n in S.LAYERS}

# ---------------------------------------------------------------------------------------------------- 1 helpers
check('joint order == echo_g G1_JOINT_NAMES', S.check_joint_order() is True)
from echo_g.evaluation import g1_motion_cls as GM
rr = load(stems[0])['robot_repr'].double(); rr = rr[:min(96, len(rr) // 32 * 32)]
ref = GM.map2latent_feature(enc.net, enc.cfg, rr.numpy(), 'cpu')
mine = enc(torch.as_tensor(rr[:, 10:])).detach().numpy()
check('FGDEncoder == official map2latent_feature', np.abs(mine - ref).max() < 1e-4 * max(1, np.abs(ref).max()),
      f'max |diff| {np.abs(mine - ref).max():.2e}, |ref| {np.abs(ref).max():.2f}')
rng = np.random.default_rng(0)
agree = n_coll = 0; Q = []
for s in stems[:40]:
    q = joints(s)
    Q.append(q[::7])
Q = np.concatenate(Q)
for q in Q:
    a = coll.contacts(q)
    coll.near(q, 1.0)                                   # poses the margin model
    b = {tuple(sorted((int(coll.m1.geom_bodyid[c.geom1]), int(coll.m1.geom_bodyid[c.geom2]))))
         for c in (coll.d1.contact[i] for i in range(coll.d1.ncon)) if c.dist < 0}
    agree += a == b; n_coll += bool(a)
check('contact set: margin model (dist < 0) == checker model', agree == len(Q), f'{agree}/{len(Q)} poses, {n_coll} in contact')
# finite-difference check of the linearised distance on poses with a near pair
errs = []
for q in Q[:400]:
    rows = coll.near(q, 0.01)
    if not rows: continue
    g, d = rows[0]
    dq = rng.normal(size=S.J) * 2e-3
    rows2 = coll.near(q + dq, 0.05)
    pred = d + g @ dq
    cand = [d2 for g2, d2 in rows2]
    if cand: errs.append(min(abs(c - pred) for c in cand) / max(abs(g @ dq), 1e-6))
    if len(errs) >= 30: break
check('near(): first-order distance prediction (relative error median < 0.3)', len(errs) > 5 and np.median(errs) < 0.3,
      f'n={len(errs)} median rel err {np.median(errs) if errs else float("nan"):.3f}')


# ---------------------------------------------------------------------------------------------- 2 safe GT ~ unchanged
def all_safe(q):
    if not S.pos_ok(q, lim, 0): return False
    if not S.diff_ok(np.zeros((0, S.J)), q, lim, 0)[0]: return False
    return all(not coll.near(x, 2e-3) for x in q)


safe = []
for s in stems:
    q = joints(s)
    if len(q) <= 400 and all_safe(q): safe.append(s)
    if len(safe) >= n_safe: break
print(f'{len(safe)} fully safe GT clips (of the first {stems.index(safe[-1]) + 1 if safe else 0} D-spk stems)')
mx = {n: 0.0 for n in S.LAYERS}; se = {n: 0.0 for n in S.LAYERS}; big = {n: 0 for n in S.LAYERS}; nfr = 0
for s in safe:
    q = joints(s); nfr += len(q)
    for n in S.LAYERS:
        L = S.make_layer(n, lim, coll, enc)
        y, st = S.run_stream(L, q, 10); times[n] += st['times']
        mx[n] = max(mx[n], float(np.abs(y - q).max())); se[n] += float(((y - q) ** 2).sum())
        big[n] += int((np.abs(y - q).max(1) > 1e-3).sum())
rm = {n: np.sqrt(se[n] / (nfr * S.J)) for n in S.LAYERS}
print(f'safe GT ({nfr} frames): ' + '; '.join(f'{n} max {mx[n]:.2e} rmse {rm[n]:.2e} frames>1e-3 {big[n]}' for n in S.LAYERS))
for n in ('none', 'clip', 'clip_coll', 'proj_l2', 'proj_perc'):
    check(f'safe GT unchanged by {n}', mx[n] <= 2e-4, f'max |dq| {mx[n]:.2e} rad, rmse {rm[n]:.2e} rad')
print(f'(not asserted) cbf_qp {mx["cbf_qp"]:.3e} rad: discrete CBF slows approaches to limits / contacts even when safe; '
      f'ruckig {mx["ruckig"]:.3e} rad: jerk-limited tracking lags the target')


# ---------------------------------------------------------------------------------------------- 3 violating clips
def contact_pose(q0, rng):
    """an upper-body pose near q0 that is in contact (random search, fixed seed)."""
    for _ in range(20000):
        q = q0.copy()
        q[15:29] += rng.normal(size=14) * 1.2
        q = np.clip(q, lim.lo, lim.hi)
        if coll.collides(q): return q
    raise RuntimeError('no contact pose found')


def make_bad(s, seed):
    rng = np.random.default_rng(seed)
    q = joints(s)[:240].copy(); T = len(q)
    qc = contact_pose(q[60], rng)
    w = np.zeros(T); a, b = 40, 100
    w[a:b] = np.sin(np.linspace(0, np.pi, b - a)) ** 2
    q = q * (1 - w[:, None]) + qc[None] * w[:, None]                  # drive the arms into contact and back
    t = np.arange(T)
    q[:, 13] += 0.9 * np.exp(-((t - 150) / 12.0) ** 2)                 # waist_roll beyond +0.52
    q[:, 16] -= 2.5 * np.exp(-((t - 190) / 10.0) ** 2)                 # left_shoulder_roll below its lower limit
    q[170:, 18] += 0.4                                                 # left elbow 0.4 rad step at frame 170
    q[:, 22:] += rng.normal(size=(T, 7)) * 0.01                        # small noise on the right arm
    return q


tmp = tempfile.mkdtemp(prefix='safeproj-test-', dir=os.environ.get('TMPDIR', '/tmp'))
bad_stems = [s for s in stems if len(joints(s)) >= 240][:4]
raw_dir = os.path.join(tmp, 'raw'); os.makedirs(raw_dir)
fb = {}; derr = {}
for i, s in enumerate(bad_stems):
    q = make_bad(s, i); x = load(s); rr = x['robot_repr'][:len(q)].double().clone(); rr[:, 10:] = torch.from_numpy(q)
    torch.save(dict(robot_repr=rr.float(), fps=30.0), f'{raw_dir}/{s}.pt')
    for n in S.LAYERS:
        L = S.make_layer(n, lim, coll, enc)
        y, st = S.run_stream(L, rr[:, 10:].numpy(), 10); times[n] += st['times']
        os.makedirs(f'{tmp}/{n}', exist_ok=True)
        r2 = rr.clone(); r2[:, 10:] = torch.from_numpy(y)
        torch.save(dict(robot_repr=r2, fps=30.0), f'{tmp}/{n}/{s}.pt')
        fb[n] = fb.get(n, 0) + st['fallback_frames']
        E = np.diff(y, 2, axis=0) * 900, np.diff(y, 3, axis=0) * 27000
        derr[n] = derr.get(n, 0) + int((np.abs(E[0]) > lim.amax + 1e-6).any(1).sum() + (np.abs(E[1]) > lim.jmax + 1e-6).any(1).sum())
open(f'{tmp}/stems.txt', 'w').write('\n'.join(bad_stems) + '\n')
out = f'{tmp}/exec.json'
cmd = [sys.executable, os.path.join(ROOT, 'scripts/echog_g1/executability.py'), '--ref-dir', raw_dir, '--stems', f'{tmp}/stems.txt',
       '--urdf', S.URDF, '--mjcf', S.MJCF, '--collision-stride', '1', '--exclusion-file', S.EXCLUSION, '--out', out] + \
      [f'{n}={tmp}/{n}' for n in S.LAYERS]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout.strip())
if r.returncode: print(r.stderr); check('independent checker ran', False)
res = json.load(open(out))
check('constructed clips do violate (raw = checker GT column)', res['GT']['pos'] > 0 and res['GT']['coll'] > 0 and res['none']['any'] > 0,
      f"raw pos {res['GT']['pos']:.3f} coll {res['GT']['coll']:.3f}")
for n in ('clip_coll', 'cbf_qp', 'proj_l2', 'proj_perc'):
    v = res[n]
    check(f'{n}: independent checker 0 violations (pos/vel/coll)', v['any'] == 0,
          f"pos {v['pos']:.4f} vel {v['vel']:.4f} coll {v['coll']:.4f}; fallback frames {fb[n]}; acc/jerk frames {derr[n]}")
for n in ('proj_l2', 'proj_perc'):
    check(f'{n}: 0 acceleration / jerk violations', derr[n] == 0, f'{derr[n]} frames')
print('(baselines) ' + '; '.join(f"{n}: any {res[n]['any']:.4f} coll {res[n]['coll']:.4f} acc/jerk frames {derr[n]} fallback {fb[n]}"
                                 for n in ('none', 'clip', 'ruckig', 'clip_coll', 'cbf_qp')))

# ---------------------------------------------------------------------------------------------- 4 causality
q = torch.load(f'{raw_dir}/{bad_stems[0]}.pt', weights_only=False)['robot_repr'].double().numpy()[:, 10:]
other = joints(bad_stems[1])[:len(q)]
K = 10
for n in S.LAYERS:
    L = S.make_layer(n, lim, coll, enc)
    y0, _ = S.run_stream(L, q, K)
    same = True
    for k in (1, 4, 9):
        q2 = q.copy(); q2[(k + 1) * K:] = other[(k + 1) * K:] + rng.normal(size=other[(k + 1) * K:].shape) * 0.05
        y2, _ = S.run_stream(L, q2, K)
        same &= np.array_equal(y0[:(k + 1) * K], y2[:(k + 1) * K])
        changed_later = not np.array_equal(y0[(k + 1) * K:], y2[(k + 1) * K:])
        same &= changed_later
    check(f'causality {n}: chunks 0..k bit-identical after replacing the future (k = 1, 4, 9)', same)

# ---------------------------------------------------------------------------------------------- 5 timing
print('per-chunk solve time (ms, CPU 1 thread, K=10 frames = 333 ms of motion):')
for n, t in times.items():
    t = np.array(t) * 1e3
    print(f'  {n:10s} n={len(t):5d} mean {t.mean():7.2f} p50 {np.percentile(t, 50):7.2f} p95 {np.percentile(t, 95):7.2f} '
          f'p99 {np.percentile(t, 99):7.2f} max {t.max():7.2f}')
check('proj_perc p99 chunk time < chunk length (333 ms)', np.percentile(np.array(times['proj_perc']), 99) < 1 / 3)
print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)

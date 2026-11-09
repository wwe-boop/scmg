"""11-08 EX002 duplex ladder L2 (docs/criteria/EX002_DUPLEX_LADDER_1107.md REVISION): rest-keyframe insertion at the interruption.
From f_i the waist (q34 0-1) and both arms (4-17) follow a quintic (minimum-jerk when the end state is at rest) from the stream's state
at f_i to the speaker's rest pose (human_ref R_sil, 34 joints) with zero end velocity / acceleration, and then HOLD the rest pose to
the end of the window (no release back to the generator until the next turn); head (2-3) and fingers (18-33) keep the continuation.
Ramp code reused from B's module 9b (exp/ctl-port ec51a7b scripts/ex002/ex002_keyframe.py: state, quintic, the 9b duration rule
T = clip(sqrt(5.77 max_j |dq_j| / a95_j), 0.3, 0.8) over the 14 arm joints, a95 = tokens/ARM_ACC_P95.json); the start state is taken
from the 5-frame moving average of the executed stream around f_i (B: the r3 decoder has 4-frame acceleration spikes).
The executed prefix (frames < f_i) is the method's own uninterrupted output U, as in retract_post_ex002.py.
    python scripts/ex002/rest_ramp_ex002.py --in-dir Y --prefix-from U --out-dir O --interrupts I.json --human H [--keyframe-code K]"""
import argparse, json, sys
from pathlib import Path
import numpy as np

KF = '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/scmg/releases/ctlport-ec51a7b/scripts/ex002'
UP = [0, 1] + list(range(4, 18))
A95 = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002/tokens/ARM_ACC_P95.json'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--in-dir', required=True); ap.add_argument('--prefix-from', required=True)
    ap.add_argument('--out-dir', required=True); ap.add_argument('--interrupts', required=True); ap.add_argument('--human', required=True)
    ap.add_argument('--keyframe-code', default=KF)
    ap.add_argument('--version', default='l2', choices=['l2', 'l2d'], help='l2d = design D3 (EX002_DUPLEX_LADDER_1107.md L2d); default l2 unchanged')
    ap.add_argument('--control', default='none', choices=['none', 'a', 'b', 'c', 'd'], help='l2d controls (a)-(d)')
    ap.add_argument('--only-ci', default='', help='comma-separated recordings (quick lane)')
    a = ap.parse_args()
    if a.version == 'l2d': return l2d(a)
    sys.path.insert(0, a.keyframe_code)
    from ex002_keyframe import state, quintic, FPS          # B's 9b ramp primitives (not forked)
    a95 = np.asarray(json.loads(Path(A95).read_text())['a95'], np.float64)               # arm joints 4..17
    H = json.loads(Path(a.human if a.human.endswith('.json') else a.human + '.json').read_text())
    rest = np.asarray(H['rest']['R_sil']['joints'], np.float64); assert len(rest) == 34, 'R_sil must be the 34-joint EX002 rest pose'
    ints = json.loads(Path(a.interrupts).read_text())['interrupts']; out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    n = 0; Ts = []
    for r in ints:
        name = f"{r['ci']}@{r['item']}@{r['frac']:.1f}.npy"; fp = Path(a.in_dir) / name
        if not fp.exists(): continue
        f = r['f_i']; y = np.load(fp).astype(np.float64); U = np.load(Path(a.prefix_from) / f"{r['ci']}.npy").astype(np.float64)
        z = np.concatenate([U[:f], y[f:]]); T = len(z)
        sm = z.copy()                       # 5-frame moving average at f-1, f, f+1 (the frames state() differentiates)
        for t_ in (f - 1, f, f + 1):
            if 2 <= t_ < T - 2: sm[t_] = z[t_ - 2:t_ + 3].mean(0)
        _, v0, a0 = state(sm[:, UP], f); p0 = z[f, UP]       # exact position at f (no step), smoothed velocity / acceleration
        dq = np.abs(rest[4:18] - z[f, 4:18]); Tr = float(np.clip(np.sqrt(5.77 * np.max(dq / a95)), 0.3, 0.8)); nr = int(round(Tr * FPS))
        o = z.copy(); zz = np.zeros(len(UP)); e = min(T, f + nr)
        o[f:e, UP] = quintic(p0, v0, a0, rest[UP], zz, zz, Tr, nr)[:e - f]
        o[e:, UP] = rest[UP][None]
        np.save(out / name, o.astype(np.float32)); n += 1; Ts.append(Tr)
    print('REST_RAMP_DONE', n, 'T median %.2f s' % float(np.median(Ts)) if Ts else '')


def l2d(a):
    """design D3: conditioned end pose (k = 5 NN in the half-A end-pose set), conditioned duration (isotonic fit on half A, R^2 rule),
    brake-then-ramp start; controls (a) uniform E_A draw, (b) fixed T 1.22 s, (c) L2's C2 quintic start, (d) random-time train pose."""
    import glob, hashlib
    sys.path.insert(0, a.keyframe_code); sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ex002_keyframe import state, quintic, FPS
    import yield_v4_ex002 as Y
    X = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/ex002-interrupt-1107'
    a95 = np.asarray(json.loads(Path(A95).read_text())['a95'], np.float64)
    sys.path.insert(0, str(Path.home() / 'projects/gesture/scmg-wt-safeproj')); sys.path.insert(0, str(Path.home() / 'bulk/pydeps/safeproj'))
    from scmg.safety_ex002_3f import EX002RMLimits
    aw = 0.5 * EX002RMLimits().amax[:2]; amax = np.concatenate([aw, a95])             # UP order: waist 0-1, arms 4-17
    v75 = 0.3971923709098323                                                           # yield_v4 it2 quiet threshold (A-picked)
    EF, EP, TA, DA = [], [], [], []
    for _, q, off in Y.tail_windows(X + '/tails/seg_halfA'):
        p = Y.pts(q[off - 1:off + Y.W]); e = (p[-25:, :4] - p[-25:, 4:5]).mean(0)
        EF.append(e.reshape(-1)); EP.append(q[off + Y.W - 25:off + Y.W].mean(0))
        w = p[:, :2]; sp = np.linalg.norm(np.diff(w, axis=0), axis=-1).max(1) * FPS; run = 0; k0 = None
        for kk in range(len(sp)):
            run = run + 1 if sp[kk] < v75 else 0
            if run >= 25: k0 = kk - 24; break
        w0 = p[1, :2] - p[1, 4]; d = float(np.linalg.norm(e[:2] - w0, axis=-1).mean())
        if k0 is not None: TA.append(k0 / FPS); DA.append(d)
    EF, EP, TA, DA = np.array(EF), np.array(EP), np.array(TA), np.array(DA)
    o_ = np.argsort(DA); xs, ys = DA[o_], TA[o_]; vals = []                  # pool-adjacent-violators (increasing)
    for y_ in ys:
        vals.append([y_, 1])
        while len(vals) > 1 and vals[-2][0] > vals[-1][0]:
            v2, n2 = vals.pop(); v1, n1 = vals.pop(); vals.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    fit = np.concatenate([[v] * n for v, n in vals]); res_ = ys - fit
    r2 = float(1 - (res_ ** 2).sum() / ((ys - ys.mean()) ** 2).sum()); cond_T = r2 >= 0.1
    def T_of(d, rng):
        if not cond_T: return float(np.clip(rng.choice(TA), 0.5, 2.0))
        return float(np.clip(np.interp(d, xs, fit) + rng.choice(res_), 0.5, 2.0))
    print(f'L2D half A: {len(EF)} end poses, {len(TA)} durations, isotonic R^2 {r2:.3f} -> {"conditioned" if cond_T else "unconditional"} duration', flush=True)
    if a.control == 'd':
        V = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1'
        M = Path.home() / 'bulk/cpfs-mirror/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1'; V = str(M) if M.exists() else V
        items = json.loads((Path(V) / 'train.json').read_text()); tr = set(json.loads((Path(V) / 'SPLIT.json').read_text())['train_ids'])
        items = [it for it in items if it['id'] in tr]; TQ = np.load(Path(V) / 'train_q.npy', mmap_mode='r')
    ints = json.loads(Path(a.interrupts).read_text())['interrupts']; out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    only = set(int(c) for c in a.only_ci.split(',')) if a.only_ci else None; n = 0; log = []
    for r in ints:
        if only is not None and r['ci'] not in only: continue
        name = f"{r['ci']}@{r['item']}@{r['frac']:.1f}.npy"; fp = Path(a.in_dir) / name
        if not fp.exists(): continue
        f = r['f_i']; y = np.load(fp).astype(np.float64); U = np.load(Path(a.prefix_from) / f"{r['ci']}.npy").astype(np.float64)
        z = np.concatenate([U[:f], y[f:]]); T = len(z)
        rng = np.random.default_rng(int(hashlib.sha1(f"{r['ci']}:{f}".encode()).hexdigest()[:8], 16))
        pf = Y.pts(z[f:f + 1]); ef = (pf[0, :4] - pf[0, 4]).reshape(-1)
        dd = np.linalg.norm(EF - ef[None], axis=1)
        if a.control == 'a': i_ = int(rng.integers(len(EF)))
        else: nn = np.argsort(dd)[:5]; i_ = int(nn[rng.integers(5)])
        end = EP[i_].copy()
        if a.control == 'd':
            it = items[int(rng.integers(len(items)))]; end = np.asarray(TQ[it['offset'] + int(rng.integers(it['frames']))], np.float64)
        de = float(np.linalg.norm(EF[i_].reshape(4, 3)[:2] - ef.reshape(4, 3)[:2], axis=-1).mean())
        Tt = 1.22 if a.control == 'b' else T_of(de, rng)
        sm = z.copy()
        for t_ in (f - 1, f, f + 1):
            if 2 <= t_ < T - 2: sm[t_] = z[t_ - 2:t_ + 3].mean(0)
        _, v0, a0 = state(sm[:, UP], f); p0 = z[f, UP]; o = z.copy(); zz = np.zeros(len(UP)); tgt = end[UP]
        if a.control == 'c':
            nr = int(round(Tt * FPS)); e_ = min(T, f + nr); o[f:e_, UP] = quintic(p0, v0, a0, tgt, zz, zz, Tt, nr)[:e_ - f]; o[e_:, UP] = tgt[None]; tb = 0.0
        else:
            tb = float(np.clip(np.max(np.pi * np.abs(v0) / (2 * amax)), 0.06, 0.5)); nb = int(round(tb * FPS))
            tt = np.arange(nb)[:, None] / FPS
            br = p0[None] + v0[None] * (tt / 2 + tb / (2 * np.pi) * np.sin(np.pi * tt / tb))
            pb = p0 + v0 * tb / 2; Tq = max(Tt - tb, 0.3); nq = int(round(Tq * FPS))
            seg = np.concatenate([br, quintic(pb, zz, zz, tgt, zz, zz, Tq, nq)], 0); e_ = min(T, f + len(seg))
            o[f:e_, UP] = seg[:e_ - f]; o[e_:, UP] = tgt[None]
        np.save(out / name, o.astype(np.float32)); n += 1; log.append(dict(ci=r['ci'], f=f, T=Tt, t_brake=tb, end_idx=i_, d=de))
    (out / 'L2D_LOG.json').write_text(json.dumps(dict(r2=r2, conditioned_duration=cond_T, n_end=len(EF), n_dur=len(TA), control=a.control, events=log)))
    print('L2D_DONE', a.control, n, 'T median %.2f' % float(np.median([e['T'] for e in log])) if log else '')


if __name__ == '__main__':
    main()

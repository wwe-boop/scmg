"""11-08 EX002 yield metric v4 (coordinator: yield3d "return to the per-joint median pose R_sil" fails its own validity rule -- humans
rarely reach R_sil after speaking). Calibrated on the A-half human tails ONLY (crc32(recording) % 2 == 0); the B half and every model
row are only scored with the frozen parameters.
Window: the interruption / speech-offset frame f, frames [f, f + 107) (2.13 s), 50 Hz, exact FK (icommon_ex002.fk: wrists).
  activity   : the per-frame hand speed s(t) = max over the two wrists; a QUIET run = >= 25 frames (0.5 s) with s < v75; after the first
               quiet run starts, no new stroke: s never exceeds v95 for the rest of the window. v75 / v95 = p75 / p95 of s over the
               A-half tail windows.
  withdrawal : end pose e = the mean over the last 25 frames of [wrist L, wrist R, elbow L, elbow R] positions relative to the hip
               (torso frame not needed: the EX002 base is fixed), 12-D, metres; pass iff its k-NN distance (k = 3, Euclidean) to the A-half
               end poses <= D90 = the leave-one-out p90 of the A-half end poses' own k-NN distances.
  yield v4   = activity AND withdrawal; latency = start of the first quiet run.
Validity gates (written before any model row is scored): B-half human tails pass >= 85 %; speech continuations (dev GT at the
interruption points while the speaker keeps talking, FRONT, B1) pass <= 15 %.
    python yield_v4_ex002.py calib --out PARAMS.json        (A half)
    python yield_v4_ex002.py audit --params PARAMS.json --out AUDIT.json
    python yield_v4_ex002.py score --params PARAMS.json --out S.json label=<dir>:<prefix dir> ...   (executed streams, as eval_interrupt)"""
import argparse, json, sys, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import icommon_ex002 as I
B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; X = B + '/ex002-interrupt-1107'
W, Q = I.WIN, 25
_K = {}


def pts(q):
    """[T, 5, 3]: wrist L, wrist R, elbow L, elbow R, hip (scmg.kinematics bodies, as arm_pose_ex002)."""
    import arm_pose_ex002 as A
    P = A.fkp(q); return np.stack([P['wr'][0], P['wr'][1], P['el'][0], P['el'][1], P['hip'][0]], 1)


def feats(q, f):
    p = pts(q[f - 1:f + W]); w = p[:, :2]; s = np.linalg.norm(np.diff(w, axis=0), axis=-1).max(1) * I.FPS     # [W] hand speed
    e = (p[-Q:, :4] - p[-Q:, 4:5]).mean(0).reshape(-1)
    e0 = (p[:3, :4] - p[:3, 4:5]).mean(0).reshape(-1)            # pose at the event (frames f-1 .. f+1)
    return s, e, e0


def quiet(s, v75, v95):
    run = 0
    for k in range(len(s)):
        run = run + 1 if s[k] < v75 else 0
        if run >= Q:
            k0 = k - Q + 1
            return (k0 / I.FPS) if s[k0:].max() <= v95 else None
    return None


def knn(E, e, k=3, skip=None):
    d = np.linalg.norm(E - e[None], axis=1)
    if skip is not None: d[skip] = np.inf
    return float(np.sort(d)[:k].mean())


def tail_windows(seg_dir):
    out = []
    for fn in sorted(glob.glob(seg_dir + '/*.npz')):
        z = np.load(fn); q = z['q34'].astype(np.float64); off = int(z['off'])
        if off >= 1 and len(q) >= off + W: out.append((Path(fn).stem, q, off))
    return out


def calib(a):
    T = tail_windows(X + '/tails/seg_halfA'); S, E = [], []
    F = [feats(q, off) for _, q, off in T]; S = np.concatenate([f_[0] for f_ in F]); E = np.stack([f_[1] for f_ in F])
    loo = np.array([knn(E, E[i], skip=i) for i in range(len(E))]); d90 = float(np.percentile(loo, 90))
    E0 = [f_[2] for f_ in F]; need = np.array([knn(E, e0) > d90 for e0 in E0])
    # iteration 2 (A half only): quiet / stroke percentiles = the smallest pair on the grid whose activity pass on the A-half
    # NEEDED events (pose at the event outside the end-pose support) is >= 0.90
    grid = [(qp, sp) for qp in (75, 80, 85, 90, 95) for sp in (95, 97, 99, 100)]; pick = None; tried = []
    for qp, sp in grid:
        v1, v2 = float(np.percentile(S, qp)), float(np.percentile(S, sp))
        r = float(np.mean([quiet(F[i][0], v1, v2) is not None for i in np.nonzero(need)[0]])) if need.any() else 0.; tried.append((qp, sp, r))
        if pick is None and r >= 0.90: pick = (qp, sp, v1, v2)
    if pick is None: qp, sp, r = max(tried, key=lambda x: x[2]); pick = (qp, sp, float(np.percentile(S, qp)), float(np.percentile(S, sp)))
    par = dict(iteration=2, quiet_pct=pick[0], stroke_pct=pick[1], v75=pick[2], v95=pick[3], D90=d90, k=3, quiet_frames=Q, end_frames=Q, n_A=len(E),
               E_A=E.tolist(), source='A-half tails (seg_halfA)', grid=tried, A_needed_share=float(need.mean()))
    par['A_activity_pass_needed'] = float(np.mean([quiet(F[i][0], pick[2], pick[3]) is not None for i in np.nonzero(need)[0]]))
    par['A_withdrawal_pass_loo'] = float(np.mean(loo <= d90))
    Path(a.out).write_text(json.dumps(par)); print('CALIB', json.dumps({k: v for k, v in par.items() if k != 'E_A'}))


def judge(par, q, f, skip=None):
    E = np.asarray(par['E_A']); s, e, e0 = feats(q, f); t = quiet(s, par['v75'], par['v95']); d = knn(E, e, par['k'], skip)
    need = knn(E, e0, par['k'], skip) > par['D90']                 # iteration 2: the pose at the event is outside the support
    return dict(activity=t is not None, withdrawal=d <= par['D90'], v4=(t is not None) and d <= par['D90'], latency=t, knn=d, needed=bool(need))


def summ(rs):
    if not rs: return dict(n=0)
    nd = [r for r in rs if r['needed']]
    return dict(n=len(rs), n_needed=len(nd), v4_needed=float(np.mean([r['v4'] for r in nd])) if nd else None, v4=float(np.mean([r['v4'] for r in rs])), activity=float(np.mean([r['activity'] for r in rs])),
                withdrawal=float(np.mean([r['withdrawal'] for r in rs])),
                latency_median=float(np.median([r['latency'] for r in rs if r['v4']])) if any(r['v4'] for r in rs) else None)


def audit(a):
    par = json.loads(Path(a.params).read_text()); res = {}
    for half in ('A', 'B'):
        T = tail_windows(X + f'/tails/seg_half{half}')
        res[f'tails_{half}'] = summ([judge(par, q, off, skip=i if half == 'A' else None) for i, (_, q, off) in enumerate(T)])
        Hh = json.load(open(X + f'/ref-half{half}/human_ref_ex002.json')); rP = I.fk(np.array(Hh['rest']['R_sil']['joints'])[None])[0]
        y3 = [I.yield3d(I.fk(q), off, rP) for _, q, off in T]                    # the old primary metric on the same windows
        res[f'tails_{half}']['yield3d_all'] = float(np.mean([t is not None for t, nd in y3]))
        res[f'tails_{half}']['yield3d_needed'] = float(np.mean([t is not None for t, nd in y3 if nd])) if any(nd for _, nd in y3) else None
        print(half, json.dumps(res[f'tails_{half}']), flush=True)
    ints = json.loads(Path(X + '/interrupts_dev_ex002.json').read_text())['interrupts']; E = B + '/ex002-sfscope-1108/enc-s0'
    H = json.load(open(X + '/ref-halfB/human_ref_ex002.json')); rest = np.array(H['rest']['R_sil']['joints']); rest_z = np.array(H['rest']['R_sil']['hand_z'])
    gts = {ci: np.load(f'{E}/psync/gt_{ci}.npy').astype(np.float64) for ci in sorted({r['ci'] for r in ints})}
    cand = []                                                     # FRONT exactly as eval_interrupt_ex002.py
    for ci_, q_ in sorted(gts.items()):
        P_ = I.fk(q_[::5]); dz = np.abs(P_[:, :, 2] - rest_z[None]).max(1); dd = np.linalg.norm(P_ - I.fk(rest[None])[0][None], axis=-1).min(1)
        for t_ in np.nonzero(dz <= 0.05)[0]: cand.append((float(dd[t_]), ci_, int(t_ * 5)))
    cand.sort(reverse=True); front = gts[cand[0][1]][cand[0][2]].copy()
    rows = dict(GT_speech=[], FRONT=[], B1=[])
    for r in ints:
        f = r['f_i']; g = gts[r['ci']]
        if len(g) < f + W: continue
        U = np.load(f"{E}/psync/s0/{r['ci']}.npy").astype(np.float64)
        rows['GT_speech'].append(judge(par, g, f)); rows['B1'].append(judge(par, I.freeze_after(U, f), f))
        rows['FRONT'].append(judge(par, I.retract_after(U, f, front, 1.8), f))
    for k, v in rows.items(): res[k] = summ(v); print(k, json.dumps(res[k]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


def score(a):
    par = json.loads(Path(a.params).read_text()); ints = json.loads(Path(X + '/interrupts_dev_ex002.json').read_text())['interrupts']; res = {}
    for spec in a.rows:
        lab, rest_ = spec.split('=', 1); d, pref = rest_.split(':'); rs = []
        for r in ints:
            fn = Path(d) / f"{r['ci']}@{r['item']}@{r['frac']:.1f}.npy"
            if not fn.exists(): continue
            f = r['f_i']; y = np.load(fn).astype(np.float64); U = np.load(f"{pref}/{r['ci']}.npy").astype(np.float64)
            z = np.concatenate([U[:f], y[f:]])[:f + W]
            if len(z) < f + W: continue
            rs.append(judge(par, z, f))
        res[lab] = summ(rs); print(lab, json.dumps(res[lab]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('calib'); p.add_argument('--out', required=True)
    p = sp.add_parser('audit'); p.add_argument('--params', required=True); p.add_argument('--out', required=True)
    p = sp.add_parser('score'); p.add_argument('--params', required=True); p.add_argument('--out', required=True); p.add_argument('rows', nargs='+')
    a = ap.parse_args(); dict(calib=calib, audit=audit, score=score)[a.cmd](a)

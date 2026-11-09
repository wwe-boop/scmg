"""11-08 EX002 arm-pose / interruption-dynamics diagnostics (coordinator review of qual-1108/ex002-enc; numbers only, no fix).
FK = scmg/kinematics.py (sphere-proxy bodies, as lively_floor_ex002.py): shoulder / elbow / wrist of each arm, waist (q34 joint 1 body,
'hip'), head (joint 3 body). Torso up axis u = unit(head - hip); humerus angle = angle(elbow - shoulder, -u) (0 deg = hanging, 90 deg =
horizontal); lateral = signed distance along the shoulder-to-shoulder axis from the shoulders' midpoint.
  base  : GT / sibag-s2 / enc-s0, offline + psync, dev 43 recordings (models: per_clip seeds 0-2 pooled):
          humerus angle median / p90 / share > 45 / > 70 deg; arm joints 1-2 angles median / p90; wrist |lateral| p90; per-joint
          amplitude (per-clip std, mean over clips) ratio vs GT for the 14 arm joints; akimbo-like hold share (both wrists within
          [-0.05, +0.15] m of the hip height and both elbows > 0.05 m outside their shoulder, held > 1 s).
  int   : per interruption row (raw / 3c-4b / + ISC executed streams): time from t_i to the first approach of the wrists to the rest
          wrists (mean wrist distance to rest starts decreasing and keeps decreasing for >= 5 frames); end pose 1.5-2.0 s after t_i:
          wrist height above the hip, elbow angle (deg, 180 = straight), vs the half-B human tails (1.5-2.0 s after the offset) and
          the rest pose R_sil of the reference.
    python scripts/ex002/arm_pose_ex002.py base|int --out O.json"""
import argparse, json, sys, glob
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
NPZ = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/sphere_proxy_vox4e.npz'
B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; X = B + '/ex002-interrupt-1107'
J = dict(sh=(4, 11), el=(7, 14), wr=(10, 17), hip=(1, 1), head=(3, 3)); _K = {}
ARMJ = list(range(4, 18))


def fkp(q):
    if not _K:
        from scmg.kinematics import Kinematics
        from scmg.common import P
        z = np.load(P(NPZ)); bj = np.asarray(z['body_joint']); b = lambda j: int(np.nonzero(bj == j)[0][0])
        _K.update(k=Kinematics(P(NPZ), 'cpu'), idx={k: (b(v[0]), b(v[1])) for k, v in J.items()})
    with torch.no_grad(): _, p = _K['k'].fk(torch.as_tensor(np.asarray(q, np.float32)))
    p = p.double().numpy(); return {k: (p[:, i0], p[:, i1]) for k, (i0, i1) in _K['idx'].items()}


def geo(q):
    P = fkp(q); hip, head = P['hip'][0], P['head'][0]; u = head - hip; u /= np.linalg.norm(u, axis=-1, keepdims=True)
    shL, shR = P['sh']; mid = (shL + shR) / 2; ax = shL - shR; ax /= np.linalg.norm(ax, axis=-1, keepdims=True)
    out = {}
    for s, h in (('L', 0), ('R', 1)):
        sh, el, wr = P['sh'][h], P['el'][h], P['wr'][h]; hu = el - sh; fo = wr - el
        ang = np.degrees(np.arccos(np.clip(np.sum(hu * -u, -1) / np.linalg.norm(hu, axis=-1), -1, 1)))
        elb = np.degrees(np.arccos(np.clip(np.sum(-hu * fo, -1) / (np.linalg.norm(hu, axis=-1) * np.linalg.norm(fo, axis=-1)), -1, 1)))
        sgn = 1. if s == 'L' else -1.
        rad = (wr - hip) - np.sum((wr - hip) * u, -1, keepdims=True) * u          # wrist offset from the torso axis (horizontal)
        out[s] = dict(ext=np.linalg.norm(rad, axis=-1), hum=ang, elbow=elb, wlat=sgn * np.sum((wr - mid) * ax, -1), elat=sgn * np.sum((el - mid) * ax, -1),
                      shlat=sgn * np.sum((sh - mid) * ax, -1), wz=np.sum((wr - hip) * u, -1), wpos=wr)
    return out


def runs(m, n):
    o = np.zeros_like(m); i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]: j += 1
            if j - i > n: o[i:j] = True
            i = j
        else: i += 1
    return o


def base(a):
    systems = dict(GT=None, sibag_s2=B + '/ex002-lively-1108/diag-sibag-s2', enc_s0=B + '/ex002-sfscope-1108/enc-s0')
    res = {}
    for P in ('offline', 'psync'):
        gtd = systems['sibag_s2'] + f'/{P}'; gts = sorted(glob.glob(gtd + '/gt_*.npy'), key=lambda f: int(Path(f).stem[3:]))
        amp_gt = None
        for name, d in systems.items():
            files = gts if name == 'GT' else [f for s in (0, 1, 2) for f in sorted(glob.glob(f'{d}/{P}/s{s}/[0-9]*.npy'))]
            hum, j12, wl, ak, fr, amps = [], [], [], 0, 0, []
            for f in files:
                q = np.load(f).astype(np.float64); g = geo(q)
                for s in 'LR': hum.append(g[s]['hum']); wl.append(np.abs(g[s]['wlat']))
                j12.append(q[:, [4, 5, 11, 12]]); amps.append(q[:, ARMJ].std(0))
                akm = np.ones(len(q), bool)
                for s in 'LR': akm &= (g[s]['wz'] > -0.05) & (g[s]['wz'] < 0.15) & (g[s]['elat'] > g[s]['shlat'] + 0.05)
                ak += int(runs(akm, 50).sum()); fr += len(q)
            hum = np.concatenate(hum); wl = np.concatenate(wl); j12 = np.concatenate(j12); amp = np.mean(amps, 0)
            if name == 'GT': amp_gt = amp
            res[f'{name}|{P}'] = dict(files=len(files), humerus_deg=dict(median=float(np.median(hum)), p90=float(np.percentile(hum, 90)),
                                      gt45=float((hum > 45).mean()), gt70=float((hum > 70).mean())),
                                      arm_j1_j2_deg=dict(median=np.degrees(np.median(j12, 0)).round(1).tolist(), p90=np.degrees(np.percentile(np.abs(j12), 90, 0)).round(1).tolist(),
                                                         cols='L j1, L j2, R j1, R j2 (q34 4, 5, 11, 12; |angle| for p90)'),
                                      wrist_lateral_p90_m=float(np.percentile(wl, 90)), akimbo_hold_share=ak / max(fr, 1),
                                      arm_amp_ratio_vs_GT=(amp / amp_gt).round(3).tolist() if amp_gt is not None else None)
            print(P, name, json.dumps({k: v for k, v in res[f'{name}|{P}'].items() if k != 'arm_amp_ratio_vs_GT'}), flush=True)
            if amp_gt is not None and name != 'GT': print('   amp ratio arm joints 4..17:', (amp / amp_gt).round(2).tolist(), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


def approach_time(g, rest_w, f):
    d = np.mean([np.linalg.norm(g[s]['wpos'] - rest_w[h], axis=-1) for h, s in enumerate('LR')], 0)
    dd = np.diff(d[f:])
    for k in range(len(dd) - 5):
        if (dd[k:k + 5] < 0).all(): return k / 50.
    return None


def outward(g, f):
    """time from t_i until the wrists stop moving away from the torso axis (mean wrist distance from the axis decreases for >= 5 frames),
    and the extra extension (m) reached before that."""
    e = np.mean([g[s]['ext'] for s in 'LR'], 0)[f:]; dd = np.diff(e)
    for k in range(len(dd) - 5):
        if (dd[k:k + 5] < 0).all(): return k / 50., float(e[k] - e[0])
    return None, float(e.max() - e[0])


def intr(a):
    import icommon_ex002 as I
    H = json.load(open(X + '/ref-halfB/human_ref_ex002.json')); rest = np.array(H['rest']['R_sil']['joints'])
    gr = geo(rest[None]); rest_w = [gr['L']['wpos'][0], gr['R']['wpos'][0]]
    ints = json.loads(Path(X + '/interrupts_dev_ex002.json').read_text())['interrupts']
    E = B + '/ex002-sfscope-1108/enc-s0'; PUr = E + '/psync/s0'; PUs = '/home/xr-25b7/bulk/causal-train-1105/runs/ex002-proj4b-enc/psync/s0'
    rows = dict(B3_raw=(E + '/int_psync_s0', PUr), B3_safe=(X + '/eval-enc-s0/SAFE_B3', PUs), B3_isc=(X + '/eval-enc-s0/SAFE_ISC_B3', PUs),
                R0causal_raw=(X + '/eval-enc-s0-halfB/ENC_r0causal', PUr), R0causal_isc=(X + '/eval-enc-s0-halfB/SAFE_ISC_R0causal', PUs),
                Funnel_raw=(X + '/eval-enc-s0-halfB/ENC_funnel', PUr), Funnel_isc=(X + '/eval-enc-s0-halfB/SAFE_ISC_Funnel', PUs),
                L2diag_raw=(X + '/eval-enc-s0-halfB/L2diag', PUr), L2diag_isc=(X + '/eval-enc-s0-halfB/SAFE_ISC_L2diag', PUs))
    res = dict(rest_R_sil=dict(wz=[float(gr[s]['wz'][0]) for s in 'LR'], elbow=[float(gr[s]['elbow'][0]) for s in 'LR'],
                               humerus=[float(gr[s]['hum'][0]) for s in 'LR']))
    for lab, (d, pu) in rows.items():
        ta, wz, el, hu, to, te = [], [], [], [], [], []
        for r in ints:
            fn = Path(d) / f"{r['ci']}@{r['item']}@{r['frac']:.1f}.npy"
            if not fn.exists(): continue
            f = r['f_i']; y = np.load(fn).astype(np.float64); U = np.load(f"{pu}/{r['ci']}.npy").astype(np.float64)
            z = np.concatenate([U[:f], y[f:]])[:f + I.WIN]; g = geo(z); t = approach_time(g, rest_w, f); ta.append(np.nan if t is None else t)
            t2, ex = outward(g, f); to.append(np.nan if t2 is None else t2); te.append(ex)
            w = slice(f + 75, f + 100)
            for s in 'LR': wz.append(g[s]['wz'][w].mean()); el.append(g[s]['elbow'][w].mean()); hu.append(g[s]['hum'][w].mean())
        ta = np.array(ta)
        to = np.array(to); te = np.array(te)
        res[lab] = dict(n=len(ta), t_outward_stop_s=dict(median=float(np.nanmedian(to)), p75=float(np.nanpercentile(to, 75)), never=float(np.isnan(to).mean())),
                        outward_extra_cm=dict(median=float(100 * np.median(te)), p75=float(100 * np.percentile(te, 75))), t_first_approach_s=dict(median=float(np.nanmedian(ta)), p75=float(np.nanpercentile(ta, 75)), never=float(np.isnan(ta).mean())),
                        end_wrist_above_hip_m=dict(median=float(np.median(wz)), p10=float(np.percentile(wz, 10)), p90=float(np.percentile(wz, 90))),
                        end_elbow_deg=dict(median=float(np.median(el)), p10=float(np.percentile(el, 10)), p90=float(np.percentile(el, 90))),
                        end_humerus_deg=float(np.median(hu)))
        print(lab, json.dumps(res[lab]), flush=True)
    wz, el, hu = [], [], []                                                     # half-B human tails, 1.5-2.0 s after the offset
    for f in sorted(glob.glob(X + '/tails/seg_halfB/*.npz')):
        z = np.load(f); q = z['q34'].astype(np.float64); off = int(z['off'])
        if len(q) < off + 100: continue
        g = geo(q); w = slice(off + 75, off + 100)
        for s in 'LR': wz.append(g[s]['wz'][w].mean()); el.append(g[s]['elbow'][w].mean()); hu.append(g[s]['hum'][w].mean())
    res['GT_tails_halfB'] = dict(n=len(wz) // 2, end_wrist_above_hip_m=dict(median=float(np.median(wz)), p10=float(np.percentile(wz, 10)), p90=float(np.percentile(wz, 90))),
                                 end_elbow_deg=dict(median=float(np.median(el)), p10=float(np.percentile(el, 10)), p90=float(np.percentile(el, 90))),
                                 end_humerus_deg=float(np.median(hu)))
    print('GT_tails_halfB', json.dumps(res['GT_tails_halfB']), 'rest', json.dumps(res['rest_R_sil']), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['base', 'int']); ap.add_argument('--out', required=True)
    a = ap.parse_args(); base(a) if a.cmd == 'base' else intr(a)

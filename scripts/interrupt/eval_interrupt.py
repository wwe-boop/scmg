"""Metrics of the G1 interruption benchmark (11-05). Run with PYTHONPATH=<ECHO-G official>/src (+ mujoco).

Per interrupt (clip, f_i) and method, the evaluated motion is the method's 30 fps robot_repr up to f_i - 1 followed by its
post-interrupt behaviour on [f_i, f_i + 64) (2.13 s):
  B1  freeze      : truncated at f_i, last frame held (an immediate stop)                     -- any offline / streaming output
  B2  retract     : truncated at f_i, waist + arms approach the rest pose R_sil critically damped (velocity at f_i - 1
                    matched, 95 % settled at T_r = human median yield time, human_ref.json), legs / base held
                    (variants B2fast: T_r = 1 s; B2mj: min-jerk from zero velocity; B2q: p/v/a-matched quintic)
  B0  (reference) : no interruption, the uninterrupted continuation (clips lasting >= 2.13 s past t_i only)
  B3  continue    : the method's own continuation (infer_interrupt.py: audio ends at t_i, unspoken words withdrawn)
Metrics (hands = wrist_yaw links, pose-only official FK, pelvis frame):
  react   [s, lower]  : reaction latency = first time after t_i at which a hand leaves the method's OWN uninterrupted
                        continuation by > 2 cm (when the robot visibly starts reacting to the interruption; censored = window)
  yield   [s, lower]  : yield time = time until both hands are at rest height (z <= z_rest + 5 cm) and stay there to the end
                        of the window; only interrupts whose hands are above rest height at t_i ("mid-gesture");
                        not reached -> counted as 2.13 s; yield_rate [%, higher] = share reached within the window
  jerk    [m/s^3, lower]: peak hand jerk in [t_i - 0.1 s, t_i + 2.13 s); abrupt_rate [%, lower] = share above the human
                        post-speech P95 (human_ref.json)
  rkd     [lower]     : retraction-kinematics distance to the human RETRACTION windows (post-speech windows whose hands are
                        above rest height at the offset and come down; rkd_all = to all post-speech windows): mean over 5 speaker-normalised
                        descriptors (icommon.kin_features: log speed ratio 0-1 s / 1-2 s after vs 1 s before, hand height
                        change, still share, log peak-jerk ratio) of the 1-D Wasserstein distance in human-std units
  fgd     [lower]     : Frechet distance (official G1 FGD encoder, 64-frame windows [f_i, f_i + 64)) to the human post-speech
                        windows (GMR-retargeted BEAT2 silences, human_ref_windows.npz)
  unsafe  [%, lower]  : share of windows with any URDF position / velocity-limit violation or MuJoCo self-collision
                        (executability.py contract: structural mesh overlaps = pairs touching in >= 50 % of GT frames excluded)
  energy  [ratio]     : mean hand speed in [t_i + 1.07 s, t_i + 2.13 s) / mean in [t_i - 1 s, t_i) (human post-speech median 0.57)
    python scripts/interrupt/eval_interrupt.py --interrupts I.json --human human_ref --fgd-ckpt F --urdf U --mjcf M
        --gt-dir G --out res.json  label=B1:<pred dir> label=B2:<pred dir> label=B3:<interrupt dir>:<uninterrupted dir> ...
"""
import argparse, json, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import icommon as I
from echo_g.evaluation import g1_motion_cls as E
from echo_g.evaluation.robot_repr import G1_JOINT_NAMES

ap = argparse.ArgumentParser()
ap.add_argument('--interrupts', required=True); ap.add_argument('--human', required=True); ap.add_argument('--fgd-ckpt', required=True)
ap.add_argument('--urdf', required=True); ap.add_argument('--mjcf', required=True); ap.add_argument('--gt-dir', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--collision-stride', type=int, default=2)
ap.add_argument('--rest', default='R_sil'); ap.add_argument('--tr', type=float, default=None, help='override T_r (s)')
ap.add_argument('--per-item', default=None, help='jsonl of per-interrupt values')
ap.add_argument('--every', type=int, default=1, help='every n-th clip of the interrupt list (sorted stems)')
ap.add_argument('methods', nargs='+')
a = ap.parse_args()
H = json.loads(Path(a.human + '.json').read_text()); _hw = np.load(a.human + '_windows.npz')
HW = _hw['rr'].astype(np.float64); HC = _hw['rr_ctx'].astype(np.float64)
H_kin_all = np.stack([I.kin_features(I.fk(w), 30) for w in HC])
H_ret = _hw['retract'].astype(bool); H_kin = H_kin_all[H_ret]          # main reference: the human RETRACTION windows
T_r = a.tr if a.tr is not None else H['T_r']; P95 = H['peak_jerk']['p95']
rest_j = {k: np.array(v['joints']) for k, v in H['rest'].items()}; rest_z = {k: np.array(v['hand_z']) for k, v in H['rest'].items()}
ints = json.loads(Path(a.interrupts).read_text())['interrupts'][: a.limit or None]
if a.every > 1:                                   # every n-th clip (the dev screening subsets of infer_retract.py --every)
    _ks = set(sorted({r['stem'] for r in ints})[::a.every]); ints = [r for r in ints if r['stem'] in _ks]

# ---- safety contract (as scripts/echog_g1/executability.py)
import mujoco
lim = {j.get('name'): j.find('limit').attrib for j in ET.parse(a.urdf).getroot().findall('joint') if j.get('type') == 'revolute'}
names = [x if x.endswith('_joint') else x + '_joint' for x in G1_JOINT_NAMES]
lo = np.array([float(lim[n]['lower']) for n in names]); hi = np.array([float(lim[n]['upper']) for n in names])
vmax = np.array([float(lim[n]['velocity']) for n in names])
mm = mujoco.MjModel.from_xml_path(a.mjcf)
for g in range(mm.ngeom):
    on = mm.geom_bodyid[g] != 0; mm.geom_contype[g] = int(on); mm.geom_conaffinity[g] = int(on)
md = mujoco.MjData(mm)
qadr = np.array([mm.jnt_qposadr[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
free = [j for j in range(mm.njnt) if mm.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]


def contacts(q):
    md.qpos[:] = 0
    for j in free: md.qpos[mm.jnt_qposadr[j] + 3] = 1.0
    md.qpos[qadr] = q; mujoco.mj_forward(mm, md); out = set()
    for i in range(md.ncon):
        c = md.contact[i]
        if c.dist < 0: out.add(tuple(sorted((mm.geom_bodyid[c.geom1], mm.geom_bodyid[c.geom2]))))
    return out


stems = sorted({r['stem'] for r in ints})
cnt, tot = {}, 0
for s in stems:
    q = I.load_rr(f'{a.gt_dir}/{s}.pt')[:, 10:]
    for t in range(0, len(q), 3):
        tot += 1
        for p in contacts(q[t]): cnt[p] = cnt.get(p, 0) + 1
structural = {p for p, c in cnt.items() if c >= 0.5 * tot}


def unsafe(rr, f):
    q = rr[f - 1:f + I.WIN, 10:]
    v = np.gradient(q, 1 / I.FPS, axis=0)[1:]; q = q[1:]
    pv = ((q < lo) | (q > hi)).any(); vv = (np.abs(v) > vmax).any()
    cv = any(contacts(q[t]) - structural for t in range(0, len(q), a.collision_stride))
    return dict(pos=bool(pv), vel=bool(vv), coll=bool(cv), any=bool(pv or vv or cv))


net, cfg = E.load_g1_ae(a.fgd_ckpt, 'cpu', 30.0)
lat = lambda W: E.map2latent_feature(net, cfg, W, 'cpu')
H_lat = np.concatenate([lat(w) for w in HW])


def post(kind, d, ref, s, r):
    """(evaluated motion [f_i + 64, 39], uninterrupted continuation or None)"""
    f = r['f_i']
    if kind == 'B0':                    # no interruption: the uninterrupted continuation (only clips that last 2.13 s past t_i)
        base = I.load_rr(f'{d}/{s}.pt'); return base[:f + I.WIN], None
    if kind in B2K or kind == 'B1':
        base = I.load_rr(f'{d}/{s}.pt')
        if len(base) < f: raise ValueError(f'{d}/{s}.pt shorter than f_i')
        if kind == 'B1': return I.freeze_after(base, f), base
        mode, tr = B2K[kind]
        return I.retract_after(base, f, rest_j['R_sil'], tr, mode), base
    mo = I.load_rr(f"{d}/{s}@{r['frac']:.1f}.pt")[:f + I.WIN]
    return mo, (I.load_rr(f'{ref}/{s}.pt') if ref else None)


# B2 variants: B2 = critically damped, T_r (reported); B2fast = damped 1.0 s; B2mj = min-jerk T_r; B2q = p/v/a-matched quintic T_r
B2K = dict(B2=('damped', T_r), B2fast=('damped', 1.0), B2mj=('minjerk', T_r), B2q=('quintic', T_r))
res = {}; per = open(a.per_item, 'w') if a.per_item else None
for spec in a.methods:
    lab, rest_ = spec.split('=', 1); parts = rest_.split(':'); kind, d = parts[0], parts[1]; ref = parts[2] if len(parts) > 2 else None
    rows, lats, kins = [], [], []
    for r in ints:
        s, f = r['stem'], r['f_i']
        try: mo, U = post(kind, d, ref, s, r)
        except FileNotFoundError: continue
        if len(mo) < f + I.WIN: continue
        P = I.fk(mo)
        row = dict(stem=s, frac=r['frac'])
        if U is not None:
            n = min(len(U), f + I.WIN); PU = I.fk(U[:n])
            dev_ = np.linalg.norm(P[f:n, list(I.HANDS)] - PU[f:n, list(I.HANDS)], axis=-1).max(1) > 0.02
            row['react'] = float(np.argmax(dev_) / I.FPS) if dev_.any() else None; row['react_cens'] = (n - f) / I.FPS
        for rk in rest_z:
            ty, need = I.yield_time(P, f, rest_z[rk]); row[f'need_{rk}'] = need; row[f'yield_{rk}'] = ty
        row['jerk'] = I.peak_jerk(P, f - I.PRE, f + I.WIN); row['acc'] = I.peak_acc(P, f - I.PRE, f + I.WIN)
        v = I.hand_speed(P).mean(1); row['energy'] = float(v[f + 32:f + I.WIN].mean() / max(v[max(0, f - 30):f].mean(), 1e-6))
        row.update({f'unsafe_{k}': x for k, x in unsafe(mo, f).items()})
        # the same contract on the 2.13 s BEFORE t_i (identical motion for B1 / B2 / B3 of one base output): separates the
        # violations the interruption handling introduces (unsafe_new: post window violates, pre window clean) from the
        # model's own (unsafe_pre)
        fp = max(1, f - I.WIN); pre = unsafe(mo[:f], fp) if f - fp >= 4 else dict(any=False)
        row['unsafe_pre'] = bool(pre['any']); row['unsafe_new'] = bool(row['unsafe_any'] and not pre['any'])
        kf = I.kin_features(P, f); row['kin'] = kf.tolist()
        rows.append(row); lats.append(lat(mo[f:f + I.WIN])); kins.append(kf)
        if per: per.write(json.dumps(dict(row, method=lab)) + '\n')
    if not rows: print(lab, 'no rows'); continue
    W = I.WIN / I.FPS
    def ysum(rk):
        sel = [x for x in rows if x[f'need_{rk}']]
        t = [x[f'yield_{rk}'] if x[f'yield_{rk}'] is not None else W for x in sel]
        return dict(n=len(sel), yield_s=float(np.mean(t)) if t else None, yield_rate=float(np.mean([x[f'yield_{rk}'] is not None for x in sel])) if sel else None)
    out = dict(n=len(rows), **{f'{k}_{rk}': v for rk in rest_z for k, v in ysum(rk).items()})
    rc = [x['react'] if x.get('react') is not None else x.get('react_cens') for x in rows if 'react_cens' in x]
    out['react_s'] = float(np.mean(rc)) if rc else None
    out['react_median'] = float(np.median(rc)) if rc else None
    jk = np.array([x['jerk'] for x in rows]); out.update(jerk_median=float(np.median(jk)), jerk_p95=float(np.percentile(jk, 95)),
                                                       abrupt_rate=float((jk > P95).mean()), acc_median=float(np.median([x['acc'] for x in rows])))
    out['energy_median'] = float(np.median([x['energy'] for x in rows]))
    for k in ('pos', 'vel', 'coll', 'any', 'pre', 'new'): out[f'unsafe_{k}'] = float(np.mean([x[f'unsafe_{k}'] for x in rows]))
    out['fgd'] = E.FGD.frechet_distance(np.concatenate(lats), H_lat)
    out['rkd'], out['rkd_per_feature'] = I.kin_distance(np.stack(kins), H_kin)
    out['rkd_all'] = I.kin_distance(np.stack(kins), H_kin_all)[0]
    out['kin_median'] = np.median(np.stack(kins), 0).tolist()
    for fr in sorted({x['frac'] for x in rows}):
        sel = [x for x in rows if x['frac'] == fr and x['need_R_sil']]
        out[f'yield_s@{fr}'] = float(np.mean([x['yield_R_sil'] if x['yield_R_sil'] is not None else W for x in sel])) if sel else None
        out[f'abrupt@{fr}'] = float(np.mean([x['jerk'] > P95 for x in rows if x['frac'] == fr]))
    res[lab] = out
    print(f"{lab:14s} n {out['n']} | react {out['react_s'] if out['react_s'] is None else round(out['react_s'], 3)} | yield {out['yield_s_R_sil']:.2f}s "
          f"rate {100 * out['yield_rate_R_sil']:.1f}% (R_gt {out['yield_s_R_gt']:.2f}s {100 * out['yield_rate_R_gt']:.1f}%) | jerk med {out['jerk_median']:.0f} "
          f"abrupt {100 * out['abrupt_rate']:.1f}% | RKD {out['rkd']:.3f} (all {out['rkd_all']:.3f}) FGD {out['fgd']:.3f} | unsafe {100 * out['unsafe_any']:.1f}% (vel {100 * out['unsafe_vel']:.1f} coll "
          f"{100 * out['unsafe_coll']:.1f} pos {100 * out['unsafe_pos']:.1f}; pre {100 * out['unsafe_pre']:.1f} new {100 * out['unsafe_new']:.1f}) | energy {out['energy_median']:.2f}", flush=True)
res['_meta'] = dict(T_r=T_r, human_jerk_p95=P95, structural=[(mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, x), mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, y)) for x, y in structural],
                    n_interrupts=len(ints), human_windows=len(HW))
# FGD noise floor: human windows split in two halves (even / odd index)
la = [lat(w) for w in HW]
res['_meta']['fgd_human_half_split'] = E.FGD.frechet_distance(np.concatenate(la[0::2]), np.concatenate(la[1::2]))
res['_meta']['rkd_human_half_split'] = I.kin_distance(H_kin[0::2], H_kin[1::2])[0]
res['_meta']['rkd_all_human_half_split'] = I.kin_distance(H_kin_all[0::2], H_kin_all[1::2])[0]
res['_meta']['rkd_all_vs_retract'] = I.kin_distance(H_kin_all, H_kin)[0]
res['_meta']['n_human_retract'] = int(H_ret.sum())
res['_meta']['human_kin_median'] = np.median(H_kin, 0).tolist(); res['_meta']['human_kin_median_all'] = np.median(H_kin_all, 0).tolist()
res['_meta']['kin_names'] = list(I.KIN_NAMES)
print('RKD human half-split floor', round(res['_meta']['rkd_human_half_split'], 3), '(all', round(res['_meta']['rkd_all_human_half_split'], 3),
      ') | all-vs-retract', round(res['_meta']['rkd_all_vs_retract'], 3), '| n retract', int(H_ret.sum()), '| human kin median', np.round(np.median(H_kin, 0), 3).tolist())
print('FGD human half-split floor', round(res['_meta']['fgd_human_half_split'], 3), '| T_r', T_r, '| jerk P95', round(P95, 1))
Path(a.out).write_text(json.dumps(res, indent=1))

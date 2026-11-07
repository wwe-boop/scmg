"""EX002 interruption metrics (port of scripts/interrupt/eval_interrupt.py; criteria docs/criteria/EX002_INTERRUPT_1107.md).
Motion files: 50 Hz q34 rad .npy. Methods (label=kind:dir[:ref]):
  B1:<no_int dir>   freeze at f_i            B2:<no_int dir>  waist + arms critically damped to R_sil (T_r = human median)
  B3:<dir>:<no_int dir>  the method's continuation <ci>@<item>@<frac>.npy, its own uninterrupted output <ci>.npy as reference
  GT:<gt dir>       the dev GT itself (gt_<ci>.npy; sanity: no interruption happened, the human keeps talking)
  GTMIS:<gt dir>    mismatched GT: another recording's GT window spliced at f_i (sanity: react / v3 near chance)
Per interruption: react (s), v3 ratio, yield (R_sil height + 5 cm), peak jerk, kin descriptors (RKD vs the human retraction windows),
unsafe = META joint limits | limits34 velocity | collision: primary "retargeter WBC clearance (ours)" (release06 EX002.clearance,
every frame); second column "MuJoCo scene, structural-contact-limited" (scene.xml contacts minus parent-child links minus pairs in
>= 50 % of dev GT frames). RKD-post vs all human post-speech windows (not comparable with G1's retraction-window RKD); RKD-retract
only if >= 30 retraction windows. Sanity rows: RKD-post human half-split floor, GT, mismatched GT.
    PYTHONPATH=<repo>:<mujoco> python eval_interrupt_ex002.py --interrupts I.json --human H --gt-dir G --out R.json label=kind:dir[:ref] ..."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import icommon_ex002 as I

ap = argparse.ArgumentParser()
ap.add_argument('--interrupts', required=True); ap.add_argument('--human', required=True); ap.add_argument('--gt-dir', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--per-item', default=None); ap.add_argument('--collision-stride', type=int, default=1)
ap.add_argument('--tr', type=float, default=None, help='B2 settle time T_r override (s); default: the human_ref median')
ap.add_argument('--prefix', default='executed', choices=['executed', 'generated'],
                help="B3 frames before f_i: 'executed' = the method's own uninterrupted output (what was already played; the chunk "
                     "restart re-decodes the last latent before the boundary, up to 3 frames before f_i), 'generated' = the file as written")
ap.add_argument('--view', default='/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1')
ap.add_argument('--scene', default='/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/ex002-rm003-v4-20261020/model-r1/scene.xml')
ap.add_argument('methods', nargs='+')
a = ap.parse_args()
H = json.loads(Path(a.human + '.json').read_text()); hw = np.load(a.human + '_windows.npz')
HK_all = np.stack([I.kin_features(I.fk(w), I.SEC) for w in hw['q_ctx'].astype(np.float64)]); HK_ret = HK_all[hw['retract'].astype(bool)]
T_r = H['T_r'] if a.tr is None else a.tr; rest = np.array(H['rest']['R_sil']['joints']); rest_z = np.array(H['rest']['R_sil']['hand_z'])
rest_P = I.fk(rest[None])[0]                                    # [2, 3] human rest wrist positions (primary 3D yield)
ints = json.loads(Path(a.interrupts).read_text())['interrupts']
meta = json.loads((Path(a.view) / 'META.json').read_text()); ex = meta['ex002']
lo, hi = np.asarray(ex['lower'], np.float64), np.asarray(ex['upper'], np.float64)
from scmg.safety import limits34
vmax = np.asarray(limits34(meta['lower'], meta['upper'], margin=0.0)['v'], np.float64)

import mujoco
mm = mujoco.MjModel.from_xml_path(a.scene); md = mujoco.MjData(mm)
L37 = ex['layout37'] if isinstance(ex['layout37'], list) else eval(ex['layout37']); li = ex['learn_index']
qadr = np.array([mm.jnt_qposadr[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in L37])


def to37(q):
    o = np.zeros(37); o[0] = 0.32; o[li] = q
    for m_, s_ in ((22, 21), (31, 30)): o[m_] = 1.389831 * o[s_] + 0.072772    # mimic x_1_4 from x_1_3 (META restore rule)
    return o


def contacts(q):
    md.qpos[:] = 0; md.qpos[qadr] = to37(q); mujoco.mj_forward(mm, md); out = set()
    for i in range(md.ncon):
        c = md.contact[i]
        if c.dist < 0: out.add(tuple(sorted((mm.geom_bodyid[c.geom1], mm.geom_bodyid[c.geom2]))))
    return out


gts = {ci: np.load(f'{a.gt_dir}/gt_{ci}.npy').astype(np.float64) for ci in sorted({r['ci'] for r in ints})}
# column 2 (diagnostic, "MuJoCo scene, structural-contact-limited"): scene.xml contacts minus adjacent links (parent-child bodies)
# minus pairs present in >= 50 % of the dev GT frames (sampled every 10th frame)
adjacent = {tuple(sorted((b, int(mm.body_parentid[b])))) for b in range(1, mm.nbody)}
cnt, tot = {}, 0
for q in gts.values():
    for t in range(0, len(q), 10):
        tot += 1
        for p in contacts(q[t]) - adjacent: cnt[p] = cnt.get(p, 0) + 1
structural = {p for p, c in cnt.items() if c >= 0.5 * tot}
# column 1 (primary, "retargeter WBC clearance (ours)"): release06 smplx2ex002_3f.robot.EX002.clearance, every frame
import os
os.environ.setdefault('MUJOCO_GL', 'disable')
sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture-dlc-20260928/smplx2ex002-r2-20261022/candidates/codex-dlc-release06-beat-nearoff')
from smplx2ex002_3f import robot as RB
if list(ex['names34'] if isinstance(ex['names34'], list) else eval(ex['names34'])) != RB.CONTROLLED: raise ValueError('names34 != retargeter CONTROLLED')
WBC = RB.EX002()


# FRONT sanity control (11-08): a fixed pose whose wrists are at the human rest height (+-5 cm) but far in front of / beside the rest
# wrist positions: chosen deterministically from the dev GT frames (every 5th) with both wrists within 5 cm of rest height,
# maximising the smaller wrist distance to rest_P. Height-only yield must pass it, the 3D yield must fail it.
_cand = []
for ci_, q_ in sorted(gts.items()):
    P_ = I.fk(q_[::5]); dz = np.abs(P_[:, :, 2] - rest_z[None]).max(1); dd = np.linalg.norm(P_ - rest_P[None], axis=-1).min(1)
    for t_ in np.nonzero(dz <= 0.05)[0]: _cand.append((float(dd[t_]), ci_, int(t_ * 5)))
_cand.sort(reverse=True); FRONT_D, FRONT_CI, FRONT_T = _cand[0]; front = gts[FRONT_CI][FRONT_T].copy()


def unsafe(q, f):
    w = q[f - 1:f + I.WIN]; v = np.gradient(w, 1 / I.FPS, axis=0)[1:]; w = w[1:]
    pv = bool(((w < lo - 1e-6) | (w > hi + 1e-6)).any()); vv = bool((np.abs(v) > vmax).any())
    cv = any(not WBC.clearance(w[t])[0] for t in range(0, len(w), a.collision_stride))                       # primary
    cs = any(contacts(w[t]) - adjacent - structural for t in range(0, len(w), a.collision_stride))             # column 2
    return dict(pos=pv, vel=vv, coll=bool(cv), any=bool(pv or vv or cv), coll_scene=bool(cs), any_scene=bool(pv or vv or cs))


def motion(kind, d, ref, r):
    f = r['f_i']; ci = r['ci']
    if kind in ('B1', 'B2', 'FRONT'):
        base = np.load(f'{d}/{ci}.npy').astype(np.float64)
        if kind == 'FRONT': return I.retract_after(base, f, front, T_r), base   # sanity control: settles at the front pose
        return (I.freeze_after(base, f) if kind == 'B1' else I.retract_after(base, f, rest, T_r)), base
    if kind == 'GT': g = gts[ci]; return g[:f + I.WIN], None
    if kind == 'GTMIS':                                 # another recording's GT window spliced at f_i
        others = [k for k in gts if k != ci]; o = gts[others[(ci * 7 + int(r['frac'] * 10)) % len(others)]]
        f2 = min(f, len(o) - I.WIN - 1); g = gts[ci][:f].copy()
        return np.concatenate([g, o[f2:f2 + I.WIN] - o[f2] + g[-1]]), gts[ci]
    y = np.load(f"{d}/{ci}@{r['item']}@{r['frac']:.1f}.npy").astype(np.float64)
    U = np.load(f'{ref}/{ci}.npy').astype(np.float64) if ref else None
    if a.prefix == 'executed':
        if U is None: raise ValueError('--prefix executed needs the uninterrupted reference dir (B3:<dir>:<ref>)')
        y = np.concatenate([U[:f], y[f:]])
    return y[:f + I.WIN], U


res = {}; per = open(a.per_item, 'w') if a.per_item else None
for spec in a.methods:
    lab, rest_ = spec.split('=', 1); parts = rest_.split(':'); kind, d = parts[0], parts[1]; ref = parts[2] if len(parts) > 2 else None
    rows, kins = [], []
    for r in ints:
        try: mo, U = motion(kind, d, ref, r)
        except FileNotFoundError: continue
        f = r['f_i']
        if len(mo) < f + I.WIN: continue
        P = I.fk(mo); row = dict(ci=r['ci'], item=r['item'], frac=r['frac'])
        if U is not None:
            n = min(len(U), f + I.WIN); PU = I.fk(U[:n])
            dev_ = np.linalg.norm(P[f:n] - PU[f:n], axis=-1).max(1) > 0.02
            row['react'] = float(np.argmax(dev_) / I.FPS) if dev_.any() else None; row['react_cens'] = (n - f) / I.FPS
            row['v3_ratio'] = I.v3_ratio(P, PU, f)
        ty, need = I.yield_time(P, f, rest_z); row.update(need=need, yield_s=ty)                       # secondary: height only
        t3, n3 = I.yield3d(P, f, rest_P); row.update(need3d=n3, yield3d_s=t3)                       # primary (11-08): 3D rest pose
        row['jerk'] = I.peak_jerk(P, f - I.PRE, f + I.WIN)
        row.update({f'unsafe_{k}': x for k, x in unsafe(mo, f).items()})
        kf = I.kin_features(P, f); row['kin'] = kf.tolist(); rows.append(row); kins.append(kf)
        if per: per.write(json.dumps(dict(row, method=lab)) + '\n')
    if not rows: print(lab, 'no rows'); continue
    W = I.WIN / I.FPS; sel = [x for x in rows if x['need']]; s3 = [x for x in rows if x['need3d']]
    out = dict(n=len(rows), n_need=len(sel), yield_rate=float(np.mean([x['yield_s'] is not None for x in sel])) if sel else None,
               yield_s=float(np.mean([x['yield_s'] if x['yield_s'] is not None else W for x in sel])) if sel else None)
    out.update(n_need3d=len(s3), yield3d_rate=float(np.mean([x['yield3d_s'] is not None for x in s3])) if s3 else None,
               yield3d_s=float(np.mean([x['yield3d_s'] if x['yield3d_s'] is not None else W for x in s3])) if s3 else None,
               yield3d_rate_all=float(np.mean([x['yield3d_s'] is not None for x in rows])))
    rc = [x['react'] if x.get('react') is not None else x.get('react_cens') for x in rows if 'react_cens' in x]
    out['react_s'] = float(np.mean(rc)) if rc else None
    v3 = np.array([x['v3_ratio'] for x in rows if x.get('v3_ratio') is not None])
    out['abrupt_v3'] = float((v3 > I.V3_RATIO).mean()) if len(v3) else None; out['v3_ratio_med'] = float(np.median(v3)) if len(v3) else None
    for k in ('pos', 'vel', 'coll', 'any', 'coll_scene', 'any_scene'): out[f'unsafe_{k}'] = float(np.mean([x[f'unsafe_{k}'] for x in rows]))
    out['rkd_post'], out['rkd_post_per_feature'] = I.kin_distance(np.stack(kins), HK_all)       # RKD-post: all post-speech windows
    out['rkd_retract'] = I.kin_distance(np.stack(kins), HK_ret)[0] if len(HK_ret) >= 30 else None   # secondary, only if >= 30 windows
    out['jerk_median'] = float(np.median([x['jerk'] for x in rows]))
    res[lab] = out
    print(f"{lab:14s} n {out['n']} | yield3d {out['yield3d_rate']} ({out['yield3d_s']}) all {out['yield3d_rate_all']:.3f} | yield-height {out['yield_rate']} ({out['yield_s']}) | v3 {out['abrupt_v3']} | RKD-post {out['rkd_post']:.3f} RKD-retract {out['rkd_retract']} | unsafe WBC {out['unsafe_any']:.4f} scene {out['unsafe_any_scene']:.4f} | react {out['react_s']}", flush=True)
res['_meta'] = dict(T_r=T_r, T_r_source='human_ref median' if a.tr is None else 'override', T_r_ex002_human=H['T_r'],
                    collision_primary='retargeter WBC clearance (ours): release06 smplx2ex002_3f EX002.clearance, runtime-rm003 349 rows',
                    collision_scene='MuJoCo scene, structural-contact-limited: scene.xml contacts minus parent-child links minus pairs in >= 50 % of dev GT frames',
                    collision_stride=a.collision_stride, n_interrupts=len(ints), n_human_windows=int(len(HK_all)), n_human_retract=int(len(HK_ret)),
                    rkd_post_half_split=I.kin_distance(HK_all[0::2], HK_all[1::2])[0],
                    rkd_retract_half_split=I.kin_distance(HK_ret[0::2], HK_ret[1::2])[0] if len(HK_ret) >= 30 else None,
                    rkd_note='RKD-post (all post-speech windows) is not comparable with the G1 retraction-window RKD',
                    yield_primary='3D: both wrists within 10 cm of the human rest wrist positions, held >= 0.5 s (icommon_ex002.yield3d); height-only = secondary',
                    rest_wrists=rest_P.tolist(), front_control=dict(ci=FRONT_CI, frame=FRONT_T, min_wrist_dist_to_rest_m=FRONT_D),
                    adjacent_pairs=len(adjacent), structural_pairs=len(structural), prefix=a.prefix, rule=__doc__.split('\n')[0])
print('RKD-post human half-split floor', round(res['_meta']['rkd_post_half_split'], 3), '| windows', len(HK_all), '| n retract', len(HK_ret), '| T_r', T_r, '(EX002 human', H['T_r'], ')')
Path(a.out).write_text(json.dumps(res, indent=1))

"""Human "speech ends -> hands come back" reference statistics + rest poses for the G1 interruption benchmark (11-05).

Inputs: the GMR-retargeted post-speech segments of human_gaps.py (<seg dir>/<id>.npz: robot_repr, off = speech-offset frame)
and, for the second rest-pose definition, ECHO-G G1 GT training clips (motion_39d_30fps).
Rest poses (29 joints; only waist + arms are used by B2 and the metrics):
  R_sil : median joint vector of human SILENCE frames (>= 1 s after the speech offset) where both hands are slow (< 0.10 m/s)
          and low (z < 0 m relative to the pelvis) = the posture BEAT2 speakers hold while not speaking
  R_gt  : median joint vector of ECHO-G GT training frames (every --gt-stride-th train clip) where both hands are still
          (< 0.05 m/s) and low (z < 0 m) = the arms-down rest seen inside the packaged (speech) clips
Reference statistics on the post-offset window [off, off + post) of each segment: hand speed profile, yield time (both hands
at rest height and staying there, see icommon.yield_time; segments whose hands are above rest height at the offset), peak
hand jerk / acceleration in [off - 0.1 s, off + 2.13 s).
    python scripts/interrupt/human_ref.py --seg-dir D --index 'D/index.*.jsonl' --gt-dir G --train-split S --out human_ref
writes human_ref.json and human_ref_windows.npz (rr: robot_repr windows [N, 64, 39] starting at the offset = FGD reference;
rr_ctx: the same windows with 1 s (30 frames) before the offset = kinematic-feature reference; retract: the retraction subset).
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import icommon as I

ap = argparse.ArgumentParser()
ap.add_argument('--seg-dir', required=True); ap.add_argument('--index', required=True); ap.add_argument('--gt-dir', required=True)
ap.add_argument('--train-split', required=True); ap.add_argument('--gt-stride', type=int, default=10); ap.add_argument('--out', required=True)
ap.add_argument('--splits', default='train_drop', help='segments of these ECHO-G splits only (train speakers; val_common = test speakers)')
a = ap.parse_args()
idx = sorted([json.loads(l) for f in glob.glob(a.index) for l in open(f)], key=lambda c: c['id'])
idx = [c for c in idx if c['split'] in a.splits.split(',')]
segs = []
for c in idx:
    d = np.load(f"{a.seg_dir}/{c['id']}.npz"); rr = d['robot_repr'].astype(np.float64); off = int(d['off'])
    P = I.fk(rr); v = I.hand_speed(P)
    segs.append(dict(c=c, rr=rr, off=off, P=P, v=v))

# ---- rest poses
sil = []
for s in segs:
    z = s['P'][:, list(I.HANDS), 2]; ok = (s['v'] < 0.10).all(1) & (z < 0.).all(1)
    ok[:s['off'] + 30] = False
    sil.append(s['rr'][ok, 10:])
sil = np.concatenate(sil); R_sil = np.median(sil, 0)
gt_still = []
for st in open(a.train_split).read().split()[::a.gt_stride]:
    rr = I.load_rr(f'{a.gt_dir}/{st}.pt'); P = I.fk(rr); v = I.hand_speed(P); z = P[:, list(I.HANDS), 2]
    ok = (v < 0.05).all(1) & (z < 0.).all(1)
    if ok.any(): gt_still.append(rr[ok, 10:])
gt_still = np.concatenate(gt_still); R_gt = np.median(gt_still, 0)
rest = {}
for k, R in (('R_sil', R_sil), ('R_gt', R_gt)):
    rr = np.zeros((1, 39)); rr[0, 0] = rr[0, 4] = 1.; rr[0, 10:] = R      # identity rot6d (first two columns of I)
    P = I.fk(rr)
    rest[k] = dict(joints=R.tolist(), hand_pos=P[0, list(I.HANDS)].tolist(), hand_z=P[0, list(I.HANDS), 2].tolist())
print('rest poses:', {k: dict(hand_z=np.round(v['hand_z'], 3).tolist(), n=(len(sil) if k == 'R_sil' else len(gt_still))) for k, v in rest.items()})
dR = np.abs(R_sil - R_gt)[I.UPPER]
print(f'R_sil vs R_gt upper-body joints: mean |d| {dR.mean():.3f} rad, max {dR.max():.3f}; hand distance '
      f'{np.linalg.norm(np.array(rest["R_sil"]["hand_pos"]) - np.array(rest["R_gt"]["hand_pos"]), axis=-1).round(3).tolist()} m')

# ---- reference statistics
L, Wp = int(1.5 * I.FPS), I.WIN
out = dict(n_segments=len(segs), speakers={}, rest=rest)
for s in segs: out['speakers'][s['c']['speaker']] = out['speakers'].get(s['c']['speaker'], 0) + 1
prof = np.array([s['v'].mean(1)[s['off'] - L:s['off'] + Wp] for s in segs if s['off'] >= L and s['off'] + Wp <= len(s['v'])])
out['speed_profile'] = dict(t=((np.arange(-L, Wp)) / I.FPS).round(3).tolist(), median=np.median(prof, 0).round(4).tolist(),
                            p25=np.percentile(prof, 25, 0).round(4).tolist(), p75=np.percentile(prof, 75, 0).round(4).tolist(), n=len(prof))
pre_v = prof[:, :L].mean(1); post_v = prof[:, L + 32:].mean(1)
out['speed_ratio_post_pre'] = dict(median=float(np.median(post_v / pre_v)), frac_drop30=float((post_v < 0.7 * pre_v).mean()))
jk = np.array([I.peak_jerk(s['P'], s['off'] - I.PRE, s['off'] + Wp) for s in segs if s['off'] + Wp <= len(s['P'])])
ac = np.array([I.peak_acc(s['P'], s['off'] - I.PRE, s['off'] + Wp) for s in segs if s['off'] + Wp <= len(s['P'])])
pk = lambda x: {f'p{q}': float(np.percentile(x, q)) for q in (5, 25, 50, 75, 95)}
out['peak_jerk'] = pk(jk); out['peak_acc'] = pk(ac)
for k in rest:
    zr = rest[k]['hand_z']; ty, ty_w, need = [], [], 0
    for s in segs:
        t_full, nd = I.yield_time(s['P'], s['off'], zr, horizon=len(s['P']) - s['off'])
        t_w, _ = I.yield_time(s['P'], s['off'], zr, horizon=Wp)
        if not nd: continue
        need += 1; ty.append(t_full); ty_w.append(t_w)
    reached = [t for t in ty if t is not None]
    out[f'yield_{k}'] = dict(n_needed=need, frac_needed=need / len(segs), reached_full=len(reached) / max(need, 1),
                             reached_2s=float(np.mean([t is not None for t in ty_w])) if ty_w else None,
                             median_reached=float(np.median(reached)) if reached else None,
                             pct_reached={f'p{q}': float(np.percentile(reached, q)) for q in (25, 50, 75)} if reached else None)
# retraction duration T_r for B2: median yield time of the human segments that do retract (hands above R_sil rest height at
# the offset, lowered and staying low before the segment ends)
out['T_r'] = out['yield_R_sil']['median_reached']
ok = [s for s in segs if s['off'] + Wp <= len(s['rr']) and s['off'] >= 30]
W = np.stack([s['rr'][s['off']:s['off'] + Wp] for s in ok]).astype(np.float32)
Wc = np.stack([s['rr'][s['off'] - 30:s['off'] + Wp] for s in ok]).astype(np.float32)     # + 1 s before the offset
# retraction subset: hands above R_sil rest height at the offset, lowered and staying low before the segment ends
ret = np.array([I.yield_time(s['P'], s['off'], rest['R_sil']['hand_z'], horizon=len(s['P']) - s['off']) for s in ok], dtype=object)
ret = np.array([bool(nd and t is not None) for t, nd in ret]); out['n_retract_windows'] = int(ret.sum())
np.savez_compressed(a.out + '_windows.npz', rr=W, rr_ctx=Wc, retract=ret, ids=np.array([s['c']['id'] for s in ok]),
                    speakers=np.array([s['c']['speaker'] for s in ok]))
Path(a.out + '.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ('speed_profile', 'rest')}, indent=1))
print('FGD reference windows', W.shape)

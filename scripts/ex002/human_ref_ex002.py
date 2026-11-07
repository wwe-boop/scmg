"""EX002 human references of the interruption benchmark (port of scripts/interrupt/human_ref.py; criteria EX002_INTERRUPT_1107.md).
Input: ex002_retarget_tails.py segments <seg>/<id>.npz (q34 [T, 34] rad at 50 Hz, off = speech-offset frame).
  R_sil   : median joint vector of the silence frames >= 1 s after the offset where both wrists are slow (< 0.10 m/s)
            (the posture held while not speaking); rest height = its wrist z (hand_z)
  windows : q windows [off - 1 s, off + 2.13 s) of every tail whose window fits (ctx = 50 frames) -> RKD reference;
            retract = wrists above rest height + 5 cm at the offset AND back at rest height (staying) inside the window
  T_r     : median yield time of the retract windows; peak_jerk p95 over [off - 0.1 s, off + 2.13 s)
    python scripts/ex002/human_ref_ex002.py --seg D/seg --out human_ref_ex002   -> .json + _windows.npz"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import icommon_ex002 as I

ap = argparse.ArgumentParser(); ap.add_argument('--seg', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
fs = sorted(glob.glob(f'{a.seg}/*.npz')); segs = []
for f in fs:
    z = np.load(f); segs.append((Path(f).stem, np.asarray(z['q34'], np.float64), int(z['off'])))
sil = []
for _, q, off in segs:
    P = I.fk(q); v = I.hand_speed(P).max(1)
    m = np.zeros(len(q), bool); m[off + I.SEC:] = True; m &= v < 0.10
    sil.append(q[m])
S = np.concatenate([x for x in sil if len(x)])
rest = np.median(S, 0); rest_z = I.fk(rest[None])[0, :, 2]
W, ret, ys, pj, ids = [], [], [], [], []
for sid, q, off in segs:
    if off < I.SEC or len(q) < off + I.WIN: continue
    w = q[off - I.SEC:off + I.WIN]; P = I.fk(w); f = I.SEC
    ty, need = I.yield_time(P, f, rest_z)
    ret.append(bool(need and ty is not None)); W.append(w); ids.append(sid); pj.append(I.peak_jerk(P, f - I.PRE, f + I.WIN))
    if need and ty is not None: ys.append(ty)
W = np.stack(W).astype(np.float32); ret = np.array(ret)
out = dict(rest=dict(R_sil=dict(joints=rest.tolist(), hand_z=rest_z.tolist(), n_frames=int(len(S)))), T_r=float(np.median(ys)) if ys else None,
           peak_jerk=dict(p95=float(np.percentile(pj, 95)), median=float(np.median(pj))), n_segments=len(segs), n_windows=len(W),
           n_retract=int(ret.sum()), yield_median=float(np.median(ys)) if ys else None, fps=I.FPS, win=I.WIN, ctx=I.SEC, rule=__doc__.split('\n')[1])
Path(a.out + '.json').write_text(json.dumps(out, indent=1)); np.savez(a.out + '_windows.npz', q_ctx=W, retract=ret, ids=np.array(ids))
print(json.dumps({k: v for k, v in out.items() if k != 'rest'}), 'rest_z', np.round(rest_z, 3).tolist())

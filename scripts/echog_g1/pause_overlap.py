"""Pause / stillness overlap metrics for the G1 main table (11-05; user: "计算停顿/静止的重合区间率").

Still frame (30 fps) = mean mmae-normalised upper-body speed < 0.15 (official FK / speed / mmae, the definition of
pause_probe.py and beat_phase.py). Pause = still run >= 0.5 s. Silence = 16 kHz RMS below clip median - 15 dB, runs
>= 0.2 s. Per method, pooled over clips:
  still%        share of still frames                                   (GT reference value)
  pauses/min    pause runs per minute of audio
  still_IoU     frame IoU of the still masks of method and GT (same clip, same frames)
  pause_R       share of GT pauses overlapped by a pause of the method            (interval recall)
  pause_P       share of the method's pauses that overlap a GT pause              (interval precision)
  pause_F1      2PR / (P + R)
  sil_P         share of the method's pauses that overlap a speech silence (GT value = how humans pause)
  sil_R         share of silences >= 0.2 s that contain a pause of the method
Chance level: pause_P_shift = pause_P with the method's pauses circularly shifted by half the clip (same count and
lengths, timing destroyed). A shifted pause past the clip end is kept as is (it can only miss), so the chance is a
slight underestimate.
    python scripts/echog_g1/pause_overlap.py --mmae-file M --ref-dir R --wav-dir W --stems S --out J label=pred_dir ...
"""
import argparse, json
import numpy as np
from echo_g.evaluation import g1_motion_cls as E

ap = argparse.ArgumentParser()
ap.add_argument('--mmae-file', required=True); ap.add_argument('--ref-dir', required=True); ap.add_argument('--wav-dir', required=True)
ap.add_argument('--stems', required=True); ap.add_argument('--out', required=True); ap.add_argument('--fps', type=float, default=30.)
ap.add_argument('--drop', type=float, default=15.); ap.add_argument('--thr', type=float, default=0.15)
ap.add_argument('panels', nargs='+')
a = ap.parse_args()
E._G1_MEAN_VEL_PATH = a.mmae_file
import librosa
ba = E.BeatAlign(a.fps); mmae = E._g1_mean_vel()
stems = [s.strip() for s in open(a.stems) if s.strip()]


def still_of(rr):
    J = E.fk_positions_ba(rr); v = (E.body_speed_central(J, a.fps)[:, ba.upper_bodies] / (mmae[ba.upper_bodies] + 1e-8)).mean(1)
    return v < a.thr


def runs(x, n):
    out, s = [], None
    for i, b in enumerate(np.r_[x, False]):
        if b and s is None: s = i
        if not b and s is not None:
            if i - s >= n: out.append((s, i))
            s = None
    return out


def hit(r, others): return any(q0 < r[1] and r[0] < q1 for q0, q1 in others)


cols = [('GT', a.ref_dir)] + [tuple(p.split('=', 1)) for p in a.panels]
K = ('frames', 'still', 'inter', 'union', 'gtp', 'gtp_hit', 'p', 'p_hit', 'p_hit_shift', 'p_sil', 'sil', 'sil_hit', 'clips')
acc = {lab: dict.fromkeys(K, 0) for lab, _ in cols}
for s in stems:
    rrs = [E.load_repr(f'{d}/{s}.pt', a.fps) for _, d in cols]
    if any(r is None for r in rrs): continue
    n = min(r.shape[0] for r in rrs)
    if n < 16: continue
    y, _ = librosa.load(E.wav_of(a.wav_dir, s), sr=16000)
    db = 20 * np.log10(librosa.feature.rms(y=y, hop_length=320)[0] + 1e-5)
    db = np.interp(np.arange(n) / a.fps, np.arange(len(db)) * 0.02, db)
    sil_runs = runs(db < np.median(db) - a.drop, int(0.2 * a.fps))
    sts = [still_of(r[:n]) for r in rrs]
    gt_st = sts[0]; gt_p = runs(gt_st, int(0.5 * a.fps))
    for (lab, _), st in zip(cols, sts):
        A = acc[lab]; P = runs(st, int(0.5 * a.fps)); h = n // 2
        Ps = [((p0 + h) % n, (p0 + h) % n + (p1 - p0)) for p0, p1 in P]          # circular half-clip shift (chance)
        A['clips'] += 1; A['frames'] += n; A['still'] += int(st.sum())
        A['inter'] += int((st & gt_st).sum()); A['union'] += int((st | gt_st).sum())
        A['gtp'] += len(gt_p); A['gtp_hit'] += sum(hit(g, P) for g in gt_p)
        A['p'] += len(P); A['p_hit'] += sum(hit(p, gt_p) for p in P); A['p_hit_shift'] += sum(hit(p, gt_p) for p in Ps)
        A['p_sil'] += sum(hit(p, sil_runs) for p in P)
        A['sil'] += len(sil_runs); A['sil_hit'] += sum(hit(q, P) for q in sil_runs)
res = {}
d = lambda x, y: x / y if y else float('nan')
for lab, _ in cols:
    A = acc[lab]; R = d(A['gtp_hit'], A['gtp']); P = d(A['p_hit'], A['p'])
    res[lab] = dict(clips=A['clips'], still_pct=100 * d(A['still'], A['frames']), pauses_per_min=d(A['p'], A['frames'] / a.fps / 60),
                    still_IoU=d(A['inter'], A['union']), pause_R=R, pause_P=P, pause_F1=d(2 * P * R, P + R),
                    pause_P_shift=d(A['p_hit_shift'], A['p']), sil_P=d(A['p_sil'], A['p']), sil_R=d(A['sil_hit'], A['sil']))
    r = res[lab]
    print(f"{lab:14s} still {r['still_pct']:5.1f}% | pauses/min {r['pauses_per_min']:5.2f} | still IoU {r['still_IoU']:.3f} | "
          f"pause P {r['pause_P']:.3f} (shift {r['pause_P_shift']:.3f}) R {r['pause_R']:.3f} F1 {r['pause_F1']:.3f} | "
          f"silence P {r['sil_P']:.3f} R {r['sil_R']:.3f}", flush=True)
json.dump(dict(res=res, stems=a.stems, thr=a.thr, drop=a.drop), open(a.out, 'w'), indent=1)

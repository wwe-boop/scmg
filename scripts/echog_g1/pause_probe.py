"""Feasibility of speech-aware pause modelling (11-05): do GT gesture pauses line up with audio silence?

Per 30 fps frame: still = mean mmae-normalized upper-body speed < 0.15 (official FK / speed / mmae, as beat_phase.py);
silence = 16 kHz RMS (hop 320 = 20 ms, resampled to the motion frames) below (clip median dB - DROP). Reports
P(still), P(still | silence), P(still | speech), the lift, and the same with silence measured CAUSALLY in the window
[t - W, t + H] (H = look-ahead the engine has, default 160 ms), plus the fraction of GT pause runs (still >= 0.5 s) that
overlap a silence of >= 0.2 s. Optional pred dirs: the same for generated motion (does the model already pause at silence?).
    python scripts/echog_g1/pause_probe.py --mmae-file M --ref-dir R --wav-dir W --stems S [--limit N] [label=pred_dir ...]
"""
import argparse
import numpy as np
from echo_g.evaluation import g1_motion_cls as E

ap = argparse.ArgumentParser()
ap.add_argument('--mmae-file', required=True); ap.add_argument('--ref-dir', required=True); ap.add_argument('--wav-dir', required=True)
ap.add_argument('--stems', required=True); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--fps', type=float, default=30.)
ap.add_argument('--drop', type=float, default=15.); ap.add_argument('--win', type=float, default=0.3); ap.add_argument('--ahead', type=float, default=0.16)
ap.add_argument('panels', nargs='*')
a = ap.parse_args()
E._G1_MEAN_VEL_PATH = a.mmae_file
import librosa
ba = E.BeatAlign(a.fps); mmae = E._g1_mean_vel()
stems = [s.strip() for s in open(a.stems) if s.strip()][: a.limit or None]


def still_of(rr):
    J = E.fk_positions_ba(rr); v = (E.body_speed_central(J, a.fps)[:, ba.upper_bodies] / (mmae[ba.upper_bodies] + 1e-8)).mean(1)
    return v < 0.15


def runs(x, n):
    out, s = [], None
    for i, b in enumerate(np.r_[x, False]):
        if b and s is None: s = i
        if not b and s is not None:
            if i - s >= n: out.append((s, i))
            s = None
    return out


cols = [('GT', a.ref_dir)] + [tuple(p.split('=', 1)) for p in a.panels]
acc = {lab: dict(st=[], sil=[], csil=[], pause_hit=0, pauses=0) for lab, _ in cols}
for s in stems:
    rrs = [E.load_repr(f'{d}/{s}.pt', a.fps) for _, d in cols]
    if any(r is None for r in rrs): continue
    n = min(r.shape[0] for r in rrs)
    if n < 16: continue
    y, _ = librosa.load(E.wav_of(a.wav_dir, s), sr=16000)
    db = 20 * np.log10(librosa.feature.rms(y=y, hop_length=320)[0] + 1e-5)
    db = np.interp(np.arange(n) / a.fps, np.arange(len(db)) * 0.02, db)
    sil = db < np.median(db) - a.drop
    w, h = int(a.win * a.fps), int(a.ahead * a.fps)
    csil = np.array([sil[max(0, t - w): min(n, t + h + 1)].mean() > 0.5 for t in range(n)])     # causal + engine look-ahead
    sil_runs = runs(sil, int(0.2 * a.fps))
    for (lab, _), r in zip(cols, rrs):
        st = still_of(r[:n]); A = acc[lab]; A['st'].append(st); A['sil'].append(sil); A['csil'].append(csil)
        for (p0, p1) in runs(st, int(0.5 * a.fps)):
            A['pauses'] += 1; A['pause_hit'] += any(q0 < p1 and p0 < q1 for q0, q1 in sil_runs)
print(f'{len(acc["GT"]["st"])} clips; silence = RMS < clip median - {a.drop} dB; causal window [-{a.win}s, +{a.ahead}s]')
for lab, _ in cols:
    A = acc[lab]; st, sil, cs = (np.concatenate(A[k]) for k in ('st', 'sil', 'csil'))
    ps, p1, p0 = st.mean(), st[sil].mean() if sil.any() else np.nan, st[~sil].mean()
    c1 = st[cs].mean() if cs.any() else np.nan
    print(f'{lab:14s} P(still) {100 * ps:.1f}% | P(still|silence) {100 * p1:.1f}% vs |speech {100 * p0:.1f}% (lift {p1 / max(p0, 1e-9):.1f}x) | '
          f'causal: P(still|silence) {100 * c1:.1f}% | silence share {100 * sil.mean():.1f}% | pauses overlapping a silence {A["pause_hit"]}/{A["pauses"]}')

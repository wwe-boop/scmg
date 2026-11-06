"""Div / BA diagnostics (11-06, no training): where do our motion beats and speed fluctuations differ from GT?

For each label (GT + prediction dirs) on the given stems, with the official G1 FK / beat definition (echo_g BeatAlign:
upper-body speed local minima above the mmae-normalised threshold):
  1. beats per second per upper body; 10-bin histogram of beat phase on the 320 ms chunk grid (anchored at clip start);
  2. speed spectrum: Welch PSD of the mean mmae-normalised upper-body speed (30 fps, detrended), band powers
     0-1 / 1-2 / 2-4 / 4-6 / 6-10 / 10-15 Hz relative to GT (frame-weighted);
  3. where beats fall: inside speech (word span +- 0.1 s) vs silence, and within +-1 frame of a chunk boundary;
  4. per speaker: beats/s and a mean-abs-deviation amplitude proxy (official BA/Div come from eval_g1_motion_cls.py);
  5. (--wav-dir) official BA (same onset/GAHR code path as BeatAlign) and a density-adjusted BA lift = BA - BA_null,
     BA_null = mean BA against the clip's own audio onsets circularly shifted by K random offsets in [1 s, T-1 s]
     (same onset count and spacing, alignment destroyed). Lift measures synchrony, not beat density / energy.
Analysis only -- nothing here selects a model.
    PYTHONPATH=<official>/src python diag_beats.py --mmae-file M --ref-dir R --words-dir D [--wav-dir W] --stems S --out J label=dir ...
"""
import argparse, json, zlib
from pathlib import Path
import numpy as np
import scipy.signal
from echo_g.evaluation import g1_motion_cls as E

ap = argparse.ArgumentParser()
ap.add_argument('--mmae-file', required=True); ap.add_argument('--ref-dir', required=True); ap.add_argument('--stems', required=True)
ap.add_argument('--words-dir', required=True); ap.add_argument('--wav-dir', default=''); ap.add_argument('--null-k', type=int, default=20); ap.add_argument('--out', required=True)
ap.add_argument('--limit', type=int, default=0); ap.add_argument('--fps', type=float, default=30.0)
ap.add_argument('--period', type=float, default=16 / 50)
ap.add_argument('panels', nargs='+')
a = ap.parse_args()
E._G1_MEAN_VEL_PATH = a.mmae_file
ba = E.BeatAlign(a.fps); mmae = E._g1_mean_vel()
stems = [s.strip() for s in open(a.stems) if s.strip()][: a.limit or None]
BANDS = [(0, 1), (1, 2), (2, 4), (4, 6), (6, 10), (10, 15)]


def beats(J):
    spd = E.body_speed_central(J, a.fps); out = []
    for b in ba.upper_bodies:
        v = spd[:, b]; norm = v / (mmae[b] + 1e-8)
        out.append(np.intersect1d(scipy.signal.argrelextrema(v, np.less, order=ba.order)[0], np.where(norm > ba.bc_threshold)[0]))
    return np.concatenate(out) if out else np.zeros(0, int)


def gahr_ba(J, onsets):
    # mirrors BeatAlign._beat_consistency (audio_to_motion, sigma, order, threshold, /len(upper_bodies))
    spd = E.body_speed_central(J, a.fps); per = []
    for b in ba.upper_bodies:
        v = spd[:, b]; norm = v / (mmae[b] + 1e-8) if mmae[b] > 1e-8 else v
        bt = np.intersect1d(scipy.signal.argrelextrema(v, np.less, order=ba.order)[0], np.where(norm > ba.bc_threshold)[0]) / a.fps
        per.append(float(np.mean(np.exp(-(np.abs(bt[:, None] - onsets[None, :]).min(0) ** 2) / (2 * ba.sigma ** 2)))) if bt.size else 0.0)
    return float(np.sum(per) / len(ba.upper_bodies))


def onsets_of(stem, n):
    import librosa
    y, _ = librosa.load(f'{a.wav_dir}/{stem}.wav', sr=16000); y = y[: int(n / a.fps * 16000)]
    return np.asarray(librosa.onset.onset_detect(y=y, sr=16000, hop_length=512, units='time'), np.float64)


def speed(J):
    return (E.body_speed_central(J, a.fps)[:, ba.upper_bodies] / (mmae[ba.upper_bodies] + 1e-8)).mean(1)


def band_power(v):
    if len(v) < 32: return None
    f, p = scipy.signal.welch(scipy.signal.detrend(v), fs=a.fps, nperseg=min(64, len(v)))
    return np.array([p[(f >= lo) & (f < hi)].sum() * (f[1] - f[0]) for lo, hi in BANDS])


def speech_mask(stem, n):
    m = np.zeros(n, bool)
    try: W = json.load(open(Path(a.words_dir) / f'{stem}.json'))['words']
    except Exception: return None
    t = np.arange(n) / a.fps
    for w in W: m |= (t >= float(w['start']) - 0.1) & (t <= float(w['end']) + 0.1)
    return m


cols = [('GT', None)] + [tuple(p.split('=', 1)) for p in a.panels]
R = {lab: dict(beats=0, frames=0, phase=np.zeros(10), bp=np.zeros(len(BANDS)), bpw=0, sp_b=0, sp_f=0, si_b=0, si_f=0,
               bd_b=0, spk={}, ba=0.0, ban=0.0, baw=0, clip=[]) for lab, _ in cols}
nb = len(ba.upper_bodies); dur = 0.0
for s in stems:
    gr = E.load_repr(f'{a.ref_dir}/{s}.pt', a.fps); prs = [E.load_repr(f'{d}/{s}.pt', a.fps) for _, d in cols[1:]]
    if gr is None or any(p is None for p in prs): continue
    n = min([gr.shape[0]] + [p.shape[0] for p in prs])
    if n < 16: continue
    sm = speech_mask(s, n); dur += n / a.fps; spk = s.split('_')[2]
    ons = onsets_of(s, n) if a.wav_dir else None; T = n / a.fps
    if ons is not None and len(ons) == 0: ons = None
    if ons is not None:
        rng = np.random.default_rng(zlib.crc32(s.encode()))
        shifts = [np.sort((ons + o) % T) for o in rng.uniform(1.0, max(T - 1.0, 1.0 + 1e-6), a.null_k)]
    for lab, r in [('GT', gr)] + [(l, p) for (l, _), p in zip(cols[1:], prs)]:
        J = E.fk_positions_ba(r[:n]); bt = beats(J); X = R[lab]
        X['beats'] += len(bt); X['frames'] += n
        ph = ((bt / a.fps) % a.period) / a.period; X['phase'] += np.histogram(ph, bins=10, range=(0, 1))[0]
        bpw = band_power(speed(J))
        if bpw is not None: X['bp'] += bpw * n; X['bpw'] += n
        if sm is not None:
            X['sp_b'] += int(sm[bt].sum()); X['si_b'] += int((~sm[bt]).sum()); X['sp_f'] += int(sm.sum()); X['si_f'] += int((~sm).sum())
        fpc = a.period * a.fps; d = np.minimum((bt % fpc), fpc - (bt % fpc)); X['bd_b'] += int((d <= 1.0).sum())
        fl = J.reshape(n, -1); dv = float(np.abs(fl - fl.mean(0)).mean(1).sum())
        e = X['spk'].setdefault(spk, np.zeros(5)); e[:3] += [len(bt), n, dv]
        if ons is not None:
            b0 = gahr_ba(J, ons); bn = float(np.mean([gahr_ba(J, o) for o in shifts]))
            X['ba'] += b0 * n; X['ban'] += bn * n; X['baw'] += n; X['clip'].append((b0, bn, n)); e[3] += b0 * n; e[4] += bn * n
out = dict(clips=len(stems), minutes=dur / 60, bands_hz=BANDS, labels={})
g = R['GT']; gbp = g['bp'] / max(g['bpw'], 1)
for lab, _ in cols:
    X = R[lab]; h = X['phase'] / max(X['phase'].sum(), 1); bp = X['bp'] / max(X['bpw'], 1)
    o = dict(beats_per_s_per_body=X['beats'] / (X['frames'] / a.fps) / nb, phase_hist=h.tolist(), phase_max_min=float(h.max() / max(h.min(), 1e-9)),
             band_power=bp.tolist(), band_power_rel_gt=(bp / np.maximum(gbp, 1e-12)).tolist(),
             beats_per_s_speech=X['sp_b'] / max(X['sp_f'] / a.fps, 1e-9) / nb, beats_per_s_silence=X['si_b'] / max(X['si_f'] / a.fps, 1e-9) / nb,
             silence_frame_share=X['si_f'] / max(X['sp_f'] + X['si_f'], 1), beat_share_at_chunk_boundary_pm1f=X['bd_b'] / max(X['beats'], 1),
             ba=X['ba'] / max(X['baw'], 1), ba_null=X['ban'] / max(X['baw'], 1), ba_lift=(X['ba'] - X['ban']) / max(X['baw'], 1),
             per_speaker={k: dict(beats_per_s=v[0] / (v[1] / a.fps) / nb, div_proxy=v[2] / v[1], ba=v[3] / v[1], ba_lift=(v[3] - v[4]) / v[1])
                          for k, v in sorted(X['spk'].items())})
    if X['clip']:  # clip bootstrap SE of the frame-weighted lift (and of lift - GT lift, paired)
        C = np.array(X['clip']); G = np.array(R['GT']['clip']); br = np.random.default_rng(0); se = []; sd = []
        for _ in range(2000):
            i = br.integers(0, len(C), len(C)); w = C[i, 2]
            l = ((C[i, 0] - C[i, 1]) * w).sum() / w.sum(); se.append(l); sd.append(l - ((G[i, 0] - G[i, 1]) * w).sum() / w.sum())
        o['ba_lift_se'] = float(np.std(se)); o['ba_lift_minus_gt_se'] = float(np.std(sd))
        o['clips_ba'] = C.tolist()
    out['labels'][lab] = o
    print(f"{lab:14s} BA {o['ba']:.4f} null {o['ba_null']:.4f} lift {o['ba_lift']:+.4f} (se {o.get('ba_lift_se', 0):.4f}) | beats/s {o['beats_per_s_per_body']:.3f} (speech {o['beats_per_s_speech']:.3f} silence {o['beats_per_s_silence']:.3f}) "
          f"| phase max/min {o['phase_max_min']:.2f} | at chunk boundary {100 * o['beat_share_at_chunk_boundary_pm1f']:.1f}% "
          f"| band power rel GT " + ' '.join(f'{lo}-{hi}Hz {x:.2f}' for (lo, hi), x in zip(BANDS, o['band_power_rel_gt'])))
    print('    per speaker beats/s: ' + '  '.join(f"{k} {v['beats_per_s']:.3f} (BA {v['ba']:.3f} lift {v['ba_lift']:+.4f})" for k, v in o['per_speaker'].items()))
Path(a.out).write_text(json.dumps(out, indent=1))

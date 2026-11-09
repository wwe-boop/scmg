"""Where do the motion beats fall relative to SCMG's 320 ms chunk grid? (11-05)

BA (official, audio_to_motion) rises with the density of motion beats (speed local minima of moving upper bodies, corr
0.97 per clip). Our generations have more beats than GT at lower speed (choppier). If chunk boundaries create speed dips,
generated beats concentrate at a fixed phase of the 320 ms grid while GT beats are uniform. For each clip: official FK
(fk_positions_ba), official beat definition (BeatAlign order / threshold / mmae), beat time t -> phase = (t mod P) / P with
P = CHUNK_FRAMES / 50 s and the grid anchored at the clip start (infer_test.py generates chunk 0 from frame 0).

    python scripts/echog_g1/beat_phase.py --mmae-file M --ref-dir R --stems S [--limit N] label=pred_dir [label=pred_dir ...]
Run with the echog python (PYTHONPATH=<official>/src). Prints a 10-bin phase histogram (share of beats) per label + GT.
"""
import argparse
import numpy as np
import scipy.signal
from echo_g.evaluation import g1_motion_cls as E

ap = argparse.ArgumentParser()
ap.add_argument('--mmae-file', required=True); ap.add_argument('--ref-dir', required=True); ap.add_argument('--stems', required=True)
ap.add_argument('--limit', type=int, default=0); ap.add_argument('--fps', type=float, default=30.0)
ap.add_argument('--period', type=float, default=16 / 50, help='chunk period in seconds (CHUNK_FRAMES / 50 Hz)')
ap.add_argument('panels', nargs='+')
a = ap.parse_args()
E._G1_MEAN_VEL_PATH = a.mmae_file
ba = E.BeatAlign(a.fps); mmae = E._g1_mean_vel()
stems = [s.strip() for s in open(a.stems) if s.strip()][: a.limit or None]


def beat_times(J):
    spd = E.body_speed_central(J, a.fps); out = []
    for b in ba.upper_bodies:
        v = spd[:, b]; norm = v / (mmae[b] + 1e-8)
        m = np.intersect1d(scipy.signal.argrelextrema(v, np.less, order=ba.order)[0], np.where(norm > ba.bc_threshold)[0])
        out.append(m / a.fps)
    return np.concatenate(out) if out else np.zeros(0)


def speed_phase(J):
    """mean mmae-normalized upper-body speed per 10 phase bins of the chunk grid -> (sum[10], count[10])"""
    spd = E.body_speed_central(J, a.fps)[:, ba.upper_bodies] / (mmae[ba.upper_bodies] + 1e-8)
    t = np.arange(J.shape[0]) / a.fps; b = np.minimum(((t % a.period) / a.period * 10).astype(int), 9)
    v = spd.mean(1); return np.bincount(b, weights=v, minlength=10), np.bincount(b, minlength=10)


def hist(ts):
    ph = (ts % a.period) / a.period; h = np.histogram(ph, bins=10, range=(0, 1))[0]
    return h / max(h.sum(), 1)


def pause_stats(J, thr=0.15, min_s=0.5):
    """frames whose mean mmae-normalized upper-body speed < thr (still), and still runs >= min_s (pauses)"""
    v = (E.body_speed_central(J, a.fps)[:, ba.upper_bodies] / (mmae[ba.upper_bodies] + 1e-8)).mean(1)
    still = v < thr; runs, n = 0, 0
    for x in np.r_[still, False]:
        if x: n += 1
        else:
            runs += n >= min_s * a.fps; n = 0
    return float(still.sum()), float(runs)


cols = [('GT', None)] + [tuple(p.split('=', 1)) for p in a.panels]
acc = {lab: [] for lab, _ in cols}; dur = 0.0; sp = {lab: np.zeros((2, 10)) for lab, _ in cols}
ps = {lab: {} for lab, _ in cols}   # speaker -> [still frames, pauses, frames, div sum]
for s in stems:
    gr = E.load_repr(f'{a.ref_dir}/{s}.pt', a.fps)
    prs = [E.load_repr(f'{d}/{s}.pt', a.fps) for _, d in cols[1:]]
    if gr is None or any(p is None for p in prs): continue
    n = min([gr.shape[0]] + [p.shape[0] for p in prs])
    if n < 8: continue
    dur += n / a.fps
    for lab, r in [('GT', gr)] + [(l, p) for (l, _), p in zip(cols[1:], prs)]:
        Jp = E.fk_positions_ba(r[:n]); acc[lab].append(beat_times(Jp)); sp[lab] += np.stack(speed_phase(Jp))
        st_, pa_ = pause_stats(Jp); fl = Jp.reshape(n, -1); dv = float(np.abs(fl - fl.mean(0)).sum())
        k = '_'.join(s.split('_')[:3]); ps[lab].setdefault(k, np.zeros(4)); ps[lab][k] += [st_, pa_, n, dv]
print(f'{len(acc["GT"])} clips, {dur / 60:.1f} min; phase bins of the {a.period * 1000:.0f} ms chunk grid (0 = chunk start)')
for lab, _ in cols:
    ts = np.concatenate(acc[lab]); h = hist(ts)
    print(f'{lab:16s} beats/s/body {len(ts) / dur / len(ba.upper_bodies):.3f} | max/min bin {h.max() / max(h.min(), 1e-9):.2f} | ' + ' '.join(f'{x:.3f}' for x in h))
print('mean normalized upper-body speed per phase bin (relative to the clip-set mean of that label):')
for lab, _ in cols:
    m = sp[lab][0] / np.maximum(sp[lab][1], 1); print(f'{lab:16s} mean {m.mean():.3f} | ' + ' '.join(f'{x / m.mean():.3f}' for x in m))
print('still frames (mean normalized upper-body speed < 0.15) and pauses (still >= 0.5 s) per minute; Div per speaker:')
for lab, _ in cols:
    tot = sum(ps[lab].values())
    per = '  '.join(f"{k.split('_')[2]}: div {v[3] / v[2]:.3f} still {100 * v[0] / v[2]:.1f}%" for k, v in sorted(ps[lab].items()))
    print(f'{lab:16s} still {100 * tot[0] / tot[2]:.1f}%  pauses/min {tot[1] / (tot[2] / a.fps / 60):.2f} | {per}')

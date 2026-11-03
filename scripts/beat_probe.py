"""Beat-metric validity test (11-03, Claude).  python scripts/beat_probe.py out.json label=dir ...  (dir: gen_Hinf_<ci>.npy, gt_<ci>.npy)
A beat metric is only usable if, on GROUND TRUTH, the true audio scores clearly above the same audio time-shifted
(+-0.6/1.0/1.6/2.4 s, circular) or taken from another clip.  Candidates (EX002 arms, 50 Hz, audio events = onset+beat
prosody features of the dev view):
  acc      current beat_align: mean normalised arm acceleration on audio events (jitter-dominated, see brief 2.0f)
  acc_lp   the same after a 4 Hz low-pass of the joint trajectory (gesture scale)
  emage    EMAGE/ECHO-G BA: motion beats = local minima of (low-passed) arm speed while moving, Gaussian
           (sigma 0.3 s) hit of every audio event by its nearest motion beat
  stroke   gesture strokes = prominent peaks of low-passed arm speed (prominence >= 0.25 x the clip's own mean speed),
           Gaussian (sigma 0.15 s) hit of every audio event; also strokes / s
Per metric: real, shifted, other-clip, real-shift gap and the share of clips where real > shifted."""
import json, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, find_peaks, argrelextrema
from scmg.data import View, WindowDataset
from scmg.common import CHUNK_FRAMES
ARMS = list(range(4, 18)); FPS = 50
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st = json.load(open(D + '/STATS_v36_1.json'))
v = View(split='val', stats=st, mmap=True)
BA, BB = butter(4, 4.0 / (FPS / 2))
SHIFTS = [int(s * FPS) for s in (-2.4, -1.6, -1.0, -0.6, 0.6, 1.0, 1.6, 2.4)]


def events(ci):
    nch = v.clips[ci]['frames'] // CHUNK_FRAMES
    s = WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=8).sample(ci, 0, np.random.default_rng(ci))
    b = np.clip(s['feat'][:, 80] + s['feat'][:, 85], 0, None)
    return b


def lp(q): return filtfilt(BA, BB, q, axis=0) if len(q) > 30 else q


def ev_times(b):
    pk, _ = find_peaks(b, height=0.5 * b.max() if b.max() > 0 else 1, distance=int(0.15 * FPS))
    return pk / FPS


def gahr(audio_t, motion_t, sigma):
    if len(audio_t) == 0: return np.nan
    if len(motion_t) == 0: return 0.
    d = np.abs(audio_t[:, None] - motion_t[None]).min(1)
    return float(np.exp(-d ** 2 / (2 * sigma ** 2)).mean())


def metrics(q, b):
    out = {}
    acc = np.linalg.norm(q[2:, ARMS] - 2 * q[1:-1, ARMS] + q[:-2, ARMS], axis=-1); acc /= max(acc.mean(), 1e-9)
    w = b[1:-1]; out['acc'] = float((acc * w).sum() / max(w.sum(), 1e-9))
    ql = lp(q[:, ARMS])
    al = np.linalg.norm(ql[2:] - 2 * ql[1:-1] + ql[:-2], axis=-1); al /= max(al.mean(), 1e-9)
    out['acc_lp'] = float((al * w).sum() / max(w.sum(), 1e-9))
    sp = np.linalg.norm(np.diff(ql, axis=0), axis=-1) * FPS
    at = ev_times(b[1:])
    mins = argrelextrema(sp, np.less, order=3)[0]; mins = mins[sp[mins] > 0.0]
    moving = np.convolve(sp > 0.3 * sp.mean(), np.ones(7), 'same') > 0
    out['emage'] = gahr(at, mins[moving[mins]] / FPS, 0.3)
    pk, _ = find_peaks(sp, prominence=0.25 * max(sp.mean(), 1e-9), distance=int(0.2 * FPS))
    out['stroke'] = gahr(at, pk / FPS, 0.15); out['stroke_rate'] = len(pk) / (len(sp) / FPS)
    return out


def score(q, b, b_other):
    r = metrics(q, b)
    sh = [metrics(q, np.roll(b, s)) for s in SHIFTS]
    ot = metrics(q, b_other)
    return r, {k: float(np.nanmean([x[k] for x in sh])) for k in r}, ot


res = {}
EV = {}
for spec in sys.argv[2:]:
    lab, d = spec.split('=', 1); rows = []
    for ci in range(43):
        try: x = np.load(f'{d}/gen_Hinf_{ci}.npy') if lab != 'gt' else np.load(f'{d}/gt_{ci}.npy')
        except FileNotFoundError: continue
        if ci not in EV: EV[ci] = events(ci)
        oc = (ci + 7) % 43
        if oc not in EV: EV[oc] = events(oc)
        b = EV[ci]; n = min(len(x), len(b)); bo = np.resize(EV[oc], n)
        rows.append(score(x[:n].astype(np.float64), b[:n], bo))
    agg = {}
    for k in rows[0][0]:
        r = np.array([a[k] for a, _, _ in rows]); s = np.array([b_[k] for _, b_, _ in rows]); o = np.array([c[k] for _, _, c in rows])
        agg[k] = dict(real=float(np.nanmean(r)), shift=float(np.nanmean(s)), other=float(np.nanmean(o)),
                      gap=float(np.nanmean(r - s)), win=float(np.nanmean(r > s)))
    res[lab] = dict(n=len(rows), **agg)
    print(lab, json.dumps({k: {kk: round(vv, 3) for kk, vv in a.items()} if isinstance(a, dict) else a for k, a in res[lab].items()}), flush=True)
json.dump(res, open(sys.argv[1], 'w'), indent=1)

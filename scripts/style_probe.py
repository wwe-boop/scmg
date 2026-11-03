"""Gesture style statistics (11-03, Claude; user targets: upper arm moves little, forearm does the work, gestures at
waist height, large amplitude, slow, few gestures per unit time).  EX002 34-D radians, 50 Hz.
    python style_probe.py out.json label=<dir with gen_Hinf_<ci>.npy | gt_<ci>.npy (label gt*)> ...  [train:<source>[:N]] ...
FK via the EX002 MuJoCo model of retarget_g1_ex002.Retargeter; positions in the torso (waist_pitch_link) frame, metres.
Per arm, then averaged:
  ua_share     upper-arm share of hand motion: mean |v elbow| / (mean |v elbow| + mean |v wrist - v elbow|)  (low = forearm-led)
  hand_h       mean wrist height above the waist joint (m);  below_chest  fraction of frames with wrist < waist + 0.25 m
  amp          wrist position spread: sqrt(trace cov) of the wrist path (m)
  v50 / v95    wrist speed percentiles (m/s), 4 Hz low-passed (gesture scale)
  rate         strokes / s: wrist-speed peaks with prominence >= 0.15 m/s, >= 0.25 s apart
  stroke_amp   mean wrist displacement between consecutive speed minima around each stroke (m)"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/code')
from retarget_g1_ex002 import Retargeter
FPS = 50; BA, BB = butter(4, 4.0 / (FPS / 2))
rt = Retargeter()


def fk(q):
    E, W = [], []
    for t in range(len(q)):
        rt.set_ex(q[t]); R = rt.d.xmat[rt.et].reshape(3, 3); o = rt.d.xpos[rt.et]
        E.append([R.T @ (rt.d.xpos[s['ee']] - o) for s in rt.sides]); W.append([R.T @ (rt.d.xpos[s['ew']] - o) for s in rt.sides])
    return np.array(E), np.array(W)                      # [T, 2, 3]


def stats(q):
    q = q[::2] if len(q) > 3000 else q                  # (only for very long clips) keep FK cheap
    fps = FPS if len(q) <= 3000 else FPS / 2
    E, W = fk(q.astype(np.float64))
    if len(E) > 30: E, W = filtfilt(BA, BB, E, axis=0), filtfilt(BA, BB, W, axis=0)
    out = {k: [] for k in ('ua_share', 'hand_h', 'below_chest', 'amp', 'v50', 'v95', 'rate', 'stroke_amp')}
    z0 = 0.0                                            # torso-frame origin = waist_pitch joint
    for a in range(2):
        ve = np.linalg.norm(np.diff(E[:, a], axis=0), axis=-1) * fps
        vw = np.linalg.norm(np.diff(W[:, a], axis=0), axis=-1) * fps
        vf = np.linalg.norm(np.diff(W[:, a] - E[:, a], axis=0), axis=-1) * fps
        out['ua_share'].append(ve.mean() / max(ve.mean() + vf.mean(), 1e-9))
        h = W[:, a, 2] - z0; out['hand_h'].append(h.mean()); out['below_chest'].append(float((h < 0.25).mean()))
        out['amp'].append(float(np.sqrt(np.trace(np.cov(W[:, a].T)))))
        out['v50'].append(np.percentile(vw, 50)); out['v95'].append(np.percentile(vw, 95))
        pk, _ = find_peaks(vw, prominence=0.15, distance=int(0.25 * fps)); out['rate'].append(len(pk) / (len(vw) / fps))
        mn, _ = find_peaks(-vw, distance=int(0.15 * fps))
        sa = [np.linalg.norm(W[mn[np.searchsorted(mn, p)], a] - W[mn[np.searchsorted(mn, p) - 1], a])
              for p in pk if 0 < np.searchsorted(mn, p) < len(mn)]
        out['stroke_amp'].append(float(np.mean(sa)) if sa else 0.)
    return {k: float(np.mean(v)) for k, v in out.items()}


def agg(rows): return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | {'n': len(rows)}


res = {}
for spec in sys.argv[2:]:
    if spec.startswith('train:'):
        import torch  # noqa: F401  (scmg.data needs it)
        from scmg.data import View
        _, src, *n = spec.split(':'); n = int(n[0]) if n else 60
        D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
        if 'V' not in globals(): V = View(split='train', stats=json.load(open(D + '/STATS_v36_1.json')), mmap=True)
        rng = np.random.default_rng(0); rows = []
        if src == 'list':
            from collections import Counter
            cnt = Counter(c['source'] for c in V.clips); hrs = Counter()
            for c in V.clips: hrs[c['source']] += c['frames'] / FPS / 3600
            print({k: (cnt[k], round(hrs[k], 1)) for k in cnt}, flush=True); continue
        cand = [i for i, c in enumerate(V.clips) if c['source'] == src and c['frames'] >= 400]
        for i in rng.permutation(cand)[:n]:
            c = V.clips[i]; q = np.asarray(V.q[c['start']:c['start'] + min(c['frames'], 1500)], np.float64)
            rows.append(stats(q))
        lab = 'train:' + src
    else:
        lab, d = spec.split('=', 1); rows = []
        for ci in range(43):
            f = Path(d) / (f'gt_{ci}.npy' if lab.startswith('gt') else f'gen_{os.environ.get("GEN_TAG", "Hinf")}_{ci}.npy')
            if f.exists(): rows.append(stats(np.load(f)))
    if rows: res[lab] = agg(rows); print(lab, json.dumps({k: round(v, 3) for k, v in res[lab].items()}), flush=True)
    else: print(lab, 'no data', flush=True)
try: json.dump(res, open(sys.argv[1], 'w'), indent=1)
except OSError as e: print('could not write', e)

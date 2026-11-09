"""SRGR on the G1 benchmark (11-07): BEAT's Semantic-Relevant Gesture Recall, computed on G1 forward kinematics.

BEAT (Liu et al., ECCV 2022) weights a per-frame PCK against the ground truth by the annotated semantic relevance of the
frame. Here:
  positions  : official ECHO-G FK (g1_motion_cls.fk_positions_ba: zero translation, identity base orientation) of the
               upper-body bodies (BeatAlign.upper_bodies), the same bodies as the official BA / Div.
  PCK_t      : share of those bodies within delta (m) of the GT body at frame t (same clip, same frame, 30 fps).
  weight w_t : the BEAT2 annotation score of the segment covering clip time t (5th column of beat_english_v2.0.0/sem:
               beat_align 0.1, semantic l / m / h e.g. 0.5 / 1.0), mapped to clip time by echog.start_sec; 0 where no
               segment covers t.
  SRGR       = sum_t w_t PCK_t / sum_t w_t        (pooled over clips)
Also reported: PCK (uniform weights), PCK_sem (frames inside deictic / iconic / metaphoric segments only),
PCK_beat (beat_align frames only), and SRGR - PCK. Thresholds: --deltas (default 0.1 0.2 m).
This is a GT-matching metric (one-to-many gestures are penalised); it is reported because it is BEAT's public protocol,
next to the matched-vs-mismatched recognition test (pair_win).
    PYTHONPATH=<ECHO-G official>/src python scripts/echog_g1/srgr_g1.py --gt-dir G --stems S --view V --out J label=dir ...
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import torch
from echo_g.evaluation import g1_motion_cls as E

ap = argparse.ArgumentParser()
ap.add_argument('--gt-dir', required=True); ap.add_argument('--stems', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--view', default='/home/xr-25b7/bulk/cpfs-mirror/dataset/zehan/gesture/echog-g1-v1')
ap.add_argument('--sem', default='/home/xr-25b7/bulk/dsh-beat2-sem/data/beat2/sem_en')
ap.add_argument('--deltas', type=float, nargs='+', default=[0.1, 0.2]); ap.add_argument('--fps', type=float, default=30.)
ap.add_argument('--mmae-file', default='/mnt/bulk/zehan/echog-dataset/official-meta/eval_assets/mmae/g1_mmae_30body_30fps.npy')
ap.add_argument('panels', nargs='+', help='label=pred_dir')
a = ap.parse_args()


def read_sem(p):
    out = []
    for line in Path(p).read_text(errors='replace').splitlines():
        f = line.split('\t')
        if len(f) < 5: continue
        m = re.match(r'\d+_(beat_align|deictic|iconic|metaphoric)', f[0].strip())
        if not m: continue
        try: s0, s1, w = float(f[1]), float(f[2]), float(f[4])
        except ValueError: continue
        out.append((m.group(1), s0, s1, w))
    return out


meta = {}
for j in [Path(a.view) / 'train.json', Path(a.view) / 'test' / 'train.json']:
    if j.exists(): meta.update({x['id']: x['echog'] for x in json.loads(j.read_text()) if 'echog' in x})
semc = {}
E._G1_MEAN_VEL_PATH = a.mmae_file
ba = E.BeatAlign(a.fps); UB = ba.upper_bodies
stems = [s.strip() for s in open(a.stems) if s.strip()]
panels = [tuple(p.split('=', 1)) for p in a.panels]


def pos(rr): return np.asarray(E.fk_positions_ba(rr))[:, UB]


K = ['w', 'wp', 'n', 'p', 'ns', 'ps', 'nb', 'pb']
acc = {lab: {d: dict.fromkeys(K, 0.) for d in a.deltas} for lab, _ in panels}
used = 0; skipped = dict(no_meta=0, no_sem=0, missing=0)
for s in stems:
    m = meta.get(s)
    if m is None: skipped['no_meta'] += 1; continue
    rec = m['recording']
    if rec not in semc:
        p = Path(a.sem) / f'{rec}.txt'; semc[rec] = read_sem(p) if p.exists() else None
    segs = semc[rec]
    if not segs: skipped['no_sem'] += 1; continue
    if not all(Path(f'{d}/{s}.pt').exists() for d in [a.gt_dir] + [d for _, d in panels]): skipped['missing'] += 1; continue
    gt = E.load_repr(f'{a.gt_dir}/{s}.pt', a.fps); preds = [E.load_repr(f'{d}/{s}.pt', a.fps) for _, d in panels]
    if gt is None or any(x is None for x in preds): skipped['missing'] += 1; continue
    n = min([len(gt)] + [len(x) for x in preds])
    t = np.arange(n) / a.fps + float(m['start_sec'])
    w = np.zeros(n); is_sem = np.zeros(n, bool); is_beat = np.zeros(n, bool)
    for kind, s0, s1, sw in segs:
        inn = (t >= s0) & (t < s1)
        if not inn.any(): continue
        w[inn] = sw
        if kind == 'beat_align': is_beat |= inn
        else: is_sem |= inn
    G = pos(gt[:n]); used += 1
    for (lab, _), x in zip(panels, preds):
        dist = np.linalg.norm(pos(x[:n]) - G, axis=-1)          # [n, bodies]
        for d in a.deltas:
            pck = (dist < d).mean(1); A = acc[lab][d]
            A['w'] += w.sum(); A['wp'] += (w * pck).sum(); A['n'] += n; A['p'] += pck.sum()
            A['ns'] += is_sem.sum(); A['ps'] += pck[is_sem].sum(); A['nb'] += is_beat.sum(); A['pb'] += pck[is_beat].sum()
res = {}
for lab, _ in panels:
    res[lab] = {}
    for d in a.deltas:
        A = acc[lab][d]; r = dict(SRGR=A['wp'] / A['w'], PCK=A['p'] / A['n'], PCK_sem=A['ps'] / max(A['ns'], 1), PCK_beat=A['pb'] / max(A['nb'], 1))
        r['SRGR_minus_PCK'] = r['SRGR'] - r['PCK']; res[lab][str(d)] = r
        print(f"{lab:22s} d={d:.2f} SRGR {r['SRGR']:.4f} PCK {r['PCK']:.4f} PCK_sem {r['PCK_sem']:.4f} PCK_beat {r['PCK_beat']:.4f} "
              f"SRGR-PCK {r['SRGR_minus_PCK']:+.4f}", flush=True)
print('clips used', used, 'skipped', skipped)
json.dump(dict(res=res, clips=used, skipped=skipped, stems=a.stems, deltas=a.deltas, bodies=[int(b) for b in UB]), open(a.out, 'w'), indent=1)

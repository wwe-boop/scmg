"""Human joint envelope of the EX002 training view (landing L1, 11-05): per-source and pooled per-joint quantiles of the
arm / waist / head joints, used by (a) the wrist-eversion metric (scripts/landing/landing_metrics.py) and (b) the
inference arm envelope (scmg/landing.py ArmEnvelope).

Reference pool ("clean human") = the BEAT2 main-body sources (audit problem-frame rate 0.8-1.8 %): beat2_en, beat2_zh,
beat2_en_additional, TRAIN items only (the fixed v3.2 val ids are excluded), frames flagged by the pose-audit ARM_MASK
excluded. Per-source quantiles (all train frames, mask not applied) are reported for the audit table only.

    python scripts/landing/human_envelope.py <out.json> [--stride 4]
Reads the view through scmg.common.P (SCMG_LOCAL_MIRROR). CPU only.
"""
import argparse, json, hashlib
import numpy as np
from scmg.common import P, CPFS

VIEW = CPFS + '/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1'
MASK = CPFS + '/USERS/zehan/gesture/pose-audit-1104/full/ARM_MASK.npy'
CLEAN = ('beat2_en', 'beat2_zh', 'beat2_en_additional')
QS = [0.1, 0.5, 1, 2.5, 5, 25, 50, 75, 95, 97.5, 99, 99.5, 99.9]
COLS = list(range(18))                      # waist 2, head 2, arms 2 x 7 (fingers are not sent to the robot)

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--stride', type=int, default=4)
a = ap.parse_args()
meta = json.load(open(P(VIEW + '/META.json'))); names = meta['joints'][:18]
items = json.load(open(P(VIEW + '/train.json'))); train = set(json.load(open(P(VIEW + '/SPLIT.json')))['train_ids'])
q = np.load(P(VIEW + '/train_q.npy'), mmap_mode='r'); mask = np.load(P(MASK), mmap_mode='r')
if len(mask) != len(q): raise SystemExit(f'ARM_MASK rows {len(mask)} != train_q rows {len(q)}')
rows = {}                                   # source -> (all-frame rows, clean rows)
for it in items:
    if it['id'] not in train: continue
    r = np.arange(it['offset'], it['offset'] + it['frames'], a.stride)
    rows.setdefault(it['source'], []).append(r)
out = dict(view=VIEW, mask=MASK, mask_rows=int(len(mask)), stride=a.stride, quantiles=QS, joints=names, clean_sources=list(CLEAN),
           per_source={}, note='quantiles in rad over TRAIN items (val ids excluded); pooled = clean sources with ARM_MASK frames removed')
pool = []
for src, rr in sorted(rows.items()):
    r = np.concatenate(rr); x = np.asarray(q[r][:, COLS], np.float64)
    out['per_source'][src] = dict(frames=int(len(r)), q=np.percentile(x, QS, axis=0).round(4).T.tolist())
    if src in CLEAN:
        keep = np.asarray(mask[r]) == 0; pool.append(x[keep])
        out['per_source'][src]['masked_frac'] = round(float(1 - keep.mean()), 4)
    print(src, len(r), flush=True)
pool = np.concatenate(pool)
out['pooled'] = dict(frames=int(len(pool)), q=np.percentile(pool, QS, axis=0).round(4).T.tolist(),
                     median=np.median(pool, 0).round(4).tolist())
s = json.dumps(out, indent=1); open(a.out, 'w').write(s)
print('wrote', a.out, 'sha256', hashlib.sha256(s.encode()).hexdigest()[:12], 'pooled frames', len(pool))

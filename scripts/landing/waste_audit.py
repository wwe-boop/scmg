"""Waste-motion audit of the EX002 training view (landing L4, Claude 11-05). Analysis only: writes per-source rates and
a row-aligned candidate mask; nothing here changes training (using the mask in training needs the user's approval).

Detectors (per train / val item, 50 Hz; thresholds fitted on the clean BEAT2 TRAIN frames, ARM_MASK frames excluded):
  silent_motion  frames of a silence run >= 1.0 s (outside the feat89 speech span flag, column 'valid') whose 0.5 s
                 moving-average arm speed exceeds the clean-human p90 arm speed DURING SPEECH (big gestures with nobody
                 talking: walking around, fidgeting, adjusting the mic, ...)
  waist_turn     frames whose waist yaw deviates from the item median by more than the clean p99.5 deviation, or
                 whose waist yaw speed exceeds the clean p99.5 (turning to a screen / to another person)
  speech_freeze  frames of >= 1.5 s still runs (arm speed < 0.25 x the item's median) entirely inside speech
                 (the "talking but frozen" pattern the user does not want the model to learn)
  any            union (each detector dilated by 2 frames, within the item)

    python scripts/landing/waste_audit.py <out dir>
"""
import argparse, json, hashlib
import numpy as np
from scmg.common import P, CPFS, PROSODY_COLS, FPS
from scmg.landing import runs, ARMS14

VIEW = CPFS + '/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1'
MASK = CPFS + '/USERS/zehan/gesture/pose-audit-1104/full/ARM_MASK.npy'
CLEAN = ('beat2_en', 'beat2_zh', 'beat2_en_additional')
DET = ('silent_motion', 'waist_turn', 'speech_freeze')

ap = argparse.ArgumentParser(); ap.add_argument('out'); a = ap.parse_args()
items = json.load(open(P(VIEW + '/train.json'))); split = json.load(open(P(VIEW + '/SPLIT.json')))
train_ids = set(split['train_ids'])
q = np.load(P(VIEW + '/train_q.npy'), mmap_mode='r'); f = np.load(P(VIEW + '/train_feat.npy'), mmap_mode='r')
amask = np.load(P(MASK), mmap_mode='r'); N = len(q)
VC = PROSODY_COLS['valid']        # feat89 speech flag: utterance span (first unit onset .. last unit offset), present for
                                  # every source (streamer has no voiced / energy prosody: those columns are 0)


def per_item(it):
    a0, n = it['offset'], it['frames']
    x = np.asarray(q[a0:a0 + n], np.float64)
    sp = np.r_[np.linalg.norm(np.diff(x[:, ARMS14], axis=0), axis=1) * FPS, 0.]; sp[-1] = sp[-2] if n > 1 else 0.
    speech = np.asarray(f[a0:a0 + n, VC]) > 0.5
    yaw = x[:, 0]; ysp = np.r_[np.abs(np.diff(yaw)) * FPS, 0.]
    return sp, speech, yaw - np.median(yaw), ysp


# ---- thresholds from clean train frames
sp_speech, ydev, yspd = [], [], []
for it in items:
    if it['source'] not in CLEAN or it['id'] not in train_ids or it['frames'] < 50: continue
    sp, speech, yd, ys = per_item(it)
    keep = np.asarray(amask[it['offset']:it['offset'] + it['frames']]) == 0
    sp_speech.append(sp[keep & speech]); ydev.append(np.abs(yd[keep])); yspd.append(ys[keep])
TH = dict(speech_speed_p90=float(np.percentile(np.concatenate(sp_speech), 90)),
          waist_dev_p995=float(np.percentile(np.concatenate(ydev), 99.5)),
          waist_speed_p995=float(np.percentile(np.concatenate(yspd), 99.5)))
print('thresholds', TH, flush=True)

# ---- detect
out = np.zeros(N, np.uint8)                  # bit 0 silent_motion, bit 1 waist_turn, bit 2 speech_freeze
stat = {}
k = int(round(0.5 * FPS)); ker = np.ones(k) / k
for it in items:
    a0, n = it['offset'], it['frames']
    if n < 10: continue
    sp, speech, yd, ys = per_item(it)
    m = np.zeros((3, n), bool)
    sm = np.convolve(sp, ker, mode='same')
    for s0, s1 in runs(~speech):
        if s1 - s0 >= FPS: m[0, s0:s1] = sm[s0:s1] > TH['speech_speed_p90']
    m[1] = (np.abs(yd) > TH['waist_dev_p995']) | (ys > TH['waist_speed_p995'])
    still = sp < 0.25 * max(np.median(sp), 1e-6)
    for s0, s1 in runs(still & speech):
        if s1 - s0 >= int(1.5 * FPS): m[2, s0:s1] = True
    for d in range(3):                        # dilate 2 frames within the item
        r = m[d].copy()
        for sh in (1, 2): r[sh:] |= m[d][:-sh]; r[:-sh] |= m[d][sh:]
        out[a0:a0 + n] |= (r.astype(np.uint8) << d)
    s = stat.setdefault(it['source'], dict(frames=0, speech_frames=0, silence_frames=0, items=0, **{d: 0 for d in DET}, any=0))
    s['frames'] += n; s['items'] += 1; s['speech_frames'] += int(speech.sum()); s['silence_frames'] += int((~speech).sum())
    for d, name in enumerate(DET): s[name] += int(((out[a0:a0 + n] >> d) & 1).sum())
    s['any'] += int((out[a0:a0 + n] > 0).sum())
rep = dict(view=VIEW, thresholds=TH, detectors=DET, bits={d: i for i, d in enumerate(DET)}, per_source={})
tot = dict(frames=0, any=0, **{d: 0 for d in DET})
for src, s in sorted(stat.items()):
    rep['per_source'][src] = dict(frames=s['frames'], hours=round(s['frames'] / FPS / 3600, 2), items=s['items'],
                                  silence_share=round(s['silence_frames'] / s['frames'], 4),
                                  **{d: round(s[d] / s['frames'], 4) for d in DET}, any=round(s['any'] / s['frames'], 4),
                                  any_and_arm_mask=None)
    for d in DET + ('any',): tot[d] = tot.get(d, 0) + s[d]
    tot['frames'] += s['frames']
rep['total'] = {d: round(tot[d] / tot['frames'], 4) for d in DET + ('any',)}
# overlap with the pose-audit arm mask (are these new frames or already masked?)
am = np.asarray(amask) > 0; anyw = out > 0
for src in stat:
    rows = np.concatenate([np.arange(it['offset'], it['offset'] + it['frames']) for it in items if it['source'] == src])
    rep['per_source'][src]['any_and_arm_mask'] = round(float((anyw[rows] & am[rows]).sum() / max(1, anyw[rows].sum())), 4)
np.save(f'{a.out}/WASTE_MASK.npy', out)
rep['mask_sha256'] = hashlib.sha256(out.tobytes()).hexdigest()
json.dump(rep, open(f'{a.out}/WASTE_AUDIT.json', 'w'), indent=1)
print(json.dumps(rep['per_source'], indent=0)); print('total', rep['total'])

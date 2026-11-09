"""Jumps in the TRAINING motion itself (user 11-02: real human data also jumps).
    python scripts/glitch_audit.py <out.json> [factor]
A frame t is a glitch if any joint's |q[t]-q[t-1]|*50 exceeds factor x the joint's velocity limit (Fable D2.2 table,
waist/head 1.0 rad/s; default factor 2). Differences across recording (clip) boundaries are ignored.
Writes per-source frame/clip rates, the worst joints, and a per-frame glitch mask (GLITCH_MASK.npy, uint8 over
train_q) that the loaders can use to drop glitch frames (and their neighbours) from the losses."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

V = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1')
VMAX = np.r_[[1.0] * 4, ([2.0] * 3 + [2.5] + [2.51] * 3) * 2, [4.0] * 16]
out = Path(sys.argv[1]); factor = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
q = np.load(V / 'train_q.npy', mmap_mode='r'); joints = json.load(open(V / 'META.json'))['joints']
items = json.loads((V / 'train.json').read_text())
clips = defaultdict(list)
for it in items: clips[it['clip']].append(it)
mask = np.zeros(len(q), np.uint8)
per_src = defaultdict(lambda: dict(frames=0, glitch_frames=0, clips=0, glitch_clips=0))
joint_hits = np.zeros(len(joints), np.int64); worst = []
for c, xs in clips.items():
    xs.sort(key=lambda x: x['offset']); a = xs[0]['offset']; n = sum(x['frames'] for x in xs)
    src = xs[0]['data_source']; seg = np.asarray(q[a:a + n], np.float32)
    s = per_src[src]; s['frames'] += n; s['clips'] += 1
    if n < 2: continue
    v = np.abs(np.diff(seg, axis=0)) * 50
    hit = v > factor * VMAX                                      # [n-1, J]
    bad = hit.any(1)
    if bad.any():
        s['glitch_frames'] += int(bad.sum()); s['glitch_clips'] += 1; joint_hits += hit.sum(0)
        idx = np.where(bad)[0] + 1
        for d in (-1, 0, 1): mask[np.clip(a + idx + d, a, a + n - 1)] = 1    # the jump frame and its neighbours
        k = int(np.argmax(v.max(1))); worst.append((float(v.max()), c, src, joints[int(np.argmax(v[k]))], int(k + 1)))
worst.sort(reverse=True)
res = dict(factor=factor, total_frames=int(len(q)), masked_frames=int(mask.sum()),
           per_source={k: dict(v, frame_rate=v['glitch_frames'] / max(v['frames'], 1), clip_rate=v['glitch_clips'] / max(v['clips'], 1))
                       for k, v in per_src.items()},
           worst_joints=sorted([(joints[i], int(joint_hits[i])) for i in range(len(joints))], key=lambda x: -x[1])[:8],
           worst_jumps=[dict(speed=w[0], clip=w[1], source=w[2], joint=w[3], frame=w[4]) for w in worst[:15]])
out.write_text(json.dumps(res, indent=1)); np.save(out.parent / 'GLITCH_MASK.npy', mask)
print(json.dumps({k: (round(v['frame_rate'], 5), round(v['clip_rate'], 4)) for k, v in res['per_source'].items()}, indent=1))
print('worst joints', res['worst_joints'][:5])

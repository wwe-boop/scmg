"""Is our beat_align (arm acceleration at prosodic onsets/beats, normalised by the clip mean; 1.0 = chance) sensitive to
audio-motion misalignment?  GT vs GT with the beat signal shifted by +-0.5 / +-1 s, another clip's motion, random walk."""
import json
import numpy as np
from scmg.common import PROSODY_COLS
from scmg.data import View, WindowDataset, VIEW
from scmg.evaluate import beat_align
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st = json.load(open(D + '/STATS_v36_1.json'))
v = View(VIEW, 'val', D + '/text-e5-v36_1', st, mmap=True); v.glitch = None
qm, qs = np.asarray(st['q_mean'], np.float32), np.asarray(st['q_std'], np.float32)
clips = [ci for ci in range(len(v.clips)) if v.clips[ci]['frames'] // 16 >= 3][:43]
G, BT = {}, {}
for ci in clips:
    T = v.clips[ci]['frames'] // 16 * 16
    s = WindowDataset(v, T, ref_frames=16, max_tokens=64).sample(ci, 0, np.random.default_rng(ci))
    G[ci] = s['q'] * qs + qm
    BT[ci] = np.clip(s['feat'][:, PROSODY_COLS['onset']] + s['feat'][:, PROSODY_COLS['beat']], 0, None)
rng = np.random.default_rng(0); res = {}
m = lambda f: float(np.mean([f(ci) for ci in clips]))
res['gt'] = m(lambda ci: beat_align(G[ci], BT[ci]))
for sh in (0.5, 1.0, -0.5, -1.0):
    res[f'shift{sh:+.1f}s'] = m(lambda ci: beat_align(G[ci], np.roll(BT[ci], int(sh * 50))))
res['other_clip'] = m(lambda ci: beat_align(G[clips[(clips.index(ci) + 7) % len(clips)]][:len(BT[ci])] if len(G[clips[(clips.index(ci) + 7) % len(clips)]]) >= len(BT[ci]) else G[ci][::-1], BT[ci]))
res['random_walk'] = m(lambda ci: beat_align(G[ci][0] + np.cumsum(rng.normal(0, 0.01, G[ci].shape), 0), BT[ci]))
res['time_reversed_motion'] = m(lambda ci: beat_align(G[ci][::-1].copy(), BT[ci]))
print(json.dumps(res, indent=1))
json.dump(res, open('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baselines/beat_align_validation.json', 'w'), indent=1)

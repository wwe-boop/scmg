"""Which audio route carries the visible beat? (11-04 Claude; user: rhythm -> replace modules to fit better)
    beat_route.py <out.json> <ckpt>@<cfg_w> [n_clips=43]
Generates the dev clips (offline, final text, inference fixes on, seed 0) three ways by wrapping the model at inference:
  full        unchanged
  no_planner  the planner sees the learned null audio tokens (mask off) -> plan cannot follow the speech rhythm
  no_local    the Local Flow audio window is the learned null audio (mask kept, as in the CFG null branch)
and scores the validated gesture-scale beat (4 Hz low-passed arm acceleration on prosodic-prominence events, real minus
circular shifts), arm speed vs GT and stillness."""
import json, sys
import numpy as np, torch
from scipy.signal import butter, filtfilt
from scmg.data import View, WindowDataset
from scmg.evaluate import load
from scmg.infer import generate
from scmg.common import CHUNK_FRAMES, PROSODY_COLS
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'; FPS = 50; ARMS = list(range(4, 18)); BA, BB = butter(4, 4.0 / (FPS / 2))
st = json.load(open(D + '/STATS_v36_1.json')); qm, qs = np.array(st['q_mean'], np.float32), np.array(st['q_std'], np.float32)
fm, fs_ = np.array(st['f_mean'], np.float32), np.array(st['f_std'], np.float32)
out = sys.argv[1]; ck, w = sys.argv[2].rsplit('@', 1); w = float(w); N = int(sys.argv[3]) if len(sys.argv) > 3 else 43


def gap(q, prom):
    ql = filtfilt(BA, BB, q[:, ARMS], axis=0); acc = np.linalg.norm(ql[2:] - 2 * ql[1:-1] + ql[:-2], axis=-1); acc /= max(acc.mean(), 1e-9)
    b = np.clip(prom[1:-1][:len(acc)], 0, None); acc = acc[:len(b)]; sc = lambda x: float((acc * x).sum() / max(x.sum(), 1e-9))
    return sc(b) - np.mean([sc(np.roll(b, int(s * FPS))) for s in (-2.4, -1.6, -1.0, -0.6, 0.6, 1.0, 1.6, 2.4)])


m, vae = load(ck, '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/trained/vae-r3/final.pt', 'cuda')
plan0, flow0 = m.plan, m.flow_velocity
def no_planner(*a, **k):
    a = list(a); a[1] = m.audio_null.to(a[1].dtype).expand_as(a[1]); a[2] = torch.zeros_like(a[2]); return plan0(*a, **k)
def no_local(*a, **k):
    a = list(a); a[5] = m.audio_null.to(a[5].dtype).expand_as(a[5]); return flow0(*a, **k)
v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', tok_cache=D + '/tok-qwen25-v36_1')
res = {}
for mode in ('full', 'no_planner', 'no_local'):
    m.plan = no_planner if mode == 'no_planner' else plan0; m.flow_velocity = no_local if mode == 'no_local' else flow0
    G, SP, ST = [], [], []
    for ci in range(min(N, len(v.clips))):
        nch = v.clips[ci]['frames'] // CHUNK_FRAMES
        s = WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
        cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend') if k in s}
        y, _, _ = generate(m, vae, s['feat'], cond, st, nch, H=10 ** 9, steps=10, seed=0, standby=st['q_mean'], ctx_t=m.ctx_t, cfg_w=w)
        gt = s['q'] * qs + qm; n = min(len(y), len(gt)); y = y[:n]
        prom = s['feat'][:n, PROSODY_COLS['prominence']] * fs_[PROSODY_COLS['prominence']] + fm[PROSODY_COLS['prominence']]
        sp = np.linalg.norm(np.diff(y[:, ARMS], axis=0), axis=1) * FPS; gs = np.linalg.norm(np.diff(gt[:n, ARMS], axis=0), axis=1) * FPS
        G.append(gap(y, prom)); SP.append(sp.mean() / gs.mean()); ST.append(float((sp < 0.25 * np.median(gs)).mean()))
    res[mode] = dict(beat_gap=float(np.mean(G)), beat_win=float(np.mean(np.array(G) > 0)), speed=float(np.mean(SP)), still=float(np.mean(ST)), n=len(G))
    print(mode, json.dumps(res[mode]), flush=True)
json.dump(res, open(out, 'w'), indent=1)

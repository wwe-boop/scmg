"""Separate the causes of low free-running motion energy.   python scripts/diag_exposure2.py <ckpt> <vae> [n]
 free          normal streaming from the standby (mean) pose
 primed        first 2 chunks forced to ground truth, then free
 plan_gt       planner sees GT history, local flow context = own outputs
 ctx_gt        local flow context = GT, planner sees own outputs
Prints speed ratio overall and per time bin (chunks 0-3, 4-7, 8-15, 16+)."""
import json, sys
import numpy as np, torch
from scmg.common import CHUNK, CHUNK_FRAMES
from scmg.data import View, WindowDataset
from scmg.evaluate import load
from scmg.infer import Streamer
from scmg.model import INF

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st = json.load(open(D + '/STATS_v36_1.json'))
v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1')
m, vae = load(sys.argv[1], sys.argv[2], 'cuda')
qm, qs = np.asarray(st['q_mean'], np.float32), np.asarray(st['q_std'], np.float32)
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
BINS = ((0, 4), (4, 8), (8, 16), (16, 999))
spd = lambda x: np.abs(np.diff(x[:, 4:18], axis=0)).mean(1)              # per-frame arm speed


def run(s, cond, mode, gl, nch):
    S = Streamer(m, vae, s['feat'], cond, st, H=INF, steps=10, standby=qm, ctx_t=getattr(m, 'ctx_t', 1.0))
    gen = []; own = []
    for k in range(nch):
        if mode == 'plan_gt': S.plan_src = [gl[i] for i in range(k * CHUNK)]          # planner: GT history only
        S.tick()
        gen.append(torch.stack(S.lat[-CHUNK:])); own.extend([x.clone() for x in S.lat[-CHUNK:]])
        if mode == 'ctx_gt' or (mode == 'primed' and k < 2):          # S.lat feeds the flow context (and the planner)
            for i in range(CHUNK): S.lat[k * CHUNK + i] = gl[k * CHUNK + i].clone()
        if mode == 'ctx_gt': S.plan_src = list(own)                   # ...but the planner keeps reading own outputs
    with torch.no_grad():
        return vae.decode(torch.cat(gen)[None] * m.lat_std + m.lat_mean)[0].cpu().numpy() * qs + qm


out = {k: [] for k in ('free', 'primed', 'plan_gt', 'ctx_gt')}
for ci in [i for i, c in enumerate(v.clips) if c['frames'] >= 400][:N]:
    nch = v.clips[ci]['frames'] // CHUNK_FRAMES
    s = WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(0))
    gt = s['q'][:nch * CHUNK_FRAMES] * qs + qm
    cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm')}
    with torch.no_grad():
        mu, _ = vae.encode(torch.as_tensor(s['q'][:nch * CHUNK_FRAMES])[None].cuda()); gl = ((mu - m.lat_mean) / m.lat_std)[0]
    sg = spd(gt)
    for mode in out:
        y = run(s, cond, mode, gl, nch); sy = spd(y[:len(gt)])
        r = [float(sy[a * 16:b * 16].mean() / max(sg[a * 16:b * 16].mean(), 1e-6)) if a * 16 < len(sy) else np.nan for a, b in BINS]
        out[mode].append([float(sy.mean() / sg.mean())] + r)
for mode, rows in out.items():
    a = np.nanmean(np.array(rows), 0)
    print(mode, 'overall %.3f | chunks 0-3 %.3f  4-7 %.3f  8-15 %.3f  16+ %.3f' % tuple(a))

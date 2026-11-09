"""Is the low motion energy autoregressive decay (exposure bias) or per-chunk conservatism?
    python scripts/diag_exposure.py <ckpt> <vae> [n_clips]
 free      : normal streaming (history = own outputs)
 oracle    : every chunk generated from GROUND-TRUTH history (committed latents replaced by GT after each tick);
             the generated chunks are collected and decoded offline
Speed ratio = mean |d q_arms| generated / ground truth."""
import json, sys
import numpy as np, torch
from scmg.common import CHUNK, CHUNK_FRAMES
from scmg.data import View, WindowDataset
from scmg.evaluate import load
from scmg.infer import Streamer, generate
from scmg.model import INF

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st = json.load(open(D + '/STATS_v36_1.json'))
v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1')
m, vae = load(sys.argv[1], sys.argv[2], 'cuda')
qm, qs = np.asarray(st['q_mean'], np.float32), np.asarray(st['q_std'], np.float32)
sp = lambda x, g: float(np.abs(np.diff(x[:, 4:18], axis=0)).mean() / np.abs(np.diff(g[:, 4:18], axis=0)).mean())
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
res = {'free': [], 'oracle': [], 'oracle_amp': [], 'free_amp': []}
amp = lambda x, g: float(x[:, 4:18].std(0).mean() / g[:, 4:18].std(0).mean())
for ci in [i for i, c in enumerate(v.clips) if c['frames'] >= 320][:N]:
    nch = v.clips[ci]['frames'] // CHUNK_FRAMES
    s = WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(0))
    gt = s['q'][:nch * CHUNK_FRAMES] * qs + qm
    cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm')}
    free, _, _ = generate(m, vae, s['feat'], cond, st, nch, H=INF, steps=10, standby=qm, ctx_t=getattr(m, 'ctx_t', 1.0))
    n = min(len(free), len(gt)); res['free'].append(sp(free[:n], gt[:n])); res['free_amp'].append(amp(free[:n], gt[:n]))
    with torch.no_grad():
        mu, _ = vae.encode(torch.as_tensor(s['q'][:nch * CHUNK_FRAMES])[None].cuda())
        gl = ((mu - m.lat_mean) / m.lat_std)[0]                      # GT latents, normalised
    S = Streamer(m, vae, s['feat'], cond, st, H=INF, steps=10, standby=qm)
    gen_lat = []
    for k in range(nch):
        S.tick()
        gen_lat.append(torch.stack(S.lat[-CHUNK:]))
        for i in range(CHUNK): S.lat[k * CHUNK + i] = gl[k * CHUNK + i].clone()   # history := ground truth
    with torch.no_grad():
        y = vae.decode((torch.cat(gen_lat)[None] * m.lat_std + m.lat_mean))[0].cpu().numpy() * qs + qm
    res['oracle'].append(sp(y[:n], gt[:n])); res['oracle_amp'].append(amp(y[:n], gt[:n]))
print(json.dumps({k: round(float(np.mean(x)), 3) for k, x in res.items()}))

"""Where do the chunk-boundary jerk spikes come from?  python scripts/diag_boundary.py <ckpt> <vae>
 A  ground truth                                  (reference ratio)
 B  GT latents, offline full decode               (VAE alone)
 C  GT latents, streamer-style tail decode, look-ahead = replicate last latent   (decoder look-ahead missing)
 D  GT latents, streamer-style tail decode, look-ahead = true next latent
 E  generated (offline audio), latent-level jump at chunk boundaries vs inside chunks"""
import json, sys
import numpy as np, torch
from scmg.common import CHUNK, CHUNK_FRAMES, STRIDE
from scmg.data import View, WindowDataset
from scmg.evaluate import load, boundary_ratio
from scmg.infer import generate, Streamer
from scmg.model import INF

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st = json.load(open(D + '/STATS_v36_1.json'))
v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1')
m, vae = load(sys.argv[1], sys.argv[2], 'cuda')
qm, qs = np.asarray(st['q_mean'], np.float32), np.asarray(st['q_std'], np.float32)
res = {k: [] for k in 'ABCDE'}; spd = {k: [] for k in ('B', 'E', 'E_steps20', 'E_cfg2', 'E_steps50')}
sp = lambda x, g: float(np.abs(np.diff(x[:, 4:18], axis=0)).mean() / np.abs(np.diff(g[:, 4:18], axis=0)).mean())
clips = [i for i, c in enumerate(v.clips) if c['frames'] >= 320][:int(sys.argv[3]) if len(sys.argv) > 3 else 12]
for ci in clips:
    nch = v.clips[ci]['frames'] // CHUNK_FRAMES
    s = WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(0))
    gt = s['q'] * qs + qm; res['A'].append(boundary_ratio(gt))
    with torch.no_grad():
        q = torch.as_tensor(s['q'])[None].cuda(); mu, _ = vae.encode(q)
        rb = vae.decode(mu)[0].cpu().numpy() * qs + qm; res['B'].append(boundary_ratio(rb)); spd['B'].append(sp(rb, gt))
        for key, oracle in (('C', False), ('D', True)):
            fr = []
            for k in range(1, nch):
                tail = mu[:, max(0, (k - 3) * CHUNK):(k + 1) * CHUNK]
                nxt = mu[:, (k + 1) * CHUNK:(k + 1) * CHUNK + 1] if oracle and (k + 1) * CHUNK < mu.shape[1] else tail[:, -1:]
                y = vae.decode(torch.cat([tail, nxt], 1))[0]
                nf = y.shape[0] - STRIDE; fr.append(y[nf - CHUNK_FRAMES:nf].cpu().numpy())
            res[key].append(boundary_ratio(np.concatenate(fr) * qs + qm))
    cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm')}
    gen, _, _ = generate(m, vae, s['feat'], cond, st, nch, H=INF, steps=10, n_future=1, standby=qm)
    res['E'].append(boundary_ratio(gen)); n = min(len(gen), len(gt)); spd['E'].append(sp(gen[:n], gt[:n]))
    g2, _, _ = generate(m, vae, s['feat'], cond, st, nch, H=INF, steps=20, n_future=0, standby=qm); spd['E_steps20'].append(sp(g2[:n], gt[:n]))
    g4, _, _ = generate(m, vae, s['feat'], cond, st, nch, H=INF, steps=50, n_future=0, standby=qm); spd['E_steps50'].append(sp(g4[:n], gt[:n]))
    g3, _, _ = generate(m, vae, s['feat'], cond, st, nch, H=INF, steps=10, n_future=0, standby=qm, temperature=0.); spd['E_cfg2'].append(sp(g3[:n], gt[:n]))
print('boundary', json.dumps({k: round(float(np.mean(x)), 3) for k, x in res.items()}))
print('speed ratio', json.dumps({k: round(float(np.mean(x)), 3) for k, x in spd.items()}))

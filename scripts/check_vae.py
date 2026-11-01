"""Gate for stage 0: deterministic reconstruction error of the VAE on the dev set, in radians per joint group, and the
streaming consistency of the decoder (decoding latents chunk by chunk with 1-latent look-ahead == offline decode).
    python scripts/check_vae.py <vae.pt>"""
import json, sys
import numpy as np, torch
from scmg.data import View, WindowDataset, collate, fixed_windows
from scmg.vae import MotionVAE
from scmg.common import STRIDE

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st = json.load(open(D + '/STATS_v36_1.json'))
ck = torch.load(sys.argv[1], map_location='cpu')
vae = MotionVAE(**ck['cfg']).cuda().eval(); vae.load_state_dict(ck['ema'])
v = View(split='val', stats=st, mmap=True)
ds = WindowDataset(v, 256, fixed=fixed_windows(v))
qs = torch.tensor(st['q_std']).cuda()
G = dict(waist=slice(0, 2), head=slice(2, 4), arms=slice(4, 18), fingers=slice(18, 34))
err = {k: [] for k in G}; n = 0
with torch.no_grad():
    for b in torch.utils.data.DataLoader(ds, 64, collate_fn=collate):
        q = b['q'].cuda(); m = b['mask'].cuda() > .5
        mu, _ = vae.encode(q); y = vae.decode(mu)
        e = ((y - q) * qs).abs()[m]                               # rad
        for k, s in G.items(): err[k].append(e[:, s].mean(1).cpu())
    res = {k: dict(mean_deg=float(torch.cat(x).mean() * 57.3), p95_deg=float(torch.cat(x).quantile(.95) * 57.3)) for k, x in err.items()}
    # streaming: decode latents prefix by prefix (as the streamer does) vs offline
    q = b['q'][:8].cuda(); mu, _ = vae.encode(q); off = vae.decode(mu)
    worst = 0.
    for i in range(4, mu.shape[1] - 1, 4):
        part = vae.decode(mu[:, :i + 1])                          # committed latents + 1 look-ahead latent
        worst = max(worst, float(((part[:, :STRIDE * i] - off[:, :STRIDE * i]) * qs).abs().max()))
    res['stream_vs_offline_max_deg'] = worst * 57.3
print(json.dumps(res, indent=1))

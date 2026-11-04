"""Write an identity-codec 'VAE' checkpoint (scmg/vae.py identity=True; 11-04, ECHO-G-style raw-space generation inside the
SCMG chunk machinery).  The latent statistics are measured on training windows exactly like scmg/train.py latent_stats
(normalised frames, so they come out ~0 / ~1).
    SCMG_J=39 SCMG_STRIDE=<4|2> python scripts/echog_g1/make_identity_codec.py <out.pt> --view <view> --stats <STATS.json>
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg.common import J, STRIDE
from scmg.data import View, WindowDataset
from scmg.vae import MotionVAE
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--view', required=True); ap.add_argument('--stats', required=True)
ap.add_argument('--windows', type=int, default=4000); ap.add_argument('--transform', default='none', choices=['none', 'dct'])
a = ap.parse_args()
cfg = dict(z=STRIDE * J, stride=STRIDE, j=J, identity=True, **({'transform': a.transform} if a.transform != 'none' else {}))
v = MotionVAE(**cfg)
view = View(a.view, 'train', None, json.loads(Path(a.stats).read_text()), mmap=True)
ds = WindowDataset(view, 256); rng = np.random.default_rng(0); mus = []
for _ in range(a.windows):
    ci = int(rng.integers(len(view.clips))); c = view.clips[ci]
    if c['frames'] < 256: continue
    q, m, _, _ = ds._window(ci, int(rng.integers(c['frames'] - 255)), 256)
    mu, _ = v.encode(torch.from_numpy(q)[None]); lm = torch.from_numpy(m).view(-1, STRIDE).min(-1).values > .5
    mus.append(mu[0][lm])
mu = torch.cat(mus)
d = dict(cfg=cfg, model={}, ema={}, step=0, vae_cfg=cfg, lat_mean=mu.mean(0).tolist(), lat_std=mu.std(0).clamp(min=1e-3).tolist(),
         note='identity codec: lossless, parameter-free; latent = STRIDE normalised frames concatenated')
Path(a.out).parent.mkdir(parents=True, exist_ok=True); torch.save(d, a.out)
print('wrote', a.out, 'z', cfg['z'], 'latents', len(mu), 'lat_mean |max|', float(mu.mean(0).abs().max()), 'lat_std range',
      float(mu.std(0).min()), float(mu.std(0).max()))

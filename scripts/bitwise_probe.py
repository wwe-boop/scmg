"""Deterministic 3-window generation probe for bit-identical regression checks (e.g. Config.text_dim default).

    python scripts/bitwise_probe.py <vae.pt> <ckpt> <text-cache> <out.npz> [--n 3]
Run it once with the old code and once with the new code on the same GPU, then compare the npz files:
all generated frames and the text_in weight bytes must be identical when text_dim stays 768.
"""
import argparse, hashlib, json
import numpy as np, torch
from scmg.common import CHUNK_FRAMES, U80
from scmg.data import View, WindowDataset
from scmg.evaluate import load
from scmg.infer import generate
from scmg.model import INF

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'

ap = argparse.ArgumentParser()
ap.add_argument('vae'); ap.add_argument('ckpt'); ap.add_argument('text_cache'); ap.add_argument('out')
ap.add_argument('--stats', default=D + '/STATS_v36_1.json')
ap.add_argument('--n', type=int, default=3); ap.add_argument('--frames', type=int, default=256)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--cfg-w', type=float, default=2.5)
a = ap.parse_args()
st = json.load(open(a.stats))
v = View(split='val', stats=st, mmap=True, text_cache=a.text_cache)
clips = [ci for ci in range(len(v.clips)) if v.clips[ci]['frames'] >= a.frames][:a.n]
if len(clips) < a.n: raise SystemExit('not enough long val clips')
m, vae = load(a.ckpt, a.vae, 'cuda'); m.eval()

ys, texts, items = [], [], []
for j, ci in enumerate(clips):
    s = WindowDataset(v, a.frames, max_tokens=256).sample(ci, 0, np.random.default_rng(1000 + j))
    cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid',
                              'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c') if k in s}
    y, _, _ = generate(m, vae, s['feat'], cond, st, a.frames // CHUNK_FRAMES, H=INF, steps=10, seed=a.seed,
                       ctx_t=m.ctx_t, cfg_w=a.cfg_w)
    ys.append(np.asarray(y, np.float32)); texts.append(np.asarray(s['text'], np.float32)); items.append(ci)
    print(f'generated clip {ci} frames {len(y)}', flush=True)

h = hashlib.sha256()
for k in ('text_in.0.weight', 'text_in.0.bias', 'text_in.1.weight', 'text_in.1.bias'):
    h.update(m.state_dict()[k].detach().cpu().numpy().tobytes())
np.savez(a.out, y=np.stack(ys), text=np.stack(texts), clips=np.asarray(items), text_in_sha=np.asarray(h.hexdigest()),
         cfg=json.dumps(dict(base_cfg=m.cfg.dict(), seed=a.seed, cfg_w=a.cfg_w, frames=a.frames)))
print(json.dumps(dict(out=a.out, clips=items, shape=list(ys[0].shape), text_in_sha=h.hexdigest())))

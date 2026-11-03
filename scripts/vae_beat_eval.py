"""Reconstruction check of motion VAEs on the 43 dev clips (11-03): beat_align of GT vs VAE(GT), velocity power above 2 Hz,
speed / p95 speed ratio, position L1 (rad).   python scripts/vae_beat_eval.py label=vae.pt ...  (SCMG_STRIDE must match)"""
import json, sys, numpy as np, torch
from scmg.data import View, WindowDataset
from scmg.vae import MotionVAE
from scmg.evaluate import beat_align, ARMS
from scmg.common import CHUNK_FRAMES, PROSODY_COLS
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'; st = json.load(open(D + '/STATS_v36_1.json'))
qm, qs = np.array(st['q_mean'], np.float32), np.array(st['q_std'], np.float32)
v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1')
clips = []
for ci in range(43):
    nch = v.clips[ci]['frames'] // CHUNK_FRAMES
    s = WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
    n = int(s['mask'].sum()) // 16 * 16
    clips.append((s['q'][:n], np.clip(s['feat'][:n, PROSODY_COLS['onset']] + s['feat'][:n, PROSODY_COLS['beat']], 0, None)))
def hf(q):
    vv = np.diff(q[:, ARMS], axis=0); P = np.abs(np.fft.rfft(vv, axis=0)) ** 2; f = np.fft.rfftfreq(len(vv), 1 / 50)
    return P[f > 2].sum() / P[f > 0.2].sum()
def sp(q): return np.linalg.norm(np.diff(q[:, ARMS], axis=0), axis=1) * 50
gt = dict(beat=np.mean([beat_align(q * qs + qm, b) for q, b in clips]), hf=np.mean([hf(q * qs + qm) for q, b in clips]))
print('GT', json.dumps({k: round(float(x), 3) for k, x in gt.items()}), flush=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
for spec in sys.argv[1:]:
    lab, p = spec.split('=', 1); ck = torch.load(p, map_location='cpu')
    vae = MotionVAE(**ck['cfg']).to(dev); vae.load_state_dict(ck['ema']); vae.eval()
    r = dict(beat=[], hf=[], speed=[], p95=[], l1=[])
    for q, b in clips:
        with torch.no_grad():
            mu, _ = vae.encode(torch.from_numpy(q)[None].float().to(dev)); y = vae.decode(mu)[0].cpu().numpy() * qs + qm
        x = q * qs + qm
        r['beat'].append(beat_align(y, b)); r['hf'].append(hf(y)); r['speed'].append(sp(y).mean() / sp(x).mean())
        r['p95'].append(np.percentile(sp(y), 95) / np.percentile(sp(x), 95)); r['l1'].append(np.abs(y - x).mean())
    print(lab, json.dumps({k: round(float(np.mean(x)), 4) for k, x in r.items()} | dict(cfg=ck.get('vae_cfg', ck['cfg']))), flush=True)

"""Does the FULL streaming pipeline reproduce clips the model has memorised?
    python scripts/overfit_check.py <ckpt> <vae> <n>   (same subset as train.py --overfit n)
Per clip: free-running (history = own outputs) and oracle-history generation vs ground truth:
arm pose MAE (deg), arm speed ratio, and the teacher-forced flow loss is in the training log."""
import json, sys
import numpy as np, torch
from scmg.common import CHUNK, CHUNK_FRAMES
from scmg.data import View, WindowDataset
from scmg.evaluate import load
from scmg.infer import Streamer, generate
from scmg.model import INF
from scmg.train import overfit_clips

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st = json.load(open(D + '/STATS_v36_1.json'))
v = View(split='train', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1')
v.clips = overfit_clips(v, int(sys.argv[3]))
m, vae = load(sys.argv[1], sys.argv[2], 'cuda')
qm, qs = np.asarray(st['q_mean'], np.float32), np.asarray(st['q_std'], np.float32)
A = slice(4, 18)
res = {k: [] for k in ('free_mae', 'free_speed', 'oracle_mae', 'oracle_speed')}
for ci in range(len(v.clips)):
    nch = min(v.clips[ci]['frames'], 256) // CHUNK_FRAMES                    # the trained 5.12 s windows
    s = WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(0))
    gt = s['q'] * qs + qm
    cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm')}
    free, _, _ = generate(m, vae, s['feat'], cond, st, nch, H=INF, steps=20, standby=gt[0], ctx_t=getattr(m, 'ctx_t', 1.0))
    with torch.no_grad():
        mu, _ = vae.encode(torch.as_tensor(s['q'])[None].cuda()); gl = ((mu - m.lat_mean) / m.lat_std)[0]
    S = Streamer(m, vae, s['feat'], cond, st, H=INF, steps=20, standby=gt[0]); gen = []
    for k in range(nch):
        S.tick(); gen.append(torch.stack(S.lat[-CHUNK:]))
        for i in range(CHUNK): S.lat[k * CHUNK + i] = gl[k * CHUNK + i].clone()
    with torch.no_grad():
        orc = vae.decode(torch.cat(gen)[None] * m.lat_std + m.lat_mean)[0].cpu().numpy() * qs + qm
    for name, y in (('free', free), ('oracle', orc)):
        n = min(len(y), len(gt))
        res[name + '_mae'].append(float(np.abs(y[:n, A] - gt[:n, A]).mean() * 57.3))
        res[name + '_speed'].append(float(np.abs(np.diff(y[:n, A], axis=0)).mean() / np.abs(np.diff(gt[:n, A], axis=0)).mean()))
print(json.dumps({k: round(float(np.mean(x)), 3) for k, x in res.items()}))
print('gt arm std (deg)', round(float(np.mean([0])), 1))

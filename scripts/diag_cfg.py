"""Free-running speed / amplitude ratio vs classifier-free guidance weight.  python scripts/diag_cfg.py <ckpt> <vae> [n]"""
import json, sys
import numpy as np
from scmg.common import CHUNK_FRAMES
from scmg.data import View, WindowDataset
from scmg.evaluate import load, boundary_ratio
from scmg.infer import generate
from scmg.model import INF

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st = json.load(open(D + '/STATS_v36_1.json'))
v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1')
m, vae = load(sys.argv[1], sys.argv[2], 'cuda')
qm, qs = np.asarray(st['q_mean'], np.float32), np.asarray(st['q_std'], np.float32)
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
A = slice(4, 18)
for w in [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1,2,3").split(",")]:
    sp, am, br = [], [], []
    for ci in [i for i, c in enumerate(v.clips) if c['frames'] >= 400][:N]:
        nch = v.clips[ci]['frames'] // CHUNK_FRAMES
        s = WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(0))
        gt = s['q'] * qs + qm
        cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm')}
        y, _, _ = generate(m, vae, s['feat'], cond, st, nch, H=INF, steps=10, standby=qm, ctx_t=getattr(m, 'ctx_t', 1.0), cfg_w=w)
        n = min(len(y), len(gt))
        sp.append(np.abs(np.diff(y[:n, A], axis=0)).mean() / np.abs(np.diff(gt[:n, A], axis=0)).mean())
        am.append(y[:n, A].std(0).mean() / gt[:n, A].std(0).mean()); br.append(boundary_ratio(y[:n]))
    print(f'cfg {w}: speed {np.mean(sp):.3f} amplitude {np.mean(am):.3f} boundary {np.mean(br):.2f}', flush=True)

"""Why are generated gestures slower than people?  python scripts/diag_speed.py <ckpt> <vae> <out.json> [n_clips]
Measures arm speed ratio (generated or reconstructed / ground truth, same clips) for:
  A  data: per-source mean arm speed of the TRAIN mix vs the dev ground truth (is the training distribution slower?)
  B  VAE round trip of dev ground truth (does the latent space smooth motion away?)
  C  generation knobs on dev: CFG weight {1, 2.5, 4}, flow steps {5, 10, 20}, audio horizon {inf, 4, 0},
     context noise ctx_t {1.0, model default}, sampling temperature of the style head {1.0, 1.3}
All on the same clips / seeds; speed = mean |dq| over arm joints (rad/frame), ratio per clip then averaged."""
import json, sys
from collections import defaultdict
import numpy as np, torch
from scmg.data import View, WindowDataset, VIEW
from scmg.evaluate import load
from scmg.infer import generate
from scmg.model import INF
from scmg.common import U80

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
ck, vae_p, out = sys.argv[1], sys.argv[2], sys.argv[3]; NC = int(sys.argv[4]) if len(sys.argv) > 4 else 20
dev = 'cuda'; A = slice(4, 18)
st = json.load(open(D + '/STATS_v36_1.json')); qm, qs = np.asarray(st['q_mean'], np.float32), np.asarray(st['q_std'], np.float32)
sp = lambda q: float(np.abs(np.diff(q[:, A], axis=0)).mean())
res = {}

# A: training-mix speed by source (random 3000 windows of 4 s, weighted as the sampler does)
tv = View(VIEW, 'train', None, st, mmap=True)
rng = np.random.default_rng(0); w = WindowDataset(tv, 200).w
by = defaultdict(list)
for _ in range(3000):
    ci = int(rng.choice(len(w), p=w)); c = tv.clips[ci]
    if c['frames'] < 200: continue
    s0 = int(rng.integers(0, c['frames'] - 200 + 1)) + c['start']
    by[c['source']].append(sp(np.asarray(tv.q[s0:s0 + 200], np.float32)))
v = View(VIEW, 'val', D + '/text-e5-v36_1', st, mmap=True, tok_cache=D + '/tok-qwen25-v36_1'); v.glitch = None
clips = [ci for ci in range(len(v.clips)) if v.clips[ci]['frames'] // 16 >= 3][:NC]
gts = {}
for ci in clips:
    c = v.clips[ci]; T = c['frames'] // 16 * 16
    gts[ci] = np.asarray(v.q[c['start']:c['start'] + T], np.float32)
dev_gt = float(np.mean([sp(g) for g in gts.values()]))
res['A_train_speed_by_source'] = {k: dict(n=len(x), speed=float(np.mean(x)), ratio_to_dev_gt=float(np.mean(x) / dev_gt)) for k, x in by.items()}
res['A_train_mix_ratio_to_dev_gt'] = float(np.mean([s for x in by.values() for s in x]) / dev_gt)

# B: VAE round trip
m, vae = load(ck, vae_p, dev)
rr = []
with torch.no_grad():
    for ci, g in gts.items():
        x = torch.tensor((g - qm) / qs, device=dev)[None]
        mu, _ = vae.encode(x); y = vae.decode(mu)[0].cpu().numpy() * qs + qm
        rr.append(sp(y[:len(g)]) / sp(g))
res['B_vae_roundtrip_speed_ratio'] = float(np.mean(rr))

# C: generation knobs
def run(**kw):
    r = []
    for ci in clips:
        c = v.clips[ci]; T = c['frames'] // 16 * 16
        s = WindowDataset(v, T, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
        cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm') if k in s}
        a = dict(H=INF, steps=10, seed=0, standby=gts[ci][0], ctx_t=getattr(m, 'ctx_t', 1.0), cfg_w=2.5,
                 cfg_text=m.cfg.planner != 'voxcpm'); a.update(kw)
        y = generate(m, vae, s['feat'], cond, st, T // 16, **a)[0]
        r.append(sp(y[:T]) / sp(gts[ci]))
    return float(np.mean(r))
base = dict()
grid = [('cfg1', dict(cfg_w=1.0)), ('cfg2.5', dict()), ('cfg4', dict(cfg_w=4.0)),
        ('steps5', dict(steps=5)), ('steps20', dict(steps=20)),
        ('H4', dict(H=4 * U80)), ('H0', dict(H=0)),
        ('ctx_t1.0', dict(ctx_t=1.0)), ('temp1.3', dict(temperature=1.3))]
res['C_knobs_speed_ratio'] = {}
for name, kw in grid:
    res['C_knobs_speed_ratio'][name] = run(**kw); print(name, round(res['C_knobs_speed_ratio'][name], 3), flush=True)
res['model_ctx_t'] = float(getattr(m, 'ctx_t', 1.0)); res['ckpt'] = ck; res['clips'] = len(clips)
open(out, 'w').write(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))

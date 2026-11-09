"""Tick latency breakdown of the streamer for one checkpoint.  python scripts/profile_tick.py <ckpt> <vae>"""
import json, sys, time
import numpy as np, torch
from scmg.data import View, WindowDataset
from scmg.evaluate import load
from scmg.infer import generate
from scmg.model import INF

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st = json.load(open(D + '/STATS_v36_1.json'))
v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1')
m, vae = load(sys.argv[1], sys.argv[2], 'cuda')
ci = next(i for i, c in enumerate(v.clips) if c['frames'] >= 400)
nch = v.clips[ci]['frames'] // 16
s = WindowDataset(v, nch * 16, max_tokens=256).sample(ci, 0, np.random.default_rng(0))
cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm')}
# component timing via forward hooks on the main submodules
acc = {}
def hook(name):
    def pre(mod, inp): torch.cuda.synchronize(); mod._t0 = time.perf_counter()
    def post(mod, inp, out): torch.cuda.synchronize(); acc[name] = acc.get(name, 0.) + time.perf_counter() - mod._t0
    return pre, post
for name, mod in (('audio', m.audio), ('text_in', m.text_in)):
    a, b = hook(name); mod.register_forward_pre_hook(a); mod.register_forward_hook(b)
orig_plan, orig_flow = m.plan, m.flow_velocity
def timed(fn, name):
    def f(*x, **k):
        torch.cuda.synchronize(); t = time.perf_counter(); r = fn(*x, **k); torch.cuda.synchronize()
        acc[name] = acc.get(name, 0.) + time.perf_counter() - t; return r
    return f
m.plan = timed(orig_plan, 'planner'); m.flow_velocity = timed(orig_flow, 'flow_step'); m.mt_memory = timed(m.mt_memory, 'mt_memory') if hasattr(m, 'mt_type') else m.mt_memory
for steps, nf in ((10, 1), (10, 0), (4, 0)):
    acc.clear()
    _, _, tm = generate(m, vae, s['feat'], cond, st, nch, H=INF, steps=steps, n_future=nf, standby=np.asarray(st['q_mean']))
    n = len(tm)
    print(f'steps={steps} n_future={nf}: tick p50 {np.median(tm)*1000:.1f} ms p95 {np.percentile(tm,95)*1000:.1f} ms;',
          ' '.join(f'{k}={v/n*1000:.1f}ms' for k, v in acc.items()), flush=True)

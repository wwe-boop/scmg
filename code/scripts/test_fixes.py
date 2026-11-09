"""Regression checks for the 11-02 fixes.   python scripts/test_fixes.py <style_ckpt> <vox_ckpt> <vae>
 1 style sampling follows the streamer seed: same seed -> identical, different seed -> different
 2 VoxCPM: the CFG unconditional branch masks the LM text ids; text changes the conditional output
 3 losses: text dropout also masks the VoxCPM token ids (tidm)"""
import json, sys
import numpy as np, torch
from scmg.common import CHUNK_FRAMES
from scmg.data import View, WindowDataset, collate
from scmg.evaluate import load
from scmg.infer import Streamer, generate
from scmg.model import INF

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st = json.load(open(D + '/STATS_v36_1.json')); qm = np.asarray(st['q_mean'], np.float32)
v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', tok_cache=D + '/tok-qwen25-v36_1')
ci = next(i for i, c in enumerate(v.clips) if c['frames'] >= 256)
s = WindowDataset(v, 256, max_tokens=256).sample(ci, 0, np.random.default_rng(0))
cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm')}
ok = True
m, vae = load(sys.argv[1], sys.argv[3], 'cuda')
assert m.cfg.style, 'test 1 needs a style model'
a1, _, _ = generate(m, vae, s['feat'], cond, st, 8, H=INF, steps=4, seed=7, standby=qm)
torch.manual_seed(123); torch.randn(1000, device='cuda')                       # disturb the GLOBAL rng
a2, _, _ = generate(m, vae, s['feat'], cond, st, 8, H=INF, steps=4, seed=7, standby=qm)
a3, _, _ = generate(m, vae, s['feat'], cond, st, 8, H=INF, steps=4, seed=8, standby=qm)
d_same, d_diff = float(np.abs(a1 - a2).max()), float(np.abs(a1 - a3).max())
print('1 style seed: same-seed max diff', d_same, '| other-seed max diff', round(d_diff, 4)); ok &= d_same == 0 and d_diff > 0
mv, _ = load(sys.argv[2], sys.argv[3], 'cuda')
S = Streamer(mv, vae, s['feat'], cond, st, H=INF, steps=4, standby=qm, cfg_w=2.0)
nid = S._null_ids(); print('2 vox null ids all masked:', bool((~nid).all()), '| cond ids on:', int(S._ids()[1].sum()))
ok &= bool((~nid).all()) and int(S._ids()[1].sum()) > 0
b1, _, _ = generate(mv, vae, s['feat'], cond, st, 6, H=INF, steps=4, seed=1, standby=qm, cfg_w=2.0)
c2 = dict(cond); c2['tidm'] = np.zeros_like(cond['tidm']); c2['tmask'] = np.zeros_like(cond['tmask'])
b2, _, _ = generate(mv, vae, s['feat'], c2, st, 6, H=INF, steps=4, seed=1, standby=qm, cfg_w=2.0)
print('2 vox text effect (cfg 2): max diff with vs without text', round(float(np.abs(b1 - b2).max()), 4)); ok &= float(np.abs(b1 - b2).max()) > 0
# 3: drop_text = 1 must zero tidm in the loss path
from scmg import losses as Lm
mv.cfg.drop_text = 1.0
seen = {}
orig = mv.plan
def spy(*a, **k):
    seen['tidm'] = k.get('tidm') if 'tidm' in k else a[9]; return orig(*a, **k)
mv.plan = spy
batch = {k: x.cuda() for k, x in collate([s, s]).items()}
mv.train(); Lm.step_losses(mv, vae, batch, Lm.StageCfg(self_force_p=0.0), dict(qm=torch.tensor(st['q_mean']).cuda(), qs=torch.tensor(st['q_std']).cuda()))
print('3 loss path: tidm after text dropout all masked:', bool((~seen['tidm']).all())); ok &= bool((~seen['tidm']).all())
print('ALL_OK' if ok else 'FAILED')

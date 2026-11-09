import json, sys
import numpy as np, torch
from scmg.data import View, WindowDataset, collate
from scmg.losses import StageCfg, step_losses
from scmg.model import SCMG, Config
from scmg.vae import MotionVAE
from scmg.infer import generate
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'; M = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models'; dev = 'cuda'
st = json.load(open(D + '/STATS_v36_1.json'))
v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', tok_cache=M + '/tok-qwen3-0.6b-v36_1')
ds = WindowDataset(v, 256, max_tokens=128)
b = {k: x.to(dev) for k, x in collate([ds.sample(i, 0, np.random.default_rng(i)) for i in (0, 3)]).items()}
vck = torch.load(sys.argv[1], map_location='cpu'); vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema']); vae.eval()
stats = dict(qm=torch.tensor(st['q_mean'], device=dev), qs=torch.tensor(st['q_std'], device=dev))
for init in ('pretrained', 'random'):
    m = SCMG(Config(z=vae.z, planner='voxcpm', vox_lm=M + '/Qwen3-0.6B-Base', vox_lm_init=init)).to(dev).train()
    with torch.autocast('cuda', torch.bfloat16): out = step_losses(m, vae, b, StageCfg(), stats)
    out['loss'].backward(); m.eval(); s = ds.sample(0, 0, np.random.default_rng(0))
    cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm')}
    y = generate(m, vae, s['feat'][:64], cond, st, 4, steps=2, standby=st['q_mean'])[0]
    print(init, 'hidden', m.vox.H, 'loss %.3f' % float(out['loss']), 'gen', y.shape, 'finite', bool(np.isfinite(y).all()), 'params %.1fM' % (sum(p.numel() for p in m.parameters()) / 1e6), flush=True)
    del m; torch.cuda.empty_cache()
print('VOXQ_OK')

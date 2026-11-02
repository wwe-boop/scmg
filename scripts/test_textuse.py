"""Smoke (TEXTUSE, 11-03): text-use switches; also reports pre-FSQ saturation |fsq_in|>3 builds, runs one training step (finite loss, gradients) and one streaming tick."""
import json, sys
import numpy as np, torch
from scmg.data import View, WindowDataset, collate
from scmg.losses import StageCfg, step_losses
from scmg.model import SCMG, Config
from scmg.vae import MotionVAE
from scmg.infer import generate
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'; dev = 'cuda'
st = json.load(open(D + '/STATS_v36_1.json'))
v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', tok_cache=D + '/tok-qwen25-v36_1')
ds = WindowDataset(v, 256, max_tokens=128)
b = {k: x.to(dev) for k, x in collate([ds.sample(i, 0, np.random.default_rng(i)) for i in (0, 3)]).items()}
vck = torch.load(sys.argv[1], map_location='cpu'); vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema']); vae.eval()
stats = dict(qm=torch.tensor(st['q_mean'], device=dev), qs=torch.tensor(st['q_std'], device=dev))
ok = True
for name, extra in [('pre', {}), ('pre-norm', dict(vox_prefsq_norm=True)), ('pre-norm-match', dict(vox_prefsq_norm=True, vox_in_match=True)),
                    ('rand-norm-match', dict(vox_lm_init='random', vox_prefsq_norm=True, vox_in_match=True))]:
    torch.manual_seed(0)
    m = SCMG(Config(z=vae.z, planner='voxcpm', **extra)).to(dev).train()
    sat = []; m.vox.fsq_in.register_forward_hook(lambda mod, i, o: sat.append(float((o.abs() > 3).float().mean())))
    with torch.autocast('cuda', torch.bfloat16):
        out = step_losses(m, vae, b, StageCfg(), stats)
    loss = out['loss'] if isinstance(out, dict) else out[0]; loss.backward()
    lm_g = sum(float(p.grad.norm()) for n, p in m.named_parameters() if n.startswith('vox.lm.') and p.grad is not None)
    m.eval(); s = ds.sample(0, 0, np.random.default_rng(0))
    cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm')}
    y = generate(m, vae, s['feat'][:64], cond, st, 4, steps=2, standby=st['q_mean'])[0]
    c = bool(torch.isfinite(loss)) and np.isfinite(y).all(); ok &= c
    print(f'{name:15s} sat {sat[0] if sat else -1:.3f} loss {float(loss):.3f} lm-grad {lm_g:.2e} gen {y.shape} ok={c}', flush=True)
    del m; torch.cuda.empty_cache()
print('ALL_OK' if ok else 'SOME_FAIL')

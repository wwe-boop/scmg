"""torch FK == MuJoCo; proxy penetration on ground-truth vs random poses; one causal-stage step with w_phys (finite, gradients)."""
import json, sys
import numpy as np, torch, mujoco
from scmg.physim import EX002Sim
from scmg.kinematics import Kinematics
from scmg.data import View, WindowDataset, collate
from scmg.losses import StageCfg, step_losses
from scmg.train import build_model
from scmg.vae import MotionVAE
npz, ck, vae_p = sys.argv[1:4]; dev = 'cuda'
sim = EX002Sim(); m, d = sim.m, sim.d; kin = Kinematics(npz, dev)
rng = np.random.default_rng(0); err = 0
for _ in range(20):
    q = rng.uniform(sim.lo, sim.hi); d.qpos[:] = 0
    for jn, v in sim.held.items(): d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = v
    d.qpos[sim.qadr] = q; mujoco.mj_forward(m, d)
    R, p = kin.fk(torch.tensor(q[None], device=dev, dtype=torch.float32))
    err = max(err, float(np.abs(p[0].cpu().numpy() - d.xpos).max()))
print('torch FK max pos error vs MuJoCo (float32)', err)
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'; st = json.load(open(D + '/STATS_v36_1.json'))
v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1')
qm, qs = np.asarray(st['q_mean'], np.float32), np.asarray(st['q_std'], np.float32)
gt = np.concatenate([v.q[c['start']:c['start'] + c['frames']] for c in v.clips[:10]])[::5] * 1.0
gt = torch.tensor(gt, device=dev, dtype=torch.float32)
pen = kin.penetration(gt, d_safe=0.0)
print('ground truth frames', len(gt), 'frames with proxy penetration > 1 mm: %.3f' % float((pen.max(1).values > 1e-3).float().mean()),
      'mean max pen %.2f mm' % float(pen.max(1).values.mean() * 1000))
vck = torch.load(vae_p, map_location='cpu'); vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema']); vae.eval()
c = torch.load(ck, map_location='cpu', weights_only=False)
mdl, base_ids = build_model(c['base_cfg'], dict(c['extra_cfg']), dict(c['extra_cfg']), c['ema']); mdl.to(dev).train()
for p_ in mdl.parameters(): p_.requires_grad_(id(p_) not in base_ids)
ds = WindowDataset(v, 256, max_tokens=128)
b = {k: x.to(dev) for k, x in collate([ds.sample(i, 0, np.random.default_rng(i)) for i in range(8)]).items()}
stats = dict(qm=torch.tensor(st['q_mean'], device=dev), qs=torch.tensor(st['q_std'], device=dev))
with torch.autocast('cuda', torch.bfloat16):
    out = step_losses(mdl, vae, b, StageCfg(p_stream=0.67, outpaint=True, w_cont=1.0, w_rhy=0.1, w_phys=1.0), stats)
out['loss'].backward()
g = sum(float(p_.grad.norm()) for p_ in mdl.parameters() if p_.grad is not None)
print({k: round(float(x), 5) for k, x in out.items() if k.startswith('phys') or k == 'loss'}, 'grad', round(g, 3))
print('OK' if np.isfinite(float(out['loss'])) and g > 0 else 'FAIL')

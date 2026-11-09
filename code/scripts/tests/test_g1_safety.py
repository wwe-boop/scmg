"""Tests for scmg/g1_safety.py (11-05).  python scripts/tests/test_g1_safety.py [gt_dir stems_file]"""
import glob, os, subprocess, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scmg import g1_safety as S

ok = True
def check(name, cond, info=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else '')); ok &= bool(cond)

# 1 limit table vs official joint names
c = S._consts('cpu', torch.float64)
check('29 joints, lo < hi', len(c['lo']) == 29 and bool((c['lo'] < c['hi']).all()))

# 2 GT motion is (almost) not penalised
gt_dir = sys.argv[1] if len(sys.argv) > 1 else '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/echog/data_root/motion_39d_30fps'
stems = [s.strip() for s in open(sys.argv[2])][:20] if len(sys.argv) > 2 else [os.path.basename(p)[:-3] for p in sorted(glob.glob(gt_dir + '/*.pt'))[:20]]
rr = [torch.as_tensor(np.asarray(torch.load(f'{gt_dir}/{s}.pt', weights_only=False)['robot_repr']), dtype=torch.float64) for s in stems]
st = [S.safety_terms(r[None]) for r in rr]
lim = torch.cat([x['lim'].flatten() for x in st]); lr = torch.cat([x['lr'].flatten() for x in st]); to = torch.cat([x['torso'].flatten() for x in st])
check('GT: no joint-limit excess', float(lim.max()) == 0.0, f'max {float(lim.max()):.4f}')
check('GT: left-right hinge active on < 0.5 % frames', float((lr > 0).double().mean()) < 0.005, f'{100*float((lr > 0).double().mean()):.3f}%')
check('GT: torso hinge active on < 1 % frames', float((to > 0).double().mean()) < 0.01, f'{100*float((to > 0).double().mean()):.3f}%')

# 3 crossed arms: penalised, and the gradient moves them apart
x = rr[0][:1].clone()
q = x[..., 10:].clone()
from echo_g.evaluation.robot_repr import G1_JOINT_NAMES
idx = {n[:-6] if n.endswith('_joint') else n: i for i, n in enumerate(G1_JOINT_NAMES)}
q[0, idx['left_shoulder_pitch']] = -1.2; q[0, idx['right_shoulder_pitch']] = -1.2
q[0, idx['left_shoulder_roll']] = -0.6; q[0, idx['right_shoulder_roll']] = 0.6     # both arms swung across the chest
q[0, idx['left_elbow']] = 1.2; q[0, idx['right_elbow']] = 1.2
y = torch.cat([x[..., :10], q], -1)[None].requires_grad_(True)
t0 = S.safety_terms(y)
check('crossed arms: d_lr below the 0.10 m threshold', float(t0['d_lr']) < S.D_LR, f'd_lr {float(t0["d_lr"]):.3f} m')
loss = S.contract_loss(y, torch.ones(1, 1, dtype=y.dtype)); tot = loss['safe_lr'] + loss['safe_lim'] + 0.5 * loss['safe_torso']; tot.backward()
g = y.grad[..., 10:]
check('gradient reaches the joints', float(g.abs().sum()) > 0)
y2 = (y - 0.05 * y.grad / y.grad.norm()).detach()
check('a gradient step reduces the loss', float(S.contract_loss(y2, torch.ones(1, 1, dtype=y.dtype))['safe_lr']) < float(loss['safe_lr']))

# 4 batched == per-clip
T = min(r.shape[0] for r in rr[:3]); bt = torch.stack([r[:T] for r in rr[:3]])
sb = S.safety_terms(bt); s1 = torch.stack([S.safety_terms(r[:T][None])['d_lr'][0] for r in rr[:3]])
check('batched d_lr == per-clip d_lr', torch.allclose(sb['d_lr'], s1, atol=1e-10))

# 5 fallback import path (DLC image without PyYAML: stubbed echo_g package)
code = ("import sys, types\n"
        "for p in ('echo_g', 'echo_g.evaluation'):\n    m = types.ModuleType(p); m.__path__ = []; sys.modules[p] = m\n"
        f"sys.path.insert(0, {os.path.join(os.path.dirname(__file__), '..', '..')!r})\n"
        "import torch; from scmg import g1_safety as S\n"
        "c = S._consts('cpu', torch.float32); print('FALLBACK_OK', len(c['lo']))\n")
r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
check('fallback import with a stubbed echo_g package', 'FALLBACK_OK 29' in r.stdout, (r.stdout + r.stderr)[-300:].strip())
print('ALL_OK' if ok else 'SOME_FAILED')

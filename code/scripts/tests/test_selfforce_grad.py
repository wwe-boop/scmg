"""11-04: on a self-forced batch every trainable tensor of the planner and the Local Flow must receive a gradient
(regression test for the autocast weight-cache bug: the no-grad generation pass cached bf16 weight copies without
grad_fn, so the gradient pass got none).  python scripts/tests/test_selfforce_grad.py   (exit 1 on failure)"""
import sys, dataclasses
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from test_flow_text_xattn import synth_batch, stats_for, dev
from scmg.model import SCMG, Config
from scmg.vae import MotionVAE
from scmg.losses import StageCfg, step_losses

torch.manual_seed(0); m = SCMG(Config(z=16, adsc=True)).to(dev).train()
torch.manual_seed(1); vae = MotionVAE(z=16).to(dev).eval()
for p in vae.parameters(): p.requires_grad_(False)
b = synth_batch()
st = dict(qm=torch.zeros(b['q'].shape[-1], device=dev), qs=torch.ones(b['q'].shape[-1], device=dev))
bad = {}
for sfp in (0.0, 1.0):
    sc = dataclasses.replace(StageCfg(), self_force_p=sfp)
    m.zero_grad(set_to_none=True); torch.manual_seed(5)
    with torch.autocast('cuda', torch.bfloat16):
        loss = step_losses(m, vae, b, sc, st)['loss']
    loss.backward()
    none = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None and (n.startswith('planner') or n.startswith('flow'))]
    tot = [n for n, p in m.named_parameters() if p.requires_grad and (n.startswith('planner') or n.startswith('flow'))]
    bad[sfp] = none; print(f'self_force_p={sfp}: planner/flow tensors without grad {len(none)}/{len(tot)}')
ok = not bad[1.0] and not bad[0.0]
print('SELFFORCE_GRAD', 'OK' if ok else 'FAIL'); sys.exit(0 if ok else 1)

"""11-05: scmg/g1_contact.py vs the official evaluator geometry, on real ECHO-G clips.
    python scripts/tests/test_g1_contact.py <motion_39d_30fps dir> <stem> [<stem> ...]"""
import sys, torch, numpy as np
sys.path.insert(0, '.')
from scmg import g1_contact as GC
GC._mods()
from echo_g.evaluation.g1_motion_cls import fk_grounded_world
D = sys.argv[1]; rrs = []
for st in sys.argv[2:]:
    x = torch.load(f'{D}/{st}.pt', weights_only=False); rr = x['robot_repr'] if isinstance(x, dict) else x
    rr = torch.as_tensor(rr).double(); rrs.append(rr)
    _, so, ro, fo = fk_grounded_world(rr, 30.0)
    sm, rm, fm = GC.sole_points(rr[None], 30.0)
    err = float((so - sm[0]).abs().max()); print(st, rr.shape, 'max |official - ours| sole pos %.2e m' % err)
    assert err < 1e-6 and np.allclose(ro, rm) and (fo == fm).all()
T = min(len(r) for r in rrs); g = torch.stack([r[:T] for r in rrs]).float(); w = torch.ones(g.shape[:2])
l0 = GC.foot_contact_loss(g, g, w, 30.0); print('gt vs gt', {k: float(v) for k, v in l0.items()})
assert float(l0['foot_slide']) < 1e-5 and float(l0['foot_height']) < 1e-5
p = (g + 0.02 * torch.randn_like(g)).requires_grad_(True)
l1 = GC.foot_contact_loss(p, g, w, 30.0); (l1['foot_slide'] + l1['foot_height']).backward()
print('perturbed', {k: float(v) for k, v in l1.items()}, 'grad finite', bool(torch.isfinite(p.grad).all()), 'grad on legs', float(p.grad[..., 10:22].abs().sum()) > 0)
assert float(l1['foot_slide']) > 0 and torch.isfinite(p.grad).all()
print('G1_CONTACT OK')

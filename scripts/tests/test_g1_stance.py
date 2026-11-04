"""11-05: scmg/g1_stance.py -- only the horizontal root velocity changes, and after the official root integration the stance
foot (lower foot of frame t-1) keeps its xy position between t-1 and t.   python scripts/tests/test_g1_stance.py <clip.pt>"""
import sys, torch
sys.path.insert(0, '.')
from scmg.g1_stance import stance_root_velocity
from scmg.g1_contact import sole_points
x = torch.load(sys.argv[1], weights_only=False); x = torch.as_tensor(x['robot_repr'] if isinstance(x, dict) else x).double()
y = stance_root_velocity(x, 30.)
assert torch.equal(torch.cat([x[:, :7], x[:, 9:]], 1), torch.cat([y[:, :7], y[:, 9:]], 1)), 'other channels changed'
s, _, f = sole_points(y[None], 30.); s = s[0]; z = s[..., 2]; fm = torch.as_tensor(f)
c = torch.stack([s[:, fm == k].mean(1) for k in (0, 1)], 1); st = torch.stack([z[:, fm == k].min(-1).values for k in (0, 1)], 1)[:-1].argmin(-1)
i = torch.arange(len(st)); d = (c[i + 1, st, :2] - c[i, st, :2]).norm(dim=-1)
assert float(d.max()) < 1e-9, float(d.max())
print('G1_STANCE OK max stance-foot xy step %.1e m' % float(d.max()))

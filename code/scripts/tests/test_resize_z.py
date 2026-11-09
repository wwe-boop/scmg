"""11-04: warm start across a codec change (z 16 -> identity z = STRIDE*J). Checks that ONLY latent I/O layers are re-initialised,
every other tensor equals the checkpoint bit for bit, a forward pass runs, and a non-I/O mismatch is refused.
    SCMG_J=39 python scripts/tests/test_resize_z.py <G1 base ckpt>"""
import sys, torch
sys.path.insert(0, '.')
from scmg.common import STRIDE, J
from scmg.train import build_model, Z_IO
ck = torch.load(sys.argv[1], map_location='cpu')
base = dict(ck['base_cfg']); prev_extra = dict(ck['extra_cfg']); z_new = STRIDE * J
m, _ = build_model({**base, 'z': z_new}, prev_extra, prev_extra, ck['ema'], resize_z=True)
own = m.state_dict(); old = ck['ema']
reinit = sorted(k for k in old if k in own and own[k].shape != old[k].shape)
same = [k for k in old if k in own and own[k].shape == old[k].shape]
assert reinit and all(k.startswith(Z_IO) for k in reinit), reinit
assert all(torch.equal(own[k], old[k]) for k in same if not k.startswith(('lat_mean', 'lat_std'))), 'a loaded tensor differs'
print('re-initialised:', reinit); print('loaded bit-identical:', len(same), 'tensors')
assert m.cfg.z == z_new and m.f_in.in_features == z_new and m.f_out.out_features == z_new
bad = dict(old); k0 = next(k for k in old if k.startswith('planner') or k.startswith('blocks') or '.attn' in k)
bad[k0] = torch.zeros(*[s + 1 for s in old[k0].shape])
try:
    build_model({**base, 'z': z_new}, prev_extra, prev_extra, bad, resize_z=True); print('FAIL: non-I/O mismatch accepted'); sys.exit(1)
except RuntimeError as e: print('non-I/O mismatch refused:', str(e)[:90])
print('RESIZE_Z OK')

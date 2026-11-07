"""11-08 skip diagnostic (StageCfg skip_diag, train.py --skip-log): default off is bit-identical; on, the training loss terms and
every parameter gradient are still bit-identical (autograd.grad on y does not touch .grad; the GT penetration reuses the same
frame / pair subset, no extra random draw) and the diagnostics are present and finite.
    SCMG_J=34 CUDA_VISIBLE_DEVICES=0 python scripts/tests/test_skip_diag.py     env BASE_TREE (code before) SP (scratch)
EX002 data-v36_1 train batch of 8 windows (256 frames), r4 base init, vae-r3, recipe sf0 + w_phys 0.1 / phys_dyn_w 0."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve(); CODE = HERE.parents[2]; sys.path.insert(0, str(CODE))
from scmg.data import View, WindowDataset, collate
E = os.environ; SP = Path(E['SP']); SP.mkdir(parents=True, exist_ok=True); FAIL = []
B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'
CK, VAE, STATS = B + '/dlc-scmg-train-r4-1102/base/best.pt', B + '/trained/vae-r3/final.pt', D + '/scmg/STATS_v36_1.json'
SC = dict(p_stream=0.67, p_async=1.0, p_async_seg=0.0, self_force_p=0, w_phys=0.1, phys_dyn_w=0)


def ok(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not cond: FAIL.append(name)


PROBE = HERE.parent / '_skipdiag_probe.py'
PROBE.write_text((HERE.parent / '_causal_loss_probe.py').read_text().replace(
    "torch.save(dict(loss={k: v.detach().float().cpu() for k, v in l.items()}, grad=g), op)",
    "torch.save(dict(loss={k: (v.detach().float().cpu() if torch.is_tensor(v) else torch.tensor(float(v))) for k, v in l.items()}, grad=g), op)").replace(
    "print('PROBE', {k: round(float(v), 6) for k, v in l.items()}, 'grads', len(g), flush=True)", "print('PROBE', len(g), flush=True)"))


def probe(code, sc, tag):
    out = SP / f'p_{tag}.pt'
    r = subprocess.run([sys.executable, str(PROBE), str(SP / 'b.pt'), str(out), CK, VAE, STATS, '{"adsc": true}', json.dumps(sc)],
                       env=dict(os.environ, PYTHONPATH=str(code)), cwd=str(code), capture_output=True, text=True)
    if r.returncode: print(r.stderr[-2500:])
    return torch.load(out, weights_only=False) if r.returncode == 0 else None


stats = json.loads(Path(STATS).read_text())
v = View(D + '/train-v1-20261101/views/data-v36_1', 'train', D + '/scmg/text-e5-v36_1', stats, mmap=True, tok_cache=D + '/scmg/tok-qwen25-v36_1')
it = iter(WindowDataset(v, 256, max_tokens=256, seed=3)); torch.save(collate([next(it) for _ in range(8)]), SP / 'b.pt')
base, new, diag = probe(E['BASE_TREE'], SC, 'base'), probe(CODE, SC, 'new'), probe(CODE, dict(SC, skip_diag=True), 'diag')
same = lambda p, q, keys: p is not None and q is not None and all(torch.equal(p['loss'][k], q['loss'][k]) for k in keys) and \
    set(p['grad']) == set(q['grad']) and all(torch.equal(p['grad'][k], q['grad'][k]) for k in p['grad'])
ok('a default (skip_diag off) losses + grads == BASE_TREE bitwise', same(base, new, base['loss'] if base else []))
ok('b skip_diag on: training loss terms + every gradient == default bitwise', same(new, diag, new['loss'] if new else []))
dk = sorted(k for k in (diag['loss'] if diag else {}) if k.startswith('_'))
need = {'_gy_phys_coll', '_gy_phys_dyn', '_gy_phys_joint', '_gy_cont', '_pen_max_cm', '_gt_pen_max_cm'}
ok('b diagnostics present and finite', diag is not None and need <= set(dk) and all(bool(torch.isfinite(diag['loss'][k])) for k in dk),
   ' '.join(f'{k}={float(diag["loss"][k]):.4g}' for k in dk) if diag else '')
hub = probe(CODE, dict(SC, phys_coll_huber=1.0), 'huber')
ok('c phys_coll_huber 1 cm: phys_coll <= squared hinge, finite grads, other terms unchanged',
   hub is not None and float(hub['loss']['phys_coll']) <= float(new['loss']['phys_coll']) + 1e-6 and all(bool(torch.isfinite(g).all()) for g in hub['grad'].values())
   and all(torch.equal(hub['loss'][k], new['loss'][k]) for k in new['loss'] if k not in ('loss', 'phys_coll')),
   f"phys_coll {float(hub['loss']['phys_coll']):.4f} vs {float(new['loss']['phys_coll']):.4f}" if hub else '')
PROBE.unlink()
print('ALL_OK' if not FAIL else f'SOME_FAIL {FAIL}'); sys.exit(1 if FAIL else 0)

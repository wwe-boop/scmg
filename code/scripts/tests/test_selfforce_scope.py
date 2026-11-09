"""11-08 StageCfg.self_force_scope (default 'all'). GPU, synthetic batch + random weights, one self-forced training step.
    python scripts/tests/test_selfforce_scope.py <legacy code tree> <pre-change code tree> [--cfg '{"intent_aux": true}']
      legacy tree     = pre-dcd50fb code (e.g. releases/bisect-1108/5d8f154): autocast weight cache filled under no_grad
      pre-change tree = this branch before self_force_scope (e.g. a git archive of a534a91)
  1 default 'all' is bit-identical to the pre-change code: loss and every gradient (self_force_p 1 and 0)
  2 'encoders' == legacy: the same parameters get no gradient, and every other gradient + the loss are bit-identical
  3 'encoders' leaves non-self-forced batches untouched (self_force_p 0: identical to 'all')
    python scripts/tests/test_selfforce_scope.py dump <out.npz> <sfp> <scope|-> <cfg json>      (internal, one code tree)"""
import json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(os.path.dirname(HERE))


def dump(out, sfp, scope, cfgj):
    import dataclasses, torch
    sys.path.insert(1, HERE)
    from test_flow_text_xattn import synth_batch, dev
    from scmg.model import SCMG, Config
    from scmg.vae import MotionVAE
    from scmg.losses import StageCfg, step_losses
    torch.manual_seed(0); m = SCMG(Config(z=16, adsc=True, **json.loads(cfgj))).to(dev).train()
    torch.manual_seed(1); vae = MotionVAE(z=16).to(dev).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    b = synth_batch()
    st = dict(qm=torch.zeros(b['q'].shape[-1], device=dev), qs=torch.ones(b['q'].shape[-1], device=dev))
    kw = dict(self_force_p=float(sfp)); kw.update({} if scope == '-' else dict(self_force_scope=scope))
    sc = dataclasses.replace(StageCfg(), **kw)
    m.zero_grad(set_to_none=True); torch.manual_seed(5)
    with torch.autocast('cuda', torch.bfloat16):
        l = step_losses(m, vae, b, sc, st)
    l['loss'].backward()
    for p in (getattr(m, '_sf_frozen', None) or []): p.grad = None          # what train.py does after backward
    d = {'__loss': l['loss'].detach().float().cpu().numpy(), '__sf': np.array(float(l['self_forced']))}
    for n, p in m.named_parameters():
        if p.requires_grad: d[n] = p.grad.float().cpu().numpy() if p.grad is not None else np.array([np.nan])
    np.savez(out, **d)


def run(tree, out, sfp, scope, cfgj):
    env = dict(os.environ, PYTHONPATH=tree)
    subprocess.run([sys.executable, os.path.abspath(__file__), 'dump', out, str(sfp), scope, cfgj], env=env, check=True, cwd='/tmp')
    return dict(np.load(out))


def same(a, b):
    if a.keys() != b.keys(): return False, 'parameter sets differ'
    bad = [k for k in a if not np.array_equal(a[k], b[k], equal_nan=True)]
    return not bad, bad[:5]


def main():
    legacy, pre = sys.argv[1], sys.argv[2]
    cfgj = sys.argv[sys.argv.index('--cfg') + 1] if '--cfg' in sys.argv else '{}'
    td = tempfile.mkdtemp(); ok = True
    for sfp in (1.0, 0.0):
        A = run(CODE, f'{td}/all{sfp}.npz', sfp, 'all', cfgj); P = run(pre, f'{td}/pre{sfp}.npz', sfp, '-', cfgj)
        r = same(A, P); print(f'1 default all == pre-change (self_force_p {sfp}):', r[0], '' if r[0] else r[1]); ok &= r[0]
    E = run(CODE, f'{td}/enc.npz', 1.0, 'encoders', cfgj); L = run(legacy, f'{td}/leg.npz', 1.0, '-', cfgj)
    nE = sorted(k for k, v in E.items() if not k.startswith('__') and np.isnan(v).all() and v.size == 1)
    nL = sorted(k for k, v in L.items() if not k.startswith('__') and np.isnan(v).all() and v.size == 1)
    r = same(E, L)
    print(f'2 encoders == legacy: no-grad sets {len(nE)} / {len(nL)} equal {nE == nL}; loss + all gradients bit-identical {r[0]}',
          '' if r[0] else r[1]); ok &= r[0] and nE == nL and E['__sf'] == 1
    E0 = run(CODE, f'{td}/enc0.npz', 0.0, 'encoders', cfgj); A0 = dict(np.load(f'{td}/all0.0.npz'))
    r = same(E0, A0); print('3 encoders on non-self-forced batches == all:', r[0]); ok &= r[0]
    print('SELFFORCE_SCOPE', 'ALL_OK' if ok else 'FAIL'); sys.exit(0 if ok else 1)


if __name__ == '__main__':
    if sys.argv[1] == 'dump': dump(*sys.argv[2:6])
    else: main()

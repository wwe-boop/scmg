"""11-09 F4 / F6 tests (docs/criteria/SEMBRIDGE_F4F6_1109.md on exp/text-path). GPU, SCMG_J=39, synthetic batch + random weights.
    SCMG_J=39 python scripts/tests/test_f4f6.py <pre-change code tree>
  1 default Config / StageCfg: loss dict, every gradient and the RNG stream bit-identical to the pre-change tree
  2 f4_align = true with w_f4 = 0 (and w_f6 = 0): bit-identical to off (shared gradients, RNG)
  3 F4 on: every recipe term (flow etc.) bit-identical to off (computed last); the F4 term alone reaches text_in, f4_txt, f4_mot
    and not the Local Flow
  4 F6 on: every recipe term bit-identical to off; the F6 term alone reaches the planner and the Local Flow; the term is 0 when
    the mismatched error already exceeds margin x matched error (margin 0 -> exactly 0)"""
import dataclasses, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(os.path.dirname(HERE))


def setup(f4):
    import torch
    sys.path.insert(1, HERE)
    from test_flow_text_xattn import synth_batch, dev
    from scmg.model import SCMG, Config
    from scmg.vae import MotionVAE
    torch.manual_seed(0); m = SCMG(Config(z=16, adsc=True, **(dict(f4_align=True) if f4 else {}))).to(dev).train()
    torch.manual_seed(1); vae = MotionVAE(z=16).to(dev).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    b = synth_batch()
    st = dict(qm=torch.zeros(b['q'].shape[-1], device=dev), qs=torch.ones(b['q'].shape[-1], device=dev))
    return m, vae, b, st, dev


def step(m, vae, b, st, sc):
    import torch
    from scmg.losses import step_losses
    m.zero_grad(set_to_none=True); torch.manual_seed(5)
    with torch.autocast('cuda', torch.bfloat16):
        return step_losses(m, vae, b, sc, st)


def dump(out, mode):
    import torch
    from scmg.losses import StageCfg
    m, vae, b, st, dev = setup(mode == 'f4on0')
    sc = StageCfg() if mode == 'base' else dataclasses.replace(StageCfg(), w_f4=0.0, w_f6=0.0)
    l = step(m, vae, b, st, sc); l['loss'].backward()
    d = {f'__{k}': np.array(float(v.detach())) for k, v in l.items() if torch.is_tensor(v) and v.numel() == 1}
    for n, p in m.named_parameters():
        if p.requires_grad and not n.startswith('f4_'): d[n] = p.grad.float().cpu().numpy() if p.grad is not None else np.array([np.nan])
    d['__rng'] = np.array(float(torch.rand(()))); np.savez(out, **d)


def run(tree, out, mode):
    subprocess.run([sys.executable, os.path.abspath(__file__), 'dump', out, mode], env=dict(os.environ, PYTHONPATH=tree, SCMG_J='39'), check=True, cwd='/tmp')
    return dict(np.load(out))


def same(a, b):
    if a.keys() != b.keys(): return False, sorted(set(a) ^ set(b))[:5]
    bad = [k for k in a if not np.array_equal(a[k], b[k], equal_nan=True)]
    return not bad, bad[:5]


def terms():
    import torch
    sys.path.insert(0, CODE)
    from scmg.losses import StageCfg
    ok = True
    m, vae, b, st, dev = setup(True)
    from test_flow_text_xattn import perturb_base
    perturb_base(m)                                       # non-zero AdaLN projections so that the planner -> flow path carries signal
    off = step(m, vae, b, st, StageCfg())
    for name, kw, key in (('3 F4', dict(w_f4=0.1), 'f4'), ('4 F6', dict(w_f6=0.05), 'f6')):
        on = step(m, vae, b, st, dataclasses.replace(StageCfg(), **kw))
        shared = [k for k in off if k != 'loss' and torch.is_tensor(off[k]) and off[k].numel() == 1]
        eq = all(torch.equal(off[k].detach(), on[k].detach()) for k in shared)
        ps = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
        gs = torch.autograd.grad(on[key], [p for _, p in ps], allow_unused=True)
        gn = {n: (0. if g is None else float(g.float().abs().sum())) for (n, _), g in zip(ps, gs)}
        grp = lambda pre: sum(v for n, v in gn.items() if n.startswith(pre))
        if key == 'f4':
            r = eq and grp('text_in') > 0 and grp('f4_txt') > 0 and grp('f4_mot') > 0 and grp('flow.') == 0 and grp('f_out') == 0
            print(f"{name}: recipe terms unchanged {eq}; grad text_in {grp('text_in'):.3g} f4_txt {grp('f4_txt'):.3g} f4_mot {grp('f4_mot'):.3g} flow {grp('flow.'):.3g}; f4 {float(on['f4']):.3f}:", r)
        else:
            r = eq and grp('planner.') > 0 and grp('flow.') > 0 and np.isfinite(float(on['f6']))
            z = step(m, vae, b, st, dataclasses.replace(StageCfg(), w_f6=0.05, f6_margin=0.0))
            r = r and float(z['f6']) == 0.0
            print(f"{name}: recipe terms unchanged {eq}; grad planner {grp('planner.'):.3g} flow {grp('flow.'):.3g}; f6 {float(on['f6']):.4f} "
                  f"ratio {float(on['f6_ratio']):.3f}; margin 0 -> {float(z['f6'])}:", r)
        ok &= r
    return ok


def main():
    pre = sys.argv[1]; td = tempfile.mkdtemp(); ok = True
    A = run(CODE, f'{td}/base.npz', 'base'); P = run(pre, f'{td}/pre.npz', 'base')
    r = same(A, P); print('1 default == pre-change (loss dict, all gradients, RNG):', r[0], '' if r[0] else r[1]); ok &= r[0]
    O = run(CODE, f'{td}/f4on0.npz', 'f4on0'); r = same(A, O); print('2 f4_align on, w 0 == off:', r[0], '' if r[0] else r[1]); ok &= r[0]
    ok &= terms()
    print('F4F6', 'ALL_OK' if ok else 'FAIL'); sys.exit(0 if ok else 1)


if __name__ == '__main__':
    if sys.argv[1] == 'dump': dump(*sys.argv[2:4])
    else: main()

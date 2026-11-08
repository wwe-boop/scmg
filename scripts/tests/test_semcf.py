"""11-09 P2 sem_cf tests (exp/text-path docs/criteria/P2_SEMCF_1109.md §3). GPU, SCMG_J=34, synthetic batch + random weights.
    SCMG_J=34 python scripts/tests/test_semcf.py <pre-change code tree>
  1 default Config / StageCfg (ctl on, aim + finger tokens, no SEM): loss dict, gradients and RNG bit-identical to the pre-change tree
  2 ctl_desc on with no descriptor in the batch == ctl_desc off (shared gradients, RNG)
  3 descriptor: numpy-FK path (points) == differentiable path (points_t) within 1e-5; gradient finite
  4 SEM tokens + sem_cf_p 0.5 + w_sem_desc 1: the outcome term is finite, reaches ctl_desc_in / ctl_ modules; counterfactual tokens
    lift imitation (flow-supervised positions drop)
    python scripts/tests/test_semcf.py dump <out.npz> <mode: base|desc0> (internal)"""
import dataclasses, json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(os.path.dirname(HERE))


def batch(sem=False):
    import torch
    sys.path.insert(1, HERE)
    from test_flow_text_xattn import synth_batch, dev
    from scmg import ctl as C
    b = synth_batch(); B = b['q'].shape[0]; kc = C.KC
    g = torch.Generator().manual_seed(7)
    t = torch.full((B, kc), float('nan')); a = t.clone(); bb = t.clone(); ap = t.clone()
    ty = torch.zeros(B, kc, dtype=torch.long); gr = torch.zeros(B, kc, dtype=torch.long)
    v = torch.zeros(B, kc, C.NV); m = torch.zeros(B, kc, C.NVM)
    for i in range(B):
        t[i, 0] = 1.5; a[i, 0] = 1.2; bb[i, 0] = 1.8; ap[i, 0] = 1.6; ty[i, 0] = C.POINT; v[i, 0, :3] = torch.rand(3, generator=g); v[i, 0, 3] = 1; m[i, 0, :2] = 1
        t[i, 1] = 2.5; a[i, 1] = 2.4; bb[i, 1] = 2.6; ty[i, 1] = C.TYPE_ID['beat']; v[i, 1, :2] = 0.3; m[i, 1, 0] = 1
    out = dict(ctl_t=t, ctl_a=a, ctl_b=bb, ctl_apex=ap, ctl_type=ty, ctl_g=gr, ctl_v=v, ctl_vm=m)
    if sem:
        d = torch.zeros(B, kc, 16); dm = torch.zeros(B, kc)
        for i in range(B):
            for j, tc in ((2, 2.6), (3, 3.6)):
                t[i, j] = tc - 0.2; a[i, j] = tc - 0.6; bb[i, j] = tc + 0.6; ty[i, j] = C.SEM; v[i, j, 3] = 1. if (i + j) % 2 else -1.; m[i, j, 1] = 1
                d[i, j] = torch.randn(16, generator=g); d[i, j, 13:] = torch.nn.functional.normalize(d[i, j, 13:], dim=0); dm[i, j] = 1
        out.update(ctl_d=d, ctl_dm=dm)
    b.update({k: x.to(dev) for k, x in out.items()})
    return b, dev


def model(desc):
    import torch
    from scmg.model import SCMG, Config
    from scmg.vae import MotionVAE
    from test_flow_text_xattn import dev
    kw = dict(ctl=True, ctl_flow=True, **(dict(ctl_desc=True) if desc else {}))
    torch.manual_seed(0); m = SCMG(Config(z=16, adsc=True, **kw)).to(dev).train()
    torch.manual_seed(1); vae = MotionVAE(z=16).to(dev).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    return m, vae


def step(m, vae, b, sc):
    import torch
    from scmg.losses import step_losses
    st = dict(qm=torch.zeros(b['q'].shape[-1], device=b['q'].device), qs=torch.ones(b['q'].shape[-1], device=b['q'].device))
    m.zero_grad(set_to_none=True); torch.manual_seed(5)
    with torch.autocast('cuda', torch.bfloat16):
        return step_losses(m, vae, b, sc, st)


def dump(out, mode):
    import torch
    from scmg.losses import StageCfg
    b, dev = batch(False); m, vae = model(mode == 'desc0')
    l = step(m, vae, b, StageCfg()); l['loss'].backward()
    d = {f'__{k}': np.array(float(v.detach())) for k, v in l.items() if torch.is_tensor(v) and v.numel() == 1}
    for n, p in m.named_parameters():
        if p.requires_grad and not n.startswith('ctl_desc_in'): d[n] = p.grad.float().cpu().numpy() if p.grad is not None else np.array([np.nan])
    d['__rng'] = np.array(float(torch.rand(()))); np.savez(out, **d)


def run(tree, out, mode):
    subprocess.run([sys.executable, os.path.abspath(__file__), 'dump', out, mode], env=dict(os.environ, PYTHONPATH=tree, SCMG_J='34'), check=True, cwd='/tmp')
    return dict(np.load(out))


def same(a, b):
    if a.keys() != b.keys(): return False, sorted(set(a) ^ set(b))[:5]
    bad = [k for k in a if not np.array_equal(a[k], b[k], equal_nan=True)]
    return not bad, bad[:5]


def rest():
    import torch
    sys.path.insert(0, CODE)
    from scmg import sem_desc as SD
    from scmg.ex002_ctl import points
    from scmg.losses import StageCfg
    ok = True
    rng = np.random.default_rng(0); q = (rng.standard_normal((3, SD.NW, 34)) * 0.3).astype(np.float32).cumsum(1) * 0.05
    h = torch.tensor([0, 1, 1])
    d1 = SD.descriptor_q(torch.tensor(q), h)
    P = [points(x) for x in q]
    d2 = SD.descriptor_pts(torch.tensor(np.stack([p['wrist'] for p in P])), torch.tensor(np.stack([p['shoulder'] for p in P])), h)
    qt = torch.tensor(q, requires_grad=True); SD.descriptor_q(qt, h).sum().backward()
    r = float((d1 - d2).abs().max()) < 1e-5 and bool(torch.isfinite(qt.grad).all())
    print(f'3 descriptor numpy-FK == torch-FK max diff {float((d1 - d2).abs().max()):.2e}, grad finite:', r); ok &= r
    td = tempfile.mkdtemp(); bank = f'{td}/SEM_BANK.json'; json.dump(dict(mu=[0.] * 13, sd=[1.] * 13), open(bank, 'w'))
    b, dev = batch(True); m, vae = model(True)
    from test_flow_text_xattn import perturb_base
    perturb_base(m)
    with torch.no_grad():                                  # the zero-init control routes (ctl_text, ctl_fmlp out) carry no gradient at init
        g_ = torch.Generator(device='cpu').manual_seed(9)
        for mod in (m.ctl_text, m.ctl_fmlp[-1]): mod.weight.copy_(torch.randn(mod.weight.shape, generator=g_).to(mod.weight.device) * 0.02)
    sc = dataclasses.replace(StageCfg(), sem_cf_p=0.5, w_sem_desc=1.0, sem_bank=bank, ctl_drop=0.0)
    l = step(m, vae, b, sc)
    ps = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    gs = torch.autograd.grad(l['sem_desc'], [p for _, p in ps], allow_unused=True, retain_graph=True)
    gn = {n: (0. if g is None else float(g.float().abs().sum())) for (n, _), g in zip(ps, gs)}
    gd = sum(v for n, v in gn.items() if n.startswith('ctl_desc_in')); gc = sum(v for n, v in gn.items() if n.startswith('ctl_'))
    r = bool(torch.isfinite(l['sem_desc'])) and float(l['sem_desc_n']) > 0 and gd > 0 and gc > 0
    print(f"4a sem_desc {float(l['sem_desc']):.3f} n {float(l['sem_desc_n']):.0f}; grad ctl_desc_in {gd:.3g} ctl_* {gc:.3g}:", r); ok &= r
    l0 = step(m, vae, b, dataclasses.replace(StageCfg(), sem_cf_p=0.0, w_sem_desc=1.0, sem_bank=bank, ctl_drop=0.0))
    l1 = step(m, vae, b, dataclasses.replace(StageCfg(), sem_cf_p=1.0, w_sem_desc=1.0, sem_bank=bank, ctl_drop=0.0))
    r = float(l1['flow']) != float(l0['flow'])
    print(f"4b sem_cf_p 1 lifts imitation (flow {float(l0['flow']):.5f} -> {float(l1['flow']):.5f}):", r); ok &= r
    return ok


def main():
    pre = sys.argv[1]; td = tempfile.mkdtemp(); ok = True
    A = run(CODE, f'{td}/base.npz', 'base'); P = run(pre, f'{td}/pre.npz', 'base')
    # the appended 'sem' type adds one embedding row: the old rows' gradients must be identical, the new row's zero
    A = dict(A); P = dict(P); A2 = dict(A)
    n0 = P['ctl_type.weight'].shape[0]; ct = A.pop('ctl_type.weight'); cp = P.pop('ctl_type.weight')
    rows = np.array_equal(ct[:n0], cp) and not ct[n0:].any() and ct.shape[0] == n0 + 1
    r = same(A, P); print(f'1 default (ctl, aim + finger tokens) == pre-change (+ ctl_type old rows identical, new row 0: {rows}):', r[0] and rows, '' if r[0] else r[1]); ok &= r[0] and rows
    O = run(CODE, f'{td}/desc0.npz', 'desc0'); r = same(A2, O); print('2 ctl_desc on, no descriptor in batch == off:', r[0], '' if r[0] else r[1]); ok &= r[0]
    ok &= rest()
    print('SEMCF', 'ALL_OK' if ok else 'FAIL'); sys.exit(0 if ok else 1)


if __name__ == '__main__':
    if sys.argv[1] == 'dump': dump(*sys.argv[2:4])
    else: main()

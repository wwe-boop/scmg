"""11-09 SemREPA tests (docs/criteria/SEMREPA_1109.md section 2). GPU, SCMG_J=39, synthetic batch + random weights.
    SCMG_J=39 python scripts/tests/test_semrepa.py <pre-change code tree>
  1 default Config / StageCfg: loss dict and every gradient bit-identical to the pre-change tree
  2 semrepa = true, w_semrepa = 0: recipe loss and every shared gradient bit-identical to off (capture changes nothing, RNG untouched)
  3 the REPA term alone: non-zero gradient to planner blocks 1-2, repa_proj, p_in; exactly zero / None to planner blocks 3-8 and
    the Local Flow
  4 teacher targets inside the loss == the standalone teacher on the same frames (max abs diff < 1e-5); the per-target causal
    planner pass (_plan_as_of) captures the same block-2 state as plan() when every target sees the same inputs
    python scripts/tests/test_semrepa.py dump <out.npz> <mode: base|on0> (internal, one code tree)"""
import json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(os.path.dirname(HERE))


def setup(semrepa):
    import torch
    sys.path.insert(1, HERE)
    from test_flow_text_xattn import synth_batch, dev
    from scmg.model import SCMG, Config
    from scmg.vae import MotionVAE
    kw = dict(semrepa=True) if semrepa else {}
    torch.manual_seed(0); m = SCMG(Config(z=16, adsc=True, **kw)).to(dev).train()
    if semrepa: m.add_modules()
    torch.manual_seed(1); vae = MotionVAE(z=16).to(dev).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    b = synth_batch()
    st = dict(qm=torch.zeros(b['q'].shape[-1], device=dev), qs=torch.ones(b['q'].shape[-1], device=dev))
    return m, vae, b, st, dev


def dump(out, mode):
    import dataclasses, torch
    from scmg.losses import StageCfg, step_losses
    m, vae, b, st, dev = setup(mode == 'on0')
    sc = StageCfg() if mode == 'base' else dataclasses.replace(StageCfg(), w_semrepa=0.0)
    m.zero_grad(set_to_none=True); torch.manual_seed(5)
    with torch.autocast('cuda', torch.bfloat16):
        l = step_losses(m, vae, b, sc, st)
    l['loss'].backward()
    d = {f'__{k}': np.array(float(v)) for k, v in l.items() if torch.is_tensor(v) and v.numel() == 1}
    for n, p in m.named_parameters():
        if p.requires_grad and not n.startswith('repa_proj.'): d[n] = p.grad.float().cpu().numpy() if p.grad is not None else np.array([np.nan])
    d['__rng'] = np.array(float(torch.rand(())))                         # the global RNG stream after the step
    np.savez(out, **d)


def run(tree, out, mode):
    env = dict(os.environ, PYTHONPATH=tree, SCMG_J='39')
    subprocess.run([sys.executable, os.path.abspath(__file__), 'dump', out, mode], env=env, check=True, cwd='/tmp')
    return dict(np.load(out))


def same(a, b):
    if a.keys() != b.keys(): return False, sorted(set(a) ^ set(b))[:5]
    bad = [k for k in a if not np.array_equal(a[k], b[k], equal_nan=True)]
    return not bad, bad[:5]


def grad_and_teacher():
    import dataclasses, torch
    sys.path.insert(0, CODE)
    from scmg.losses import StageCfg, step_losses
    from scmg.semrepa import Teacher, chunk_windows, teacher_targets, WIN
    from scmg.common import CHUNK_FRAMES
    import torch.nn.functional as F
    ok = True
    m, vae, b, st, dev = setup(True)
    torch.manual_seed(11); t = Teacher(np.random.default_rng(0).uniform(0.5, 1.5, 17).astype(np.float32))
    td = tempfile.mkdtemp(); tp = f'{td}/T.pt'
    torch.save(dict(model=t.state_dict(), std=t.std.tolist(), arch=dict(win=WIN)), tp)
    sc = dataclasses.replace(StageCfg(), w_semrepa=0.5, semrepa_teacher=tp)
    m.zero_grad(set_to_none=True); torch.manual_seed(5)
    with torch.autocast('cuda', torch.bfloat16):
        l = step_losses(m, vae, b, sc, st)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    gs = torch.autograd.grad(l['semrepa'], [p for n, p in m.named_parameters() if p.requires_grad], allow_unused=True)
    gn = {n: (0. if g is None else float(g.float().abs().sum())) for n, g in zip(names, gs)}
    blk = lambda i: sum(v for n, v in gn.items() if n.startswith(f'planner.{i}.'))
    nz = [blk(0) > 0, blk(1) > 0, sum(v for n, v in gn.items() if n.startswith('repa_proj.')) > 0, gn.get('p_in.weight', 0) > 0]
    z = [blk(i) == 0 for i in range(2, len(m.planner))] + [sum(v for n, v in gn.items() if n.startswith(('flow.', 'f_'))) == 0]
    r = all(nz) and all(z)
    print(f"3 REPA-only gradient: blocks 1-2 / proj / p_in non-zero {nz}; blocks 3-{len(m.planner)} + flow zero {all(z)}; "
          f"semrepa {float(l['semrepa']):.4f}:", r); ok &= r
    # 4a teacher targets inside the loss == standalone
    K = b['q'].shape[1] // CHUNK_FRAMES; q = b['q'].float()
    tt = load = None
    from scmg.semrepa import load_teacher
    T = load_teacher(tp, dev)
    tgt = teacher_targets(T, q, K)
    k = 3; c = k * CHUNK_FRAMES + CHUNK_FRAMES // 2
    idx = np.clip(np.arange(c - WIN // 2, c + WIN // 2), 0, q.shape[1] - 1)
    with torch.no_grad(): e = F.normalize(T(q[:, idx][..., list(range(22, 39))])['emb'], dim=-1)
    d4 = float((e - tgt[:, k]).abs().max())
    r = d4 < 1e-5; print(f'4a teacher target vs standalone max abs diff {d4:.2e}:', r); ok &= r
    # 4b _plan_as_of capture == plan() capture with identical per-target inputs
    m.eval(); B = b['q'].shape[0]; d = m.cfg.d; Kc = 6
    g_ = torch.Generator().manual_seed(3)
    lat = torch.randn(B, Kc, 4, 16, generator=g_).to(dev); N = Kc * 4
    a = torch.randn(B, N, d, generator=g_).to(dev); am = torch.ones(B, Kc, N, dtype=torch.bool, device=dev)
    text = torch.randn(B, 8, d, generator=g_).to(dev); tmk = torch.ones(B, 8, dtype=torch.bool, device=dev); g = torch.randn(B, d, generator=g_).to(dev)
    with torch.no_grad():
        m._repa_capture = 2; h1 = m.plan(lat, a, am, text, tmk, g); c1 = m._repa_h.clone()
        h2 = m.plan(lat, a[:, None].expand(B, Kc, N, d).contiguous(), am, text, tmk, g); c2 = m._repa_h.clone(); m._repa_capture = 0
    d5 = float((c1 - c2).abs().max()); dh = float((h1 - h2).abs().max())
    r = d5 < 1e-4 and c1.shape == (B, Kc, d); print(f'4b causal-pass capture vs plan() max abs diff {d5:.2e} (h {dh:.2e}):', r); ok &= r
    return ok


def main():
    pre = sys.argv[1]; td = tempfile.mkdtemp(); ok = True
    A = run(CODE, f'{td}/base.npz', 'base'); P = run(pre, f'{td}/pre.npz', 'base')
    r = same(A, P); print('1 default == pre-change (loss dict, all gradients, RNG):', r[0], '' if r[0] else r[1]); ok &= r[0]
    O = run(CODE, f'{td}/on0.npz', 'on0')
    O = {k: v for k, v in O.items() if k != '__semrepa'}
    r = same(A, O); print('2 semrepa on, w 0 == off (loss dict, shared gradients, RNG):', r[0], '' if r[0] else r[1]); ok &= r[0]
    ok &= grad_and_teacher()
    print('SEMREPA', 'ALL_OK' if ok else 'FAIL'); sys.exit(0 if ok else 1)


if __name__ == '__main__':
    if sys.argv[1] == 'dump': dump(*sys.argv[2:4])
    else: main()

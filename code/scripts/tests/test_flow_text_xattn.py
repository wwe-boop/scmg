"""cfg.flow_text_xattn (11-04): ECHO-G word-time text cross-attention in every Local Flow layer.

    python scripts/tests/test_flow_text_xattn.py                 # all checks, EX002 (J=34) and G1 (J=39) in subprocesses
    python scripts/tests/test_flow_text_xattn.py --j 34          # one embodiment in this process
    python scripts/tests/test_flow_text_xattn.py dump <ckpt> <vae> <out.npz> [--n 3]
          inference (offline CFG 2.5 + engine x2/H=2) on the first n val clips + one ADSC training step, fixed seeds;
          run once with PYTHONPATH=<code before the change> and once with this code (DUMP_ADD='{"flow_text_xattn": true}':
          the checkpoint + the freshly added module, as a fine-tune starts), then
    python scripts/tests/test_flow_text_xattn.py compare a.npz b.npz        # bit-for-bit
Synthetic checks (random weights, synthetic batch, GPU):
  1 zero-init equivalence: flow_text_xattn on (fresh, gate 0) vs off, same weights / seed -> training-step losses (ADSC,
    self-forcing + CFG self-forcing) and Streamer outputs (offline CFG, engine x2 + deploy timing) bit-identical;
    building the module does not move the global RNG (base weights identical without copying)
  2 causality under ADSC visibility (gates switched on): changing text tokens that have NOT arrived by chunk k0 leaves the
    flow output of chunks < k0 bit-identical (same plan h), changes chunks >= k0; changing an arrived token changes the
    output; the same for committed Streamer chunks (x2 token streaming); untimed tokens -> exactly the global attention
  3 forward + backward of a training step with the module on: finite loss, gradients reach the new parameters
"""
import argparse, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
dev = 'cuda'


def synth_batch(B=2, T=256, Kt=24, seed=0):
    import torch
    from scmg.common import J, FEAT, SPK, TEXT_DIM
    g = torch.Generator().manual_seed(seed)
    tt = torch.sort(torch.rand(B, Kt, generator=g) * T / 50., 1).values
    tt[:, 3] = float('nan')                                       # one token without alignment
    tm = torch.zeros(B, Kt, dtype=torch.bool); tm[:, :Kt - 4] = True
    tt = torch.where(tm, tt, torch.full_like(tt, float('nan')))
    b = dict(q=torch.randn(B, T, J, generator=g) * .5, mask=torch.ones(B, T), feat=torch.randn(B, T, FEAT, generator=g),
             text=torch.randn(B, Kt, TEXT_DIM, generator=g), text_c=torch.randn(B, Kt, TEXT_DIM, generator=g), ttime=tt,
             tmask=tm, tseg=torch.where(tm, torch.zeros_like(tt), tt), spk=torch.randn(B, SPK, generator=g),
             spk_valid=torch.ones(B), source=torch.ones(B, dtype=torch.long), ref=torch.zeros(B, 128, J), ref_valid=torch.zeros(B))
    return {k: v.to(dev) for k, v in b.items()}


def models(adsc=True):
    import torch
    from scmg.model import SCMG, Config
    from scmg.vae import MotionVAE
    torch.manual_seed(0); off = SCMG(Config(z=16, adsc=adsc))
    torch.manual_seed(0); on = SCMG(Config(z=16, adsc=adsc, flow_text_xattn=True))
    same = all(torch.equal(p, on.state_dict()[k]) for k, p in off.state_dict().items())
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert not unexpected and all(k.startswith('f_text.') for k in missing), (missing[:3], unexpected[:3])
    torch.manual_seed(1); vae = MotionVAE(z=16).to(dev).eval()
    return off.to(dev), on.to(dev), vae, same


def perturb_base(m, seed=3):
    """Non-zero AdaLN projections of the base (zero-init) so that every path carries signal."""
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if (n.endswith('ada.weight') or n in ('source.weight',)) and not n.startswith('f_text'):
                p.copy_(torch.randn(p.shape, generator=g).to(p) * 0.02)


def open_gates(m, seed=4):
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for blk in m.f_text:
            blk.ada.weight.copy_(torch.randn(blk.ada.weight.shape, generator=g).to(blk.ada.weight) * 0.05)
            blk.ada.bias.fill_(0.5)
            blk.mix_raw.fill_(2.0)                                 # mix -> ~0.44 so the time prior matters


def stats_for():
    from scmg.common import J
    return dict(q_mean=[0.] * J, q_std=[1.] * J, f_mean=[0.] * 89, f_std=[1.] * 89)


def run_steps(m, vae, b, sc, seed):
    import torch
    from scmg.losses import step_losses
    st = dict(qm=torch.zeros(b['q'].shape[-1], device=dev), qs=torch.ones(b['q'].shape[-1], device=dev))
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    with torch.autocast('cuda', torch.bfloat16):
        return step_losses(m, vae, b, sc, st)


def stream(m, vae, b, i, n, **kw):
    import torch
    from scmg.infer import Streamer
    c = {k: b[k][i] for k in ('text', 'text_c', 'ttime', 'tmask', 'tseg', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid')}
    S = Streamer(m, vae, b['feat'][i], c, stats_for(), seed=7, steps=4, **kw)
    for _ in range(n): S.tick()
    return torch.stack(S.lat)


def check(name, ok, msg=''):
    print(f'  {"OK  " if ok else "FAIL"} {name} {msg}', flush=True)
    return ok


def synthetic(j):
    import torch
    from scmg.common import CHUNK, CHUNK_FRAMES, J, U80
    from scmg.losses import StageCfg
    from scmg.model import arrival_mask, INF
    assert J == j, (J, j)
    print(f'== SCMG_J={J}', flush=True)
    ok = True
    off, on, vae, same = models(adsc=True)
    ok &= check('module build leaves the global RNG untouched (base weights equal without copying)', same)
    perturb_base(off); on.load_state_dict(off.state_dict(), strict=False)
    n_new = sum(p.numel() for n_, p in on.named_parameters() if n_.startswith('f_text.'))
    n_base = sum(p.numel() for p in off.parameters())
    print(f'  params: base {n_base / 1e6:.3f}M, flow_text_xattn +{n_new / 1e6:.3f}M (+{100 * n_new / n_base:.1f}%)')
    b = synth_batch()
    sc = StageCfg(p_stream=0.67, p_async=1.0, p_async_seg=0.0, self_force_p=1.0, self_force_cfg=2.0)
    # ---- 1 zero-init equivalence
    for mode in ('train', 'eval'):
        for m in (off, on): m.train(mode == 'train')
        for seed in (0, 1, 2):
            lo, ln = run_steps(off, vae, b, sc, seed), run_steps(on, vae, b, sc, seed)
            eq = all(torch.equal(lo[k], ln[k]) for k in lo)
            ok &= check(f'1 training step losses on == off ({mode}, seed {seed})', eq,
                        f"flow {float(lo['flow'].detach()):.6f} vs {float(ln['flow'].detach()):.6f}")
    off.eval(); on.eval()
    for kw in (dict(H=INF, cfg_w=2.5), dict(H=2 * U80, cfg_w=2.5, text_mode='x2', x2_lag=0.15, deploy_timing=True)):
        a, c = stream(off, vae, b, 0, 10, **kw), stream(on, vae, b, 0, 10, **kw)
        ok &= check(f"1 Streamer on == off ({kw.get('text_mode', 'offline')}, H={kw['H']})", torch.equal(a, c),
                    f'max |diff| {float((a - c).abs().max()):.1e}')
    # ---- 2 causality (gates open)
    open_gates(on)
    B, T = b['q'].shape[:2]; K = T // CHUNK_FRAMES
    with torch.no_grad():
        m = on
        k0 = 6; now = torch.arange(K, device=dev).float() * CHUNK_FRAMES / 50.
        arr = torch.where(b['tmask'], torch.arange(b['tmask'].shape[1], device=dev).float()[None] * 0.12, b['ttime'])
        tvis = arrival_mask(arr, b['tmask'], torch.zeros(B, dtype=torch.bool, device=dev), now)
        late = b['tmask'] & (arr > now[k0 - 1])                    # not arrived by any chunk < k0
        assert late.any() and (b['tmask'] & ~late).any()
        feat = b['feat']; a = m.audio(feat, causal=torch.ones(B, dtype=torch.bool, device=dev))
        H = torch.full((B,), 2 * U80, device=dev); Nn = a.shape[1]
        amask = m.chunk_audio_mask(K, Nn, H, dev)
        def tok(raw):
            t, tm = m.text_tokens(raw, b['ttime'], b['tmask'])
            return t, tm, m.token_seconds(b['ttime'])
        text0, tm0, ts0 = tok(b['text'])
        g = m.global_cond(b['spk'], b['spk_valid'], b['source'])
        mu, _ = vae.encode(b['q']); lat = ((mu - m.lat_mean) / m.lat_std).view(B, K, CHUNK, -1)
        h = m.plan(lat, a, amask, text0, tm0, g, tvis=tvis)
        aw, awm = m.local_audio(a, H, K); aw, awm = aw.flatten(0, 1), awm.flatten(0, 1)
        M = B * K; L = m.cfg.ctx + CHUNK
        gen = torch.Generator(device=dev).manual_seed(5)
        xt = torch.randn(M, L, 16, device=dev, generator=gen); t = torch.full((M, L), 0.3, device=dev)
        known = torch.zeros(M, L, dtype=torch.bool, device=dev)
        def flow(text, tsec, vis=tvis):
            return m.flow_velocity(xt, t, known, h.flatten(0, 1), g[:, None].expand(-1, K, -1).flatten(0, 1), aw, awm,
                                   ftext=m.flow_text(text, tm0, vis, tsec, K)).view(B, K, L, -1)
        v0 = flow(text0, ts0)
        raw2 = torch.where(late[..., None], b['text'] + torch.randn_like(b['text']), b['text'])
        text2, _, _ = tok(raw2)
        v2 = flow(text2, ts0)
        d_early = float((v2[:, :k0] - v0[:, :k0]).abs().max()); d_late = float((v2[:, k0:] - v0[:, k0:]).abs().max())
        ok &= check('2 flow: unarrived tokens changed -> chunks < k0 bit-identical', torch.equal(v2[:, :k0], v0[:, :k0]), f'{d_early:.1e}')
        ok &= check('2 flow: ... and chunks >= k0 change', d_late > 1e-4, f'{d_late:.1e}')
        # unarrived token TIMES must not matter either
        ts3 = torch.where(late, ts0 + 3.0, ts0)
        v3 = flow(text0, ts3)
        ok &= check('2 flow: unarrived token times changed -> chunks < k0 bit-identical', torch.equal(v3[:, :k0], v0[:, :k0]))
        early_tok = b['tmask'] & (arr <= now[1])
        raw4 = torch.where(early_tok[..., None], b['text'] + torch.randn_like(b['text']), b['text'])
        text4, _, _ = tok(raw4)
        v4 = flow(text4, ts0)
        d4 = float((v4[:, 1:] - v0[:, 1:]).abs().max())
        ok &= check('2 flow: arrived token changed -> output changes', d4 > 1e-4, f'{d4:.1e}')
        v5 = flow(text0, ts0, vis=torch.zeros_like(tvis))
        vref = m.flow_velocity(xt, t, known, h.flatten(0, 1), g[:, None].expand(-1, K, -1).flatten(0, 1), aw, awm).view(B, K, L, -1)
        ok &= check('2 flow: no visible token == no text route (exact 0 contribution)', torch.equal(v5, vref))
        # untimed tokens: weights == global attention exactly
        blk = m.f_text[0]
        q = torch.randn(1, 8, 5, 64, device=dev); kk = torch.randn(1, 8, 7, 64, device=dev)
        keep = torch.ones(1, 5, 7, dtype=torch.bool, device=dev)
        w_nan = blk.weights(q, kk, keep, torch.full((1, 5, 7), float('nan'), device=dev))
        w_glob = blk.weights(q, kk, keep, torch.randn(1, 5, 7, device=dev))    # reference: timed tokens (prior on)
        from scmg.model import _masked_softmax
        p = blk.params(); qn = torch.nn.functional.normalize(q, dim=-1); kn = torch.nn.functional.normalize(kk, dim=-1)
        glob = _masked_softmax((qn @ kn.transpose(-1, -2)) * p['temp'][None, :, None, None], keep[:, None])
        ok &= check('2 untimed tokens -> global attention only', torch.allclose(w_nan, glob, atol=1e-7) and not torch.allclose(w_glob, glob, atol=1e-4))
    # Streamer (x2 token streaming, deploy timing): committed chunks before the arrival of the changed tokens unchanged
    kw = dict(H=2 * U80, cfg_w=2.5, text_mode='x2', x2_lag=0.15, x2_rate=4.0, deploy_timing=True)
    from scmg.model import x2_arrival
    arrS = x2_arrival(b['ttime'][:1], b['tmask'][:1], torch.tensor([4.0], device=dev), torch.tensor([0.15], device=dev))[0]
    kc = 8; tcut = kc * CHUNK_FRAMES / 50.
    lateS = b['tmask'][0] & (arrS > tcut)
    if not lateS.any(): print('  (no late token for the Streamer check)')
    bS = dict(b); bS['text_c'] = b['text_c'].clone(); bS['text_c'][0, lateS] += torch.randn_like(bS['text_c'][0, lateS])
    sA, sB = stream(on, vae, b, 0, 14, **kw), stream(on, vae, bS, 0, 14, **kw)
    nl = (kc + 1) * CHUNK                                          # committed latents of chunks 0..kc
    ok &= check(f'2 Streamer x2: tokens arriving after chunk {kc} changed -> committed chunks <= {kc} bit-identical',
                torch.equal(sA[:nl], sB[:nl]), f'later chunks diff {float((sA[nl:] - sB[nl:]).abs().max()):.1e}')
    ok &= check('2 Streamer x2: ... later chunks change', float((sA[nl:] - sB[nl:]).abs().max()) > 1e-4)
    # ---- 3 forward / backward
    # self_force_p = 0: in a self-forced batch NO planner / flow Linear gets a gradient (base included): the bf16 autocast
    # weight cache is filled by the no_grad self-forcing pass and reused by the grad pass (pre-existing, see report)
    on.train(); on.zero_grad(set_to_none=True)
    l = run_steps(on, vae, b, StageCfg(p_stream=0.67, p_async=1.0, p_async_seg=0.0, self_force_p=0.), 11)
    l['loss'].backward()
    gn = {n: float(p.grad.norm()) if p.grad is not None else 0. for n, p in on.named_parameters() if n.startswith('f_text.')}
    fin = all(np.isfinite(v) for v in gn.values()) and bool(torch.isfinite(l['loss']))
    ok &= check('3 forward + backward (gates open): finite, every new parameter gets a gradient', fin and all(v > 0 for v in gn.values()),
                f"loss {float(l['loss']):.4f}, {sum(v > 0 for v in gn.values())}/{len(gn)} f_text tensors with grad")
    fresh = models(adsc=True)[1]; fresh.train()
    for seed in (12, 13, 14):                                      # offline text (a streaming batch may have no text yet)
        l = run_steps(fresh, vae, b, StageCfg(self_force_p=0.), seed); l['loss'].backward()
    gg = sum(float(blk.ada.weight.grad.norm()) for blk in fresh.f_text)
    ok &= check('3 zero-init gate: the AdaLN gate receives gradient at step 0', gg > 0, f'|grad| {gg:.2e}')
    # informational (not a pass/fail of this module): self-forced batch -> which parameters get no gradient at all
    for mm, lab in ((off, 'off'), (on, 'on')):
        mm.train(); mm.zero_grad(set_to_none=True)
        l = run_steps(mm, vae, b, StageCfg(p_stream=0.67, p_async=1.0, p_async_seg=0.0, self_force_p=1.), 13); l['loss'].backward()
        nog = [n for n, p in mm.named_parameters() if p.grad is None]
        print(f'  info: self-forced batch ({lab}): {len(nog)}/{len(list(mm.parameters()))} parameter tensors get NO gradient '
              f"(planner {sum(n.startswith('planner.') for n in nog)}, flow {sum(n.startswith('flow.') for n in nog)}, f_text {sum(n.startswith('f_text.') for n in nog)})")
    return ok


# ---------------------------------------------------------------- before / after dump on a real checkpoint
def dump(ck, vae_path, out, n):
    import torch
    from scmg.common import CHUNK_FRAMES, U80, EMBODIMENT
    from scmg.data import View, WindowDataset, collate, VIEW
    from scmg.evaluate import load
    from scmg.infer import Streamer
    from scmg.losses import StageCfg, step_losses
    D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
    view = os.environ.get('DUMP_VIEW', VIEW); stp = os.environ.get('DUMP_STATS', D + '/STATS_v36_1.json')
    txt = os.environ.get('DUMP_TEXT', D + '/text-e5-v36_1')
    st = json.load(open(stp))
    m, vae = load(ck, vae_path, dev); m.eval()
    for k_, v_ in json.loads(os.environ.get('DUMP_ADD', '{}')).items():   # e.g. '{"flow_text_xattn": true}': fresh zero-gated module
        setattr(m.cfg, k_, v_); m.add_modules(); m.to(dev).eval(); print('added', k_, flush=True)
    v = View(view, split='val', stats=st, mmap=True, text_cache=txt)
    res = {}
    picked = []
    for ci in range(len(v.clips)):
        nch = min(v.clips[ci]['frames'] // CHUNK_FRAMES, 24)
        if nch < 8: continue
        s = WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
        if s['mask'].min() < 1 or s['tmask'].sum() == 0: continue
        picked.append((ci, nch, s))
        if len(picked) == n: break
    with torch.no_grad():
        for ci, nch, s in picked:
            cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tseg', 'text_c') if k in s}
            for name, kw in (('offline', dict(H=10 ** 9)), ('engine', dict(H=2 * U80, text_mode='x2', x2_lag=0.15, deploy_timing=True))):
                S = Streamer(m, vae, s['feat'], cond, st, steps=10, seed=0, standby=st['q_mean'], ctx_t=m.ctx_t, cfg_w=2.5, **kw)
                for _ in range(nch): S.tick()
                S.flush()
                res[f'{name}_clip{ci}'] = torch.stack(S.frames).float().cpu().numpy()
                print(name, ci, res[f'{name}_clip{ci}'].shape, flush=True)
    # one ADSC training step (sbag lane StageCfg) on the same clips' first windows, fixed seeds, train mode
    ds = WindowDataset(v, 256, max_tokens=128)
    b = {k: x.to(dev) for k, x in collate([ds.sample(ci, 0, np.random.default_rng(ci)) for ci, _, _ in picked]).items()}
    stt = dict(qm=torch.tensor(st['q_mean'], device=dev), qs=torch.tensor(st['q_std'], device=dev))
    sc = StageCfg(p_stream=0.67, p_async=1.0, p_async_seg=0.0)
    m.train()
    for seed in (0, 1):
        torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        with torch.autocast('cuda', torch.bfloat16):
            l = step_losses(m, vae, b, sc, stt)
        for k, x in l.items(): res[f'train_s{seed}_{k}'] = np.asarray(float(x))
        print('train', seed, {k: round(float(x), 6) for k, x in l.items()}, flush=True)
    np.savez(out, **res)


def compare(a, b):
    A, Bz = np.load(a), np.load(b)
    ok = sorted(A.files) == sorted(Bz.files)
    for k in sorted(A.files):
        same = A[k].shape == Bz[k].shape and np.array_equal(A[k], Bz[k])
        ok &= same
        print(f'{"OK  " if same else "DIFF"} {k} {A[k].shape} max|diff| {np.abs(A[k] - Bz[k]).max() if A[k].shape == Bz[k].shape else "shape"}')
    print('BIT-IDENTICAL' if ok else 'DIFFERENT'); return ok


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'dump':
        ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('ckpt'); ap.add_argument('vae'); ap.add_argument('out')
        ap.add_argument('--n', type=int, default=3); a = ap.parse_args()
        dump(a.ckpt, a.vae, a.out, a.n); sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
    ap = argparse.ArgumentParser(); ap.add_argument('--j', type=int, default=None); a = ap.parse_args()
    if a.j is None:
        rc = 0
        for j in (34, 39):
            env = dict(os.environ, SCMG_J=str(j), PYTHONPATH=CODE + os.pathsep + os.environ.get('PYTHONPATH', ''))
            env.pop('SCMG_TEXT_CAUSAL', None)
            rc |= subprocess.call([sys.executable, os.path.abspath(__file__), '--j', str(j)], env=env)
        print('ALL OK' if rc == 0 else 'SOME CHECKS FAILED'); sys.exit(rc)
    os.environ['SCMG_J'] = str(a.j)
    sys.path.insert(0, CODE)
    sys.exit(0 if synthetic(a.j) else 1)

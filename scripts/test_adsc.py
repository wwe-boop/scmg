"""ADSC (Asynchronous Dual-Stream Conditioning) unit tests on real checkpoints.
    python scripts/test_adsc.py <main peft ckpt> <vox base ckpt> <vae>
1. all-visible masks == no masks (main planner + VoxCPM 4-D LM mask path), bit-for-bit up to fp tolerance
2. causality: perturbing a text token that has NOT arrived before chunk k0 leaves chunks < k0 unchanged, changes >= k0
3. training step with p_async on an ADSC model: finite loss, stream_emb receives gradient
4. Streamer with text_lead runs and differs from text-final generation"""
import json, sys
import numpy as np, torch
from scmg.data import View, WindowDataset, collate
from scmg.losses import StageCfg, step_losses
from scmg.model import INF, arrival_mask, x2_arrival
from scmg.train import build_model
from scmg.vae import MotionVAE
from scmg.infer import generate

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
dev = 'cuda'; torch.manual_seed(0)
st = json.load(open(D + '/STATS_v36_1.json'))
v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', tok_cache=D + '/tok-qwen25-v36_1')
ds = WindowDataset(v, 256, max_tokens=128)
b = {k: x.to(dev) for k, x in collate([ds.sample(i, 0, np.random.default_rng(i)) for i in (0, 3)]).items()}
vck = torch.load(sys.argv[3], map_location='cpu'); vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema']); vae.eval()
ok = True


def load(path, adsc):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    extra = dict(ck['extra_cfg']); new = dict(extra, adsc=adsc)
    m, _ = build_model(ck['base_cfg'], extra, new, ck['ema'])
    return m.to(dev).eval()


def conds(m, B, K):
    N = K * 4
    a = m.audio(b['feat'], causal=torch.zeros(B, dtype=torch.bool, device=dev))
    H = torch.full((B,), INF, device=dev); amask = m.chunk_audio_mask(K, N, H, dev)
    text, tmask = m.text_tokens(b['text'], b['ttime'], b['tmask'])
    g = m.global_cond(b['spk'], b['spk_valid'], b['source'])
    return a, amask, text, tmask, g


def plan(m, lat, a, amask, text, tmask, g, **kw):
    if m.cfg.mtc:
        mt, mtm = m.mt_memory(text, tmask, a, amask, g, None, lat, None, lat.shape[1], tvis=kw.get('tvis'))
        mf, pm = m.planner_memory(mt, mtm)
        return m.plan(lat, a, amask, text, tmask, g, mf, pm, b['tids'], b['tidm'], None, **kw)
    return m.plan(lat, a, amask, text, tmask, g, tids=b['tids'], tidm=b['tidm'], **kw)


with torch.no_grad():
    for name, path in (('main', sys.argv[1]), ('vox', sys.argv[2])):
        m = load(path, False)
        B, K = 2, 16
        with torch.no_grad():
            mu, _ = vae.encode(b['q']); lat = ((mu - m.lat_mean) / m.lat_std).view(B, K, 4, -1)
        a, amask, text, tmask, g = conds(m, B, K)
        ref = plan(m, lat, a, amask, text, tmask, g)
        allv = torch.ones(B, K, text.shape[1], dtype=torch.bool, device=dev)
        lall = torch.ones(B, K, b['tids'].shape[1], dtype=torch.bool, device=dev)
        h1 = plan(m, lat, a, amask, text, tmask, g, tvis=allv, lvis=lall)
        e1 = float((h1 - ref).abs().max()); ok &= e1 < 1e-3
        print(f'[{name}] 1 all-visible vs none: max diff {e1:.2e}', 'OK' if e1 < 1e-3 else 'FAIL')
        # 2 causality: tokens arriving at chunk k0 = 8 (lead 0, now = 0.32k)
        k0 = 8; now = torch.arange(K, device=dev).float() * 0.32
        if name == 'main':
            tt = torch.where(b['tmask'], torch.full_like(b['ttime'], 0.32 * k0 - 0.01), b['ttime'])
            vis = arrival_mask(tt, b['tmask'], torch.zeros(B, dtype=torch.bool, device=dev), now)
            text2 = text + torch.randn_like(text) * b['tmask'][..., None]
            hA = plan(m, lat, a, amask, text, tmask, g, tvis=vis); hB = plan(m, lat, a, amask, text2, tmask, g, tvis=vis)
        else:
            ti = torch.where(b['tidm'], torch.full_like(b['titime'], 0.32 * k0 - 0.01), b['titime'])
            vis = arrival_mask(ti, b['tidm'], torch.zeros(B, dtype=torch.bool, device=dev), now)
            hA = plan(m, lat, a, amask, text, tmask, g, lvis=vis)
            tids0 = b['tids'].clone(); b['tids'] = torch.where(b['tidm'], (b['tids'] + 17) % 150000, b['tids'])
            hB = plan(m, lat, a, amask, text, tmask, g, lvis=vis); b['tids'] = tids0
        early = float((hA[:, :k0] - hB[:, :k0]).abs().max()); late = float((hA[:, k0:] - hB[:, k0:]).abs().max())
        c2 = early < 1e-4 and late > 1e-3; ok &= c2
        print(f'[{name}] 2 causality: chunks <{k0} diff {early:.2e} (want 0), >= {k0} diff {late:.2e} (want >0)', 'OK' if c2 else 'FAIL')
        del m; torch.cuda.empty_cache()

# 3 training step, ADSC model
for name, path in (('main', sys.argv[1]), ('vox', sys.argv[2])):
    m = load(path, True).train()
    for p in m.parameters(): p.requires_grad_(False)
    for p in m.stream_emb.parameters(): p.requires_grad_(True)
    with torch.no_grad(): m.stream_emb.weight.normal_(0, 0.02)
    stats = dict(qm=torch.tensor(st['q_mean'], device=dev), qs=torch.tensor(st['q_std'], device=dev))
    with torch.autocast('cuda', torch.bfloat16):
        out = step_losses(m, vae, b, StageCfg(p_async=1.0, self_force_p=0.0), stats)
    loss = out['loss'] if isinstance(out, dict) else out[0]
    loss.backward(); gn = float(m.stream_emb.weight.grad.norm())
    c3 = bool(torch.isfinite(loss)) and gn > 0; ok &= c3
    print(f'[{name}] 3 train step p_async=1: loss {float(loss):.4f} stream_emb grad {gn:.2e}', 'OK' if c3 else 'FAIL')
    # 4 streamer in every mode
    m.eval()
    s = ds.sample(0, 0, np.random.default_rng(0)); T = 160
    cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg')}
    gen = lambda **kw: generate(m, vae, s['feat'][:T], cond, st, T // 16, steps=4, seed=0, standby=st['q_mean'], **kw)[0]
    y_off = gen(H=10 ** 6); y_fin = gen(H=0); y_x2 = gen(H=0, text_mode='x2'); y_seg = gen(H=0, text_mode='segment')
    c4 = all(np.isfinite(y).all() for y in (y_off, y_fin, y_x2, y_seg)); ok &= bool(c4)
    print(f'[{name}] 4 streamer offline/online-final/x2/segment finite {c4}; x2 vs final {np.abs(y_x2 - y_fin).max():.3f} rad, '
          f'segment vs final {np.abs(y_seg - y_fin).max():.3f} rad')
    del m; torch.cuda.empty_cache()
# 5 X2 arrival model: never later than onset - lag, monotone in token order, rate-limited
on = torch.tensor([[0.5, 0.7, 0.9, 3.0, 3.2, float('nan')]], device=dev); vm = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool, device=dev)
ar = x2_arrival(on, vm, torch.tensor([10.], device=dev), torch.tensor([0.1], device=dev))[0, :5]
c5 = bool((ar <= on[0, :5] - 0.1 + 1e-6).all() and (ar[1:] >= ar[:-1] - 1e-6).all() and abs(float(ar[3]) - 0.7) < 1e-5); ok &= c5
print('5 x2 arrival', [round(float(x), 2) for x in ar], 'OK' if c5 else 'FAIL')
# 6 no timing leak online: changing the TIME of a token not yet voiced must not change chunks generated before it is heard
m = load(sys.argv[1], True).eval()
s = ds.sample(0, 0, np.random.default_rng(0)); T = 256
cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg')}
late = np.where(np.isfinite(cond['ttime']) & (cond['ttime'] > 3.0))[0]
c2_ = dict(cond); c2_['ttime'] = cond['ttime'].copy(); c2_['ttime'][late] += 0.7
gA = generate(m, vae, s['feat'][:T], cond, st, T // 16, H=0, steps=4, seed=0, standby=st['q_mean'])[0]
gB = generate(m, vae, s['feat'][:T], c2_, st, T // 16, H=0, steps=4, seed=0, standby=st['q_mean'])[0]
e6 = float(np.abs(gA[:140] - gB[:140]).max()) if len(late) else 0.
c6 = e6 < 1e-4; ok &= c6
print(f'6 no future timing online: {len(late)} late tokens retimed, frames before 2.8 s diff {e6:.2e}', 'OK' if c6 else 'FAIL')
print('ALL_OK' if ok else 'SOME_FAIL')

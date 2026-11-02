"""Module-by-module checks on a model over-fitted to N train clips (train.py --overfit N).
    python scripts/module_probe.py <ckpt> <vae> <N>
A VAE           arm reconstruction error on these clips (deg)
B local flow    one-chunk x1 estimate error (latent MSE / latent variance) per noise level t, GT context + GT plan
C planner       aux head R^2 for the chunk-mean latent, GT history
D audio         free generation with the clip's own audio vs another clip's audio: arm MAE to GT (deg)
E text          free generation without text: arm MAE to GT (deg)
F causality     streaming H=0 / H=4: perturb audio after frame F; frames that may not depend on it must be identical
G pipeline      free-running vs oracle-history: arm MAE (deg) and speed ratio"""
import json, sys
import numpy as np, torch
from scmg.common import CHUNK, CHUNK_FRAMES, STRIDE
from scmg.data import View, WindowDataset, collate
from scmg.evaluate import load
from scmg.infer import Streamer, generate
from scmg.model import INF
from scmg.train import overfit_clips

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st = json.load(open(D + '/STATS_v36_1.json'))
v = View(split='train', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1')
N = int(sys.argv[3]); overfit_clips(v, N)
m, vae = load(sys.argv[1], sys.argv[2], 'cuda')
qm, qs = np.asarray(st['q_mean'], np.float32), np.asarray(st['q_std'], np.float32)
A = slice(4, 18); deg = 57.2958; dev = 'cuda'
out = {}
ds = WindowDataset(v, 256, max_tokens=256)
samples = [ds.sample(ci, 0, np.random.default_rng(0)) for ci in range(N)]
mae = lambda y, g: float(np.abs(y[:min(len(y), len(g)), A] - g[:min(len(y), len(g)), A]).mean() * deg)
spdr = lambda y, g: float(np.abs(np.diff(y[:len(g), A], axis=0)).mean() / np.abs(np.diff(g[:len(y), A], axis=0)).mean())
COND = ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm')

with torch.no_grad():
    b = {k: x.to(dev) for k, x in collate(samples).items()}
    q = b['q']; Bn, T, _ = q.shape; Nl = T // STRIDE; K = Nl // CHUNK
    mu, _ = vae.encode(q)
    # A
    y = vae.decode(mu) * torch.tensor(qs, device=dev) + torch.tensor(qm, device=dev)
    out['A_vae_arm_deg'] = float(((y - (q * torch.tensor(qs, device=dev) + torch.tensor(qm, device=dev))).abs()[..., A]).mean() * deg)
    x1 = (mu - m.lat_mean) / m.lat_std; lat_k = x1.view(Bn, K, CHUNK, -1)
    a = m.audio(b['feat'], causal=torch.zeros(Bn, dtype=torch.bool, device=dev))
    H = torch.full((Bn,), INF, device=dev)
    amask = m.chunk_audio_mask(K, Nl, H, dev)
    text, tmask = m.text_tokens(b['text'], b['ttime'], b['tmask'])
    g = m.global_cond(b['spk'], b['spk_valid'], b['source'])
    h = m.plan(lat_k, a, amask, text, tmask, g, tids=b['tids'], tidm=b['tidm'])
    # C
    pred = m.p_aux(h); tgt = lat_k.flatten(-2)
    out['C_planner_aux_R2'] = float(1 - ((pred - tgt) ** 2).mean() / tgt.var(dim=(0, 1)).mean())
    # B
    M = Bn * K; L = m.cfg.ctx + CHUNK
    ctx = torch.cat([torch.zeros_like(lat_k[:, :1]), lat_k[:, :-1]], 1).flatten(0, 1)[:, -m.cfg.ctx:]
    x1i = torch.cat([ctx, lat_k.flatten(0, 1)], 1)
    known = torch.zeros(M, L, dtype=torch.bool, device=dev); known[:, :m.cfg.ctx] = (torch.arange(K, device=dev).repeat(Bn) > 0)[:, None]
    aw, awm = m.local_audio(a, H, K); aw, awm = aw.flatten(0, 1), awm.flatten(0, 1)
    gi = g[:, None].expand(-1, K, -1).flatten(0, 1)
    var = float(lat_k.var())
    for tv in (0.1, 0.3, 0.5, 0.7, 0.9):
        tt = torch.full((M, L), tv, device=dev); tt[known] = 1.
        noise = torch.randn_like(x1i, generator=torch.Generator(device=dev).manual_seed(0))
        xt = torch.where(known[..., None], x1i, tv * x1i + (1 - tv) * noise)
        vel = m.flow_velocity(xt, tt, known, h.flatten(0, 1), gi, aw, awm)
        x1h = xt + (1 - tv) * vel
        out[f'B_flow_x1_relmse_t{tv}'] = float(((x1h - x1i)[:, m.cfg.ctx:] ** 2).mean() / var)

# D, E, G (free-running streaming, 20 steps) and F (causality)
d_own, d_other, e_notext, g_free, g_orc, g_free_sp, g_orc_sp = [], [], [], [], [], [], []
for i, s in enumerate(samples[:12]):
    gt = s['q'] * qs + qm; nch = 256 // CHUNK_FRAMES
    cond = {k: s[k] for k in COND}
    kw = dict(H=INF, steps=20, standby=gt[0], ctx_t=getattr(m, 'ctx_t', 1.0))
    own, _, _ = generate(m, vae, s['feat'], cond, st, nch, **kw)
    other, _, _ = generate(m, vae, samples[(i + 1) % len(samples)]['feat'], cond, st, nch, **kw)
    c2 = dict(cond); c2['tmask'] = np.zeros_like(cond['tmask']); c2['tidm'] = np.zeros_like(cond['tidm'])
    nt, _, _ = generate(m, vae, s['feat'], c2, st, nch, **kw)
    d_own.append(mae(own, gt)); d_other.append(mae(other, gt)); e_notext.append(mae(nt, gt))
    g_free.append(mae(own, gt)); g_free_sp.append(spdr(own, gt))
    with torch.no_grad():
        mu, _ = vae.encode(torch.as_tensor(s['q'])[None].cuda()); gl = ((mu - m.lat_mean) / m.lat_std)[0]
    S = Streamer(m, vae, s['feat'], cond, st, H=INF, steps=20, standby=gt[0]); gen = []
    for k in range(nch):
        S.tick(); gen.append(torch.stack(S.lat[-CHUNK:]))
        for j in range(CHUNK): S.lat[k * CHUNK + j] = gl[k * CHUNK + j].clone()
    with torch.no_grad():
        orc = vae.decode(torch.cat(gen)[None] * m.lat_std + m.lat_mean)[0].cpu().numpy() * qs + qm
    g_orc.append(mae(orc, gt)); g_orc_sp.append(spdr(orc, gt))
out.update(D_audio_own_deg=float(np.mean(d_own)), D_audio_other_deg=float(np.mean(d_other)),
           E_no_text_deg=float(np.mean(e_notext)), G_free_deg=float(np.mean(g_free)), G_oracle_deg=float(np.mean(g_orc)),
           G_free_speed=float(np.mean(g_free_sp)), G_oracle_speed=float(np.mean(g_orc_sp)))
# F: causality. Chunk k may use audio latents < k*CHUNK + H; emitted frames lag one latent.
s = samples[0]; cond = {k: s[k] for k in COND}; nch = 256 // CHUNK_FRAMES
for Hl in (0, 4):
    base, _, _ = generate(m, vae, s['feat'], cond, st, nch, H=Hl, steps=10, standby=qm, seed=3)
    F = 160                                          # perturb audio from frame 160 (latent 40)
    f2 = s['feat'].copy(); f2[F:] = np.random.default_rng(1).normal(size=f2[F:].shape)
    pert, _, _ = generate(m, vae, f2, cond, st, nch, H=Hl, steps=10, standby=qm, seed=3)
    # chunk k sees audio latents < k*CHUNK + Hl ; perturbation at latent 40 -> chunks with k*CHUNK + Hl <= 40 unaffected
    k_last = (F // STRIDE - Hl) // CHUNK                # first chunk that may see the perturbation
    safe = k_last * CHUNK_FRAMES - STRIDE              # frames emitted before that chunk (1-latent emission delay)
    out[f'F_causal_H{Hl}_maxdiff_before'] = float(np.abs(base[:safe] - pert[:safe]).max())
    out[f'F_causal_H{Hl}_meandiff_after'] = float(np.abs(base[safe + 32:] - pert[safe + 32:]).mean())
print(json.dumps({k: round(v_, 4) for k, v_ in out.items()}, indent=1))

"""One training step's losses for every stage.

Total = flow matching (unknown latents)                                       [all stages]
      + w_aux  * planner chunk-mean prediction                                [all stages]
      + w_cont * decoded motion reconstruction + boundary pos/vel/acc continuity (frozen VAE decoder)
      + w_rhy  * audio rhythm alignment (hinge: predicted acceleration peaks on audio onsets/beats >= ground truth's)
      + w_kl   * style KL (free bits)                                          [style]
      + state-conditioned recovery: the flow/continuity targets become the simulated recovery trajectory  [state]
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .common import CHUNK, CHUNK_FRAMES, STRIDE, PROSODY_COLS, U80
from .model import INF
from .sim import simulate, state_vector, ARMS
from .vae import recon_losses


@dataclass
class StageCfg:
    p_stream: float = 0.0          # chunk-aware causal training: probability of a streaming (causal) sample
    horizons: tuple = (0, 2, 4, 8)  # audio horizon past the chunk start, in 80 ms units (0 = strictly causal)
    outpaint: bool = False         # random known/unknown masks
    p_noctx: float = 0.1           # instances without any known context
    w_aux: float = 0.1
    w_cont: float = 0.0
    w_rhy: float = 0.0
    w_kl: float = 0.05
    free_bits: float = 0.05
    p_state: float = 0.0           # fraction of instances with a simulated execution deviation
    state_hist_measured: bool = False   # ablation: feed the measured history as context latents instead of a state token
    sigma_tf: float = 0.2          # planner teacher-forcing noise (exposure bias)


def logit_normal(n, device, generator=None):
    return torch.sigmoid(torch.randn(n, device=device, generator=generator))


def segments(x, K, before, after):
    """x [B, T, ...] -> per chunk [B, K, before+after, ...] frames [16k - before, 16k + after) (edge-replicated)."""
    T = x.shape[1]
    idx = (torch.arange(K, device=x.device)[:, None] * CHUNK_FRAMES - before + torch.arange(before + after, device=x.device)[None])
    valid = (idx >= 0) & (idx < T)
    return x[:, idx.clamp(0, T - 1)], valid


def step_losses(model, vae, batch, sc: StageCfg, stats):
    cfg = model.cfg; dev = batch['q'].device
    q, mask, feat = batch['q'], batch['mask'], batch['feat']
    B, T, Jn = q.shape; N = T // STRIDE; K = N // CHUNK
    qm, qs = stats['qm'], stats['qs']
    with torch.no_grad():
        mu, _ = vae.encode(q)
        x1 = (mu - model.lat_mean) / model.lat_std                                   # [B, N, z]
    lmask = mask.view(B, N, STRIDE).min(-1).values > .5
    lat_k = x1.view(B, K, CHUNK, -1)

    # ---- condition dropout / modes
    r = lambda p: torch.rand(B, device=dev) < p
    drop_text, drop_audio, drop_spk = r(cfg.drop_text), r(cfg.drop_audio), r(cfg.drop_spk)
    stream = r(sc.p_stream)
    hz = U80 * torch.tensor(sc.horizons, device=dev)[torch.randint(len(sc.horizons), (B,), device=dev)]
    H = torch.where(stream, hz, torch.full_like(hz, INF))
    a = model.audio(feat, causal=stream)
    a = torch.where(drop_audio[:, None, None], model.audio_null.expand_as(a), a)
    amask_k = model.chunk_audio_mask(K, N, H, dev) & ~drop_audio[:, None, None]
    text, tmask = model.text_tokens(batch['text'], batch['ttime'], batch['tmask'] & ~drop_text[:, None],
                                    drop=r(cfg.drop_time))
    src = torch.where(r(0.1), torch.zeros_like(batch['source']), batch['source'])
    g = model.global_cond(batch['spk'], batch['spk_valid'] * (~drop_spk), src)

    # ---- state simulation (stage 4): replaces targets of the selected chunks by recovery trajectories
    seg_q, seg_valid = segments(q, K, CHUNK_FRAMES * 3, CHUNK_FRAMES)                # [B, K, 64, J] normalised
    seg_q = seg_q.flatten(0, 1); seg_valid = seg_valid[None].expand(B, -1, -1).flatten(0, 1)
    M = B * K
    tgt_lat = lat_k.flatten(0, 1)                                                     # [M, CHUNK, z]
    ctx_lat = torch.cat([torch.zeros_like(lat_k[:, :1]), lat_k[:, :-1]], 1).flatten(0, 1)[:, -cfg.ctx:]
    gt_frames = seg_q[:, -2 * CHUNK_FRAMES:]                                          # [M, 32, J] ctx chunk + target
    state_tok = state_ada = None
    if cfg.state:
        plan_rad = seg_q * qs + qm
        boundary = 3 * CHUNK_FRAMES
        sel = torch.rand(M, device=dev) < sc.p_state
        probs = (0., .3, .25, .25, .2)
        act, sq, sdq, target, kind = simulate(plan_rad, boundary, probs=probs)
        target = torch.where(sel[:, None, None], target, plan_rad)
        sq = torch.where(sel[:, None], sq, plan_rad[:, boundary - 1])
        sdq = torch.where(sel[:, None], sdq, (plan_rad[:, boundary - 1] - plan_rad[:, boundary - 2]) * 50.)
        tn = (target - qm) / qs
        with torch.no_grad():
            mu_t, _ = vae.encode(tn)
            lat_t = (mu_t - model.lat_mean) / model.lat_std                         # [M, 16, z]
        tgt_lat = torch.where(sel[:, None, None], lat_t[:, -CHUNK:], tgt_lat)
        if sc.state_hist_measured:
            ctx_lat = torch.where(sel[:, None, None], lat_t[:, -CHUNK - cfg.ctx:-CHUNK], ctx_lat)
        gt_frames = tn[:, -2 * CHUNK_FRAMES:]
        sv = state_vector(sq, sdq, plan_rad[:, boundary - 1], qm, qs)
        st_tok, st_ada = model.state_ad(sv)
        state_tok = st_tok.view(B, K, -1); state_ada = st_ada
        if sc.state_hist_measured:
            state_tok = None; state_ada = None

    # ---- planner (teacher forcing with noise)
    lat_in = lat_k + sc.sigma_tf * torch.rand(B, 1, 1, 1, device=dev) * torch.randn_like(lat_k)
    sref = style_tok = None; kl = None
    if cfg.style:
        ok = (batch['ref_valid'] > .5) & ~r(cfg.drop_ref)
        sref = torch.where(ok[:, None], model.style_enc(batch['ref']), model.style_null)
    mt = mt_mask = None
    if cfg.mtc:
        mt, mt_mask = model.mt_memory(text, tmask, a, amask_k, g, sref, lat_k, state_tok, K)
        mtp, mtpm = mt.flatten(0, 1), mt_mask.flatten(0, 1)
        mflat, pm = model.planner_memory(mt, mt_mask)
        h = model.plan(lat_in, a, amask_k, text, tmask, g, mflat, pm)
    else:
        h = model.plan(lat_in, a, amask_k, text, tmask, g)
    if cfg.style:
        style_tok, kl, _ = model.style_head(sref[:, None].expand(-1, K, -1), h, x_target=lat_k)
        style_tok = style_tok.flatten(0, 1)

    # ---- local flow instances
    L = cfg.ctx + CHUNK
    x1i = torch.cat([ctx_lat, tgt_lat], 1)                                            # [M, L, z]
    known = torch.zeros(M, L, dtype=torch.bool, device=dev)
    kk = torch.arange(K, device=dev).repeat(B)
    has_ctx = (kk > 0) & ~(torch.rand(M, device=dev) < sc.p_noctx)
    if sc.outpaint:
        c = torch.randint(0, cfg.ctx + 1, (M,), device=dev)                           # keep the last c context latents
        known[:, :cfg.ctx] = torch.arange(cfg.ctx, device=dev)[None] >= (cfg.ctx - c)[:, None]
        extra = torch.randint(1, CHUNK, (M,), device=dev) * (torch.rand(M, device=dev) < .3)
        known[:, cfg.ctx:] = torch.arange(CHUNK, device=dev)[None] < extra[:, None]
    else:
        known[:, :cfg.ctx] = True
    known[:, :cfg.ctx] &= has_ctx[:, None]
    t = logit_normal(M, dev)[:, None].expand(M, L).clone(); t[known] = 1.
    noise = torch.randn_like(x1i)
    xt = torch.where(known[..., None], x1i, t[..., None] * x1i + (1 - t[..., None]) * noise)
    aw, awm = model.local_audio(a, H, K)
    aw, awm = aw.flatten(0, 1), awm.flatten(0, 1)
    gi = g[:, None].expand(-1, K, -1).flatten(0, 1)
    v = model.flow_velocity(xt, t, known, h.flatten(0, 1), gi, aw, awm, style_tok, state_ada,
                            mtp if cfg.mtc else None, mtpm[:, None].expand(-1, L, -1) if cfg.mtc else None)
    tgt_valid = lmask.view(B, K, CHUNK).flatten(0, 1)
    sup = ~known
    sup[:, cfg.ctx:] &= tgt_valid
    sup[:, :cfg.ctx] = False                                                          # past context never supervised
    err = (v - (x1i - noise)).square().mean(-1)
    out = dict(flow=(err * sup).sum() / sup.sum().clamp(min=1))
    aux = (model.p_aux(h) - lat_k.flatten(-2)).square().mean(-1)
    cm = lmask.view(B, K, CHUNK).all(-1).float()
    out['aux'] = (aux * cm).sum() / cm.sum().clamp(min=1)
    loss = out['flow'] + sc.w_aux * out['aux']

    # ---- decoded continuity / rhythm
    if sc.w_cont > 0 or sc.w_rhy > 0:
        x1hat = torch.where(known[..., None], x1i, xt + (1 - t[..., None]) * v)
        y = vae.decode(x1hat * model.lat_std + model.lat_mean)                       # [M, 32, J]
        fv = seg_valid[:, -2 * CHUNK_FRAMES:].float() * torch.cat(
            [has_ctx[:, None].float().expand(M, CHUNK_FRAMES), tgt_valid.float().repeat_interleave(STRIDE, 1)], 1)
        w_inst = t[:, -1]                                                             # trust low-noise estimates more
        rl = recon_losses(y[:, CHUNK_FRAMES - 4:], gt_frames[:, CHUNK_FRAMES - 4:], fv[:, CHUNK_FRAMES - 4:] * w_inst[:, None])
        out.update({f'cont_{k}': v_ for k, v_ in rl.items()})
        loss = loss + sc.w_cont * (rl['rec'] + 0.3 * rl['vel'] + 0.1 * rl['acc'])
        if sc.w_rhy > 0:
            yt = y[:, CHUNK_FRAMES:].reshape(B, K * CHUNK_FRAMES, Jn)
            gt = gt_frames[:, CHUNK_FRAMES:].reshape(B, K * CHUNK_FRAMES, Jn)
            beat = (feat[:, :K * CHUNK_FRAMES, PROSODY_COLS['onset']] + feat[:, :K * CHUNK_FRAMES, PROSODY_COLS['beat']]).clamp(min=0)
            fm = mask[:, :K * CHUNK_FRAMES]
            def ba(m):
                acc = (m[:, 2:, ARMS] - 2 * m[:, 1:-1, ARMS] + m[:, :-2, ARMS]).norm(dim=-1)
                acc = acc / (acc * fm[:, 1:-1]).mean(1, keepdim=True).clamp(min=1e-4)
                return (acc * beat[:, 1:-1] * fm[:, 1:-1]).sum(1) / (beat[:, 1:-1] * fm[:, 1:-1]).sum(1).clamp(min=1e-3)
            out['rhy'] = F.relu(ba(gt).detach() - ba(yt)).mean()
            loss = loss + sc.w_rhy * out['rhy']
    if kl is not None:
        klm = kl.clamp(min=sc.free_bits).sum(-1)
        out['kl'] = (klm * cm).sum() / cm.sum().clamp(min=1)
        out['kl_raw'] = (kl.sum(-1) * cm).sum() / cm.sum().clamp(min=1)
        loss = loss + sc.w_kl * out['kl']
    out['loss'] = loss
    return out

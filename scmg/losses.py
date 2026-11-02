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
    sigma_tf: float = 0.5          # planner teacher-forcing noise (exposure bias; r5: 0.2 -> 0.5)
    ctx_noise_p: float = 0.9       # r5: probability that the KNOWN context latents are noised (exposure bias: with clean
    ctx_t: tuple = (0.75, 1.0)     # GT context in training, free-running speed fell to 0.28 of GT vs 0.99 with GT history)
    ctx_infer_t: float = 0.9       # context noise level used at inference by models trained with ctx noise
    self_force_p: float = 0.5      # r5: probability that a batch uses ONE-STEP SELF-FORCED history (each chunk generated
    self_force_steps: int = 4      # by the current model from GT history, no grad, then used as the next chunk's history)


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
    tidm_d = batch['tidm'] & ~drop_text[:, None] if 'tidm' in batch else None   # VoxCPM LM text: same text dropout
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
        act, _, _, target, kind = simulate(plan_rad, boundary, probs=probs)
        target = torch.where(sel[:, None, None], target, plan_rad)
        # inference timing: the streamer emits a frame one latent late, so when chunk k is generated the robot has executed
        # up to frame 16k - STRIDE; the last STRIDE frames before the boundary are committed-but-not-yet-sent PLAN frames.
        # State token and measured history are built exactly like that here (they were aligned to 16k before: mismatch).
        e_ = boundary - STRIDE                                                        # first not-yet-executed frame
        hist_rad = torch.where(sel[:, None, None], act, plan_rad[:, :boundary])
        sq = hist_rad[:, e_ - 1]; sdq = (hist_rad[:, e_ - 1] - hist_rad[:, e_ - 2]) * 50.
        q_plan_s = plan_rad[:, e_ - 1]
        tn = (target - qm) / qs
        with torch.no_grad():
            mu_t, _ = vae.encode(tn)
            lat_t = (mu_t - model.lat_mean) / model.lat_std                         # [M, 16, z]
        tgt_lat = torch.where(sel[:, None, None], lat_t[:, -CHUNK:], tgt_lat)
        if sc.state_hist_measured:                    # measured = executed frames + the pending plan frames, as at inference
            meas = torch.cat([hist_rad[:, :e_], plan_rad[:, e_:boundary]], 1)
            with torch.no_grad():
                mu_m, _ = vae.encode((meas - qm) / qs)
            lat_m = (mu_m - model.lat_mean) / model.lat_std
            ctx_lat = torch.where(sel[:, None, None], lat_m[:, -cfg.ctx:], ctx_lat)
        gt_frames = tn[:, -2 * CHUNK_FRAMES:]
        sv = state_vector(sq, sdq, q_plan_s, qm, qs)
        st_tok, st_ada = model.state_ad(sv)
        state_tok = st_tok.view(B, K, -1); state_ada = st_ada
        if sc.state_hist_measured:
            state_tok = None; state_ada = None

    # ---- planner (teacher forcing with noise)
    sref = style_tok = None; kl = None
    if cfg.style:
        ok = (batch['ref_valid'] > .5) & ~r(cfg.drop_ref)
        sref = torch.where(ok[:, None], model.style_enc(batch['ref']), model.style_null)
    mt = mt_mask = None

    def run_planner(hist):
        nonlocal mt, mt_mask
        lat_in = hist + sc.sigma_tf * torch.rand(B, 1, 1, 1, device=dev) * torch.randn_like(hist)
        if cfg.mtc:
            mt, mt_mask = model.mt_memory(text, tmask, a, amask_k, g, sref, hist, state_tok, K)
            mflat, pm = model.planner_memory(mt, mt_mask)
            return model.plan(lat_in, a, amask_k, text, tmask, g, mflat, pm, batch.get('tids'), tidm_d, sref)
        return model.plan(lat_in, a, amask_k, text, tmask, g, tids=batch.get('tids'), tidm=tidm_d, sref=sref)

    aw, awm = model.local_audio(a, H, K)
    aw, awm = aw.flatten(0, 1), awm.flatten(0, 1)
    gi = g[:, None].expand(-1, K, -1).flatten(0, 1)
    Lf = cfg.ctx + CHUNK
    hist = lat_k
    if sc.self_force_p > 0 and not cfg.state and float(torch.rand(())) < sc.self_force_p:
        # exposure-bias fix: generate every chunk from GT history (parallel, no grad), then train chunk k on the
        # GENERATED chunk k-1 as its history (planner input, multi-token history token and local-flow context)
        with torch.no_grad():
            h0 = run_planner(lat_k)
            st0 = model.style_head(sref[:, None].expand(-1, K, -1), h0)[0].flatten(0, 1) if cfg.style else None
            ctx0 = torch.cat([torch.zeros_like(lat_k[:, :1]), lat_k[:, :-1]], 1).flatten(0, 1)[:, -cfg.ctx:]
            kn0 = torch.zeros(M, Lf, dtype=torch.bool, device=dev); kn0[:, :cfg.ctx] = (torch.arange(K, device=dev).repeat(B) > 0)[:, None]
            x = torch.cat([ctx0, torch.randn(M, CHUNK, cfg.z, device=dev)], 1)
            mt0 = (mt.flatten(0, 1), mt_mask.flatten(0, 1)[:, None].expand(-1, Lf, -1)) if cfg.mtc else (None, None)
            for i in range(sc.self_force_steps):
                tt = torch.full((M, Lf), i / sc.self_force_steps, device=dev); tt[kn0] = 1.
                v0 = model.flow_velocity(x, tt, kn0, h0.flatten(0, 1), gi, aw, awm, st0, None, *mt0)
                x = torch.where(kn0[..., None], x, x + v0 / sc.self_force_steps)
            gen = x[:, cfg.ctx:].view(B, K, CHUNK, -1)
            gen = torch.where(lmask.view(B, K, CHUNK)[..., None], gen, lat_k)
        hist = gen
        ctx_lat = torch.cat([torch.zeros_like(gen[:, :1]), gen[:, :-1]], 1).flatten(0, 1)[:, -cfg.ctx:]
        out_sf = 1.
    else:
        out_sf = 0.
    h = run_planner(hist)
    if cfg.mtc: mtp, mtpm = mt.flatten(0, 1), mt_mask.flatten(0, 1)
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
    if sc.ctx_noise_p > 0:                       # noisy-but-known context: its noise level is given to the model via t
        noisy = known.clone(); noisy[:, cfg.ctx:] = False
        noisy &= (torch.rand(M, device=dev) < sc.ctx_noise_p)[:, None]
        tc = (sc.ctx_t[0] + (sc.ctx_t[1] - sc.ctx_t[0]) * torch.rand(M, device=dev))[:, None].expand(M, L)
        xt = torch.where(noisy[..., None], tc[..., None] * x1i + (1 - tc[..., None]) * noise, xt)
        t = torch.where(noisy, tc, t)
    v = model.flow_velocity(xt, t, known, h.flatten(0, 1), gi, aw, awm, style_tok, state_ada,
                            mtp if cfg.mtc else None, mtpm[:, None].expand(-1, L, -1) if cfg.mtc else None)
    tgt_valid = lmask.view(B, K, CHUNK).flatten(0, 1)
    sup = ~known
    sup[:, cfg.ctx:] &= tgt_valid
    sup[:, :cfg.ctx] = False                                                          # past context never supervised
    err = (v - (x1i - noise)).square().mean(-1)
    out = dict(flow=(err * sup).sum() / sup.sum().clamp(min=1), self_forced=torch.tensor(out_sf, device=dev))
    aux = (model.p_aux(h) - lat_k.flatten(-2)).square().mean(-1)
    cm = lmask.view(B, K, CHUNK).all(-1).float()
    out['aux'] = (aux * cm).sum() / cm.sum().clamp(min=1)
    loss = out['flow'] + sc.w_aux * out['aux']

    # ---- decoded continuity / rhythm
    if sc.w_cont > 0 or sc.w_rhy > 0:
        x1hat = torch.where(known[..., None], x1i, xt + (1 - t[..., None]) * v)
        # decode with the context the streamer has: 3 more history chunks before the known context (the decoder looks
        # ~7 latents back) and the next chunk's first latent as look-ahead (inference emits a frame only once the latent
        # after it is committed). The bare [ctx | target] window gave 9.9x boundary jerk / 0.77 deg error even for
        # ground-truth latents (scripts/vae_consistency.py) -> the loss was fighting decoder artefacts.
        Hx = 3 * CHUNK; Nl = lat_k.shape[1] * CHUNK
        src = hist.reshape(B, Nl, -1)                                                 # GT, or self-forced history
        padded = torch.cat([torch.zeros_like(src[:, :Hx + CHUNK]), src, x1.new_zeros(B, 1, src.shape[-1])], 1)
        kk_ = torch.arange(K, device=dev)
        ex_idx = kk_[:, None] * CHUNK + torch.arange(Hx, device=dev)[None]            # padded idx of latents [4k-16, 4k-4)
        extra = padded[:, ex_idx].flatten(0, 1)                                       # [M, Hx, z]
        nx = torch.clamp(kk_ + 1, max=K - 1) * CHUNK
        x1_full = torch.cat([x1, x1[:, -1:]], 1)
        nxt = torch.where((kk_ + 1 < K)[None, :, None], x1_full[:, nx], x1hat.view(B, K, L, -1)[:, :, -1]).flatten(0, 1)[:, None]
        zz = torch.cat([extra, x1hat, nxt], 1)
        yfull = vae.decode(zz * model.lat_std + model.lat_mean)
        y = yfull[:, Hx * STRIDE:(Hx + L) * STRIDE]                                   # frames of [ctx | target]
        fv = seg_valid[:, -2 * CHUNK_FRAMES:].float() * torch.cat(
            [has_ctx[:, None].float().expand(M, CHUNK_FRAMES), tgt_valid.float().repeat_interleave(STRIDE, 1)], 1)
        fv = fv * (kk.view(M) >= 4).float()[:, None]                                  # needs 4 chunks of history
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

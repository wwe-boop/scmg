"""One training step's losses for every stage.

Total = flow matching (unknown latents)                                       [all stages]
      + w_aux  * planner chunk-mean prediction                                [all stages]
      + w_cont * decoded motion reconstruction + boundary pos/vel/acc continuity (frozen VAE decoder)
      + w_rhy  * audio rhythm alignment (hinge: predicted acceleration peaks on audio onsets/beats >= ground truth's)
      + w_kl   * style KL (free bits)                                          [style]
      + state-conditioned recovery: the flow/continuity targets become the simulated recovery trajectory  [state]
"""
from dataclasses import dataclass, replace
import numpy as np
import torch
import torch.nn.functional as F

from .common import CHUNK, CHUNK_FRAMES, STRIDE, PROSODY_COLS, U80
from .model import INF, arrival_mask, stream_state, x2_arrival
from .sim import simulate, state_vector, ARMS
from .vae import recon_losses
from .data import require_causal_text


@dataclass
class StageCfg:
    cfg_joint_drop: float = 0.0
    cfg_align_audio: bool = False
    p_hist_drop: float = 0.0
    w_dfm: float = 0.0
    w_mg: float = 0.0
    self_force_steps_random: bool = False  # uniform integer in [4, 10]
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
    w_phys: float = 0.0            # stage-1 physical regularisation on the decoded chunk (11-02 user proposal): hardware joint
                                   # limits, velocity/acceleration/jerk beyond the robot's limits, self-collision (FK + calibrated
                                   # sphere proxy, scmg/kinematics.py); boundary continuity is the existing w_cont term
    phys_frames: int = 256         # frames per step that go through FK / collision (random subset of the decoded targets)
    phys_pairs: int = 40000        # sphere pairs per step (random subset; unbiased up to a constant)
    p_async: float = 0.0           # ADSC: among STREAMING samples, probability that the text also streams (else text final);
                                   # streaming samples never see timing of not-yet-voiced tokens (offline samples do)
    p_async_seg: float = 0.3       # text arrival unit: TTS segment (robot TTS / sentence-level engines) vs X2Streaming-TTS
    seg_lead: tuple = (0.0, 1.0)   # token streaming (rate below, commitment lag); segment text arrives 0-1 s before its audio
    x2_rate: tuple = (5.0, 40.0)   # committed text tokens / s (LLM + causal commitment)
    x2_lag: tuple = (0.05, 0.4)    # commitment + first-audio delay of a token (s)
    p_adsc_stream: float = 0.67    # ADSC lanes: share of streaming (online) samples; the rest is the offline mode
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


def step_losses(model, vae, batch, sc: StageCfg, stats, ema_model=None, diagnostic=None, diagnostic_seed=777):
    require_causal_text(batch.get('text_c'), sc.p_async > 0 and sc.p_async_seg < 1, 'ADSC training')
    cfg = model.cfg; dev = batch['q'].device
    if not 0 <= sc.cfg_joint_drop <= 1 or not 0 <= sc.p_hist_drop <= 1:
        raise ValueError('drop probabilities must be in [0, 1]')
    if not 0 <= sc.w_dfm < 1 or sc.w_mg < 0 or sc.self_force_steps < 1:
        raise ValueError('invalid loss weight / self-force steps')
    aligned = sc.cfg_align_audio or sc.cfg_joint_drop > 0 or diagnostic is not None
    if sc.p_hist_drop and not hasattr(model, 'hist_null'):
        raise ValueError('p_hist_drop requires Config.hist_null')
    if sc.w_mg and (ema_model is None or not aligned or sc.p_async or cfg.state):
        raise ValueError('MG requires EMA, S1 alignment, and non-ADSC/non-state stage')
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
    if sc.cfg_joint_drop > 0:
        joint = r(sc.cfg_joint_drop)
        drop_text = drop_text | joint; drop_audio = drop_audio | joint
    if diagnostic is not None:
        drop_text = torch.full_like(drop_text, diagnostic == 'no_text')
        drop_audio = torch.full_like(drop_audio, diagnostic == 'no_audio')
        drop_spk = torch.zeros_like(drop_spk)
    hist_drop = r(sc.p_hist_drop) if sc.p_hist_drop else None
    if sc.p_async > 0:                    # ADSC lanes: offline / online text-final / online text-streaming
        stream = r(sc.p_adsc_stream); asy = stream & r(sc.p_async)
    else:
        stream = r(sc.p_stream); asy = torch.zeros_like(stream)
    hz = U80 * torch.tensor(sc.horizons, device=dev)[torch.randint(len(sc.horizons), (B,), device=dev)]
    H = torch.where(stream, hz, torch.full_like(hz, INF))
    a = model.audio(feat, causal=stream)
    a = torch.where(drop_audio[:, None, None], model.audio_null.expand_as(a), a)
    amask_k = model.chunk_audio_mask(K, N, H, dev) & ~drop_audio[:, None, None]
    time_drop = r(cfg.drop_time)
    if diagnostic is not None: time_drop = torch.zeros_like(time_drop)
    text, tmask = model.text_tokens(batch['text'], batch['ttime'], batch['tmask'] & ~drop_text[:, None],
                                    drop=time_drop)
    tidm_d = batch['tidm'] & ~drop_text[:, None] if 'tidm' in batch else None   # VoxCPM LM text: same text dropout
    tvis = lvis = sstate = None
    if sc.p_async > 0:                    # ADSC: what has arrived at the start of each chunk (window-relative seconds)
        U = lambda lohi: lohi[0] + (lohi[1] - lohi[0]) * torch.rand(B, device=dev)
        seg = r(sc.p_async_seg); rate, lag, slead = U(sc.x2_rate), U(sc.x2_lag), U(sc.seg_lead)
        now = torch.arange(K, device=dev).float() * CHUNK_FRAMES / 50.
        final = ~asy
        tvalid = batch['tmask'] & ~drop_text[:, None]
        arr = torch.where(seg[:, None], batch['tseg'] - slead[:, None], x2_arrival(batch['ttime'], tvalid, rate, lag))
        tvis = arrival_mask(arr, tvalid, final, now)
        if tidm_d is not None and 'titime' in batch:
            larr = torch.where(seg[:, None], batch['tiseg'] - slead[:, None], x2_arrival(batch['titime'], tidm_d, rate, lag))
            lvis = arrival_mask(larr, tidm_d, final, now)
        nvalid = mask.sum(1)                                                          # frames before the clip end
        Hc = torch.clamp(H, max=10 ** 6)
        reach = (torch.arange(K, device=dev)[None] * CHUNK + Hc[:, None]) * STRIDE
        afin = (H[:, None] >= INF) | ((nvalid[:, None] < T) & (reach >= nvalid[:, None]))
        if hasattr(model, 'stream_emb'): sstate = stream_state(tvis, tvalid, afin)
        # timing exists only for voiced tokens when streaming (audio, and with it the anchors, arrived up to now + H)
        heard = now[None] + Hc[:, None].float() * STRIDE / 50.                        # [B, K]
        voiced = torch.isfinite(batch['ttime'])[:, None] & (torch.nan_to_num(batch['ttime'], 1e9)[:, None] <= heard[:, :, None])
        voiced = voiced | ~stream[:, None, None]
        raw = batch['text']
        if 'text_c' in batch:             # token streaming: prefix-causal features (a bidirectional encoder sees later words)
            raw = torch.where((asy & ~seg)[:, None, None], batch['text_c'], raw)
            tt_, _ = model.text_tokens(raw, batch['ttime'], tmask, drop=r(cfg.drop_time))
            text = torch.where((asy & ~seg)[:, None, None], tt_, text)
        tu, _ = model.text_tokens(raw, batch['ttime'], tmask, drop=torch.ones(B, dtype=torch.bool, device=dev))
        text = torch.cat([text, tu], 1); tmask = torch.cat([tmask, tmask], 1)
        tvis = torch.cat([tvis & voiced, tvis & ~voiced], -1)
    src = torch.where(r(0.1), torch.zeros_like(batch['source']), batch['source'])
    if diagnostic is not None: src = batch['source']
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
        # inference timing: the streamer emits a frame one latent late, so when chunk k is generated the robot has executed
        # up to frame 16k - STRIDE; the last STRIDE frames before the boundary are committed-but-not-yet-sent PLAN frames.
        # 11-02 review fix: the deviation is simulated ONLY up to the last observed frame (e_ - 1) and the recovery target
        # starts from that observed state; frames e_..boundary-1 are the pending plan (no unobserved disturbance there).
        e_ = boundary - STRIDE                                                        # first not-yet-executed frame
        act, _, _, target, kind = simulate(plan_rad, e_, probs=probs)
        target = torch.where(sel[:, None, None], target, plan_rad)
        act = torch.cat([act, plan_rad[:, e_:boundary]], 1)                           # executed + pending plan frames
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
        if diagnostic is not None: ok = batch['ref_valid'] > .5
        sref = torch.where(ok[:, None], model.style_enc(batch['ref']), model.style_null)
    mt = mt_mask = None

    lat_in = None
    def run_planner(hist):
        nonlocal mt, mt_mask, lat_in
        lat_in = hist + sc.sigma_tf * torch.rand(B, 1, 1, 1, device=dev) * torch.randn_like(hist)
        if hist_drop is not None:
            lat_in = torch.where(hist_drop[:, None, None, None], model.hist_null[None, None], lat_in)
            hist = torch.where(hist_drop[:, None, None, None], model.hist_null[None, None], hist)
        if cfg.mtc:
            mt, mt_mask = model.mt_memory(text, tmask, a, amask_k, g, sref, hist, state_tok, K, tvis=tvis)
            mflat, pm = model.planner_memory(mt, mt_mask)
            return model.plan(lat_in, a, amask_k, text, tmask, g, mflat, pm, batch.get('tids'), tidm_d, sref,
                              tvis=tvis, lvis=lvis, sstate=sstate)
        return model.plan(lat_in, a, amask_k, text, tmask, g, tids=batch.get('tids'), tidm=tidm_d, sref=sref,
                          tvis=tvis, lvis=lvis, sstate=sstate)

    aw, awm = model.local_audio(a, H, K)
    if aligned: awm = awm & ~drop_audio[:, None, None]
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
            sf_steps = int(torch.randint(4, 11, ())) if sc.self_force_steps_random else sc.self_force_steps
            for i in range(sf_steps):
                tt = torch.full((M, Lf), i / sf_steps, device=dev); tt[kn0] = 1.
                v0 = model.flow_velocity(x, tt, kn0, h0.flatten(0, 1), gi, aw, awm, st0, None, *mt0)
                x = torch.where(kn0[..., None], x, x + v0 / sf_steps)
            gen = x[:, cfg.ctx:].view(B, K, CHUNK, -1)
            gen = torch.where(lmask.view(B, K, CHUNK)[..., None], gen, lat_k)
        hist = gen
        ctx_lat = torch.cat([torch.zeros_like(gen[:, :1]), gen[:, :-1]], 1).flatten(0, 1)[:, -cfg.ctx:]
        out_sf = 1.
    else:
        out_sf = 0.
    h = run_planner(hist)
    if hist_drop is not None:
        ctx_lat = torch.where(hist_drop.repeat_interleave(K)[:, None, None], model.hist_null[-cfg.ctx:][None], ctx_lat)
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
    # Dev uses a dedicated generator: identical t/noise even across style vs non-style architectures.
    dg = torch.Generator(device=dev).manual_seed(diagnostic_seed) if diagnostic is not None else None
    t = logit_normal(M, dev, generator=dg)[:, None].expand(M, L).clone(); t[known] = 1.
    noise = torch.randn_like(x1i) if dg is None else torch.randn(x1i.shape, dtype=x1i.dtype, device=dev, generator=dg)
    xt = torch.where(known[..., None], x1i, t[..., None] * x1i + (1 - t[..., None]) * noise)
    if sc.ctx_noise_p > 0:                       # noisy-but-known context: its noise level is given to the model via t
        noisy = known.clone(); noisy[:, cfg.ctx:] = False
        noisy &= (torch.rand(M, device=dev) < sc.ctx_noise_p)[:, None]
        tc = (sc.ctx_t[0] + (sc.ctx_t[1] - sc.ctx_t[0]) * torch.rand(M, device=dev))[:, None].expand(M, L)
        xt = torch.where(noisy[..., None], tc[..., None] * x1i + (1 - tc[..., None]) * noise, xt)
        t = torch.where(noisy, tc, t)
    if hist_drop is not None:
        hd = hist_drop.repeat_interleave(K)[:, None, None]
        # Keep the null clean even when context-noising is enabled.
        xt = torch.cat([torch.where(hd, model.hist_null[-cfg.ctx:][None], xt[:, :cfg.ctx]), xt[:, cfg.ctx:]], 1)
        t[:, :cfg.ctx] = torch.where(hd[..., 0], torch.ones_like(t[:, :cfg.ctx]), t[:, :cfg.ctx])
    v = model.flow_velocity(xt, t, known, h.flatten(0, 1), gi, aw, awm, style_tok, state_ada,
                            mtp if cfg.mtc else None, mtpm[:, None].expand(-1, L, -1) if cfg.mtc else None)
    tgt_valid = lmask.view(B, K, CHUNK).flatten(0, 1)
    sup = ~known
    sup[:, cfg.ctx:] &= tgt_valid
    sup[:, :cfg.ctx] = False                                                          # past context never supervised
    target = x1i - noise
    if sc.w_mg:
        # EMA recomputes ALL conditions with its own weights, on the same xt/t/history.
        em = ema_model
        with torch.no_grad(), torch.random.fork_rng(devices=[dev.index or 0] if dev.type == 'cuda' else []):
            eg = em.global_cond(batch['spk'], batch['spk_valid'] * (~drop_spk), src)
            er = torch.where(ok[:, None], em.style_enc(batch['ref']), em.style_null) if cfg.style else None
            ea = em.audio(feat, causal=stream)
            et, etm = em.text_tokens(batch['text'], batch['ttime'], batch['tmask'], drop=time_drop)
            teacher_style = None
            def teacher(uncond):
                nonlocal teacher_style
                aa = em.audio_null.expand_as(ea) if uncond else ea
                am = torch.zeros_like(amask_k) if uncond else em.chunk_audio_mask(K, N, H, dev)
                tm = torch.zeros_like(etm) if uncond else etm
                ids = torch.zeros_like(batch['tidm']) if uncond and 'tidm' in batch else batch.get('tidm')
                memory = memory_mask = pmem = pmm = None
                ehist = hist.detach()
                ein = lat_in.detach()
                if hist_drop is not None:
                    ehist = torch.where(hist_drop[:, None, None, None], em.hist_null[None, None], ehist)
                    ein = torch.where(hist_drop[:, None, None, None], em.hist_null[None, None], ein)
                if cfg.mtc:
                    memory, memory_mask = em.mt_memory(et, tm, aa, am, eg, er, ehist, None, K)
                    pmem, pmm = em.planner_memory(memory, memory_mask)
                hh = em.plan(ein, aa, am, et, tm, eg, pmem, pmm, batch.get('tids'), ids, er)
                ww, wm = em.local_audio(aa, H, K)
                if uncond: wm = torch.zeros_like(wm)
                # Inference preserves the conditional style token in the empty branch.
                if cfg.style and not uncond:
                    teacher_style = em.style_head(er[:, None].expand(-1, K, -1), hh, x_target=lat_k)[0].flatten(0, 1)
                st = teacher_style
                return em.flow_velocity(xt, t, known, hh.flatten(0, 1), eg[:, None].expand(-1,K,-1).flatten(0,1),
                    ww.flatten(0,1), wm.flatten(0,1), st, None,
                    memory.flatten(0,1) if cfg.mtc else None,
                    memory_mask.flatten(0,1)[:,None].expand(-1,L,-1) if cfg.mtc else None)
            rng = torch.get_rng_state(); crng = torch.cuda.get_rng_state(dev) if dev.type == 'cuda' else None
            vc = teacher(False)
            torch.set_rng_state(rng)
            if crng is not None: torch.cuda.set_rng_state(crng, dev)
            vu = teacher(True)
            keep = (~drop_audio & ~drop_text).repeat_interleave(K)[:,None,None]
            correction = torch.where(keep, sc.w_mg * (vc - vu), torch.zeros_like(vc)).detach()
        target = target + correction
    err = (v - target).square().mean(-1)
    out = dict(flow=(err * sup).sum() / sup.sum().clamp(min=1), self_forced=torch.tensor(out_sf, device=dev))
    if diagnostic is not None:
        out['flow_sum'] = (err * sup).sum(); out['flow_count'] = sup.sum()
        for bi, (lo, hi) in enumerate(zip((0., .2, .5, .8), (.2, .5, .8, 1.000001))):
            bm = sup & (t >= lo) & (t < hi)
            out[f't{bi}_sum'] = (err * bm).sum(); out[f't{bi}_count'] = bm.sum()
    if sc.w_mg: out['mg_correction_rms'] = correction.square().mean().sqrt()
    aux = (model.p_aux(h) - lat_k.flatten(-2)).square().mean(-1)
    cm = lmask.view(B, K, CHUNK).all(-1).float()
    out['aux'] = (aux * cm).sum() / cm.sum().clamp(min=1)
    loss = out['flow'] + sc.w_aux * out['aux']
    if sc.w_dfm:
        # Recording identity, not a different window of the same recording.
        clips = batch['clip']; eligible = clips[:, None] != clips[None, :]
        if not eligible.any(1).all():
            raise ValueError('DFM requires at least two distinct recordings in each batch')
        neg = torch.multinomial(eligible.float(), 1).squeeze(1)
        neg_target = (x1i - noise).view(B, K, L, -1)[neg].reshape(M, L, -1).detach()
        neg_valid = tgt_valid.view(B, K, CHUNK)[neg].reshape(M, CHUNK)
        dm = sup.clone(); dm[:, cfg.ctx:] &= neg_valid
        out['dfm'] = -((v - neg_target).square().mean(-1) * dm).sum() / dm.sum().clamp(min=1)
        loss = loss + sc.w_dfm * out['dfm']

    # ---- decoded continuity / rhythm
    if sc.w_cont > 0 or sc.w_rhy > 0 or sc.w_phys > 0:
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
        if sc.w_phys > 0:
            out.update(phys_reg(y, fv * w_inst[:, None], qm, qs, sc))
            loss = loss + sc.w_phys * (out['phys_joint'] + out['phys_dyn'] + out['phys_coll'])
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


_PHYS = {}


def phys_reg(y, fw, qm, qs, sc):
    """y [M, F, J] decoded [ctx | target] frames (normalised), fw [M, F] frame weights (validity x low-noise trust).
    Returns phys_joint (hardware-limit hinge^2), phys_dyn (|v|,|a|,|j| beyond the robot's limits, relative, hinge^2),
    phys_coll (self-penetration^2 in cm^2 over a random frame / pair subset)."""
    import json, os
    from .common import CPFS
    from .data import VIEW
    from .kinematics import Kinematics
    from .safety import limits34
    dev = y.device
    if dev not in _PHYS:
        meta = json.load(open(f'{VIEW}/META.json'))
        L = limits34(meta['lower'], meta['upper'], margin=0.0)
        npz = os.environ.get('SCMG_PROXY', CPFS + '/CKPT/zehan/gesture-scmg/models/sphere_proxy_vox4e.npz')
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=dev)
        _PHYS[dev] = dict(lo=t(L['lo']), hi=t(L['hi']), v=t(L['v']), a=t(L['a']), j=t(L['j']), kin=Kinematics(npz, dev))
    P = _PHYS[dev]
    q = y.float() * qs + qm                                                          # rad
    w = fw.float()
    wn = w.sum().clamp(min=1)
    jl = (torch.relu(q - P['hi']) ** 2 + torch.relu(P['lo'] - q) ** 2).sum(-1)
    v = torch.diff(q, dim=1) * 50.; a = torch.diff(v, dim=1) * 50.; jk = torch.diff(a, dim=1) * 50.
    hv = lambda x, lim: (torch.relu(x.abs() / lim - 1.) ** 2).sum(-1)
    dyn = (hv(v, P['v']) * w[:, 1:]).sum() / wn + (hv(a, P['a']) * w[:, 2:]).sum() / wn + 0.1 * (hv(jk, P['j']) * w[:, 3:]).sum() / wn
    out = dict(phys_joint=(jl * w).sum() / wn, phys_dyn=dyn)
    tg = q[:, q.shape[1] // 2:].reshape(-1, q.shape[-1]); tw = w[:, w.shape[1] // 2:].reshape(-1)
    idx = torch.nonzero(tw > 0.5).flatten()
    if len(idx) == 0:
        out['phys_coll'] = y.sum() * 0.; return out
    idx = idx[torch.randperm(len(idx), device=dev)[:sc.phys_frames]]
    kin = P['kin']; npair = kin.pairs.shape[0]
    pi = torch.randint(0, npair, (min(sc.phys_pairs, npair),), device=dev)
    pen = kin.penetration(tg[idx], pi, d_safe=0.005) * 100.                          # cm
    out['phys_coll'] = (pen ** 2).sum(-1).mean() * (npair / len(pi))
    return out

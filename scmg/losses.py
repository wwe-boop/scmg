"""One training step's losses for every stage.

Total = flow matching (unknown latents)                                       [all stages]
      + w_aux  * planner chunk-mean prediction                                [all stages]
      + w_cont * decoded motion reconstruction + boundary pos/vel/acc continuity (frozen VAE decoder)
      + w_rhy  * audio rhythm alignment (hinge: predicted acceleration peaks on audio onsets/beats >= ground truth's)
      + w_kl   * style KL (free bits)                                          [style]
      + state-conditioned recovery: the flow/continuity targets become the simulated recovery trajectory  [state]
"""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from .common import CHUNK, CHUNK_FRAMES, STRIDE, PROSODY_COLS, U80, J
from .model import INF, arrival_mask, stream_state, x2_arrival
from .sim import simulate, state_vector, ARMS
from .vae import recon_losses
from .data import require_causal_text


@dataclass
class StageCfg:
    p_stream: float = 0.0          # chunk-aware causal training: probability of a streaming (causal) sample
    horizons: tuple = (0, 2, 4, 8)  # audio horizon past the chunk start, in 80 ms units (0 = strictly causal)
    outpaint: bool = False         # random known/unknown masks
    p_noctx: float = 0.1           # instances without any known context
    w_aux: float = 0.1
    w_int: float = 0.0             # 11-03: chunk intent CE (needs cfg.intent_aux + batch intent); text-dropped samples excluded
    sem_w: float = 0.0             # 11-03: flow-loss weight 1 + sem_w on chunks that contain an executed semantic gesture
    int_none_w: float = 0.2        # CE class weight of 'none' (0): ~90 % of labelled chunks
    # 11-03 user: Semantic Gesture Head (cfg.sem_head / sem_cond) and selective audio dropout
    w_gest: float = 0.0            # CE of the coarse g_k head (text-dropped samples excluded)
    gest_none_w: float = 0.3       # CE class weight of g = none
    gest_cw: tuple = ()            # 11-05: optional per-class CE weights (len n_sem; overrides gest_none_w), () = old behaviour
    sem_tf: float = 0.5            # P(Local Flow gets the TRUE g_k one-hot) on labelled chunks; else the head's prediction
    sem_drop: float = 0.1          # P(g_k -> null (zeros)) per sample, so g can be guided like a condition
    gest_grad_scale: float = 1.0   # 11-03 review: scale of the gesture-CE gradient that reaches the plan h (head itself trains fully)
    w_word: float = 0.0            # 11-03 word-level intent CE on the LM text-token states (cfg.vox_word_aux, batch tilab)
    word_none_w: float = 0.2       # CE class weight of 'none' (0)
    w_slot: float = 0.0            # 11-03 soft-intent-slot presence BCE (cfg.vox_slots, needs the gest frame labels)
    fsq_sat_w: float = 0.0         # 11-03 penalty on FSQ inputs beyond |z| > fsq_sat_bound (saturated FSQ cuts the text off)
    fsq_sat_bound: float = 1.2           # tanh(x)*4 reaches the outermost FSQ level from |x| >= 1.35
    text_focus_p: float = 0.0      # 11-03 P(batch is TEXT-FOCUSED): mel (and TTS rows) hidden, prosody kept, no text dropout
    sel_audio_premask: bool = False  # remove hidden chunks from encoder input as well as direct audio routes
    p_sel_audio: float = 0.0       # P(per text-bearing sample) that the audio of every gesture chunk is hidden (planner + flow)
    w_foot: float = 0.0            # 11-05 G1 only: foot-contact consistency on the decoded [ctx | target] frames (scmg/g1_contact.py:
                                   # sole xy velocity + lowest-sole height vs ground truth on GT-contact frames, official geometry)
    w_tdelta: float = 0.0          # 11-04 ECHO-G temporal loss (temporal_loss_weight 0.5 there): squared error of the change of
                                   # the velocity-field error between adjacent steps; with the identity codec (z = STRIDE*J)
                                   # also between adjacent FRAMES inside / across latents (raw-space temporal coherence)
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
    phys_dyn_w: float = 1.0        # 11-04 landing L1: weight of the velocity/acc/jerk term inside w_phys (0 = joint limits +
                                   # self-collision only; the full term made the upper arm ~20% smaller than the human)
    phys_coll_w: float = 1.0       # weight of the self-collision term inside w_phys
    p_async: float = 0.0           # ADSC: among STREAMING samples, probability that the text also streams (else text final);
                                   # streaming samples never see timing of not-yet-voiced tokens (offline samples do)
    p_async_seg: float = 0.3       # text arrival unit: TTS segment (robot TTS / sentence-level engines) vs X2Streaming-TTS
    seg_lead: tuple = (0.0, 1.0)   # token streaming (rate below, commitment lag); segment text arrives 0-1 s before its audio
    x2_rate: tuple = (5.0, 40.0)   # committed text tokens / s (LLM + causal commitment)
    x2_lag: tuple = (0.05, 0.4)    # commitment + first-audio delay of a token (s)
    p_adsc_stream: float = 0.67    # ADSC lanes: share of streaming (online) samples; the rest is the offline mode
    self_force_p: float = 0.5      # r5: probability that a batch uses ONE-STEP SELF-FORCED history (each chunk generated
    self_force_steps: int = 4      # by the current model from GT history, no grad, then used as the next chunk's history)
    self_force_rounds: int = 1     # 11-03: >1 = MULTI-STEP self-forcing: round r regenerates every chunk from round r-1's generated
                                   # history, so after R rounds each chunk's history is an R-step own rollout (hold-attractor fix)
    self_force_cfg: float = 0.0   # >0: joint audio/text CFG; local null-audio visibility matches Streamer(cfg_null_fix=True)
    self_force_mix: float = 0.0   # per round, independently replace each generated chunk with GT with this probability
    self_force_noise: float = 0.0 # Gaussian std in normalised latent space; applied before GT mixing / padding restore
    dim_mask: tuple = ()           # 11-04 ECHO-G G1 'leg masking' report lane (default off): motion channels with ZERO weight in
                                   # the decoded-motion losses (w_cont / rhythm unaffected: arms only); pair with the VAE's
                                   # dim_mask so the latents do not carry them at all (view META.json lists legs / root indices)


def logit_normal(n, device, generator=None):
    return torch.sigmoid(torch.randn(n, device=device, generator=generator))


def segments(x, K, before, after):
    """x [B, T, ...] -> per chunk [B, K, before+after, ...] frames [16k - before, 16k + after) (edge-replicated)."""
    T = x.shape[1]
    idx = (torch.arange(K, device=x.device)[:, None] * CHUNK_FRAMES - before + torch.arange(before + after, device=x.device)[None])
    valid = (idx >= 0) & (idx < T)
    return x[:, idx.clamp(0, T - 1)], valid


def chunk_intent_labels(frames, n_classes):
    """Dominant GESTURE intent (id >= 2) with >= 4 frames (80 ms), ties -> lowest id; else none (0) if every frame is labelled,
    -1 (ignored) otherwise.  A chunk with 4 gesture frames and 12 none frames is that gesture (11-03 Claude)."""
    counts = F.one_hot(frames.clamp(min=0), n_classes) * (frames >= 2)[..., None]   # gesture intents only (0 none, 1 rest)
    counts = counts.sum(-2)
    count, label = counts.max(-1)
    none = torch.where((frames >= 0).all(-1), torch.zeros_like(label), torch.full_like(label, -1))
    return torch.where(count >= 4, label, none)


def step_losses(model, vae, batch, sc: StageCfg, stats):
    require_causal_text(batch.get('text_c'), sc.p_async > 0 and sc.p_async_seg < 1, 'ADSC training')
    cfg = model.cfg; dev = batch['q'].device
    q, mask, feat = batch['q'], batch['mask'], batch['feat']
    B, T, Jn = q.shape; N = T // STRIDE; K = N // CHUNK
    from .bagger import target_chunks
    supervise, bagger = target_chunks(batch, K)
    if bagger is not None and cfg.state:
        raise ValueError('BAgger corrective targets cannot be combined with state simulation')
    qm, qs = stats['qm'], stats['qs']
    with torch.no_grad():
        mu, _ = vae.encode(q)
        x1 = (mu - model.lat_mean) / model.lat_std                                   # [B, N, z]
    lmask = mask.view(B, N, STRIDE).min(-1).values > .5
    lat_k = x1.view(B, K, CHUNK, -1)

    # ---- condition dropout / modes
    r = lambda p: torch.rand(B, device=dev) < p
    drop_text, drop_audio, drop_spk = r(cfg.drop_text), r(cfg.drop_audio), r(cfg.drop_spk)
    if bagger is not None:
        drop_text = drop_text | bagger
        drop_spk = drop_spk | batch.get('bagger_null', torch.zeros_like(bagger))
        drop_audio = drop_audio | batch.get('bagger_null', torch.zeros_like(bagger))
    if sc.text_focus_p > 0 and float(torch.rand(())) < sc.text_focus_p:
        # text-focused batch (alternates with joint batches): the speech CONTENT channels are hidden -- mel (80 cols, 0 = the
        # normalised mean) and TTS hidden rows -- while prosody (onset/energy/f0/beat) keeps the timing; text is never dropped
        feat = feat.clone(); feat[..., :80] = 0.
        if 'tts_hm' in batch: batch = dict(batch, tts_hm=torch.zeros_like(batch['tts_hm']))
        drop_text = torch.zeros_like(drop_text)
    if sc.p_async > 0:                    # ADSC lanes: offline / online text-final / online text-streaming
        stream = r(sc.p_adsc_stream); asy = stream & r(sc.p_async)
    else:
        stream = r(sc.p_stream); asy = torch.zeros_like(stream)
    hz = U80 * torch.tensor(sc.horizons, device=dev)[torch.randint(len(sc.horizons), (B,), device=dev)]
    H = torch.where(stream, hz, torch.full_like(hz, INF))
    if not sc.sel_audio_premask:  # keep the default execution / RNG order unchanged
        a = model.audio_tokens(feat, stream, batch.get('tts_h'), batch.get('tts_hm'))
        a = torch.where(drop_audio[:, None, None], model.audio_null.expand_as(a), a)
    amask_k = model.chunk_audio_mask(K, N, H, dev) & ~drop_audio[:, None, None]
    glab = hide = None
    if 'gest' in batch:                   # coarse g_k per chunk: majority class with >= 80 ms, 0 none, -1 unknown
        fg = batch['gest'].long().view(B, K, CHUNK_FRAMES)
        cnt = F.one_hot(fg.clamp(min=0), model.cfg.n_sem)[..., 1:].sum(2)                # [B, K, n_sem - 1]
        glab = torch.where(cnt.max(-1).values >= 4, cnt.argmax(-1) + 1,
                           torch.where((fg >= 0).all(-1), torch.zeros_like(fg[..., 0]), torch.full_like(fg[..., 0], -1)))
        if sc.p_sel_audio > 0:            # selective audio dropout: gesture chunks of text-bearing samples lose their audio
            hide = (r(sc.p_sel_audio) & ~drop_text & ~drop_audio)[:, None] & (glab >= 1)   # [B, K]
            amask_k = amask_k & ~hide[:, :, None]
    if sc.sel_audio_premask:
        th, hm = batch.get('tts_h'), batch.get('tts_hm')
        if hide is not None:
            hidden_frames = hide.repeat_interleave(CHUNK_FRAMES, 1)
            feat = feat.masked_fill(hidden_frames[..., None], 0.)
            if th is not None:
                hidden_rows = hidden_frames[:, ::4]
                th = th.masked_fill(hidden_rows[..., None], 0.)
                hm = (~hidden_rows).to(th.dtype) if hm is None else hm.masked_fill(hidden_rows, 0.)
        a = model.audio_tokens(feat, stream, th, hm)
        a = torch.where(drop_audio[:, None, None], model.audio_null.expand_as(a), a)
        if hide is not None:
            hidden_tokens = hide.repeat_interleave(CHUNK, 1)
            a = torch.where(hidden_tokens[..., None], model.audio_null.expand_as(a), a)
            amask_k = amask_k & ~hidden_tokens[:, None, :]  # planner AND MTC lose the hidden source tokens
    dtime = r(cfg.drop_time)
    text, tmask = model.text_tokens(batch['text'], batch['ttime'], batch['tmask'] & ~drop_text[:, None], drop=dtime)
    tsec = model.token_seconds(batch['ttime'], dtime)                                # flow_text_xattn: usable token times
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
            dtime_c = r(cfg.drop_time)
            tt_, _ = model.text_tokens(raw, batch['ttime'], tmask, drop=dtime_c)
            text = torch.where((asy & ~seg)[:, None, None], tt_, text)
            tsec = torch.where((asy & ~seg)[:, None], model.token_seconds(batch['ttime'], dtime_c), tsec)
        tu, _ = model.text_tokens(raw, batch['ttime'], tmask, drop=torch.ones(B, dtype=torch.bool, device=dev))
        text = torch.cat([text, tu], 1); tmask = torch.cat([tmask, tmask], 1)
        tsec = torch.cat([tsec, torch.full_like(tsec, float('nan'))], 1)              # the untimed copy has no time
        tvis = torch.cat([tvis & voiced, tvis & ~voiced], -1)
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
        sref = torch.where(ok[:, None], model.style_enc(batch['ref']), model.style_null)
    mt = mt_mask = plan_input = None

    def run_planner(hist):
        nonlocal mt, mt_mask, plan_input
        lat_in = hist + sc.sigma_tf * torch.rand(B, 1, 1, 1, device=dev) * torch.randn_like(hist)
        if bagger is not None: lat_in = torch.where(bagger[:, None, None, None], lat_k, lat_in)
        plan_input = lat_in             # CFG branches share the SAME teacher-noised planner history
        if cfg.mtc:
            mt, mt_mask = model.mt_memory(text, tmask, a, amask_k, g, sref, hist, state_tok, K, tvis=tvis)
            mflat, pm = model.planner_memory(mt, mt_mask)
            return model.plan(lat_in, a, amask_k, text, tmask, g, mflat, pm, batch.get('tids'), tidm_d, sref,
                              tvis=tvis, lvis=lvis, sstate=sstate)
        return model.plan(lat_in, a, amask_k, text, tmask, g, tids=batch.get('tids'), tidm=tidm_d, sref=sref,
                          tvis=tvis, lvis=lvis, sstate=sstate)

    ftext = model.flow_text(text, tmask, tvis, tsec, K)                             # None without flow_text_xattn
    aw, awm = model.local_audio(a, H, K)
    if hide is not None:                  # the hidden chunks' Local Flow audio window = the null audio (as for dropped audio)
        aw = torch.where(hide[:, :, None, None], model.audio_null.to(aw.dtype).expand_as(aw), aw)
    aw, awm = aw.flatten(0, 1), awm.flatten(0, 1)
    gi = g[:, None].expand(-1, K, -1).flatten(0, 1)
    Lf = cfg.ctx + CHUNK
    hist = lat_k
    if sc.self_force_p > 0 and not cfg.state and float(torch.rand(())) < sc.self_force_p:
        # exposure-bias fix: generate every chunk from GT history (parallel, no grad), then train chunk k on the
        # GENERATED chunk k-1 as its history (planner input, multi-token history token and local-flow context)
        # 11-04 fix: the no-grad generation pass must NOT fill autocast's weight cache -- with cache_enabled the bf16
        # weight copies made here (no grad_fn) were reused by the gradient pass below, so on self-forced batches the
        # planner and the Local Flow got NO gradient (324 of 406 tensors None).  Same autocast dtype, cache off.
        from .model import autocast_dtype
        _ac = torch.autocast('cuda', dtype=autocast_dtype(), enabled=torch.is_autocast_enabled(), cache_enabled=False)
        with torch.no_grad(), _ac:
            src = lat_k
            for _round in range(max(1, sc.self_force_rounds)):
                h0 = run_planner(src)
                st0 = model.style_head(sref[:, None].expand(-1, K, -1), h0)[0].flatten(0, 1) if cfg.style else None
                ctx0 = torch.cat([torch.zeros_like(src[:, :1]), src[:, :-1]], 1).flatten(0, 1)[:, -cfg.ctx:]
                kn0 = torch.zeros(M, Lf, dtype=torch.bool, device=dev); kn0[:, :cfg.ctx] = (torch.arange(K, device=dev).repeat(B) > 0)[:, None]
                x = torch.cat([ctx0, torch.randn(M, CHUNK, cfg.z, device=dev)], 1)
                mt0 = (mt.flatten(0, 1), mt_mask.flatten(0, 1)[:, None].expand(-1, Lf, -1)) if cfg.mtc else (None, None)
                if sc.self_force_cfg > 0:
                    # Mirror Streamer: planner hides audio/text (including VoxCPM ids), but retains speaker/style,
                    # history and arrival state. Local Flow keeps the null-audio visibility mask (cfg_null_fix=True).
                    a_u = model.audio_null.expand_as(a); am_u = torch.zeros_like(amask_k)
                    tm_u = torch.zeros_like(tmask)
                    tidm_u = torch.zeros_like(tidm_d) if tidm_d is not None else None
                    if cfg.mtc:
                        mt_u, mtm_u = model.mt_memory(text, tm_u, a_u, am_u, g, sref, src, state_tok, K, tvis=tvis)
                        mf_u, pm_u = model.planner_memory(mt_u, mtm_u)
                        h_u = model.plan(plan_input, a_u, am_u, text, tm_u, g, mf_u, pm_u, batch.get('tids'), tidm_u,
                                         sref, tvis=tvis, lvis=lvis, sstate=sstate)
                        mt0_u = (mt_u.flatten(0, 1), mtm_u.flatten(0, 1)[:, None].expand(-1, Lf, -1))
                    else:
                        h_u = model.plan(plan_input, a_u, am_u, text, tm_u, g, tids=batch.get('tids'), tidm=tidm_u,
                                         sref=sref, tvis=tvis, lvis=lvis, sstate=sstate)
                        mt0_u = (None, None)
                    h0_u = model.sem_h(h_u).flatten(0, 1)
                    ftext_u = model.flow_text(text, tm_u, tvis, tsec, K)              # text hidden like the planner's
                    aw_u = model.audio_null.expand_as(aw); awm_u = awm.clone()
                for i in range(sc.self_force_steps):
                    tt = torch.full((M, Lf), i / sc.self_force_steps, device=dev); tt[kn0] = 1.
                    v0 = model.flow_velocity(x, tt, kn0, model.sem_h(h0).flatten(0, 1), gi, aw, awm, st0, None, *mt0,
                                             ftext=ftext)
                    if sc.self_force_cfg > 0:
                        v_u = model.flow_velocity(x, tt, kn0, h0_u, gi, aw_u, awm_u, st0, None, *mt0_u, ftext=ftext_u)
                        v0 = v_u + sc.self_force_cfg * (v0 - v_u)
                    x = torch.where(kn0[..., None], x, x + v0 / sc.self_force_steps)
                gen = x[:, cfg.ctx:].view(B, K, CHUNK, -1)
                if sc.self_force_noise > 0:
                    gen = gen + sc.self_force_noise * torch.randn_like(gen)
                if sc.self_force_mix > 0:
                    use_gt = torch.rand(B, K, 1, 1, device=dev) < sc.self_force_mix
                    gen = torch.where(use_gt, lat_k, gen)
                gen = torch.where(lmask.view(B, K, CHUNK)[..., None], gen, lat_k)
                src = gen
        hist = gen
        if bagger is not None:
            gen = torch.where(bagger[:, None, None, None], lat_k, gen)
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
    if bagger is not None:
        bm = bagger[:, None].expand(B, K).reshape(M)
        has_ctx = torch.where(bm, kk > 0, has_ctx)
        known[bm, :cfg.ctx] = has_ctx[bm, None]
        known[bm, cfg.ctx:] = (~supervise).reshape(M)[bm, None]
    t = logit_normal(M, dev)[:, None].expand(M, L).clone(); t[known] = 1.
    noise = torch.randn_like(x1i)
    xt = torch.where(known[..., None], x1i, t[..., None] * x1i + (1 - t[..., None]) * noise)
    if sc.ctx_noise_p > 0:                       # noisy-but-known context: its noise level is given to the model via t
        noisy = known.clone(); noisy[:, cfg.ctx:] = False
        noisy &= (torch.rand(M, device=dev) < sc.ctx_noise_p)[:, None]
        if bagger is not None: noisy &= ~bm[:, None]
        tc = (sc.ctx_t[0] + (sc.ctx_t[1] - sc.ctx_t[0]) * torch.rand(M, device=dev))[:, None].expand(M, L)
        xt = torch.where(noisy[..., None], tc[..., None] * x1i + (1 - tc[..., None]) * noise, xt)
        t = torch.where(noisy, tc, t)
    h_flow = h; g_logits = None
    if cfg.sem_head:
        hg = h if sc.gest_grad_scale >= 1 else h * sc.gest_grad_scale + h.detach() * (1 - sc.gest_grad_scale)
        g_logits = model.sem_head(hg.float())                                        # [B, K, n_sem]
        if cfg.sem_cond:
            gp = torch.softmax(g_logits, -1).detach()                                # flow loss does not train the head
            if glab is not None:
                tf = r(sc.sem_tf)[:, None] & (glab >= 0) & ~drop_text[:, None]
                gp = torch.where(tf[..., None], F.one_hot(glab.clamp(min=0), cfg.n_sem).float(), gp)
            gp = gp * (~r(sc.sem_drop)).float()[:, None, None]                       # null g_k
            h_flow = model.sem_h(h, gp)
    v = model.flow_velocity(xt, t, known, h_flow.flatten(0, 1), gi, aw, awm, style_tok, state_ada,
                            mtp if cfg.mtc else None, mtpm[:, None].expand(-1, L, -1) if cfg.mtc else None, ftext=ftext)
    tgt_valid = lmask.view(B, K, CHUNK).flatten(0, 1)
    if supervise is not None: tgt_valid = tgt_valid & supervise.reshape(M, 1)
    sup = ~known
    sup[:, cfg.ctx:] &= tgt_valid
    sup[:, :cfg.ctx] = False                                                          # past context never supervised
    err = (v - (x1i - noise)).square().mean(-1)
    ilab = None
    if 'intent' in batch and (sc.w_int > 0 or sc.sem_w > 0):
        fi = batch['intent'].long().view(B, K, CHUNK_FRAMES)
        has = (fi >= 2).sum(-1) >= 4  # semantic weighting remains an existence test
        ilab = chunk_intent_labels(fi, cfg.n_intent)
    if ilab is not None and sc.sem_w > 0:
        wi = (1 + sc.sem_w * has.float()).flatten()[:, None]                   # [M, 1]
        flow = (err * sup * wi).sum() / (sup * wi).sum().clamp(min=1)
    else:
        flow = (err * sup).sum() / sup.sum().clamp(min=1)
    out = dict(flow=flow, self_forced=torch.tensor(out_sf, device=dev))
    if sc.w_tdelta > 0:
        d = v - (x1i - noise)                                                        # [M, L, z] velocity error
        pair = sup[:, 1:] & sup[:, :-1]
        td = ((d[:, 1:] - d[:, :-1]).square().mean(-1) * pair).sum() / pair.sum().clamp(min=1)
        if d.shape[-1] == STRIDE * J:                                                # identity codec: frame-level deltas
            df = d.reshape(d.shape[0], -1, J); sf = sup.repeat_interleave(STRIDE, 1)
            pf = sf[:, 1:] & sf[:, :-1]
            td = ((df[:, 1:] - df[:, :-1]).square().mean(-1) * pf).sum() / pf.sum().clamp(min=1)
        out['tdelta'] = td
    aux = (model.p_aux(h) - lat_k.flatten(-2)).square().mean(-1)
    cm = lmask.view(B, K, CHUNK).all(-1).float()
    if supervise is not None: cm = cm * supervise
    out['aux'] = (aux * cm).sum() / cm.sum().clamp(min=1)
    loss = out['flow'] + sc.w_aux * out['aux'] + (sc.w_tdelta * out['tdelta'] if sc.w_tdelta > 0 else 0.)
    vaux = getattr(getattr(model, 'vox', None), '_aux', None) or {}
    if sc.w_word > 0 and 'word_logits' in vaux and 'tilab' in batch:
        wl = vaux['word_logits']; tl = batch['tilab'].long()[:, :wl.shape[1]]
        tl = torch.where(batch['tidm'][:, :wl.shape[1]] & ~drop_text[:, None], tl, torch.full_like(tl, -1))
        if supervise is not None:
            times = batch.get('titime')
            if times is None: raise ValueError('BAgger word auxiliary requires token times')
            after = times[:, :wl.shape[1]] * 50 >= batch['loss_from_chunk'][:, None] * CHUNK_FRAMES
            tl = torch.where(~bagger[:, None] | after, tl, torch.full_like(tl, -1))
        cw = torch.ones(wl.shape[-1], device=dev); cw[0] = sc.word_none_w
        out['word'] = F.cross_entropy(wl.flatten(0, 1), tl.flatten(), weight=cw, ignore_index=-1) if (tl >= 0).any() else wl.sum() * 0.
        pos = tl >= 2
        out['word_acc'] = ((wl.argmax(-1) == tl) & pos).float().sum() / pos.float().sum().clamp(min=1)
        loss = loss + sc.w_word * out['word']
    if sc.w_slot > 0 and 'slot_logits' in vaux and glab is not None:
        slot_known = (glab >= 0).all(1) & ~drop_text                                # whole window labelled, text visible
        if supervise is not None: slot_known &= supervise.all(1)
        if slot_known.any():
            pres = torch.stack([(glab == c).any(1) for c in range(1, 7)], 1).float()      # gesture classes present in the window
            sl = vaux['slot_logits'][..., 1:].max(1).values                               # [B, 6] best slot per class
            out['slot'] = F.binary_cross_entropy_with_logits(sl[slot_known], pres[slot_known])
            loss = loss + sc.w_slot * out['slot']
    if sc.fsq_sat_w > 0 and 'fsq_pre' in vaux:
        sat = F.relu(vaux['fsq_pre'].float().abs() - sc.fsq_sat_bound).pow(2)
        if supervise is None:
            out['fsq_sat'] = sat.mean()
        else:
            sm = supervise.reshape(*supervise.shape, *([1] * (sat.ndim - 2))).expand_as(sat)
            out['fsq_sat'] = (sat * sm).sum() / sm.sum().clamp(min=1)
        loss = loss + sc.fsq_sat_w * out['fsq_sat']
    if g_logits is not None and glab is not None and sc.w_gest > 0:
        gl = torch.where(drop_text[:, None] | ~cm.bool(), torch.full_like(glab, -1), glab)
        cw = torch.ones(cfg.n_sem, device=dev); cw[0] = sc.gest_none_w
        if sc.gest_cw: cw = torch.tensor([float(x) for x in sc.gest_cw], device=dev)
        out['gest'] = F.cross_entropy(g_logits.flatten(0, 1), gl.flatten(), weight=cw, ignore_index=-1) if (gl >= 0).any() \
            else g_logits.sum() * 0.
        pos = gl >= 1
        out['gest_acc'] = ((g_logits.argmax(-1) == gl) & pos).float().sum() / pos.float().sum().clamp(min=1)
        for k in range(cfg.n_sem): out[f'glab_{k}'] = (gl == k).float().mean()                   # chunk label distribution (logged)
        loss = loss + sc.w_gest * out['gest']
    if ilab is not None and sc.w_int > 0 and hasattr(model, 'int_head'):
        il = torch.where(drop_text[:, None] | ~cm.bool(), torch.full_like(ilab, -1), ilab)   # no text -> intent unknowable
        cw = torch.ones(model.cfg.n_intent, device=dev); cw[0] = sc.int_none_w
        logits = model.int_head(h.float())
        out['intent'] = F.cross_entropy(logits.flatten(0, 1), il.flatten(), weight=cw, ignore_index=-1) if (il >= 0).any() \
            else logits.sum() * 0.
        pos = il >= 2
        out['intent_acc'] = ((logits.argmax(-1) == il) & pos).float().sum() / pos.float().sum().clamp(min=1)
        loss = loss + sc.w_int * out['intent']

    # ---- decoded continuity / rhythm
    if sc.w_cont > 0 or sc.w_rhy > 0 or sc.w_phys > 0 or sc.w_foot > 0:
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
        if supervise is not None: fv = fv * supervise.reshape(M, 1)
        w_inst = t[:, -1]                                                             # trust low-noise estimates more
        dw = None
        if sc.dim_mask:
            dw = torch.ones(Jn, device=dev); dw[list(sc.dim_mask)] = 0.
        rl = recon_losses(y[:, CHUNK_FRAMES - 4:], gt_frames[:, CHUNK_FRAMES - 4:], fv[:, CHUNK_FRAMES - 4:] * w_inst[:, None], dw)
        out.update({f'cont_{k}': v_ for k, v_ in rl.items()})
        loss = loss + sc.w_cont * (rl['rec'] + 0.3 * rl['vel'] + 0.1 * rl['acc'])
        if sc.w_foot > 0:
            from .common import EMBODIMENT
            if EMBODIMENT != 'g1': raise ValueError('StageCfg.w_foot is the G1 foot-contact term (SCMG_J=39)')
            from .g1_contact import foot_contact_loss
            fc = foot_contact_loss(y.float() * qs + qm, gt_frames.float() * qs + qm, fv * w_inst[:, None], 50.)
            out.update(fc); loss = loss + sc.w_foot * (fc['foot_slide'] + 10. * fc['foot_height'])   # m/s and m -> comparable scale
        if sc.w_phys > 0:
            out.update(phys_reg(y, fv * w_inst[:, None], qm, qs, sc))
            loss = loss + sc.w_phys * (out['phys_joint'] + sc.phys_dyn_w * out['phys_dyn'] + sc.phys_coll_w * out['phys_coll'])
        if sc.w_rhy > 0:
            yt = y[:, CHUNK_FRAMES:].reshape(B, K * CHUNK_FRAMES, Jn)
            gt = gt_frames[:, CHUNK_FRAMES:].reshape(B, K * CHUNK_FRAMES, Jn)
            beat = (feat[:, :K * CHUNK_FRAMES, PROSODY_COLS['onset']] + feat[:, :K * CHUNK_FRAMES, PROSODY_COLS['beat']]).clamp(min=0)
            fm = mask[:, :K * CHUNK_FRAMES]
            if supervise is not None: fm = fm * supervise.repeat_interleave(CHUNK_FRAMES, 1)
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
    from .common import CPFS, require_ex002
    require_ex002('StageCfg.w_phys (EX002 limits + FK collision proxy)')
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

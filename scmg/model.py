"""SCMG: State-Conditioned Streaming Motion Generation.

  feat(50 Hz) --AudioEncoder (full | causal)--> audio tokens a_i (12.5 Hz, one per motion latent)
  text --frozen e5 (cache)--> semantic tokens            spk emb / reference motion --> style tokens
  Global Motion Planner: causal transformer over motion CHUNKS (4 latents = 320 ms); input of chunk k =
      proj(latents of chunk k-1) + pooled visible audio of chunk k; cross-attn to semantic tokens -> plan h_k
  Local Motion Flow: flow-matching DiT over [context latents (end of chunk k-1, KNOWN) | chunk k latents (UNKNOWN)];
      AdaLN(time, h_k, global style), cross-attn to the local audio window -> velocity on unknown latents
  Pretrained MotionVAE (frozen) decodes latents to joints.

Added modules (each switchable, zero-gated so that switching on keeps the base output exactly):
  mtc      Multi-Token Conditioning: semantic / audio / style / history / state as separate typed tokens in a NEW
           gated cross-attention of every planner and flow block (base text / audio cross-attn untouched)
  style    MotionStyleEncoder(reference motion) + Style Distribution Head: per-chunk prior N(mu_p, s_p) from
           (style ref, plan h_k); posterior adds the target chunk; reparameterised z_k -> style token; KL
  chunk-aware causal training: per sample, full context (bidirectional audio encoder) or streaming with audio
           horizon H latents past the chunk start (H = 0 strict causal), causal audio encoder
  outpaint random known/unknown time masks over [context | target] + boundary pos/vel/acc continuity loss
  state    Robot State Adapter: MLP(q, dq, deviation-from-plan, contact) -> state token (AdaLN + mtc memory)
"""
import math
from dataclasses import dataclass, field, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F


def autocast_dtype():
    """Current CUDA autocast dtype on every torch the project runs (DLC image torch 2.3 has no get_autocast_dtype)."""
    return torch.get_autocast_dtype('cuda') if hasattr(torch, 'get_autocast_dtype') else torch.get_autocast_gpu_dtype()

from .common import J, FEAT, SPK, TEXT_DIM, N_SOURCE, STRIDE, CHUNK, CHUNK_FRAMES, PROSODY_COLS, FPS, U80
from .nn import Block, Encoder, MLP, sincos, apply_lora

INF = 10 ** 6


@dataclass
class Config:
    n_source: int = 7          # preserve legacy init/RNG exactly; BAgger explicitly expands to N_SOURCE
    z: int = 16                    # VAE latent dim
    d: int = 512
    heads: int = 8
    audio_layers: int = 4
    planner_layers: int = 8
    flow_layers: int = 6
    ctx: int = CHUNK               # known context latents before the target chunk
    stride: int = STRIDE           # frames per latent the model was built for (checked at load)
    flow_audio_back: int = 4 * U80 # local audio window: latents [4k - back, 4k + CHUNK + H)
    # added modules
    mtc: bool = False
    style: bool = False
    style_z: int = 8
    state: bool = False
    state_dim: int = 3 * J + 2     # q, dq, q - q_plan, contact(2, zeros when unavailable)
    planner: str = 'chunk'         # 'chunk' (from-scratch chunk transformer) | 'voxcpm' (TSLM + FSQ + RALM + LocEnc)
    lm_lr_scale: float = 0.1       # learning-rate scale of the pretrained LM (voxcpm planner)
    lora_r: int = 0                # >0: LoRA on planner/flow linear layers (base frozen by the trainer)
    # VoxCPM planner ablations (11-02, user: which VoxCPM modules make it work); defaults = the full design
    vox_lm: str = ''               # VoxCPM text-LM directory ('' = Qwen2.5-0.5B); MiniCPM4 also needs the trust flag
    vox_lm_trust_remote_code: bool = False  # opt-in for pinned local MiniCPM4 code; legacy Qwen path unchanged
    vox_lm_init: str = 'pretrained'   # 'random': selected LM architecture, no pretrained weights
    vox_fsq: bool | str = True     # True: FSQ; 'tanh': bounded continuous; False: unbounded continuous
    vox_ralm: bool = True          # False: plan = FSQ skeleton only (no residual LM)
    vox_locenc: bool = True        # False: previous chunk enters through a linear layer (as in the chunk planner)
    vox_text: bool = True          # False: no text tokens in the LM sequence (audio / motion only)
    vox_prefsq_norm: bool = False  # 11-03: RMSNorm on the LM output before FSQ (fixes tanh saturation with a pretrained LM)
    vox_in_match: bool = False     # 11-03: non-text LM inputs scaled to the RMS of the LM's token embeddings (learnable gain)
    vox_ref_gate: bool = False     # VoxCPM style reference: zero-gated add onto the speaker token (11-02 review fix) instead of
                                   # an extra LM token (which changed the base output at initialisation)
    adsc: bool = False             # Asynchronous Dual-Stream Conditioning: per-chunk text/audio visibility + stream-state token
    # 11-03 opt-in: Qwen3-TTS Talker hidden states (data.View(tts_hidden=...), one 2048-d row per 80 ms) projected
    # (LayerNorm -> Linear, zero-init) and ADDED to the audio-encoder input tokens; dropped / masked exactly like audio
    vox_text_xattn: bool = False   # 11-03 text route around the FSQ: chunks cross-attend to LM text-token states (voxplanner)
    vox_slots: int = 0             # 11-03 soft intent slots (S learned queries over the text states; chunks read the slots)
    vox_word_aux: bool = False     # 11-03 word-level intent head on the LM text-token states (losses w_word)
    sem_head: bool = False         # 11-03 user (CoordSpeaker-style): Semantic Gesture Head p(g_k | T, A, M<k) on the planner output
    sem_cond: bool = False         # ... and its g_k representation conditions the Local Flow (zero-init embedding added to h)
    n_sem: int = 7                 # scmg/semclass.py: none / point L / point R / expand / contract / emphasis / other
    intent_aux: bool = False       # 11-03 opt-in: chunk-level intent classifier on the planner output (losses w_int)
    n_intent: int = 214            # VOCAB ids 0..213 (0 none, 1 unknown, 2.. classes)
    tts_h: bool = False
    tts_h_dim: int = 2048
    audio_feat: bool = True        # False (only meaningful with tts_h): mel/prosody features zeroed at the audio encoder input
    # condition dropout (classifier-free guidance)
    drop_text: float = 0.1
    drop_audio: float = 0.1
    drop_spk: float = 0.2
    drop_ref: float = 0.5
    drop_time: float = 0.5         # token time embedding -> 'unknown' (text is NOT hard-aligned)
    # 11-04 ECHO-G-style text route into the Local Flow (default off = the exact old model): after every flow block a
    # word-time cross-attention over the planner's text tokens (global cosine attention mixed with a Gaussian prior on
    # latent time - token time, per head), output added through a zero-initialised AdaLN gate. Chunk k sees exactly the
    # planner's text of chunk k (tmask & ADSC tvis); tokens without a known time get global attention only.
    flow_text_xattn: bool = False

    def dict(self):
        return asdict(self)


# ECHO-G word-time attention bounds (release30-bounded-qk-global-local-wordtime-v2.1, word_time_attention.py)
WT = dict(temp=(1.0, 16.0, 8.0), sigma=(0.25, 2.0, 0.75), lag=0.5, mix=(0.5, 0.1), max_log_penalty=12.0, eps=1e-6)


def _logit(v, lo, hi):
    p = (v - lo) / (hi - lo)
    return math.log(p / (1 - p))


def _masked_softmax(logits, keep):
    """keep [B, 1, Q, K] bool; rows without a key -> all-zero weights; masked keys get weight exactly 0."""
    has = keep.any(-1, keepdim=True)
    safe = torch.where(has, logits.masked_fill(~keep, -torch.inf), torch.zeros_like(logits))
    return safe.softmax(-1).masked_fill(~keep, 0.)


class WordTimeXAttn(nn.Module):
    """ECHO-G WordTimeCrossAttentionBlock's text sub-layer for the SCMG Local Flow. Per head: weights =
    (1 - m s) softmax(tau cos(q, k)) + m s softmax(tau cos(q, k) + log N(dt - lag; 0, sigma)), dt = query time - token time
    (s), s = max_j N(...) over visible timed tokens (support), tau in [1, 16], sigma in [0.25, 2] s, lag in +-0.5 s,
    m <= 0.5 learnable. Tokens with unknown time (NaN) carry no prior (global only). Output gated by AdaLN(c) (shift,
    scale, gate; zero-init -> the sub-layer is an exact identity at initialisation). No dropout (keeps RNG streams of
    the base untouched). Query rows with no visible token add exactly 0."""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.n = nn.LayerNorm(d, elementwise_affine=False)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.ada = nn.Linear(d, 3 * d)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)
        tl, th, t0 = WT['temp']; sl, sh, s0 = WT['sigma']; mx, m0 = WT['mix']
        self.temp_raw = nn.Parameter(torch.full((heads,), _logit(t0, tl, th)))
        self.sigma_raw = nn.Parameter(torch.full((heads,), _logit(s0, sl, sh)))
        self.lag_raw = nn.Parameter(torch.zeros(heads))
        self.mix_raw = nn.Parameter(torch.full((heads,), _logit(m0, 0., mx)))

    def params(self):
        tl, th, _ = WT['temp']; sl, sh, _ = WT['sigma']
        return dict(temp=tl + (th - tl) * self.temp_raw.float().sigmoid(), sigma=sl + (sh - sl) * self.sigma_raw.float().sigmoid(),
                    lag=WT['lag'] * self.lag_raw.float().tanh(), mix=WT['mix'][0] * self.mix_raw.float().sigmoid())

    def weights(self, q, k, keep, dt):
        """q [B, H, Q, dh], k [B, H, K, dh], keep [B, Q, K] bool, dt [B, Q, K] s (NaN = token time unknown) -> [B, H, Q, K]."""
        p = self.params(); keep = keep[:, None]
        q = F.normalize(q.float(), dim=-1, eps=WT['eps']); k = F.normalize(k.float(), dim=-1, eps=WT['eps'])
        logits = (q @ k.transpose(-1, -2)) * p['temp'][None, :, None, None]
        glob = _masked_softmax(logits, keep)
        timed = torch.isfinite(dt)[:, None]
        z = (torch.nan_to_num(dt, 0.)[:, None].float() - p['lag'][None, :, None, None]) / p['sigma'][None, :, None, None]
        lp = torch.where(timed, -0.5 * z.square(), torch.full_like(z, -WT['max_log_penalty']))
        loc = _masked_softmax(logits + lp.clamp_min(-WT['max_log_penalty']), keep)
        support = torch.where(keep & timed, lp.exp(), torch.zeros_like(lp)).amax(-1, keepdim=True)
        em = p['mix'][None, :, None, None] * support
        return (1 - em) * glob + em * loc

    def attend(self, q, k, v, keep, dt):
        """weights(...) @ v with two fused attention kernels (same function; the explicit weights() is the reference):
        (1 - m s) SDPA(tau q^, k^, v | keep) + m s SDPA(tau q^, k^, v | keep, + log prior). Rows without a visible key -> 0."""
        p = self.params(); keep = keep[:, None]                                      # [B, 1, Q, K]
        has = keep.any(-1, keepdim=True)
        km = keep | ~has                                                             # avoid empty rows (zeroed by the caller)
        qn = F.normalize(q.float(), dim=-1, eps=WT['eps']) * p['temp'][None, :, None, None]
        kn = F.normalize(k.float(), dim=-1, eps=WT['eps'])
        timed = torch.isfinite(dt)[:, None]
        z = (torch.nan_to_num(dt, 0.)[:, None].float() - p['lag'][None, :, None, None]) / p['sigma'][None, :, None, None]
        lp = torch.where(timed, -0.5 * z.square(), torch.full_like(z, -WT['max_log_penalty'])).clamp_min(-WT['max_log_penalty'])
        em = p['mix'][None, :, None, None] * torch.where(keep & timed, lp.exp(), torch.zeros_like(lp)).amax(-1, keepdim=True)
        dtype = autocast_dtype() if torch.is_autocast_enabled() else q.dtype
        qn, kn, v = qn.to(dtype), kn.to(dtype), v.to(dtype)
        og = F.scaled_dot_product_attention(qn, kn, v, attn_mask=km, scale=1.0)
        bias = torch.where(km, lp, torch.full_like(lp, -torch.inf)).to(dtype)
        ol = F.scaled_dot_product_attention(qn, kn, v, attn_mask=bias, scale=1.0)
        return (1 - em) * og.float() + em * ol.float()

    def forward(self, x, c, kv, keep, tsec, qsec):
        """x [M, L, d] flow tokens, c [M, L, d] or [M, d] AdaLN condition; kv [Bt, Kt, d] text tokens, keep [M, Kt] tokens
        visible to instance m, tsec [Bt, Kt] token time (s, NaN unknown), qsec [M, L] time of every flow position (s, same
        origin); instances are grouped by sample: m = b * G + g (M = Bt * G)."""
        M, L, D = x.shape; Bt, Kt = kv.shape[:2]; G = M // Bt
        if G * Bt != M: raise ValueError(f'flow instances {M} are not a multiple of the text batch {Bt}')
        a = self.ada(F.silu(c))
        if a.ndim == 2: a = a[:, None]
        sh, sc, gt = a.chunk(3, -1)
        y = (self.n(x) * (1 + sc) + sh).reshape(Bt, G * L, D)
        keep_q = keep.view(Bt, G, 1, Kt).expand(Bt, G, L, Kt).reshape(Bt, G * L, Kt)
        dt = qsec.reshape(Bt, G * L)[:, :, None].float() - tsec[:, None, :].float()
        H = self.h
        q = self.q(y).view(Bt, G * L, H, -1).transpose(1, 2)
        k = self.k(kv).view(Bt, Kt, H, -1).transpose(1, 2); v = self.v(kv).view(Bt, Kt, H, -1).transpose(1, 2)
        o = self.attend(q, k, v, keep_q, dt).to(x.dtype).transpose(1, 2).reshape(Bt, G * L, D)
        o = self.o(o) * keep_q.any(-1, keepdim=True).to(x.dtype)                     # no visible text -> exactly 0
        return x + gt * o.view(M, L, D)


def gate(n):
    return nn.Parameter(torch.zeros(n))


class GatedXBlock(nn.Module):
    """Base Block + an extra gated cross-attention over the multi-token memory (tanh gate, init 0)."""

    def __init__(self, base: Block, d, heads):
        super().__init__()
        self.base = base
        from .nn import Attention
        self.nx = nn.LayerNorm(d); self.xa = Attention(d, heads); self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x, c, self_mask=None, mem=None, mem_mask=None, mt=None, mt_mask=None):
        x = self.base(x, c, self_mask, mem, mem_mask)
        if mt is not None:
            x = x + torch.tanh(self.g) * self.xa(self.nx(x), kv=mt, mask=mt_mask)
        return x


class AudioEncoder(nn.Module):
    """[B, T, 89] -> [B, N, d]; token i = frames [4i, 4i+4) (patchified, no cross-token leakage before attention)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.inp = nn.Linear(FEAT * STRIDE, cfg.d)
        self.enc = Encoder(cfg.d, cfg.heads, cfg.audio_layers)
        self.d = cfg.d

    def forward(self, feat, causal, avail=None, add=None):
        """causal: bool [B] (True = token i attends j <= i); avail: bool [B, N] keys that have arrived;
        add [B, N, d]: extra per-token input (TTS hidden states, SCMG.audio_tokens), None = the original encoder."""
        B, T, _ = feat.shape; N = T // STRIDE
        x = self.inp(feat.reshape(B, N, STRIDE * FEAT)) + sincos(torch.arange(N, device=feat.device), self.d)[None]
        if add is not None: x = x + add
        i = torch.arange(N, device=feat.device)
        m = (i[None, :] <= i[:, None])[None] | ~causal[:, None, None]       # [B, N, N]
        if avail is not None: m = m & avail[:, None, :]
        return self.enc(x, m)


class MotionStyleEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.inp = nn.Linear(J * STRIDE, cfg.d); self.enc = Encoder(cfg.d, cfg.heads, 2); self.d = cfg.d

    def forward(self, ref):
        B, R, _ = ref.shape; N = R // STRIDE
        x = self.inp(ref.reshape(B, N, STRIDE * J)) + sincos(torch.arange(N, device=ref.device), self.d)[None]
        return self.enc(x).mean(1)


class StyleHead(nn.Module):
    """Per-chunk conditional style distribution: prior p(z | style ref, h_k), posterior q(z | style ref, h_k, x_k)."""

    def __init__(self, cfg: Config):
        super().__init__()
        d, zd = cfg.d, cfg.style_z
        self.prior = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, 2 * zd))
        self.post = nn.Sequential(nn.Linear(2 * d + CHUNK * cfg.z, d), nn.SiLU(), nn.Linear(d, 2 * zd))
        self.out = nn.Linear(zd, d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, sref, h, x_target=None, temperature=1.0, generator=None):
        pin = torch.cat([sref, h], -1)
        mp, lp = self.prior(pin).chunk(2, -1); lp = lp.clamp(-8, 4)
        if x_target is None:                     # inference: draw z from the caller's generator (seeded streamer)
            eps = torch.randn(mp.shape, device=mp.device, dtype=mp.dtype, generator=generator)
            z = mp + temperature * eps * (0.5 * lp).exp(); kl = None
        else:
            mq, lq = self.post(torch.cat([pin, x_target.flatten(-2)], -1)).chunk(2, -1); lq = lq.clamp(-8, 4)
            z = mq + torch.randn_like(mq) * (0.5 * lq).exp()
            kl = 0.5 * (lp - lq + (lq.exp() + (mq - mp).square()) / lp.exp() - 1)      # [..., zd]
        return self.out(z), kl, (mp, lp)


class StateAdapter(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cfg.state_dim, cfg.d), nn.SiLU(), nn.Linear(cfg.d, cfg.d), nn.SiLU(),
                                 nn.Linear(cfg.d, cfg.d))
        self.ada = nn.Linear(cfg.d, cfg.d)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, s):
        tok = self.mlp(s)
        return tok, self.ada(tok)


def arrival_mask(arr, valid, final, now):
    """ADSC text visibility [B, K, Kt]. arr [B, Kt] arrival time of each token (s, same origin as `now`; NaN = never
    timed); final [B] bool: text final (everything visible); now [B, K] or [K] chunk start times."""
    now = now if now.dim() == 2 else now[None].expand(arr.shape[0], -1)
    vis = torch.where(torch.isfinite(arr)[:, None, :], torch.nan_to_num(arr, 0.)[:, None, :] <= now[:, :, None],
                      torch.zeros((), dtype=torch.bool, device=arr.device))
    return (vis | final[:, None, None]) & valid[:, None, :]


def x2_arrival(onset, valid, rate, lag):
    """Committed-text arrival under token-level streaming TTS (X2Streaming-TTS: text chunks from the LLM pass causal
    commitment, then are voiced; the TTS never speaks a token before it arrived). onset [B, Kt] speech onset of each
    token (s), rate [B] committed tokens / s, lag [B] commit + first-audio delay (s).
    arrival_j = min(onset_j - lag, onset_first - lag + j / rate)."""
    idx = (valid.long().cumsum(1) - 1).clamp(min=0).float()
    on = torch.where(valid & torch.isfinite(onset), onset, torch.full_like(onset, float('inf')))
    first = on.min(1).values
    first = torch.where(torch.isfinite(first), first, torch.zeros_like(first))
    cursor = first[:, None] - lag[:, None] + idx / rate[:, None]
    return torch.where(torch.isfinite(on), torch.minimum(on - lag[:, None], cursor), cursor)


def dual_text(model, text, ttime, tmask, tvis, voiced):
    """Timing only exists for tokens already voiced (TTS text_progress anchors come with the audio): every token gets a
    TIMED and an UNTIMED copy; chunk k reads the timed copy iff the token was voiced by then. Returns text [B, 2Kt, d],
    tmask [B, 2Kt], tvis [B, K, 2Kt]."""
    tt, _ = model.text_tokens(text, ttime, tmask)
    tu, _ = model.text_tokens(text, ttime, tmask, drop=torch.ones(text.shape[0], dtype=torch.bool, device=text.device))
    return torch.cat([tt, tu], 1), torch.cat([tmask, tmask], 1), torch.cat([tvis & voiced, tvis & ~voiced], -1)


def stream_state(tvis, tvalid, audio_final):
    """[B, K] long: text state (0 nothing arrived / 1 partial / 2 final) * 2 + audio state (0 streaming / 1 final)."""
    nv = tvis.sum(-1); nt = tvalid.sum(-1, keepdim=True)
    ts = torch.where(nv >= nt, 2, torch.where(nv > 0, 1, 0))
    return ts * 2 + audio_final.long()


class SCMG(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg; d = cfg.d
        if cfg.stride != STRIDE: raise ValueError(f'checkpoint built for SCMG_STRIDE={cfg.stride}, environment has {STRIDE}')
        self.audio = AudioEncoder(cfg)
        # semantic tokens (frozen e5 features come in; this projection is trainable)
        self.text_in = nn.Sequential(nn.LayerNorm(TEXT_DIM), nn.Linear(TEXT_DIM, d))
        self.text_time = nn.Linear(d, d); self.text_time_unk = nn.Parameter(torch.zeros(d))
        # global (AdaLN) style of the base: speaker embedding + data source
        self.spk_in = nn.Sequential(nn.LayerNorm(SPK), nn.Linear(SPK, d)); self.spk_null = nn.Parameter(torch.zeros(d))
        self.source = nn.Embedding(cfg.n_source, d); nn.init.zeros_(self.source.weight)
        # planner
        if cfg.planner == 'voxcpm':
            from .voxplanner import VoxPlanner
            self.vox = VoxPlanner(cfg); self.planner = nn.ModuleList()
        else:
            self.p_in = nn.Linear(CHUNK * cfg.z, d); self.p_start = nn.Parameter(torch.zeros(d))
            self.p_audio = nn.Linear(d, d)
            self.planner = nn.ModuleList([Block(d, cfg.heads, cross=True) for _ in range(cfg.planner_layers)])
            self.p_norm = nn.LayerNorm(d)
        self.p_aux = nn.Linear(d, CHUNK * cfg.z)             # auxiliary: predict chunk mean latents from the plan
        # local flow
        L = cfg.ctx + CHUNK
        self.f_in = nn.Linear(cfg.z, d); self.f_role = nn.Embedding(2, d); self.f_pos = nn.Parameter(torch.zeros(L, d))
        self.f_t = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.f_plan = nn.Linear(d, d)
        self.flow = nn.ModuleList([Block(d, cfg.heads, cross=True) for _ in range(cfg.flow_layers)])
        self.f_norm = nn.LayerNorm(d, elementwise_affine=False); self.f_ada = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.f_ada.weight); nn.init.zeros_(self.f_ada.bias)
        self.f_out = nn.Linear(d, cfg.z)
        self.audio_null = nn.Parameter(torch.zeros(d))
        self.register_buffer('lat_mean', torch.zeros(cfg.z)); self.register_buffer('lat_std', torch.ones(cfg.z))
        self.add_modules()

    def forward(self, vae, batch, sc, stats):
        """Training losses (one call, so DistributedDataParallel sees the whole computation)."""
        from .losses import step_losses
        return step_losses(self, vae, batch, sc, stats)

    # ------------------------------------------------------------------ module surgery (stages 2-4)
    def add_modules(self):
        cfg, d = self.cfg, self.cfg.d
        if cfg.mtc and not isinstance(self.flow[0], GatedXBlock):
            self.planner = nn.ModuleList([GatedXBlock(b, d, cfg.heads) for b in self.planner])
            self.flow = nn.ModuleList([GatedXBlock(b, d, cfg.heads) for b in self.flow])
            self.mt_type = nn.Embedding(6, d)                 # 0 text 1 audio 2 speaker 3 ref-style 4 history 5 state
            self.mt_hist = nn.Linear(CHUNK * cfg.z, d)
        if cfg.style and not hasattr(self, 'style_enc'):
            self.style_enc = MotionStyleEncoder(cfg); self.style_null = nn.Parameter(torch.zeros(d))
            self.style_head = StyleHead(cfg)
        if cfg.state and not hasattr(self, 'state_ad'):
            self.state_ad = StateAdapter(cfg)
        if cfg.planner == 'voxcpm' and cfg.vox_ref_gate and not hasattr(self.vox, 'ref_gate'):
            self.vox.ref_gate = nn.Parameter(torch.zeros(1))
        if cfg.adsc and not hasattr(self, 'stream_emb'):
            self._init_adsc()
        if hasattr(self, 'vox') and (cfg.vox_text_xattn or cfg.vox_slots or cfg.vox_word_aux):
            self.vox.add_text_path(cfg)
        if hasattr(self, 'vox') and cfg.vox_prefsq_norm and self.vox.prefsq_norm is None:
            # fine-tuning a checkpoint trained without it (11-03): add the pre-FSQ RMSNorm (unit gain)
            from .voxplanner import RMSNorm
            self.vox.prefsq_norm = RMSNorm(self.vox.lm.config.hidden_size)
        if cfg.sem_head and not hasattr(self, 'sem_head'):
            self.sem_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.n_sem))
        if cfg.sem_head and cfg.sem_cond and not hasattr(self, 'sem_emb'):
            self.sem_emb = nn.Linear(cfg.n_sem, d, bias=False); nn.init.zeros_(self.sem_emb.weight)   # starts as the plain model
        if cfg.intent_aux and not hasattr(self, 'int_head'):
            self.int_head = nn.Linear(d, cfg.n_intent)
        if cfg.tts_h and not hasattr(self, 'tts_in'):
            # zero-init output: switching tts_h on starts from exactly the audio-only function (gradients still reach it)
            self.tts_in = nn.Sequential(nn.LayerNorm(cfg.tts_h_dim), nn.Linear(cfg.tts_h_dim, d))
            nn.init.zeros_(self.tts_in[1].weight); nn.init.zeros_(self.tts_in[1].bias)
        if getattr(cfg, 'flow_text_xattn', False) and not hasattr(self, 'f_text'):
            # own RNG stream: adding the module (fine-tune) or building it (from scratch) leaves the global RNG untouched
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261104)
                self.f_text = nn.ModuleList([WordTimeXAttn(d, cfg.heads) for _ in range(cfg.flow_layers)])
            self.f_text.to(self.f_in.weight.device)
        if cfg.lora_r > 0 and not getattr(self, '_lora', False):
            for m in (self.planner, self.flow, self.audio) + ((self.vox.ralm,) if hasattr(self, 'vox') else ()):
                apply_lora(m, r=cfg.lora_r)
            self._lora = True

    def _init_adsc(self):
        """Stream-state token (text: 0 nothing yet / 1 partial / 2 final) x (audio: 0 streaming / 1 final), zero-init so
        loading a non-ADSC checkpoint starts from exactly the old function."""
        self.stream_emb = nn.Embedding(6, self.cfg.d); nn.init.zeros_(self.stream_emb.weight)

    # ------------------------------------------------------------------ semantic gesture conditioning
    def sem_h(self, h, g=None):
        """Planner vector handed to the Local Flow: h + E g_k. g [.., n_sem] (one-hot / probabilities / zeros = null);
        None = the head's own prediction softmax(sem_head(h)). Identity for models without sem_cond."""
        if not (self.cfg.sem_head and self.cfg.sem_cond): return h
        if g is None: g = torch.softmax(self.sem_head(h.float()), -1)
        return h + self.sem_emb(g.to(self.sem_emb.weight.dtype)).to(h.dtype)

    # ------------------------------------------------------------------ conditions
    def audio_tokens(self, feat, causal, tts_h=None, tts_hm=None):
        """Audio tokens [B, N, d]. Default (cfg.tts_h False): exactly self.audio(feat, causal).
        tts_h [B, T/4, tts_h_dim] (one row per 80 ms), tts_hm [B, T/4] (1 = row exists): the projected row r is added to the
        encoder input of the token(s) of frames [4r, 4r+4) -- the same token the audio of those frames enters, so the
        causal encoder / chunk visibility masks / audio dropout (audio_null) / CFG null branch apply to it unchanged."""
        if not self.cfg.tts_h:
            return self.audio(feat, causal=causal)
        if tts_h is None:
            raise ValueError('model was trained with tts_h (TTS hidden states): pass --tts-hidden / batch tts_h')
        e = self.tts_in(tts_h.float())
        if tts_hm is not None: e = e * tts_hm.to(e.dtype)[..., None]
        if U80 > 1: e = e.repeat_interleave(U80, 1)             # 25 Hz latents: one 80 ms row -> U80 tokens
        N = feat.shape[1] // STRIDE
        if e.shape[1] != N: raise ValueError(f'tts_h rows {e.shape[1] // U80} do not match {feat.shape[1]} frames')
        if not self.cfg.audio_feat: feat = torch.zeros_like(feat)
        return self.audio(feat, causal=causal, add=e)

    def text_tokens(self, text, ttime, tmask, drop=None):
        x = self.text_in(text)
        known = torch.isfinite(ttime)
        if drop is not None: known = known & ~drop[:, None]
        te = self.text_time(sincos(torch.nan_to_num(ttime, 0.) * FPS / STRIDE, self.cfg.d))
        return x + torch.where(known[..., None], te, self.text_time_unk), tmask

    @staticmethod
    def token_seconds(ttime, drop=None):
        """Token times the model may use (s; NaN = unknown): exactly the tokens text_tokens() gives a time embedding."""
        known = torch.isfinite(ttime)
        if drop is not None: known = known & ~drop[:, None]
        return torch.where(known, ttime, torch.full_like(ttime, float('nan')))

    @staticmethod
    def chunk_text_mask(tmask, tvis, K):
        """bool [B, Kt] valid text (+ ADSC tvis [B, K, Kt]) -> text visible to chunk k [B, K, Kt] (planner AND flow)."""
        tm = tmask[:, None, :].expand(tmask.shape[0], K, -1)
        return tm & tvis if tvis is not None else tm

    @staticmethod
    def flow_seconds(k, L, ctx):
        """Time (s, window origin) of the centre of every Local Flow position of chunk(s) k: latent 4k - ctx + j.
        k [n] long -> [n, L]."""
        i = k[:, None] * CHUNK - ctx + torch.arange(L, device=k.device)[None]
        return (i.float() + 0.5) * STRIDE / FPS

    def flow_text(self, text, tmask, tvis, tsec, K):
        """Flow text memory for the B*K training instances (m = b*K + k), None without flow_text_xattn."""
        if not hasattr(self, 'f_text'): return None
        keep = self.chunk_text_mask(tmask, tvis, K).flatten(0, 1)
        q = self.flow_seconds(torch.arange(K, device=text.device), self.cfg.ctx + CHUNK, self.cfg.ctx)
        return text, keep, tsec, q.repeat(text.shape[0], 1)

    def global_cond(self, spk, spk_valid, source):
        s = torch.where(spk_valid[:, None] > .5, self.spk_in(spk), self.spk_null)
        return s + self.source(source)

    def chunk_audio_mask(self, K, N, H, device):
        """bool [B, K, N]: audio tokens visible to chunk k = tokens < 4k + H (H = INF full context)."""
        i = torch.arange(N, device=device)
        lim = torch.arange(K, device=device)[None, :, None] * CHUNK + H[:, None, None]
        return i[None, None, :] < lim

    # ------------------------------------------------------------------ planner
    def plan(self, lat, a, amask, text, tmask, g, mt=None, mt_mask=None, tids=None, tidm=None, sref=None,
             tvis=None, lvis=None, sstate=None):
        """lat [B, K, CHUNK, z] latents of chunks 0..K-1 (teacher forcing: chunk k's input is chunk k-1);
        a [B, N, d]; amask [B, K, N]; returns h [B, K, d]."""
        B, K = lat.shape[:2]
        own = (amask & self._own_chunk(K, a.shape[1], a.device)[None]).float()          # visible audio of chunk k
        pooled = torch.einsum('bkn,bnd->bkd', own, a) / own.sum(-1, keepdim=True).clamp(min=1)
        pooled = torch.where(own.sum(-1, keepdim=True) > 0, pooled, self.audio_null)
        semb = self.stream_emb(sstate) if (sstate is not None and hasattr(self, 'stream_emb')) else None
        if self.cfg.planner == 'voxcpm':
            return self.vox(lat, pooled, tids, tidm, g, sref, lvis=lvis, semb=semb)[0]
        prev = torch.cat([self.p_start.expand(B, 1, -1), self.p_in(lat[:, :-1].flatten(-2))], 1)
        x = prev + self.p_audio(pooled) + sincos(torch.arange(K, device=lat.device), self.cfg.d)[None]
        if semb is not None: x = x + semb
        causal = torch.ones(K, K, dtype=torch.bool, device=lat.device).tril()[None]
        tm = self.chunk_text_mask(tmask, tvis, K)                                     # ADSC: text arrived by chunk k
        for blk in self.planner:
            if isinstance(blk, GatedXBlock): x = blk(x, g, causal, text, tm, mt, mt_mask)
            else: x = blk(x, g, causal, text, tm)
        return self.p_norm(x)

    @staticmethod
    def _own_chunk(K, N, device):
        i = torch.arange(N, device=device); k = torch.arange(K, device=device)
        return (i[None] // CHUNK) == k[:, None]

    # ------------------------------------------------------------------ local flow
    def flow_velocity(self, xt, t, known, h, g, aw, aw_mask, style_tok=None, state_ada=None, mt=None, mt_mask=None,
                      ftext=None):
        """xt [M, L, z] (known positions hold clean latents); t [M, L] (1 on known); h [M, d]; g [M, d];
        aw [M, W, d] local audio window, aw_mask [M, W]; ftext (flow_text_xattn): (text [Bt, Kt, d], keep [M, Kt],
        token s [Bt, Kt], position s [M, L]) or None (= no text visible)."""
        x = self.f_in(xt) + self.f_role(known.long()) + self.f_pos[None]
        c = self.f_t(sincos(t * 1000., self.cfg.d)) + (self.f_plan(h) + g)[:, None]
        if style_tok is not None: c = c + style_tok[:, None]
        if state_ada is not None: c = c + state_ada[:, None]
        am = aw_mask[:, None, :].expand(-1, x.shape[1], -1)
        ftx = self.f_text if (ftext is not None and hasattr(self, 'f_text')) else None
        if ftx is not None:
            # keep only token columns visible to SOME instance (masked keys contribute exactly 0; most of the 128 / 256
            # slots are padding or not-yet-arrived) -- one host sync per call; no visible token at all = no text route
            text_, keep_, tsec_, qsec_ = ftext
            cols = keep_.any(0).nonzero().squeeze(1)
            if cols.numel() == 0: ftx = None
            elif cols.numel() < keep_.shape[1]:
                ftext = (text_.index_select(1, cols), keep_.index_select(1, cols), tsec_.index_select(1, cols), qsec_)
        for i, blk in enumerate(self.flow):
            if isinstance(blk, GatedXBlock): x = blk(x, c, None, aw, am, mt, mt_mask)
            else: x = blk(x, c, None, aw, am)
            if ftx is not None: x = ftx[i](x, c, *ftext)
        sh, sc = self.f_ada(F.silu(c)).chunk(2, -1)
        return self.f_out(self.f_norm(x) * (1 + sc) + sh)

    def local_audio(self, a, H, K):
        """Gather per-chunk audio windows: tokens [4k - back, 4k + CHUNK + max lookahead), masked by visibility.
        Returns aw [B, K, W, d], mask [B, K, W]."""
        B, N, d = a.shape; back = self.cfg.flow_audio_back
        W = back + 2 * CHUNK                                   # current chunk + up to one chunk of extra look-ahead
        start = torch.arange(K, device=a.device) * CHUNK - back
        idx = start[:, None] + torch.arange(W, device=a.device)[None]            # [K, W]
        valid = (idx >= 0) & (idx < N)
        g = a[:, idx.clamp(0, N - 1)]                                            # [B, K, W, d]
        vis = idx[None] < (torch.arange(K, device=a.device)[None, :, None] * CHUNK + H[:, None, None])
        return g, valid[None] & vis

    # ------------------------------------------------------------------ multi-token memory
    @staticmethod
    def planner_memory(mt, mt_mask):
        """[B, K, M, d] per-chunk memories -> flat [B, K*M, d] + block-diagonal mask [B, K, K*M] (chunk k sees row k)."""
        B, K, M, _ = mt.shape
        eye = torch.eye(K, dtype=torch.bool, device=mt.device).repeat_interleave(M, 1)          # [K, K*M]
        return mt.flatten(1, 2), eye[None] & mt_mask.flatten(1)[:, None, :]

    def mt_memory(self, text, tmask, a, amask_k, g_spk, sref, hist_lat, state_tok, K, tvis=None):
        """Per chunk typed memory: text tokens, visible audio tokens (pooled per chunk to keep it small), speaker,
        reference style, last committed chunk, robot state. Returns mt [B, K, M, d], mask [B, K, M]."""
        B = text.shape[0]; T = self.mt_type.weight
        own = (amask_k & self._own_chunk(K, a.shape[1], a.device)[None]).float()
        prev_own = torch.cat([torch.zeros_like(own[:, :1]), own[:, :-1]], 1)
        pa = lambda m: torch.einsum('bkn,bnd->bkd', m, a) / m.sum(-1, keepdim=True).clamp(min=1)
        audio_toks = torch.stack([pa(prev_own), pa(own)], 2) + T[1]                 # [B, K, 2, d]
        audio_mask = torch.stack([prev_own.sum(-1) > 0, own.sum(-1) > 0], 2)
        tt = (text + T[0])[:, None].expand(B, K, -1, -1); tm = tmask[:, None].expand(B, K, -1)
        if tvis is not None: tm = tm & tvis
        parts = [tt, audio_toks, (g_spk + T[2])[:, None, None].expand(B, K, 1, -1)]
        masks = [tm, audio_mask, torch.ones(B, K, 1, dtype=torch.bool, device=a.device)]
        if sref is not None:
            parts.append((sref + T[3])[:, None, None].expand(B, K, 1, -1)); masks.append(masks[-1])
        if hist_lat is not None:                                                      # [B, K, CHUNK, z]
            prev = torch.cat([torch.zeros_like(hist_lat[:, :1]), hist_lat[:, :-1]], 1)
            hm = torch.ones(B, K, 1, dtype=torch.bool, device=a.device); hm[:, 0] = False
            parts.append((self.mt_hist(prev.flatten(-2)) + T[4])[:, :, None]); masks.append(hm)
        if state_tok is not None:                                                     # [B, K, d]
            parts.append((state_tok + T[5])[:, :, None]); masks.append(torch.ones(B, K, 1, dtype=torch.bool, device=a.device))
        return torch.cat(parts, 2), torch.cat(masks, 2)

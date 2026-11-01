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

from .common import J, FEAT, SPK, TEXT_DIM, N_SOURCE, STRIDE, CHUNK, CHUNK_FRAMES, PROSODY_COLS, FPS, U80
from .nn import Block, Encoder, MLP, sincos, apply_lora

INF = 10 ** 6


@dataclass
class Config:
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
    lora_r: int = 0                # >0: LoRA on planner/flow linear layers (base frozen by the trainer)
    # condition dropout (classifier-free guidance)
    drop_text: float = 0.1
    drop_audio: float = 0.1
    drop_spk: float = 0.2
    drop_ref: float = 0.5
    drop_time: float = 0.5         # token time embedding -> 'unknown' (text is NOT hard-aligned)

    def dict(self):
        return asdict(self)


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

    def forward(self, feat, causal, avail=None):
        """causal: bool [B] (True = token i attends j <= i); avail: bool [B, N] keys that have arrived."""
        B, T, _ = feat.shape; N = T // STRIDE
        x = self.inp(feat.reshape(B, N, STRIDE * FEAT)) + sincos(torch.arange(N, device=feat.device), self.d)[None]
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

    def forward(self, sref, h, x_target=None, temperature=1.0):
        pin = torch.cat([sref, h], -1)
        mp, lp = self.prior(pin).chunk(2, -1); lp = lp.clamp(-8, 4)
        if x_target is None:
            z = mp + temperature * torch.randn_like(mp) * (0.5 * lp).exp(); kl = None
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
        self.source = nn.Embedding(N_SOURCE, d); nn.init.zeros_(self.source.weight)
        # planner
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
        if cfg.mtc and not isinstance(self.planner[0], GatedXBlock):
            self.planner = nn.ModuleList([GatedXBlock(b, d, cfg.heads) for b in self.planner])
            self.flow = nn.ModuleList([GatedXBlock(b, d, cfg.heads) for b in self.flow])
            self.mt_type = nn.Embedding(6, d)                 # 0 text 1 audio 2 speaker 3 ref-style 4 history 5 state
            self.mt_hist = nn.Linear(CHUNK * cfg.z, d)
        if cfg.style and not hasattr(self, 'style_enc'):
            self.style_enc = MotionStyleEncoder(cfg); self.style_null = nn.Parameter(torch.zeros(d))
            self.style_head = StyleHead(cfg)
        if cfg.state and not hasattr(self, 'state_ad'):
            self.state_ad = StateAdapter(cfg)
        if cfg.lora_r > 0 and not getattr(self, '_lora', False):
            for m in (self.planner, self.flow, self.audio): apply_lora(m, r=cfg.lora_r)
            self._lora = True

    # ------------------------------------------------------------------ conditions
    def text_tokens(self, text, ttime, tmask, drop=None):
        x = self.text_in(text)
        known = torch.isfinite(ttime)
        if drop is not None: known = known & ~drop[:, None]
        te = self.text_time(sincos(torch.nan_to_num(ttime, 0.) * FPS / STRIDE, self.cfg.d))
        return x + torch.where(known[..., None], te, self.text_time_unk), tmask

    def global_cond(self, spk, spk_valid, source):
        s = torch.where(spk_valid[:, None] > .5, self.spk_in(spk), self.spk_null)
        return s + self.source(source)

    def chunk_audio_mask(self, K, N, H, device):
        """bool [B, K, N]: audio tokens visible to chunk k = tokens < 4k + H (H = INF full context)."""
        i = torch.arange(N, device=device)
        lim = torch.arange(K, device=device)[None, :, None] * CHUNK + H[:, None, None]
        return i[None, None, :] < lim

    # ------------------------------------------------------------------ planner
    def plan(self, lat, a, amask, text, tmask, g, mt=None, mt_mask=None):
        """lat [B, K, CHUNK, z] latents of chunks 0..K-1 (teacher forcing: chunk k's input is chunk k-1);
        a [B, N, d]; amask [B, K, N]; returns h [B, K, d]."""
        B, K = lat.shape[:2]
        prev = torch.cat([self.p_start.expand(B, 1, -1), self.p_in(lat[:, :-1].flatten(-2))], 1)
        own = (amask & self._own_chunk(K, a.shape[1], a.device)[None]).float()          # visible audio of chunk k
        pooled = torch.einsum('bkn,bnd->bkd', own, a) / own.sum(-1, keepdim=True).clamp(min=1)
        pooled = torch.where(own.sum(-1, keepdim=True) > 0, pooled, self.audio_null)
        x = prev + self.p_audio(pooled) + sincos(torch.arange(K, device=lat.device), self.cfg.d)[None]
        causal = torch.ones(K, K, dtype=torch.bool, device=lat.device).tril()[None]
        tm = tmask[:, None, :].expand(B, K, -1)
        for blk in self.planner:
            if isinstance(blk, GatedXBlock): x = blk(x, g, causal, text, tm, mt, mt_mask)
            else: x = blk(x, g, causal, text, tm)
        return self.p_norm(x)

    @staticmethod
    def _own_chunk(K, N, device):
        i = torch.arange(N, device=device); k = torch.arange(K, device=device)
        return (i[None] // CHUNK) == k[:, None]

    # ------------------------------------------------------------------ local flow
    def flow_velocity(self, xt, t, known, h, g, aw, aw_mask, style_tok=None, state_ada=None, mt=None, mt_mask=None):
        """xt [M, L, z] (known positions hold clean latents); t [M, L] (1 on known); h [M, d]; g [M, d];
        aw [M, W, d] local audio window, aw_mask [M, W]."""
        x = self.f_in(xt) + self.f_role(known.long()) + self.f_pos[None]
        c = self.f_t(sincos(t * 1000., self.cfg.d)) + (self.f_plan(h) + g)[:, None]
        if style_tok is not None: c = c + style_tok[:, None]
        if state_ada is not None: c = c + state_ada[:, None]
        am = aw_mask[:, None, :].expand(-1, x.shape[1], -1)
        for blk in self.flow:
            if isinstance(blk, GatedXBlock): x = blk(x, c, None, aw, am, mt, mt_mask)
            else: x = blk(x, c, None, aw, am)
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

    def mt_memory(self, text, tmask, a, amask_k, g_spk, sref, hist_lat, state_tok, K):
        """Per chunk typed memory: text tokens, visible audio tokens (pooled per chunk to keep it small), speaker,
        reference style, last committed chunk, robot state. Returns mt [B, K, M, d], mask [B, K, M]."""
        B = text.shape[0]; T = self.mt_type.weight
        own = (amask_k & self._own_chunk(K, a.shape[1], a.device)[None]).float()
        prev_own = torch.cat([torch.zeros_like(own[:, :1]), own[:, :-1]], 1)
        pa = lambda m: torch.einsum('bkn,bnd->bkd', m, a) / m.sum(-1, keepdim=True).clamp(min=1)
        audio_toks = torch.stack([pa(prev_own), pa(own)], 2) + T[1]                 # [B, K, 2, d]
        audio_mask = torch.stack([prev_own.sum(-1) > 0, own.sum(-1) > 0], 2)
        tt = (text + T[0])[:, None].expand(B, K, -1, -1); tm = tmask[:, None].expand(B, K, -1)
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

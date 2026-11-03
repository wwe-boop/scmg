"""VoxCPM-style global planner (cfg.planner = 'voxcpm'): replaces the from-scratch chunk transformer.

  sequence = [speaker tok | ref-style tok | text tokens (pretrained LM embeddings) | chunk_0 ... chunk_{K-1}]
  chunk_k input = LocEnc(latents of chunk k-1) + proj(visible audio of chunk k)
  TSLM  = Qwen2.5-0.5B by default, or pinned MiniCPM4/VoxCPM TSLM; causal text + motion in ONE sequence
  FSQ   = finite scalar quantisation of the TSLM output at chunk positions (semi-discrete semantic/rhythm skeleton,
          straight-through gradients)
  RALM  = small causal residual LM over chunk positions: input FSQ(h_k) + LocEnc(prev) + audio, output r_k
  plan h_k = proj(FSQ(h_k)) + r_k  -> conditions the Local Motion Flow (LocDiT) exactly like the default planner
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CHUNK, SPK, P
from .nn import Encoder, sincos

LM = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/models/scmg/Qwen2.5-0.5B'


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class FSQ(nn.Module):
    def __init__(self, dim=8, levels=9):
        super().__init__()
        self.dim, self.half = dim, (levels - 1) / 2

    def forward(self, x):
        z = torch.tanh(x) * self.half
        q = z + (torch.round(z) - z).detach()                       # straight-through rounding
        return q / self.half


class LocEnc(nn.Module):
    """Chunk latents [.., CHUNK, z] -> one vector (CLS token of a 2-layer encoder)."""

    def __init__(self, z, d, heads):
        super().__init__()
        self.inp = nn.Linear(z, d); self.cls = nn.Parameter(torch.zeros(1, 1, d)); self.enc = Encoder(d, heads, 2)
        self.pos = nn.Parameter(torch.zeros(1, CHUNK + 1, d))

    def forward(self, lat):
        sh = lat.shape[:-2]
        x = self.inp(lat.reshape(-1, CHUNK, lat.shape[-1]))
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos
        return self.enc(x)[:, 0].reshape(*sh, -1)


class VoxPlanner(nn.Module):
    def __init__(self, cfg, lm_path=LM, fsq_dim=8, fsq_levels=9, ralm_layers=4):
        super().__init__()
        from transformers import AutoModel
        d = cfg.d
        import os
        lp = P(getattr(cfg, 'vox_lm', '') or lm_path)
        if getattr(cfg, 'vox_lm_trust_remote_code', False):
            from transformers import AutoConfig
            config_path = os.environ.get('SCMG_LM_CONFIG') or lp
            lm_cfg = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
            if os.environ.get('SCMG_LM_CONFIG') or getattr(cfg, 'vox_lm_init', 'pretrained') == 'random':
                self.lm = AutoModel.from_config(lm_cfg, trust_remote_code=True, torch_dtype=torch.float32,
                                                attn_implementation='sdpa')
            else:
                self.lm = AutoModel.from_pretrained(lp, config=lm_cfg, trust_remote_code=True,
                                                    torch_dtype=torch.float32, attn_implementation='sdpa')
            if self.lm.config.model_type == 'minicpm':
                # SCMG recomputes the entire sequence; MiniCPM's default cache requires a Cache object.
                self.lm.config.use_cache = False
        elif os.environ.get('SCMG_LM_CONFIG'):              # weights come from the SCMG checkpoint anyway (no LM read)
            from transformers import AutoConfig
            self.lm = AutoModel.from_config(AutoConfig.from_pretrained(os.environ['SCMG_LM_CONFIG']), torch_dtype=torch.float32,
                                            attn_implementation='sdpa')
        elif getattr(cfg, 'vox_lm_init', 'pretrained') == 'random':
            from transformers import AutoConfig
            self.lm = AutoModel.from_config(AutoConfig.from_pretrained(lp), torch_dtype=torch.float32, attn_implementation='sdpa')
        else:
            self.lm = AutoModel.from_pretrained(lp, torch_dtype=torch.float32, attn_implementation='sdpa')
        self.ab = dict(fsq=getattr(cfg, 'vox_fsq', True), ralm=getattr(cfg, 'vox_ralm', True),
                       locenc=getattr(cfg, 'vox_locenc', True), text=getattr(cfg, 'vox_text', True))
        if not (self.ab['fsq'] is True or self.ab['fsq'] is False or self.ab['fsq'] == 'tanh'):
            raise ValueError("vox_fsq must be True, False, or 'tanh'")
        if not self.ab['locenc']: self.p_lin = nn.Linear(CHUNK * cfg.z, d)
        H = self.lm.config.hidden_size; self.H = H
        self.locenc = LocEnc(cfg.z, d, cfg.heads)
        self.start = nn.Parameter(torch.zeros(d))
        self.chunk_in = nn.Linear(d, H); self.audio_in = nn.Linear(d, d)
        self.spk_in = nn.Linear(d, H); self.ref_in = nn.Linear(d, H)
        self.fsq_in = nn.Linear(H, fsq_dim); self.fsq = FSQ(fsq_dim, fsq_levels); self.fsq_out = nn.Linear(fsq_dim, d)
        # 11-03 text-use fixes (SANA arXiv 2410.10629 / massive activations arXiv 2402.17762): decoder-LM hidden states have a
        # large, outlier-dominated scale -> 38% of pre-FSQ values saturate tanh with the pretrained LM (code review).
        #   vox_prefsq_norm: RMSNorm on the LM output before fsq_in;  vox_in_match: non-text inputs (speaker / reference /
        #   chunk tokens) are RMS-normalised and scaled to the RMS of the LM's own token embeddings (learnable gain).
        self.prefsq_norm = RMSNorm(H) if getattr(cfg, 'vox_prefsq_norm', False) else None   # own class: DLC image has torch 2.3 (no nn.RMSNorm)
        if getattr(cfg, 'vox_in_match', False):
            with torch.no_grad():
                w = self.lm.get_input_embeddings().weight
                r = float(w.float().pow(2).mean().sqrt())
            self.in_gain = nn.Parameter(torch.tensor(r))
        self.ralm_in = nn.Linear(3 * d, d)
        self.ralm = Encoder(d, cfg.heads, ralm_layers)
        self.d = d

    def add_text_path(self, cfg):
        """11-03 opt-in text routes that BYPASS the FSQ skeleton (saturated FSQ cut the text off completely):
        vox_text_xattn  each chunk's residual-LM input cross-attends to the LM text-token states (word level)
        vox_slots = S   S learned soft-intent slots pool the text states; chunks read the slots (+ slot presence head)
        vox_word_aux    word-level intent head on the LM text-token states (losses w_word)
        Outputs are zero-initialised: switching a route on starts from exactly the previous function."""
        H = self.lm.config.hidden_size; d = self.d
        xa, S = getattr(cfg, 'vox_text_xattn', False), int(getattr(cfg, 'vox_slots', 0))
        if (xa or S) and not hasattr(self, 'x_attn'):
            self.t_proj = nn.Linear(H, d); self.t_norm = nn.LayerNorm(d)
            self.x_attn = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
            self.x_out = nn.Linear(d, d); nn.init.zeros_(self.x_out.weight); nn.init.zeros_(self.x_out.bias)
        if S and not hasattr(self, 'slot_q'):
            self.slot_q = nn.Parameter(torch.randn(S, d) * 0.02)
            self.slot_attn = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
            self.slot_cls = nn.Linear(d, 7)                       # coarse gesture classes (scmg/semclass.py) present in the window
        if getattr(cfg, 'vox_word_aux', False) and not hasattr(self, 'word_head'):
            self.word_head = nn.Linear(H, getattr(cfg, 'n_intent', 214))

    def forward(self, lat, pooled_audio, tids, tidm, g, sref=None, lvis=None, semb=None):
        """lat [B, K, CHUNK, z] (teacher forcing: chunk k's input is chunk k-1); pooled_audio [B, K, d] visible audio
        of each chunk; tids/tidm [B, Kt] LM token ids + mask; g [B, d] global speaker/source cond; sref [B, d] or None.
        Returns h [B, K, d] and the quantised skeleton codes [B, K, fsq_dim]."""
        B, K = lat.shape[:2]
        enc = self.locenc(lat[:, :-1]) if self.ab['locenc'] else self.p_lin(lat[:, :-1].flatten(-2))
        prev = torch.cat([self.start.expand(B, 1, -1), enc], 1)                                # [B, K, d]
        if not self.ab['text']: tidm = torch.zeros_like(tidm)
        au = self.audio_in(pooled_audio)
        if semb is not None: au = au + semb                                                  # ADSC stream state
        emb = self.lm.get_input_embeddings()
        text = emb(tids.clamp(min=0))                                                         # [B, Kt, H]
        if self.lm.config.model_type == 'minicpm':
            # Official VoxCPM scales TEXT before mixing modalities; HF inputs_embeds bypasses this scale.
            text = text * (self.lm.config.scale_emb if getattr(self.lm.config, 'use_mup', True) else 1.0)
        pre = [self.spk_in(g)[:, None]]
        if sref is not None:
            if hasattr(self, 'ref_gate'): pre[0] = pre[0] + torch.tanh(self.ref_gate) * self.ref_in(sref)[:, None]
            else: pre.append(self.ref_in(sref)[:, None])                                   # legacy (r4-r6 checkpoints)
        chunks = self.chunk_in(prev + au)                                                     # [B, K, H]
        if hasattr(self, 'in_gain'):
            nrm = lambda t: t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * self.in_gain
            chunks = nrm(chunks); pre = [nrm(t) for t in pre]
        x = torch.cat(pre + [text.to(chunks.dtype), chunks], 1)
        m = torch.cat([torch.ones(B, len(pre), dtype=torch.bool, device=x.device), tidm,
                       torch.ones(B, K, dtype=torch.bool, device=x.device)], 1)
        pos = (m.long().cumsum(1) - 1).clamp(min=0)
        if lvis is None:
            am = m.long()
        else:                                    # ADSC: chunk k attends only to the LM tokens that had arrived by its start
            S = x.shape[1]; P = len(pre); Kt = tids.shape[1]
            allow = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()[None] & m[:, None, :]
            allow = allow.clone()
            allow[:, P + Kt:, P:P + Kt] &= lvis
            allow |= torch.eye(S, dtype=torch.bool, device=x.device)[None]             # no empty rows (padding queries)
            am = torch.zeros(allow.shape, dtype=x.dtype, device=x.device).masked_fill(~allow, torch.finfo(x.dtype).min)[:, None]
        lh = self.lm(inputs_embeds=x, attention_mask=am, position_ids=pos).last_hidden_state
        hs = lh[:, -K:]
        self._aux = {}
        if hasattr(self, 'word_head') or hasattr(self, 'x_attn'):
            Pn = len(pre); ts = lh[:, Pn:Pn + tids.shape[1]]                                  # LM states at the text tokens
            if hasattr(self, 'word_head'): self._aux['word_logits'] = self.word_head(ts.float())
        if self.prefsq_norm is not None: hs = self.prefsq_norm(hs)
        if self.ab['fsq'] == 'tanh':
            q = torch.tanh(self.fsq_in(hs))
        else:
            zpre = self.fsq_in(hs); self._aux['fsq_pre'] = zpre
            q = self.fsq(zpre) if self.ab['fsq'] else zpre
        base = self.fsq_out(q)
        if not self.ab['ralm']: return base, q
        r_in = self.ralm_in(torch.cat([base, prev, au], -1)) + sincos(torch.arange(K, device=x.device), self.d)[None]
        if hasattr(self, 'x_attn'):
            Kt = tids.shape[1]; has = tidm.any(1)
            kv = self.t_norm(self.t_proj(ts.to(r_in.dtype))) + sincos(torch.arange(Kt, device=x.device), self.d)[None].to(r_in.dtype)
            kpm = ~tidm.bool(); kpm[~has, 0] = False                                            # no all-masked rows (zeroed below)
            if hasattr(self, 'slot_q'):
                slots, _ = self.slot_attn(self.slot_q[None].expand(B, -1, -1).to(kv.dtype), kv, kv, key_padding_mask=kpm)
                slots = slots * has[:, None, None].to(slots.dtype)
                self._aux['slot_logits'] = self.slot_cls(slots.float())
                xo, _ = self.x_attn(r_in, slots, slots)
            else:
                xo, _ = self.x_attn(r_in, kv, kv, key_padding_mask=kpm)
            r_in = r_in + self.x_out(xo * has[:, None, None].to(xo.dtype))
        causal = torch.ones(K, K, dtype=torch.bool, device=x.device).tril()[None]
        r = self.ralm(r_in, causal)
        return base + r, q

"""ECHO-G trained streaming variant (11-05): shared model / condition truncation / window sampling / loss / sampler.

StreamSGDiT = the official SpeechGroundedDiT (echo_g.model, read-only, imported) + a prefix-motion condition:
    extra input [known_clean (normalised) x known_mask, known_mask]  ->  known_projection (Linear, ZERO-initialised)
    added to the official motion input embedding.  With zero weights the forward is bit-identical to the official one
    ((a + 0) + b == a + b in IEEE), so the official best.pt loads as a StreamSGDiT that IS the official model.

Streaming protocol (training and inference share these functions):
    blocks of block_ms (320 ms; G1 30 FPS: block k = frames [ceil(k*9.6), ceil((k+1)*9.6)); EX002 50 FPS: 16 frames),
    block k is decided at now_k = k * block_s.  Window = [ws, fe) with ws = max(0, c - ctx), c = block start,
    fe = block end + tail.  Frames [ws, c) are KNOWN (already committed): given clean through the new condition and
    replaced by their clean values in x_t (known_xt='clean', see choose_xt), never in the loss.  Frames [c, fe) are generated.
    audio: frame f of the window keeps its feature iff f / fps < now + ahead, else zero (the official null audio).
    text : token j visible iff x2 arrival_j <= now (SCMG rule: min(start_j - lag, start_0 - lag + j / rate)) AND its word has
           been voiced (end_j <= now + ahead: ECHO-G's word-time attention needs the token centre); time_distance =
           absolute frame time - token centre (official convention) for the window frames; no visible token -> the official
           null text (one zero token, distance 0).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from echo_g.flow import apply_condition_dropout
from echo_g.model import SpeechGroundedDiT, modulate, timestep_embedding

STREAM_SCHEMA = 'echo-g-stream-v1'


@dataclass
class StreamConfig:
    block_ms: float = 320.0          # decision period (SCMG chunk)
    ctx_s: float = 8.0               # max committed context in the window (inference uses it whenever the past is long enough)
    p_ctx_full: float = 0.5          # training: P(ctx = ctx_s); else ctx ~ U[1 block, ctx_s] (short contexts mid-clip)
    tail_max_s: float = 1.28         # training: generated frames after the block ~ U{0 .. tail_max}
    tail_weight: float = 1.0         # loss weight of the tail frames (block frames weigh 1)
    infer_tail_ms: float = 640.0     # inference tail (inside the trained range)
    p_cold: float = 0.15             # training: P(c = 0) (cold start, no known frames)
    ahead_s: float = 0.16            # audio / voiced-word look-ahead (SCMG engine H = 2 x 80 ms)
    p_ahead_more: float = 0.2        # training: P(ahead ~ U[ahead_s, ahead_more_max_s]) (no overfit to one value)
    ahead_more_max_s: float = 0.64
    x2_rate: float = 15.0            # SCMG x2 text arrival: tokens / s
    x2_lag: float = 0.15             #                      lag before the token onset, s
    known_xt: str = 'clean'          # x_t on known frames: 'clean' (x1) | 'noise' (official flow path t x1 + (1 - t) eps)
    cut_boost: float = 0.0           # training: loss weight x (1 + cut_boost * exp(-j / cut_tau_frames)) at offset j from the cut
    cut_tau_frames: float = 3.0      #           (continuity of the first generated frames; 0 = off)
    known_aug_std: float = 0.0       # training: Gaussian noise (normalised units) on the known prefix (exposure-bias knob; 0 = off)

    def validate(self) -> None:
        if self.known_xt not in ('clean', 'noise'): raise ValueError('known_xt must be clean or noise')
        if not (0 <= self.p_cold <= 1 and 0 <= self.p_ctx_full <= 1 and 0 <= self.p_ahead_more <= 1): raise ValueError('probabilities')
        if self.block_ms <= 0 or self.ctx_s <= 0 or self.tail_max_s < 0 or self.ahead_s < 0: raise ValueError('durations')

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> 'StreamConfig':
        c = cls(**(d or {})); c.validate(); return c

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------------------------------------- model
class StreamSGDiT(SpeechGroundedDiT):
    def __init__(self, config):
        super().__init__(config)
        self.known_projection = nn.Linear(config.motion_dim + 1, config.hidden_dim)
        nn.init.zeros_(self.known_projection.weight); nn.init.zeros_(self.known_projection.bias)

    def forward(self, noised_motion, flow_time, audio, text_tokens, motion_mask, text_padding_mask, time_distance,
                known=None, known_mask=None):
        """official SpeechGroundedDiT.forward (same operations, same order) + known_projection on the motion embedding.
        known [B, T, D] normalised clean motion (any value where not known), known_mask [B, T] bool."""
        frames = noised_motion.shape[1]
        if not 1 <= text_tokens.shape[1] <= self.config.max_text_tokens:
            raise ValueError('text token count exceeds model capacity')
        motion_in = self.motion_projection(noised_motion)
        if known is not None:
            m = known_mask.to(noised_motion.dtype).unsqueeze(-1)
            motion_in = motion_in + self.known_projection(torch.cat([known * m, m], dim=-1))
        hidden = (motion_in + self.audio_projection(self.audio_norm(audio))
                  + self.positional_encoding(frames, noised_motion.device, noised_motion.dtype))
        condition = self.timestep_mlp(timestep_embedding(flow_time, self.hidden))
        text = self.text_projection(self.text_norm(text_tokens))
        motion_padding_mask = ~motion_mask
        for block in self.blocks:
            hidden = block(hidden, condition, text, motion_padding_mask, text_padding_mask, time_distance)
        shift, scale = self.final_adaptive_norm(condition).chunk(2, dim=-1)
        hidden = modulate(self.final_norm(hidden), shift, scale)
        return self.output(hidden)


def load_official_into(model: StreamSGDiT, state: dict[str, torch.Tensor]) -> None:
    """official SGDiT weights -> StreamSGDiT; only the zero-initialised known_projection is left untouched."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or set(missing) != {'known_projection.weight', 'known_projection.bias'}:
        raise ValueError(f'official checkpoint mismatch: missing={missing} unexpected={unexpected}')


def load_stream_model(path, device, config_path=None):
    """official best.pt (-> zero-init StreamSGDiT == official model) or a stream checkpoint.
    returns model, ExperimentConfig, mean, std, StreamConfig, raw checkpoint dict (without weights)."""
    from echo_g.config import ExperimentConfig
    ck = torch.load(path, map_location='cpu', weights_only=False)
    exp = ck['experiment']
    if 'config' in exp:
        config = ExperimentConfig.from_dict(exp['config'])
    else:
        if config_path is None: raise ValueError('--config required for a checkpoint without an embedded config')
        config = ExperimentConfig.from_yaml(config_path)
    model = StreamSGDiT(config.model)
    if 'stream' in ck:
        model.load_state_dict(ck['ema'], strict=True); sc = StreamConfig.from_dict(ck['stream'])
    else:
        load_official_into(model, ck['ema']); sc = StreamConfig()
    model.to(device).eval()
    mean = torch.as_tensor(ck['motion_mean']).float().reshape(-1); std = torch.as_tensor(ck['motion_std']).float().reshape(-1)
    meta = {k: v for k, v in ck.items() if k not in ('ema', 'model', 'optimizer', 'scheduler')}
    return model, config, mean, std, sc, meta


# --------------------------------------------------------------------------------------------- streaming geometry
def block_start(k: int, block_s: float, fps: float) -> int:
    return int(math.ceil(k * block_s * fps - 1e-6))


def n_blocks(F: int, block_s: float, fps: float) -> int:
    k = 0
    while block_start(k, block_s, fps) < F: k += 1
    return k


def frames_before(t: float, fps: float) -> int:
    """number of frames f >= 0 with f / fps < t"""
    return 0 if t <= 0 else (10 ** 9 if math.isinf(t) else int(math.ceil(t * fps - 1e-6)))


def x2_arrival(start: np.ndarray, rate: float, lag: float) -> np.ndarray:
    """SCMG scmg.model.x2_arrival for one utterance (tokens in order)."""
    j = np.arange(len(start), dtype=np.float64)
    return np.minimum(start - lag, start.min() - lag + j / rate) if len(start) else start


def visible_tokens(token_times: np.ndarray, now: float, ahead: float, mode: str, rate: float, lag: float) -> np.ndarray:
    if mode == 'final': return np.ones(len(token_times), bool)
    arr = x2_arrival(token_times[:, 0].astype(np.float64), rate, lag)
    vis = arr <= now + 1e-9
    if mode == 'voiced': vis &= token_times[:, 1].astype(np.float64) <= now + ahead + 1e-9
    elif mode != 'x2': raise ValueError(mode)
    return vis


def window_conditions(audio_abs: torch.Tensor, text: torch.Tensor, token_times: np.ndarray, centres: torch.Tensor,
                      ws: int, fe: int, now: float, ahead: float, fps: float, text_mode: str = 'voiced',
                      rate: float = 15.0, lag: float = 0.15, max_tokens: int = 256) -> dict[str, torch.Tensor]:
    """conditions of window [ws, fe) decided at `now` (s). audio_abs: features indexed by ABSOLUTE frame (released full-clip
    features, or the features of the arrived waveform prefix; rows past its length count as not arrived)."""
    W = fe - ws; dev = audio_abs.device
    au = torch.zeros(W, audio_abs.shape[1], device=dev, dtype=audio_abs.dtype)
    hi = min(fe, frames_before(now + ahead, fps), audio_abs.shape[0])
    if hi > ws: au[:hi - ws] = audio_abs[ws:hi]
    vis = visible_tokens(token_times, now, ahead, text_mode, rate, lag)
    idx = np.nonzero(vis)[0][-max_tokens:]
    if len(idx):
        it = torch.as_tensor(idx, device=dev)
        tx = text[it]; tpm = torch.zeros(len(idx), dtype=torch.bool, device=dev)
        ft = torch.arange(ws, fe, device=dev, dtype=torch.float32) / fps
        td = ft[:, None] - centres.to(dev)[it][None]
    else:                                                     # official null text
        tx = torch.zeros(1, text.shape[1], device=dev, dtype=text.dtype); tpm = torch.zeros(1, dtype=torch.bool, device=dev)
        td = torch.zeros(W, 1, device=dev)
    return dict(audio=au, text=tx, text_padding_mask=tpm, time_distance=td, n_visible=int(len(idx)), audio_frames=max(0, hi - ws))


# ------------------------------------------------------------------------------------------------ training windows
def sample_window(sample: dict[str, Any], rng: np.random.Generator, sc: StreamConfig, fps: float,
                  tail_weight: float | None = None) -> dict[str, Any]:
    """one streaming training window from a full official sample (RobotSpeechDataset item + 'token_times')."""
    F = int(sample['frames']); bs = sc.block_ms / 1000.0; K = n_blocks(F, bs, fps)
    k = 0 if (K <= 1 or rng.random() < sc.p_cold) else int(rng.integers(1, K))
    c = block_start(k, bs, fps); f1 = min(F, block_start(k + 1, bs, fps))
    ctx_max = int(round(sc.ctx_s * fps)); blk_f = max(1, int(round(bs * fps)))
    ctx = ctx_max if rng.random() < sc.p_ctx_full else int(rng.integers(min(blk_f, ctx_max), ctx_max + 1))
    ws = max(0, c - ctx)
    fe = min(F, f1 + int(rng.integers(0, int(round(sc.tail_max_s * fps)) + 1)))
    now = k * bs
    ahead = sc.ahead_s if rng.random() >= sc.p_ahead_more else float(rng.uniform(sc.ahead_s, sc.ahead_more_max_s))
    cond = window_conditions(sample['audio'], sample['text'], sample['token_times'], sample['token_centers'], ws, fe, now,
                             ahead, fps, 'voiced', sc.x2_rate, sc.x2_lag)
    f = torch.arange(ws, fe)
    known = f < c
    w = torch.where(known, 0.0, torch.where(f < f1, 1.0, sc.tail_weight if tail_weight is None else tail_weight))
    if sc.cut_boost > 0: w = w * torch.where(known, 1.0, 1.0 + sc.cut_boost * torch.exp(-(f - c).clamp_min(0).float() / sc.cut_tau_frames))
    return dict(motion=sample['motion'][ws:fe], known_mask=known, loss_weight=w.float(), audio=cond['audio'],
                text=cond['text'], text_padding_mask=cond['text_padding_mask'], time_distance=cond['time_distance'],
                stem=sample['stem'], meta=dict(k=k, c=c, ws=ws, f1=f1, fe=fe, now=now, ahead=ahead, n_visible=cond['n_visible'],
                                                audio_frames=cond['audio_frames']))


def collate_windows(ws: list[dict[str, Any]]) -> dict[str, Any]:
    B = len(ws); T = max(len(w['motion']) for w in ws); N = max(w['text'].shape[0] for w in ws)
    D = ws[0]['motion'].shape[1]; Da = ws[0]['audio'].shape[1]; Dt = ws[0]['text'].shape[1]
    out = dict(motion=torch.zeros(B, T, D), audio=torch.zeros(B, T, Da), text=torch.zeros(B, N, Dt),
               motion_mask=torch.zeros(B, T, dtype=torch.bool), known_mask=torch.zeros(B, T, dtype=torch.bool),
               loss_weight=torch.zeros(B, T), text_padding_mask=torch.ones(B, N, dtype=torch.bool), time_distance=torch.zeros(B, T, N))
    for i, w in enumerate(ws):
        t, n = len(w['motion']), w['text'].shape[0]
        out['motion'][i, :t] = w['motion']; out['audio'][i, :t] = w['audio']; out['text'][i, :n] = w['text']
        out['motion_mask'][i, :t] = True; out['known_mask'][i, :t] = w['known_mask']; out['loss_weight'][i, :t] = w['loss_weight']
        out['text_padding_mask'][i, :n] = w['text_padding_mask']; out['time_distance'][i, :t, :n] = w['time_distance']
    out['stems'] = [w['stem'] for w in ws]; out['meta'] = [w['meta'] for w in ws]
    return out


def stream_flow_loss(model, batch, condition_dropout: float, temporal_weight: float, known_xt: str = 'clean', known_aug_std: float = 0.0):
    """official flow_matching_loss restricted to the generated frames (weighted), with the known prefix as condition."""
    target = batch['motion']; km = batch['known_mask']; w = batch['loss_weight']
    kc = target if known_aug_std <= 0 else torch.where(km[..., None], target + known_aug_std * torch.randn_like(target), target)
    audio, text, tpm, td = apply_condition_dropout(batch['audio'], batch['text'], batch['text_padding_mask'],
                                                   batch['time_distance'], condition_dropout)
    noise = torch.randn_like(target)
    t = torch.rand(target.shape[0], device=target.device)
    xt = t[:, None, None] * target + (1.0 - t[:, None, None]) * noise
    if known_xt == 'clean': xt = torch.where(km[..., None], kc, xt)
    v = model(xt, t, audio, text, batch['motion_mask'], tpm, td, known=kc, known_mask=km)
    tv = target - noise
    D = target.shape[-1]
    flow = ((v - tv).square() * w[..., None]).sum() / (w.sum().clamp_min(1e-6) * D)
    adj = torch.minimum(w[:, 1:], w[:, :-1])
    temporal = (((v[:, 1:] - v[:, :-1]) - (tv[:, 1:] - tv[:, :-1])).square() * adj[..., None]).sum() / (adj.sum().clamp_min(1e-6) * D)
    return flow + temporal_weight * temporal, flow, temporal


# ------------------------------------------------------------------------------------------------------ sampling
@torch.no_grad()
def sample_window_euler(model, cond: dict[str, torch.Tensor], noise: torch.Tensor, known: torch.Tensor, known_mask: torch.Tensor,
                        steps: int, known_xt: str = 'clean') -> torch.Tensor:
    """official Euler sampler (guidance 1) on one batch of windows with known frames. noise/known [B, T, D], known_mask [B, T].
    With no known frame and a zero known_projection this is echo_g.flow.sample_euler exactly."""
    m = known_mask[..., None]
    x = torch.where(m, known, noise) if known_xt == 'clean' else noise.clone()
    dt = 1.0 / steps
    for i in range(steps):
        t = i * dt
        if known_xt == 'noise': x = torch.where(m, t * known + (1 - t) * noise, x)
        ft = torch.full((x.shape[0],), t, device=x.device, dtype=torch.float32)
        x = x + dt * model(x, ft, cond['audio'], cond['text'], cond['motion_mask'], cond['text_padding_mask'],
                           cond['time_distance'], known=known, known_mask=known_mask)
        if known_xt == 'clean': x = torch.where(m, known, x)
    return torch.where(m, known, x)

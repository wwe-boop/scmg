"""Streaming, receding-horizon inference.

Every tick (one motion chunk = 320 ms):
  1. audio that has arrived (horizon H latents past the chunk start; INF = offline full context) is encoded;
  2. the planner runs over a sliding window of committed chunks -> plan h_k;
  3. style z_k is sampled from the conditional prior; the latest robot state (if any) becomes the state token;
  4. the local flow out-paints chunk k from the committed context (known) -> optionally `n_future` further chunks
     are predicted provisionally (the long prediction window handed to the controller as preview);
  5. ONLY chunk k is committed and decoded; committed motion is never modified; provisional chunks are discarded
     and regenerated next tick from the new audio and the new robot state.
The sliding window mirrors training: WIN chunks (= training window), current chunk at index WIN - 1 - LOOK so that
LOOK chunks of audio look-ahead fit inside the window.
"""
import os, time
import numpy as np
import torch

from .common import CHUNK, CHUNK_FRAMES, STRIDE, U80
from .model import INF, arrival_mask, stream_state, x2_arrival
from .sim import state_vector
from .data import require_causal_text

WIN, LOOK = 16, 2


class Streamer:
    def __init__(self, model, vae, feat, cond, stats, H=INF, steps=10, temperature=1.0, n_future=0, seed=0, noise_scale=1.0,
                 standby=None, device='cuda', hist_measured=False, ctx_t=1.0, cfg_w=1.0, cfg_text=True, cfg_t=None,
                 text_mode='final', x2_rate=15.0, x2_lag=0.15, seg_lead=0.3, deploy_timing=False,
                 elastic=None, window_lm_text=False, g_override=None, trim_start=True, cfg_null_fix=True, still_refresh=None,
                 audio_off=False, causal_inputs=None, asr_lag=0.5, trace=None, cfg_ctl=None, inpaint=None, ctl_lookahead=0.0):
        """feat [T, 89] normalised (whole utterance; access is restricted to what has arrived);
        cond: dict(text [K,768], ttime [K] absolute s, tmask [K], spk [192], spk_valid, source, ref [R,J], ref_valid)
        (+ tts_h [T/4, 2048], tts_hm [T/4] for models with cfg.tts_h: TTS hidden rows aligned to feat frame 0, row r = frames
        [4r, 4r+4); a row is visible exactly when the audio of its frames is -- the same arrival rule as feat).
        11-05 causal streaming inputs (design v5 section 3; defaults = the unchanged path, bit-identical):
          causal_inputs: scmg.causal_feat.CausalInputs -- every tick recomputes feat89 from the arrived PCM prefix
            (t_k = (4k + H) latents, words complete by t_k, fixed prosody statistics, committed beats) and the speaker
            vector from the PCM prefix (empty embedding < 1 s); `feat` then only sets the clip length.
          text_mode 'voiced' / 'asr': a token is visible once every word it overlaps is (scmg.causal_feat.visible_words =
            ECHO-G stream_core.visible_tokens 'voiced'; 'asr' = word end + asr_lag) at the current tick: every window chunk
            reads the visibility at min(its own start, the current chunk start), so no condition tensor of tick k
            depends on a word finished after t_k; prefix-causal e5 rows (text_c); cond needs wtime [n_words, 2]
            (start, end s) and tword [K, 2] (first / last word index of each token row, -1 = none). ADSC text state:
            0 nothing / 1 partial / 2 only once the audio is final and everything is visible (the total token count is
            future information while the audio streams).
          trace: a list -> one dict of the tick's condition tensors (cpu) appended per tick (tests)."""
        require_causal_text(cond.get('text_c'), text_mode in ('x2', 'voiced', 'asr'), f'Streamer text_mode={text_mode}')
        self.m, self.vae, self.dev = model.eval(), vae.eval(), device
        self.feat = torch.as_tensor(feat, device=device).float()
        self.cond = {k: torch.as_tensor(v, device=device) for k, v in cond.items()}
        if getattr(model.cfg, 'tts_h', False) and 'tts_h' not in self.cond:
            raise ValueError('model was trained with tts_h (TTS hidden states): cond needs tts_h / tts_hm')
        self.qm = torch.tensor(stats['q_mean'], device=device); self.qs = torch.tensor(stats['q_std'], device=device)
        self.H, self.steps, self.temp, self.n_future = H, steps, temperature, n_future
        self.noise_scale = float(noise_scale)   # 11-04: scale of the initial flow noise (1 = standard sampling; <1 = lower diversity)
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.lat = []                     # committed latents (normalised), one [z] per latent step
        self.frames = []                  # emitted joint frames (rad)
        self.k = 0; self.n_emitted = 0
        self.noise_override = None        # callable(streamer, chunk index, default noise [1, CHUNK, z]) -> noise or None
        self.st_hist = {}                 # chunk index -> robot-state token used when that chunk was generated
        self.timing = []
        self.cfg_w = cfg_w                   # classifier-free guidance (1 = off)
        self.cfg_t = cfg_t                   # separate TEXT guidance (InstructPix2Pix-style): v_u + cfg_w (v_a - v_u) + cfg_t (v_at - v_a); None = joint CFG
        # online text arrival (ADSC): 'final' = whole text known; 'x2' = X2Streaming-TTS token streaming (committed text
        # at x2_rate tokens/s, never later than onset - x2_lag); 'segment' = TTS segment text arrives seg_lead s before its audio.
        # deploy_timing: token times exist only once voiced (TTS text_progress anchors) -- automatic for ADSC models.
        self.text_mode, self.deploy_timing = text_mode, deploy_timing
        self.causal, self.trace, self.asr_lag = causal_inputs, trace, asr_lag
        self.x2_rate, self.x2_lag = x2_rate, x2_lag
        if text_mode in ('voiced', 'asr'):
            if H >= INF or elastic is not None: raise ValueError(f'text_mode {text_mode} needs a fixed streaming horizon H')
            if 'wtime' not in cond or 'tword' not in cond: raise ValueError(f'text_mode {text_mode} needs cond wtime / tword')
            self._wt = np.asarray(cond['wtime'], np.float64).reshape(-1, 2); self._tw = np.asarray(cond['tword'], np.int64)
            if len(self._tw) != len(cond['tmask']): raise ValueError('tword rows != tmask rows')
        if causal_inputs is not None:
            if H >= INF: raise ValueError('causal_inputs needs a streaming horizon H < INF')
            if still_refresh is not None: raise ValueError('still_refresh reads the whole-clip speech flag: not with causal_inputs')
            if causal_inputs.Tpad != self.feat.shape[0]: raise ValueError(f'causal_inputs Tpad {causal_inputs.Tpad} != feat rows {self.feat.shape[0]}')
        # elastic look-ahead (latents, max): per tick H = largest trained horizon (0/2/4/8 x 80 ms) covered by the audio that has
        # actually arrived -- the first chunk starts as soon as any audio is there, later chunks use the TTS buffer ahead
        self.elastic = elastic
        cc = self.cond; one = lambda x: torch.tensor([float(x)], device=device)
        self.arr_text = self.arr_lm = None
        if text_mode == 'x2':
            self.arr_text = x2_arrival(cc['ttime'][None].float(), cc['tmask'][None].bool(), one(x2_rate), one(x2_lag))[0]
            if 'titime' in cc: self.arr_lm = x2_arrival(cc['titime'][None].float(), cc['tidm'][None].bool(), one(x2_rate), one(x2_lag))[0]
        elif text_mode == 'segment':
            self.arr_text = cc['tseg'].float() - seg_lead
            if 'tiseg' in cc: self.arr_lm = cc['tiseg'].float() - seg_lead
        self.window_lm_text = window_lm_text
        self.trim_start = trim_start         # 11-03 review N3: no pre-clip pseudo chunks -- the window starts at the clip start (as a
                                             # training window with s = 0) until it is full; default False = the old protocol
        self.cfg_null_fix = cfg_null_fix     # 11-03 review N1: CFG null branch attends to the null audio tokens with the normal
                                             # visibility mask (as dropped audio in training); default False = old (mask all)
        self.g_override = g_override         # fixed g_k [n_sem] for every chunk (zeros = null g; None = the head's prediction)
        # 11-05 input-modality ablation (off = bit-identical): audio_off = every branch sees the NULL audio exactly as a
        # drop_audio training sample (tokens -> audio_null, planner / MTC audio mask all False, Local Flow window = null
        # tokens with the normal window mask); the audio still sets the clip length and the clock. Text-only = audio_off
        # with the text kept (drop_text / drop_audio are sampled independently in training, so it is a trained case).
        self.audio_off = audio_off
        self.ctl_lookahead = float(ctl_lookahead)   # 11-07: annotation control tokens visible (and timed) >= this many s ahead
        # 11-07 fixed-action library insertion (off = None, bit-identical): RePaint-style latent inpainting inside the Local Flow
        # sampler. inpaint = dict(lat [N, z] NORMALISED target latents on the global latent grid, w [N] soft weights in [0, 1],
        # seed). For every generated position with w > 0: x starts at w eps + (1 - w) x and after every Euler step becomes
        # w (t' lat + (1 - t') eps) + (1 - w) x (t' = the time just reached; eps = a fixed draw of its own generator, so the main
        # noise stream is unchanged); the sampler generates the rest (and the soft edges) around it.
        self.inpaint = None
        if inpaint is not None:
            self.inpaint = dict(lat=torch.as_tensor(inpaint['lat'], device=device).float(),
                                w=torch.as_tensor(inpaint['w'], device=device).float(),
                                g=torch.Generator(device=device).manual_seed(int(inpaint.get('seed', 0)) + 7919))
        # 11-05 explicit control tokens (cond ctl_t [Kc] absolute s, ctl_type, ctl_v, ctl_vm; models with cfg.ctl). Without them
        # nothing below changes. cfg_ctl None: the tokens are part of the conditional branch of the existing CFG; a number: a
        # separate guidance term cfg_ctl * (v(with tokens) - v(without tokens)) on top of it (one more Local Flow pass).
        self.has_ctl = bool(getattr(model.cfg, 'ctl', False)) and 'ctl_type' in self.cond
        from .ctl import YIELD as _YID
        self._yield_id = _YID
        self.has_ret = bool(getattr(model.cfg, 'ret', False)) and 'ret_t' in self.cond      # 11-06 retrieved exemplars (same CFG group)
        self.cfg_ctl = cfg_ctl if (self.has_ctl or self.has_ret) else None
        if self.cfg_ctl is not None and model.cfg.mtc: raise ValueError('cfg_ctl is not implemented for mtc models')
        self.cfg_text = cfg_text             # False: guide on audio only (models never trained with text dropped, e.g. r4 VoxCPM)
        self.ctx_t = ctx_t                   # <1: noise the committed context like in training (exposure-bias fix)
        # 11-04 static-history refresh (off = None, bit-identical): (thr, t_low). When the current chunk contains speech and the
        # committed context latents barely change (mean step norm < thr, normalised latent units), noise that context with
        # t_low instead of ctx_t so the model is not anchored to a held pose (hold attractor; recovery after pauses).
        self.still_refresh = still_refresh; self.n_refresh = 0
        if still_refresh is not None:
            fm, fsd = float(stats['f_mean'][88]), float(stats['f_std'][88])
            self._speech = (self.feat[:, 88] * fsd + fm) > 0.5          # per-frame speech flag (feat89 column 88)
        self.hist_measured = hist_measured   # ablation: context latents = encoded EXECUTED motion, no state adapter
        z = model.cfg.z
        if standby is not None:           # rest pose as initial context (encoded like any history)
            q = ((torch.as_tensor(standby, device=device).float() - self.qm) / self.qs)[None, None].expand(1, CHUNK_FRAMES, -1)
            mu, _ = vae.encode(q.contiguous())
            self.hist0 = ((mu - model.lat_mean) / model.lat_std)[0]
        else:
            self.hist0 = None

    def _null_ids(self):
        """Unconditional branch of CFG: the VoxCPM LM must not see the text either."""
        t = self._ids()[1]
        if t is None or not self.cfg_text: return t
        return torch.zeros_like(t)

    def _ids(self):
        c = self.cond
        if 'tids' not in c: return None, None
        mask = c['tidm'].bool()
        if self.window_lm_text:
            if 'tiend' not in c or 'tiseg' not in c:
                raise ValueError('window_lm_text requires untruncated clip ids with tiseg/tiend')
            start = getattr(self, '_c0', self.k - (WIN - 1 - LOOK)) * CHUNK_FRAMES / 50.
            end = start + getattr(self, '_W', WIN) * CHUNK_FRAMES / 50.
            mask = mask & (c['tiseg'] < end) & (c['tiend'] > start)
            mask = mask & (mask.long().cumsum(0) <= 160)
        return c['tids'][None].long(), mask[None]

    @torch.no_grad()
    def _window(self):
        m, k = self.m, self.k
        c0 = k - (WIN - 1 - LOOK)                                   # first chunk of the window (may be < 0)
        W = WIN
        if self.trim_start and c0 < 0: W, c0 = WIN + c0, 0              # window = clip start .. chunk k + look-ahead
        self._W, self._c0 = W, c0
        f0 = c0 * CHUNK_FRAMES; T = W * CHUNK_FRAMES
        idx = torch.arange(f0, f0 + T, device=self.dev)
        Tf = self.feat.shape[0]
        arrived = Tf if self.H >= INF else min(Tf, (k * CHUNK + self.H) * STRIDE)
        ok = (idx >= 0) & (idx < arrived)
        feat = torch.where(ok[:, None], self.feat[idx.clamp(0, Tf - 1)], torch.zeros_like(self.feat[:1]))
        if 'tts_h' in self.cond and getattr(m.cfg, 'tts_h', False):
            # TTS hidden rows of the window (row r = frames [f0 + 4r, f0 + 4r + 4)); same arrival rule as the audio frames:
            # visible iff its first frame has arrived (arrived is a multiple of STRIDE: for STRIDE 4 = all 4 frames arrived).
            # h[k] is computed before codec frame k is emitted, so this never uses a state the TTS has not produced yet.
            th = self.cond['tts_h']; Rf = th.shape[0]
            rows = torch.div(idx[::4], 4, rounding_mode='floor')
            vis = (rows >= 0) & (rows < Rf) & (rows * 4 < arrived)
            hm = self.cond['tts_hm'][rows.clamp(0, Rf - 1)].float() if 'tts_hm' in self.cond else torch.ones_like(rows, dtype=torch.float)
            hw = torch.where(vis[:, None], th[rows.clamp(0, Rf - 1)], torch.zeros_like(th[:1]))   # not-yet-arrived rows: zeros
            self._tts = (hw[None], (hm * vis)[None])
        else:
            self._tts = (None, None)
        return c0, f0, feat[None]

    def _word_tvis(self, c0, W):
        """[1, W, K] token visibility (text_mode voiced / asr) of the window chunks, evaluated at min(chunk start, now)."""
        from .causal_feat import visible_words, token_visible
        now_k = self.k * CHUNK_FRAMES / 50.; ahead = self.H * STRIDE / 50.
        out = np.zeros((W, len(self._tw)), bool)
        for w in range(W):
            now = min((c0 + w) * CHUNK_FRAMES / 50., now_k)
            out[w] = token_visible(visible_words(self._wt, now, ahead, self.text_mode, self.x2_rate, self.x2_lag, self.asr_lag), self._tw)
        return torch.as_tensor(out, device=self.dev)[None]

    @torch.no_grad()
    def _emit(self, final=False):
        m = self.m
        L1 = len(self.lat); L0 = max(0, L1 - 4 * CHUNK)
        zz = torch.stack(self.lat[L0:L1])[None] * m.lat_std + m.lat_mean
        with torch.autocast('cuda', enabled=False):                  # decode in fp32 even under the bf16 option
            y = self.vae.decode(zz.float())[0] * self.qs + self.qm  # global frames [L0*STRIDE, L1*STRIDE)
        end = L1 * STRIDE if final else (L1 - 1) * STRIDE           # last latent waits for its look-ahead
        fr = y[self.n_emitted - L0 * STRIDE:end - L0 * STRIDE]
        for f in fr: self.frames.append(f)
        self.n_emitted = end
        return fr

    def flush(self):
        """End of utterance: emit the frames of the last latent (decoded with replicated look-ahead)."""
        return self._emit(final=True)

    @torch.no_grad()
    def tick(self, robot_q=None, robot_dq=None, executed=None, arrived_s=None):
        """SCMG_BF16=1: run the whole tick under bf16 autocast (latency; quality checked separately).
        arrived_s: seconds of speech audio received so far (elastic look-ahead)."""
        if self.elastic is not None:
            ahead = int(np.floor((arrived_s if arrived_s is not None else 1e9) / 0.08 + 1e-6)) - self.k * CHUNK // U80
            self.H = max(h for h in (0, 2, 4, 8) if h <= max(0, min(ahead, self.elastic))) * U80
        if os.environ.get('SCMG_BF16') == '1':
            with torch.autocast('cuda', torch.bfloat16):
                return self._tick(robot_q, robot_dq, executed)
        return self._tick(robot_q, robot_dq, executed)

    def _tick(self, robot_q=None, robot_dq=None, executed=None):
        """Advance one chunk. robot_q / robot_dq: measured state (rad, rad/s) at the end of the executed motion;
        executed: [F, J] executed frames so far (rad), used only by the history = measured ablation."""
        m, cfg = self.m, self.m.cfg
        torch.cuda.synchronize(); t0 = time.perf_counter()
        spk_k = None
        if self.causal is not None:                                  # recompute from the arrived prefix (counted in the tick time)
            Tk = min(self.feat.shape[0], (self.k * CHUNK + self.H) * STRIDE)
            self.feat = torch.as_tensor(self.causal.feat(Tk), device=self.dev).float()
            sv, sval = self.causal.spk(Tk)
            spk_k = (torch.as_tensor(sv, device=self.dev).float(), torch.tensor(float(sval), device=self.dev))
        c0, f0, feat = self._window()
        W = self._W                                                  # W, or shorter at the clip start with trim_start
        cur = W - 1 - LOOK                                           # index of chunk k in the window
        N = W * CHUNK
        causal = torch.tensor([self.H < INF], device=self.dev)
        a = m.audio_tokens(feat, causal, *self._tts)                  # == m.audio(feat, causal) for models without tts_h
        Hrel = torch.tensor([self.H if self.H < INF else INF], device=self.dev)
        amask = m.chunk_audio_mask(W, N, Hrel, self.dev)
        if self.audio_off:                                           # = the CFG null branch's audio, now in every branch
            a = m.audio_null.expand_as(a); amask = torch.zeros_like(amask)
        # chunks before the clip start do not exist: hide their audio (zeros) is enough; text times relative to window
        c = self.cond
        tt = c['ttime'] - f0 / 50.
        raw = c['text_c'] if self.text_mode in ('x2', 'voiced', 'asr') else c['text']   # token streaming: causal features
        text, tmask = m.text_tokens(raw[None].float(), tt[None].float(), c['tmask'][None].bool())
        tsec = m.token_seconds(tt[None].float())                    # flow_text_xattn: token times (window origin)
        tvis = lvis = sstate = None; dual = False
        adsc_m = hasattr(m, 'stream_emb')
        if self.text_mode != 'final' or self.deploy_timing or adsc_m:
            now = torch.arange(W, device=self.dev).float() * CHUNK_FRAMES / 50.         # window-relative chunk starts
            final = torch.tensor([self.text_mode == 'final'], device=self.dev)
            tm0 = c['tmask'][None].bool()
            nan = lambda n: torch.full((1, n), float('nan'), device=self.dev)
            if self.text_mode in ('voiced', 'asr'):
                tvis = self._word_tvis(c0, W) & tm0[:, None, :]
            else:
                arr = (self.arr_text - f0 / 50.)[None] if self.arr_text is not None else nan(tm0.shape[1])
                tvis = arrival_mask(arr, tm0, final, now)
            if 'tidm' in c:
                larr = (self.arr_lm - f0 / 50.)[None] if self.arr_lm is not None else nan(c['tidm'].shape[0])
                lvis = arrival_mask(larr, c['tidm'][None].bool(), final, now)
            if adsc_m:
                kg = torch.arange(W, device=self.dev) + c0
                afin = torch.full((1, W), self.H >= INF, device=self.dev) | ((kg * CHUNK + min(self.H, 10 ** 6)) * STRIDE >= self.feat.shape[0])[None]
                if self.text_mode in ('voiced', 'asr'):              # text 'final' (2) only once the audio is final
                    nv = tvis.sum(-1); nt = tm0.sum(-1, keepdim=True)
                    sstate = torch.where(afin & (nv >= nt), 2, torch.where(nv > 0, 1, 0)) * 2 + afin.long()
                else:
                    sstate = stream_state(tvis, tm0, afin)
                if getattr(self, 'floor_k0', None) is not None:         # 11-06 floor token from chunk floor_k0 on (default off)
                    sstate = sstate + 6 * self.floor_val * (kg >= self.floor_k0).long()[None]
            if (self.deploy_timing or adsc_m) and self.H < INF:              # timing only for tokens voiced by now + H
                heard = now + self.H * STRIDE / 50.
                ttw = tt.float()
                voiced = (torch.isfinite(ttw)[None] & (torch.nan_to_num(ttw, 1e9)[None] <= heard[:, None]))[None]
                tu, _ = m.text_tokens(raw[None].float(), ttw[None], tm0, drop=torch.ones(1, dtype=torch.bool, device=self.dev))
                text = torch.cat([text, tu], 1); tmask = torch.cat([tmask, tmask], 1)
                tvis = torch.cat([tvis & voiced, tvis & ~voiced], -1)
                tsec = torch.cat([tsec, torch.full_like(tsec, float('nan'))], 1); dual = True
        ctl_u = None; tm_noctl = tmask
        if self.has_ctl:                                             # control tokens: same arrival / voicing rules as their word
            from .ctl import anchor_arrival, visibility
            ctl_tt = (c['ctl_t'].float() - f0 / 50.)[None]; cm = (c['ctl_type'] > 0)[None] & torch.isfinite(ctl_tt)
            _cd = (c['ctl_d'][None].float(), c['ctl_dm'][None].float() * cm.float()) if 'ctl_d' in c else (None, None)   # 11-09 P2
            ctl_u = m.ctl_embed(c['ctl_type'].long()[None] * cm.long(), c['ctl_v'][None].float(), c['ctl_vm'][None].float(), *_cd)
            nowc = torch.arange(W, device=self.dev).float() * CHUNK_FRAMES / 50.
            arr_c = (anchor_arrival((self.arr_text - f0 / 50.)[None], tt[None].float(), c['tmask'][None].bool(), ctl_tt)
                     if (self.arr_text is not None and self.text_mode != 'final') else ctl_tt)
            if bool((c['ctl_type'] == self._yield_id).any()):         # 11-07 duplex: a YIELD event arrives at its own time
                arr_c = torch.where((c['ctl_type'] == self._yield_id)[None], ctl_tt, arr_c)
            heard_c = (nowc + (self.H * STRIDE / 50. if self.H < INF else 1e9))[None]
            stream_c = torch.tensor([bool((self.deploy_timing or adsc_m) and self.H < INF)], device=self.dev)
            if self.ctl_lookahead > 0:
                from .ctl import TYPE_NAMES
                from .ctl import SEM
                g0 = (c['ctl_type'] > 0)[None] & (c['ctl_type'] <= len([n for n in TYPE_NAMES[1:] if not n.startswith('lex_') and n != 'sem']))[None]
                g0 = g0 | (c['ctl_type'] == SEM)[None]                  # 11-09 P2: SEM tokens get the look-ahead (the legacy id rule unchanged)
                arr_c = torch.where(g0, torch.minimum(arr_c.float(), torch.nan_to_num(ctl_tt, 0.) - self.ctl_lookahead), arr_c.float())
            ckeep, ctimed = visibility(cm, ctl_tt, arr_c, torch.tensor([self.text_mode == 'final'], device=self.dev), nowc, heard_c, stream_c)
            if self.ctl_lookahead > 0:        # its time is known ctl_lookahead s before it is spoken (voiced-rule shifted)
                early = (torch.nan_to_num(ctl_tt, 1e9)[:, None, :] - self.ctl_lookahead <= heard_c[:, :, None])
                ctimed = torch.where(g0[:, None, :], ctimed | (ckeep & early), ctimed)
            tt_c, tu_c = m.ctl_text_tokens(ctl_u, ctl_tt)
            ctn = torch.where(cm, ctl_tt, torch.full_like(ctl_tt, float('nan'))); n0 = text.shape[1]
            gate_ = getattr(cfg, 'ctl_gate', False) or getattr(cfg, 'ctl_route', 'both') != 'both'     # D5 (default off)
            if dual:
                cm_t, tv_t, tv_u = m.ctl_text_vis(cm, ctl_tt, ckeep, ctimed, nowc)
                text = torch.cat([text, tt_c.to(text.dtype), tu_c.to(text.dtype)], 1); tmask = torch.cat([tmask, cm_t, cm_t], 1)
                tvis = torch.cat([tvis, tv_t, tv_u], -1); tsec = torch.cat([tsec, ctn, torch.full_like(ctn, float('nan'))], 1)
            else:
                if gate_:
                    if tvis is None: raise ValueError('ctl_gate / ctl_route need chunk visibility (tvis)')
                    cm_t, tv_t, _ = m.ctl_text_vis(cm, ctl_tt, ckeep, ctimed, nowc)
                    text = torch.cat([text, tt_c.to(text.dtype)], 1); tmask = torch.cat([tmask, cm_t], 1); tsec = torch.cat([tsec, ctn], 1)
                    tvis = torch.cat([tvis, tv_t], -1)
                else:
                    text = torch.cat([text, tt_c.to(text.dtype)], 1); tmask = torch.cat([tmask, cm], 1); tsec = torch.cat([tsec, ctn], 1)
                    if tvis is not None: tvis = torch.cat([tvis, ckeep], -1)
            tm_noctl = tmask.clone(); tm_noctl[:, n0:] = False
        if self.has_ret:                                             # retrieved exemplars: same arrival / voicing rules as their word
            from .ctl import anchor_arrival, visibility
            ret_tt = (c['ret_t'].float() - f0 / 50.)[None]; rmk = torch.isfinite(ret_tt)
            nowr = torch.arange(W, device=self.dev).float() * CHUNK_FRAMES / 50.
            arr_r = (anchor_arrival((self.arr_text - f0 / 50.)[None], tt[None].float(), c['tmask'][None].bool(), ret_tt)
                     if (self.arr_text is not None and self.text_mode != 'final') else ret_tt)
            _, rtimed = visibility(rmk, ret_tt, arr_r, torch.tensor([self.text_mode == 'final'], device=self.dev), nowr,
                                   (nowr + (self.H * STRIDE / 50. if self.H < INF else 1e9))[None],
                                   torch.tensor([bool((self.deploy_timing or adsc_m) and self.H < INF)], device=self.dev))
        st_flow = st_plan = None                                     # 11-07 sem_time: conditional branch only (0 without visible words)
        if getattr(cfg, 'sem_time', False): st_flow, st_plan = m.sem_time_cond(text, tmask, tvis, tsec, W)
        spk, spk_valid = spk_k if spk_k is not None else (c['spk'], c['spk_valid'])
        g = m.global_cond(spk[None].float(), spk_valid[None].float(), c['source'][None].long())
        if self.trace is not None:
            cv = (tvis[0, cur] if tvis is not None else tmask[0]).cpu()
            self.trace.append(dict(k=self.k, c0=c0, feat=feat.cpu(), spk=spk.cpu(), spk_valid=spk_valid.cpu(), raw=raw.cpu(),
                                   ttime=c['ttime'].cpu(), tmask=tmask.cpu(), tvis=None if tvis is None else tvis.cpu(), cur_vis=cv,
                                   sstate=None if sstate is None else sstate.cpu()))
        # committed latents of the window chunks (zeros before the start)
        z = cfg.z
        def window(src):
            w = torch.zeros(W, CHUNK, z, device=self.dev)
            allc = torch.stack(src).view(-1, CHUNK, z) if src else None
            for j in range(cur):
                kc = c0 + j
                if allc is not None and 0 <= kc < allc.shape[0]: w[j] = allc[kc]
                elif kc == -1 and self.hist0 is not None: w[j] = self.hist0
            return w
        lat = window(self.lat)
        # diagnostics: the planner may read a different history (e.g. ground truth) than the local flow context
        plan_lat = window(self.plan_src) if getattr(self, 'plan_src', None) is not None else lat
        sref = None
        if cfg.style:
            sref = m.style_enc(c['ref'][None].float()) if float(c['ref_valid']) > .5 else m.style_null[None]
        st_tok = st_ada = None
        lat_ctx = lat
        if self.hist_measured and executed is not None and len(executed) >= CHUNK_FRAMES and self.lat:
            # measured history = executed frames + the committed-but-not-yet-emitted plan frames (last latent), exactly as
            # in training; only the LOCAL FLOW CONTEXT uses it, the planner keeps reading the committed history
            L1 = len(self.lat); L0 = max(0, L1 - 4 * CHUNK)
            y = self.vae.decode(torch.stack(self.lat[L0:L1])[None] * m.lat_std + m.lat_mean)[0] * self.qs + self.qm
            pend = y[self.n_emitted - L0 * STRIDE:L1 * STRIDE]
            ex = torch.cat([torch.as_tensor(np.asarray(executed), device=self.dev).float(), pend], 0)[-4 * CHUNK_FRAMES:]
            ex = ex[len(ex) % STRIDE:]
            mu, _ = self.vae.encode(((ex - self.qm) / self.qs)[None])
            meas = ((mu - m.lat_mean) / m.lat_std)[0, -CHUNK:]
            if cur - 1 >= 0: lat_ctx = lat.clone(); lat_ctx[cur - 1] = meas
        if cfg.state and robot_q is not None and not self.hist_measured:
            q_plan = self.frames[-1] if self.frames else torch.as_tensor(robot_q, device=self.dev).float()
            sv = state_vector(torch.as_tensor(robot_q, device=self.dev).float()[None],
                              torch.as_tensor(robot_dq, device=self.dev).float()[None], q_plan[None].float(), self.qm, self.qs)
            st_tok, st_ada = m.state_ad(sv)
        out = []
        n_gen = 1 + self.n_future
        for j in range(n_gen):
            kw = cur + j
            if kw >= W: break
            mt = mtm = None
            stw = None
            if st_tok is not None:
                # 11-02 review fix: every window position gets the state token of ITS OWN chunk (as in training), not a
                # copy of the current one; chunks without a recorded state (before the start) use the current token
                self.st_hist[self.k] = st_tok.detach()
                stw = torch.stack([self.st_hist.get(c0 + jj, st_tok)[0] for jj in range(W)])[None]
            if cfg.mtc:
                mt, mtm = m.mt_memory(text, tmask, a, amask, g, sref, lat[None], stw, W, tvis=tvis)
                mflat, pm = m.planner_memory(mt, mtm)
                h = m.plan((plan_lat if j == 0 else lat)[None], a, amask, text, tmask, g, mflat, pm, *self._ids(), sref, tvis=tvis, lvis=lvis, sstate=sstate)
            else:
                h = m.plan((plan_lat if j == 0 else lat)[None], a, amask, text, tmask, g, tids=self._ids()[0], tidm=self._ids()[1], sref=sref, tvis=tvis, lvis=lvis, sstate=sstate,
                           **({} if st_plan is None else dict(padd=st_plan)))
            hk = h[:, kw]
            if self.cfg_ctl is not None:                            # conditional branch without the control tokens
                hk_c0 = m.plan((plan_lat if j == 0 else lat)[None], a, amask, text, tm_noctl, g, tids=self._ids()[0], tidm=self._ids()[1],
                               sref=sref, tvis=tvis, lvis=lvis, sstate=sstate)[:, kw]
            if self.cfg_w != 1:                                     # unconditional branch: audio + text dropped
                a_u = m.audio_null.expand_as(a); am_u = torch.zeros_like(amask); tm_u = torch.zeros_like(tmask) if self.cfg_text else tmask
                if cfg.mtc:
                    mt_u, mtm_u = m.mt_memory(text, tm_u, a_u, am_u, g, sref, lat[None], stw, W, tvis=tvis)
                    mf_u, pm_u = m.planner_memory(mt_u, mtm_u)
                    h_u = m.plan(lat[None], a_u, am_u, text, tm_u, g, mf_u, pm_u, self._ids()[0], self._null_ids(), sref, tvis=tvis, lvis=lvis, sstate=sstate)
                else:
                    h_u = m.plan(lat[None], a_u, am_u, text, tm_u, g, tids=self._ids()[0], tidm=self._null_ids(), sref=sref, tvis=tvis, lvis=lvis, sstate=sstate)
                hk_u = h_u[:, kw]
                if self.cfg_t is not None:                          # audio-only branch (text dropped, audio kept)
                    if cfg.mtc:
                        mt_a, mtm_a = m.mt_memory(text, tm_u, a, amask, g, sref, lat[None], stw, W, tvis=tvis)
                        mf_a, pm_a = m.planner_memory(mt_a, mtm_a)
                        h_a = m.plan(lat[None], a, amask, text, tm_u, g, mf_a, pm_a, self._ids()[0], self._null_ids(), sref, tvis=tvis, lvis=lvis, sstate=sstate)
                    else:
                        h_a = m.plan(lat[None], a, amask, text, tm_u, g, tids=self._ids()[0], tidm=self._null_ids(), sref=sref, tvis=tvis, lvis=lvis, sstate=sstate)
                    hk_a = h_a[:, kw]
            style_tok = None
            if cfg.style:
                style_tok, _, _ = m.style_head(sref, hk, temperature=self.temp, generator=self.g)
            # context = last ctx latents before chunk kw
            prev = (lat_ctx if j == 0 else lat)[kw - 1] if kw >= 1 else torch.zeros(CHUNK, z, device=self.dev)
            if kw == 0 and c0 + kw - 1 == -1 and self.hist0 is not None: prev = self.hist0   # trimmed window at the clip start
            has_ctx = (c0 + kw - 1 >= 0 and len(self.lat) > 0) or (c0 + kw - 1 == -1 and self.hist0 is not None) or j > 0
            L = cfg.ctx + CHUNK
            known = torch.zeros(1, L, dtype=torch.bool, device=self.dev); known[:, :cfg.ctx] = has_ctx
            x = torch.randn(1, L, z, device=self.dev, generator=self.g)
            if self.noise_scale != 1.0: x = x * self.noise_scale
            if self.noise_override is not None:      # stage-2 latent steering (DSRL): a policy chooses the initial noise
                eps = self.noise_override(self, self.k + j, x[:, cfg.ctx:].clone())
                if eps is not None: x[:, cfg.ctx:] = eps.to(x)
            c_ = prev[-cfg.ctx:][None]
            ct = self.ctx_t
            if self.still_refresh is not None and has_ctx and cfg.ctx > 1:
                f0 = (self.k + j) * CHUNK_FRAMES; sp = self._speech[f0:f0 + CHUNK_FRAMES]
                if len(sp) and bool(sp.any()) and float((c_[0, 1:] - c_[0, :-1]).norm(dim=-1).mean()) < self.still_refresh[0]:
                    ct = min(ct, self.still_refresh[1]); self.n_refresh += 1
            if ct < 1: c_ = ct * c_ + (1 - ct) * torch.randn(c_.shape, device=self.dev, generator=self.g)
            x[:, :cfg.ctx] = torch.where(known[:, :cfg.ctx, None], c_, x[:, :cfg.ctx])
            ip = None; ikw = {}
            if self.inpaint is not None:                         # target / weights of this window's positions (global latent grid)
                gl = (c0 + kw) * CHUNK - cfg.ctx + torch.arange(L, device=self.dev)
                okg = (gl >= 0) & (gl < len(self.inpaint['w']))
                wf = torch.where(okg, self.inpaint['w'][gl.clamp(0, len(self.inpaint['w']) - 1)], torch.zeros_like(gl, dtype=torch.float))
                tg = self.inpaint['lat'][gl.clamp(0, len(self.inpaint['w']) - 1)][None]
                if hasattr(m, 'f_mask') and bool((wf >= 0.999).any()):
                    # 11-07 item 5b, mask-conditioned model: the weight-1 core is CLEAN KNOWN latents flagged by f_mask (as in
                    # training); only the soft ramps are inpainted RePaint-style
                    core = (wf >= 0.999)[None]
                    x = torch.where((core & ~known)[..., None], tg, x); known = known | core
                    ikw = dict(ins=core); wf = wf * (~core[0]).float()
                wv = (wf * (~known[0]).float())[None, :, None]
                if bool((wv > 0).any()):
                    eps = torch.randn(x.shape, device=self.dev, generator=self.inpaint['g'])
                    x = wv * eps + (1 - wv) * x; ip = (wv, tg, eps)
            aw, awm = m.local_audio(a, Hrel, W)
            aw, awm = aw[:, kw], awm[:, kw]
            if self.cfg_w != 1:
                aw_u = m.audio_null.expand_as(aw); awm_u = awm.clone() if self.cfg_null_fix else torch.zeros_like(awm)
                mtk_u = mtmk_u = None
                if cfg.mtc: mtk_u, mtmk_u = mt_u[:, kw], mtm_u[:, kw][:, None].expand(-1, cfg.ctx + CHUNK, -1)
                if self.cfg_t is not None:
                    mtk_a = mtmk_a = None
                    if cfg.mtc: mtk_a, mtmk_a = mt_a[:, kw], mtm_a[:, kw][:, None].expand(-1, cfg.ctx + CHUNK, -1)
            mtk = mtmk = None
            if cfg.mtc: mtk, mtmk = mt[:, kw], mtm[:, kw][:, None].expand(-1, L, -1)
            ft = ft_u = None
            if hasattr(m, 'f_text'):                                 # the planner's text visibility of chunk kw (incl. CFG branches)
                qsec = m.flow_seconds(torch.tensor([kw], device=self.dev), L, cfg.ctx)
                ft = (text, m.chunk_text_mask(tmask, tvis, W)[:, kw], tsec, qsec)
                if self.cfg_w != 1: ft_u = (text, m.chunk_text_mask(tm_u, tvis, W)[:, kw], tsec, qsec)
                if self.cfg_ctl is not None: ft_c0 = (text, m.chunk_text_mask(tm_noctl, tvis, W)[:, kw], tsec, qsec)
            elif self.cfg_ctl is not None: ft_c0 = None
            ctlkw = {}
            if ctl_u is not None:
                cc = m.ctl_flow_cond(ctl_u, ctl_tt, ckeep[:, kw:kw + 1], ctimed[:, kw:kw + 1],
                                     m.flow_seconds(torch.tensor([kw], device=self.dev), L, cfg.ctx))
                if cc is not None: ctlkw = dict(ctl_c=cc[:, 0])
            if st_flow is not None:
                ctlkw = dict(ctl_c=st_flow[:, kw] if 'ctl_c' not in ctlkw else ctlkw['ctl_c'] + st_flow[:, kw].to(ctlkw['ctl_c'].dtype))
            if self.has_ret:
                rc = m.ret_flow_cond(c['ret_lat'][None].float(), ret_tt, rtimed[:, kw:kw + 1],
                                     m.flow_seconds(torch.tensor([kw], device=self.dev), L, cfg.ctx))
                if rc is not None: ctlkw = dict(ctl_c=rc[:, 0] if 'ctl_c' not in ctlkw else ctlkw['ctl_c'] + rc[:, 0].to(ctlkw['ctl_c'].dtype))
            ada = st_ada if j == 0 else None                         # the measured state only describes "now"
            if cfg.sem_head and cfg.sem_cond:                        # Local Flow sees the predicted g_k (each CFG branch its own)
                hk = m.sem_h(hk, self.g_override)
                if self.cfg_w != 1:
                    hk_u = m.sem_h(hk_u, self.g_override)
                    if self.cfg_t is not None: hk_a = m.sem_h(hk_a, self.g_override)
                if self.cfg_ctl is not None: hk_c0 = m.sem_h(hk_c0, self.g_override)
            for s in range(self.steps):
                tval = s / self.steps
                t = torch.full((1, L), tval, device=self.dev); t[known] = 1.
                if ct < 1: t[:, :cfg.ctx] = torch.where(known[:, :cfg.ctx], torch.full_like(t[:, :cfg.ctx], ct), t[:, :cfg.ctx])
                v = m.flow_velocity(x, t, known, hk, g, aw, awm, style_tok, ada, mtk, mtmk, ftext=ft, **ctlkw, **ikw)
                if self.cfg_ctl is not None:
                    v_full = v; v_c0 = m.flow_velocity(x, t, known, hk_c0, g, aw, awm, style_tok, ada, mtk, mtmk, ftext=ft_c0, **ikw); v = v_c0
                if self.cfg_w != 1:
                    v_u = m.flow_velocity(x, t, known, hk_u, g, aw_u, awm_u, style_tok, ada, mtk_u, mtmk_u, ftext=ft_u, **ikw)
                    if self.cfg_t is not None:
                        v_a = m.flow_velocity(x, t, known, hk_a, g, aw, awm, style_tok, ada, mtk_a, mtmk_a, ftext=ft_u, **ikw)
                        v = v_u + self.cfg_w * (v_a - v_u) + self.cfg_t * (v - v_a)
                    else:
                        v = v_u + self.cfg_w * (v - v_u)
                if self.cfg_ctl is not None: v = v + self.cfg_ctl * (v_full - v_c0)
                x = torch.where(known[..., None], x, x + v / self.steps)
                if ip is not None:
                    tn = (s + 1) / self.steps; wv, tg, eps = ip
                    x = wv * (tn * tg + (1 - tn) * eps) + (1 - wv) * x
            new = x[0, cfg.ctx:]
            lat[kw] = new
            out.append(new)
        committed = out[0]
        for i in range(CHUNK): self.lat.append(committed[i].clone())
        # decode with the decoder's 1-latent look-ahead satisfied by COMMITTED latents only: a frame is emitted once the
        # latent after it is committed (fixed 80 ms output delay). Emitting with a replicated or provisional look-ahead
        # gave 5-6x jerk at chunk boundaries (scripts/diag_boundary.py C/E); the decoder also looks ~7 latents back, so a
        # 4-chunk tail is decoded (a 2-chunk tail leaked window-start padding: 9-10x).
        fr = self._emit(final=False)
        self.provisional = [o for o in out[1:]]
        torch.cuda.synchronize(); self.timing.append(time.perf_counter() - t0)
        self.k += 1
        return fr


@torch.no_grad()
def generate(model, vae, feat, cond, stats, n_chunks, robot=None, **kw):
    """Run the streamer over a whole utterance. robot: optional object with .step(frames) -> (q, dq) (closed loop)."""
    s = Streamer(model, vae, feat, cond, stats, **kw)
    q = dq = None
    executed = []
    for i in range(n_chunks):
        fr = s.tick(q, dq, np.concatenate(executed) if executed else None)
        if i == n_chunks - 1: fr = torch.cat([fr, s.flush()], 0)
        if robot is not None and len(fr):
            q, dq, ex = robot.step(fr)
            executed.append(ex)
    cmd = torch.stack(s.frames).cpu().numpy()
    return cmd, (np.concatenate(executed) if executed else None), np.array(s.timing)

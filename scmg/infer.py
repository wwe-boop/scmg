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
    def __init__(self, model, vae, feat, cond, stats, H=INF, steps=10, temperature=1.0, n_future=0, seed=0,
                 standby=None, device='cuda', hist_measured=False, ctx_t=1.0, cfg_w=1.0, cfg_text=True, cfg_t=None,
                 text_mode='final', x2_rate=15.0, x2_lag=0.15, seg_lead=0.3, deploy_timing=False,
                 elastic=None, window_lm_text=False, g_override=None):
        """feat [T, 89] normalised (whole utterance; access is restricted to what has arrived);
        cond: dict(text [K,768], ttime [K] absolute s, tmask [K], spk [192], spk_valid, source, ref [R,J], ref_valid)
        (+ tts_h [T/4, 2048], tts_hm [T/4] for models with cfg.tts_h: TTS hidden rows aligned to feat frame 0, row r = frames
        [4r, 4r+4); a row is visible exactly when the audio of its frames is -- the same arrival rule as feat)."""
        require_causal_text(cond.get('text_c'), text_mode == 'x2', 'Streamer text_mode=x2')
        self.m, self.vae, self.dev = model.eval(), vae.eval(), device
        self.feat = torch.as_tensor(feat, device=device).float()
        self.cond = {k: torch.as_tensor(v, device=device) for k, v in cond.items()}
        if getattr(model.cfg, 'tts_h', False) and 'tts_h' not in self.cond:
            raise ValueError('model was trained with tts_h (TTS hidden states): cond needs tts_h / tts_hm')
        self.qm = torch.tensor(stats['q_mean'], device=device); self.qs = torch.tensor(stats['q_std'], device=device)
        self.H, self.steps, self.temp, self.n_future = H, steps, temperature, n_future
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
        self.g_override = g_override         # fixed g_k [n_sem] for every chunk (zeros = null g; None = the head's prediction)
        self.cfg_text = cfg_text             # False: guide on audio only (models never trained with text dropped, e.g. r4 VoxCPM)
        self.ctx_t = ctx_t                   # <1: noise the committed context like in training (exposure-bias fix)
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
            start = (self.k - (WIN - 1 - LOOK)) * CHUNK_FRAMES / 50.
            end = start + WIN * CHUNK_FRAMES / 50.
            mask = mask & (c['tiseg'] < end) & (c['tiend'] > start)
            mask = mask & (mask.long().cumsum(0) <= 160)
        return c['tids'][None].long(), mask[None]

    @torch.no_grad()
    def _window(self):
        m, k = self.m, self.k
        c0 = k - (WIN - 1 - LOOK)                                   # first chunk of the window (may be < 0)
        f0 = c0 * CHUNK_FRAMES; T = WIN * CHUNK_FRAMES
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
        c0, f0, feat = self._window()
        cur = WIN - 1 - LOOK                                         # index of chunk k in the window
        N = WIN * CHUNK
        causal = torch.tensor([self.H < INF], device=self.dev)
        a = m.audio_tokens(feat, causal, *self._tts)                  # == m.audio(feat, causal) for models without tts_h
        Hrel = torch.tensor([self.H if self.H < INF else INF], device=self.dev)
        amask = m.chunk_audio_mask(WIN, N, Hrel, self.dev)
        # chunks before the clip start do not exist: hide their audio (zeros) is enough; text times relative to window
        c = self.cond
        tt = c['ttime'] - f0 / 50.
        raw = c['text_c'] if self.text_mode == 'x2' else c['text']   # token streaming: causal features
        text, tmask = m.text_tokens(raw[None].float(), tt[None].float(), c['tmask'][None].bool())
        tvis = lvis = sstate = None
        adsc_m = hasattr(m, 'stream_emb')
        if self.text_mode != 'final' or self.deploy_timing or adsc_m:
            now = torch.arange(WIN, device=self.dev).float() * CHUNK_FRAMES / 50.         # window-relative chunk starts
            final = torch.tensor([self.text_mode == 'final'], device=self.dev)
            tm0 = c['tmask'][None].bool()
            nan = lambda n: torch.full((1, n), float('nan'), device=self.dev)
            arr = (self.arr_text - f0 / 50.)[None] if self.arr_text is not None else nan(tm0.shape[1])
            tvis = arrival_mask(arr, tm0, final, now)
            if 'tidm' in c:
                larr = (self.arr_lm - f0 / 50.)[None] if self.arr_lm is not None else nan(c['tidm'].shape[0])
                lvis = arrival_mask(larr, c['tidm'][None].bool(), final, now)
            if adsc_m:
                kg = torch.arange(WIN, device=self.dev) + c0
                afin = torch.full((1, WIN), self.H >= INF, device=self.dev) | ((kg * CHUNK + min(self.H, 10 ** 6)) * STRIDE >= self.feat.shape[0])[None]
                sstate = stream_state(tvis, tm0, afin)
            if (self.deploy_timing or adsc_m) and self.H < INF:              # timing only for tokens voiced by now + H
                heard = now + self.H * STRIDE / 50.
                ttw = tt.float()
                voiced = (torch.isfinite(ttw)[None] & (torch.nan_to_num(ttw, 1e9)[None] <= heard[:, None]))[None]
                tu, _ = m.text_tokens(raw[None].float(), ttw[None], tm0, drop=torch.ones(1, dtype=torch.bool, device=self.dev))
                text = torch.cat([text, tu], 1); tmask = torch.cat([tmask, tmask], 1)
                tvis = torch.cat([tvis & voiced, tvis & ~voiced], -1)
        g = m.global_cond(c['spk'][None].float(), c['spk_valid'][None].float(), c['source'][None].long())
        # committed latents of the window chunks (zeros before the start)
        z = cfg.z
        def window(src):
            w = torch.zeros(WIN, CHUNK, z, device=self.dev)
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
            if kw >= WIN: break
            mt = mtm = None
            stw = None
            if st_tok is not None:
                # 11-02 review fix: every window position gets the state token of ITS OWN chunk (as in training), not a
                # copy of the current one; chunks without a recorded state (before the start) use the current token
                self.st_hist[self.k] = st_tok.detach()
                stw = torch.stack([self.st_hist.get(c0 + jj, st_tok)[0] for jj in range(WIN)])[None]
            if cfg.mtc:
                mt, mtm = m.mt_memory(text, tmask, a, amask, g, sref, lat[None], stw, WIN, tvis=tvis)
                mflat, pm = m.planner_memory(mt, mtm)
                h = m.plan((plan_lat if j == 0 else lat)[None], a, amask, text, tmask, g, mflat, pm, *self._ids(), sref, tvis=tvis, lvis=lvis, sstate=sstate)
            else:
                h = m.plan((plan_lat if j == 0 else lat)[None], a, amask, text, tmask, g, tids=self._ids()[0], tidm=self._ids()[1], sref=sref, tvis=tvis, lvis=lvis, sstate=sstate)
            hk = h[:, kw]
            if self.cfg_w != 1:                                     # unconditional branch: audio + text dropped
                a_u = m.audio_null.expand_as(a); am_u = torch.zeros_like(amask); tm_u = torch.zeros_like(tmask) if self.cfg_text else tmask
                if cfg.mtc:
                    mt_u, mtm_u = m.mt_memory(text, tm_u, a_u, am_u, g, sref, lat[None], stw, WIN, tvis=tvis)
                    mf_u, pm_u = m.planner_memory(mt_u, mtm_u)
                    h_u = m.plan(lat[None], a_u, am_u, text, tm_u, g, mf_u, pm_u, self._ids()[0], self._null_ids(), sref, tvis=tvis, lvis=lvis, sstate=sstate)
                else:
                    h_u = m.plan(lat[None], a_u, am_u, text, tm_u, g, tids=self._ids()[0], tidm=self._null_ids(), sref=sref, tvis=tvis, lvis=lvis, sstate=sstate)
                hk_u = h_u[:, kw]
                if self.cfg_t is not None:                          # audio-only branch (text dropped, audio kept)
                    if cfg.mtc:
                        mt_a, mtm_a = m.mt_memory(text, tm_u, a, amask, g, sref, lat[None], stw, WIN, tvis=tvis)
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
            has_ctx = (c0 + kw - 1 >= 0 and len(self.lat) > 0) or (c0 + kw - 1 == -1 and self.hist0 is not None) or j > 0
            L = cfg.ctx + CHUNK
            known = torch.zeros(1, L, dtype=torch.bool, device=self.dev); known[:, :cfg.ctx] = has_ctx
            x = torch.randn(1, L, z, device=self.dev, generator=self.g)
            if self.noise_override is not None:      # stage-2 latent steering (DSRL): a policy chooses the initial noise
                eps = self.noise_override(self, self.k + j, x[:, cfg.ctx:].clone())
                if eps is not None: x[:, cfg.ctx:] = eps.to(x)
            c_ = prev[-cfg.ctx:][None]
            if self.ctx_t < 1: c_ = self.ctx_t * c_ + (1 - self.ctx_t) * torch.randn(c_.shape, device=self.dev, generator=self.g)
            x[:, :cfg.ctx] = torch.where(known[:, :cfg.ctx, None], c_, x[:, :cfg.ctx])
            aw, awm = m.local_audio(a, Hrel, WIN)
            aw, awm = aw[:, kw], awm[:, kw]
            if self.cfg_w != 1:
                aw_u = m.audio_null.expand_as(aw); awm_u = torch.zeros_like(awm)
                mtk_u = mtmk_u = None
                if cfg.mtc: mtk_u, mtmk_u = mt_u[:, kw], mtm_u[:, kw][:, None].expand(-1, cfg.ctx + CHUNK, -1)
                if self.cfg_t is not None:
                    mtk_a = mtmk_a = None
                    if cfg.mtc: mtk_a, mtmk_a = mt_a[:, kw], mtm_a[:, kw][:, None].expand(-1, cfg.ctx + CHUNK, -1)
            mtk = mtmk = None
            if cfg.mtc: mtk, mtmk = mt[:, kw], mtm[:, kw][:, None].expand(-1, L, -1)
            ada = st_ada if j == 0 else None                         # the measured state only describes "now"
            if cfg.sem_head and cfg.sem_cond:                        # Local Flow sees the predicted g_k (each CFG branch its own)
                hk = m.sem_h(hk, self.g_override)
                if self.cfg_w != 1:
                    hk_u = m.sem_h(hk_u, self.g_override)
                    if self.cfg_t is not None: hk_a = m.sem_h(hk_a, self.g_override)
            for s in range(self.steps):
                tval = s / self.steps
                t = torch.full((1, L), tval, device=self.dev); t[known] = 1.
                if self.ctx_t < 1: t[:, :cfg.ctx] = torch.where(known[:, :cfg.ctx], torch.full_like(t[:, :cfg.ctx], self.ctx_t), t[:, :cfg.ctx])
                v = m.flow_velocity(x, t, known, hk, g, aw, awm, style_tok, ada, mtk, mtmk)
                if self.cfg_w != 1:
                    v_u = m.flow_velocity(x, t, known, hk_u, g, aw_u, awm_u, style_tok, ada, mtk_u, mtmk_u)
                    if self.cfg_t is not None:
                        v_a = m.flow_velocity(x, t, known, hk_a, g, aw, awm, style_tok, ada, mtk_a, mtmk_a)
                        v = v_u + self.cfg_w * (v_a - v_u) + self.cfg_t * (v - v_a)
                    else:
                        v = v_u + self.cfg_w * (v - v_u)
                x = torch.where(known[..., None], x, x + v / self.steps)
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

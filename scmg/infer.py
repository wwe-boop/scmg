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
import time
import numpy as np
import torch

from .common import CHUNK, CHUNK_FRAMES, STRIDE
from .model import INF
from .sim import state_vector

WIN, LOOK = 16, 2


class Streamer:
    def __init__(self, model, vae, feat, cond, stats, H=INF, steps=10, temperature=1.0, n_future=0, seed=0,
                 standby=None, device='cuda', hist_measured=False):
        """feat [T, 89] normalised (whole utterance; access is restricted to what has arrived);
        cond: dict(text [K,768], ttime [K] absolute s, tmask [K], spk [192], spk_valid, source, ref [R,J], ref_valid)."""
        self.m, self.vae, self.dev = model.eval(), vae.eval(), device
        self.feat = torch.as_tensor(feat, device=device).float()
        self.cond = {k: torch.as_tensor(v, device=device) for k, v in cond.items()}
        self.qm = torch.tensor(stats['q_mean'], device=device); self.qs = torch.tensor(stats['q_std'], device=device)
        self.H, self.steps, self.temp, self.n_future = H, steps, temperature, n_future
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.lat = []                     # committed latents (normalised), one [z] per latent step
        self.frames = []                  # emitted joint frames (rad)
        self.k = 0
        self.timing = []
        self.hist_measured = hist_measured   # ablation: context latents = encoded EXECUTED motion, no state adapter
        z = model.cfg.z
        if standby is not None:           # rest pose as initial context (encoded like any history)
            q = ((torch.as_tensor(standby, device=device).float() - self.qm) / self.qs)[None, None].expand(1, CHUNK_FRAMES, -1)
            mu, _ = vae.encode(q.contiguous())
            self.hist0 = ((mu - model.lat_mean) / model.lat_std)[0]
        else:
            self.hist0 = None

    def _ids(self):
        c = self.cond
        if 'tids' not in c: return None, None
        return c['tids'][None].long(), c['tidm'][None].bool()

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
        return c0, f0, feat[None]

    @torch.no_grad()
    def tick(self, robot_q=None, robot_dq=None, executed=None):
        """Advance one chunk. robot_q / robot_dq: measured state (rad, rad/s) at the end of the executed motion;
        executed: [F, J] executed frames so far (rad), used only by the history = measured ablation."""
        m, cfg = self.m, self.m.cfg
        torch.cuda.synchronize(); t0 = time.perf_counter()
        c0, f0, feat = self._window()
        cur = WIN - 1 - LOOK                                         # index of chunk k in the window
        N = WIN * CHUNK
        causal = torch.tensor([self.H < INF], device=self.dev)
        a = m.audio(feat, causal=causal)
        Hrel = torch.tensor([self.H if self.H < INF else INF], device=self.dev)
        amask = m.chunk_audio_mask(WIN, N, Hrel, self.dev)
        # chunks before the clip start do not exist: hide their audio (zeros) is enough; text times relative to window
        c = self.cond
        tt = c['ttime'] - f0 / 50.
        text, tmask = m.text_tokens(c['text'][None].float(), tt[None].float(), c['tmask'][None].bool())
        g = m.global_cond(c['spk'][None].float(), c['spk_valid'][None].float(), c['source'][None].long())
        # committed latents of the window chunks (zeros before the start)
        z = cfg.z
        lat = torch.zeros(WIN, CHUNK, z, device=self.dev)
        allc = torch.stack(self.lat).view(-1, CHUNK, z) if self.lat else None
        for j in range(cur):
            kc = c0 + j
            if allc is not None and 0 <= kc < allc.shape[0]: lat[j] = allc[kc]
            elif kc == -1 and self.hist0 is not None: lat[j] = self.hist0
        sref = None
        if cfg.style:
            sref = m.style_enc(c['ref'][None].float()) if float(c['ref_valid']) > .5 else m.style_null[None]
        st_tok = st_ada = None
        if self.hist_measured and executed is not None and len(executed) >= CHUNK_FRAMES and self.lat:
            ex = torch.as_tensor(np.asarray(executed[-4 * CHUNK_FRAMES:]), device=self.dev).float()
            ex = ex[len(ex) % STRIDE:]
            mu, _ = self.vae.encode(((ex - self.qm) / self.qs)[None])
            meas = ((mu - m.lat_mean) / m.lat_std)[0, -CHUNK:]
            if cur - 1 >= 0: lat[cur - 1] = meas                       # previous chunk = what was actually executed
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
            if st_tok is not None: stw = st_tok[:, None].expand(1, WIN, -1)
            if cfg.mtc:
                mt, mtm = m.mt_memory(text, tmask, a, amask, g, sref, lat[None], stw, WIN)
                mflat, pm = m.planner_memory(mt, mtm)
                h = m.plan(lat[None], a, amask, text, tmask, g, mflat, pm, *self._ids(), sref)
            else:
                h = m.plan(lat[None], a, amask, text, tmask, g, tids=self._ids()[0], tidm=self._ids()[1], sref=sref)
            hk = h[:, kw]
            style_tok = None
            if cfg.style:
                style_tok, _, _ = m.style_head(sref, hk, temperature=self.temp)
            # context = last ctx latents before chunk kw
            prev = lat[kw - 1] if kw >= 1 else torch.zeros(CHUNK, z, device=self.dev)
            has_ctx = (c0 + kw - 1 >= 0 and len(self.lat) > 0) or (c0 + kw - 1 == -1 and self.hist0 is not None) or j > 0
            L = cfg.ctx + CHUNK
            known = torch.zeros(1, L, dtype=torch.bool, device=self.dev); known[:, :cfg.ctx] = has_ctx
            x = torch.randn(1, L, z, device=self.dev, generator=self.g)
            x[:, :cfg.ctx] = torch.where(known[:, :cfg.ctx, None], prev[-cfg.ctx:][None], x[:, :cfg.ctx])
            aw, awm = m.local_audio(a, Hrel, WIN)
            aw, awm = aw[:, kw], awm[:, kw]
            mtk = mtmk = None
            if cfg.mtc: mtk, mtmk = mt[:, kw], mtm[:, kw][:, None].expand(-1, L, -1)
            ada = st_ada if j == 0 else None                         # the measured state only describes "now"
            for s in range(self.steps):
                tval = s / self.steps
                t = torch.full((1, L), tval, device=self.dev); t[known] = 1.
                v = m.flow_velocity(x, t, known, hk, g, aw, awm, style_tok, ada, mtk, mtmk)
                x = torch.where(known[..., None], x, x + v / self.steps)
            new = x[0, cfg.ctx:]
            lat[kw] = new
            out.append(new)
        committed = out[0]
        for i in range(CHUNK): self.lat.append(committed[i].clone())
        # decode: last 2 chunks of committed (+ provisional next latent as the decoder's look-ahead)
        tail = torch.stack(self.lat[-2 * CHUNK:])
        look = out[1][:1] if len(out) > 1 else committed[-1:]
        zz = torch.cat([tail, look], 0)[None] * m.lat_std + m.lat_mean
        y = self.vae.decode(zz)[0] * self.qs + self.qm                # [(2*CHUNK+1)*4, J]
        nf = y.shape[0] - STRIDE
        fr = y[nf - CHUNK_FRAMES:nf]
        for f in fr: self.frames.append(f)
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
    for _ in range(n_chunks):
        fr = s.tick(q, dq, np.concatenate(executed) if executed else None)
        if robot is not None:
            q, dq, ex = robot.step(fr)
            executed.append(ex)
    cmd = torch.stack(s.frames).cpu().numpy()
    return cmd, (np.concatenate(executed) if executed else None), np.array(s.timing)

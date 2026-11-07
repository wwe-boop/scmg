"""Zero-training retraction for the G1 interruption benchmark (11-05): SCMG streaming + interruption handling.

Builds on infer_interrupt.py (B3: the audio ends / turns silent at t_i, unspoken words withdrawn). Additions, all active
ONLY after the interruption (before it, and without --interrupts, the code path is the unmodified Streamer):
  --restart latent  (reaction latency): do not wait for the next 320 ms chunk boundary. At t_i + --react-delay (compute
        time of one tick) the plan is cut at the next latent boundary l_r (80 ms grid): committed latents >= l_r and the
        frames decoded from them are dropped, and streaming re-starts with a chunk that begins at latent l_r. Implemented
        by re-basing the streamer clock by o = l_r mod 4 latents (feat / token times / arrival times / history shifted by
        o * 80 ms), so the model sees a standard [4 context | 4 target] Local Flow window (training windows start at any
        latent offset). Frames < 4 l_r (50 Hz) are kept exactly as already emitted. Cost: when the window reaches back to
        the clip start, the first o latents of the clip leave the planner window (<= 240 ms, clips interrupted < ~4 s).
  --splice n        : with --restart latent, the jump between the kept frames and the first regenerated frame (vs. the
        constant-velocity continuation) is blended out over n 50 Hz frames (cosine weight; only post-restart frames change).
  --restart chunk   : B3 timing (next chunk boundary >= t_i).
  --audio end | silence | off : after t_i the audio stream ends (feat truncated) / is digital silence / is the model's
        NULL audio in every branch (Streamer.audio_off, = a drop_audio training sample; the clip clock is unchanged).
  --retract PATTERN (rest-pose inpainting in the Local Flow): a target joint trajectory from the frame at the restart to
        the rest pose (--rest R_sil | R_gt, waist + arms critically damped, velocity matched, 95 % settled after --tr s;
        legs / base orientation held, base velocity / yaw rate 0) is VAE-encoded (with 1.3 s of executed motion before it)
        into normalised latents; inside every post-interruption flow run the selected target-chunk positions follow the
        target by REPLACEMENT inpainting (at flow time t the position is t * z_target + (1 - t) * its initial noise,
        mixed with weight --w; committed value = w z_target + (1 - w) z_flow), the other positions are generated freely.
        PATTERN: end (last latent of every chunk), mid (latents 1 and 3), all (every latent: ~B2 through the VAE),
        tr (only latents at or after t_i + T_r, target = the settled rest), ramp (every latent, weight rising smoothly
        from 0 at t_i to --w at t_i + T_r: a latent-space cross-fade from the model's own continuation to the target).
Output as infer_interrupt.py: <out>/<stem>@<frac>.pt (30 fps, frames 0 .. f_i + 66) + .q50.npy; --no-interrupt writes the
full-clip infer_test.py output (bitwise check).
11-07 --protocol psync (the main-table causal protocol, infer_test.py --protocol psync, same flags --wav-dir / --words-dir /
--pstats / --campplus / --spk-refresh / --causal-feat): per-tick feat89 / CAM++ of the arrived PCM prefix (scmg.causal_feat.
CausalInputs), voiced text (a token once its words ended within the arrived audio). Before the interruption the run IS the
infer_test psync run (no code path differs; scripts/tests/test_retract_psync.py checks it bitwise). From the interruption on
(PostInterrupt below): the PCM ends at t_i, so feat89 / speaker vector are those of the audio before t_i (a fresh instance of
the same feat class evaluated at min(Tk, t_i); the speaker grade rule unchanged at min(Tk, t_i)), the kept tokens are the
tokens whose words all ended by t_i (the voiced rule at t_i, no look-ahead), and the word times move with the clock re-basing.
--prefix-feats (the engine-protocol approximation of the same thing) is not used with psync.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scmg.common import J, CHUNK, CHUNK_FRAMES, STRIDE, EMBODIMENT, U80
from scmg.data import View, WindowDataset
from scmg.evaluate import load
from scmg.infer import Streamer, LOOK
from scmg.model import INF
from scmg import g1
from infer_interrupt import SIL_RAW, spoken_tokens

UPPER = list(range(10 + 12, 10 + 29))          # waist + arms channels of the 39-d frame


class PostInterrupt:
    """causal inputs after the interruption (psync): the audio stream ended at frame n50 (absolute), the streamer clock is
    shifted by sh frames (latent restart). feat(Tk) / spk(Tk) take the RE-BASED Tk; feat rows = the streamer's feat rows
    (audio end: n50 - sh, off: unchanged) so the audio-final state is the one of the engine protocol (feat length)."""

    def __init__(self, ci, fresh, n50, sh, rows):
        self.ci, self.fresh, self.n50, self.sh, self.rows = ci, fresh, int(n50), int(sh), int(rows)
        self.Tpad = ci.Tpad; self.calls = []

    def feat(self, Tk):
        Ta = min(int(Tk) + self.sh, self.n50); self.calls.append(Ta)
        F = self.fresh.feat(Ta)                      # [Tpad, 89], rows >= Ta zero
        return F[self.sh:self.sh + self.rows]

    def spk(self, Tk):
        return self.ci.spk(min(int(Tk) + self.sh, self.n50))


def clip_seed(seed, mode, stem):
    """11-06 protocol: same definition as scripts/echog_g1/infer_test.py --seed-mode (DivBA worktree): 'run' = the run seed for
    every clip (old behaviour), 'per_clip' = (seed + crc32(stem)) mod 2^63. The interrupted run and its uninterrupted
    counterfactual of a clip use the same clip seed, so the pre-interruption frames stay bitwise paired."""
    import zlib
    if mode == 'run': return seed
    if mode == 'per_clip': return (seed + zlib.crc32(stem.encode())) % (2 ** 63)
    raise ValueError(f'seed mode {mode}')


class RetractStreamer(Streamer):
    def __init__(self, *args, t_int=None, keep_tok=None, audio='end', fm=None, fs=None, n_total=None, restart='chunk',
                 react_delay=0.04, retract=None, rest=None, tr=1.0, w=1.0, splice=0, w0=0., prefix=None, floor=0, guide=0.,
                 comp=None, diff=0., fresh_feat=None, **kw):
        super().__init__(*args, **kw)
        self.fresh_feat = fresh_feat      # psync: () -> a new CausalInputs of the clip (feat after the interruption)
        self.t_int, self.keep_tok, self.audio, self.n_total = t_int, keep_tok, audio, n_total
        self.sil = None if fm is None else torch.as_tensor((SIL_RAW - fm) / fs, device=self.dev).float()
        self.restart, self.delay, self.retract, self.rest, self.tr, self.w = restart, react_delay, retract, rest, tr, w
        self.applied = False; self.k_int = None; self.o = 0; self.prefix = []; self.l_r = None
        self.tgt = None; self._eps = {}; self._flow = None
        self.splice = splice; self._sp = None; self.w0 = w0
        if floor and not hasattr(self.m, 'floor_emb'): raise ValueError('--floor needs a model trained with cfg.floor')
        self.prefix_npz = prefix; self.floor = floor; self.fm_, self.fs_ = fm, fs
        self.guide = float(guide)          # 11-06 yield guidance (post-interruption only; 0 = off)
        self.comp, self.diff = comp, float(diff)   # 11-06 model-difference guidance: companion streamer of the reference model
        if self.guide and self.cfg_t is not None: raise ValueError('--yield-guide supports the joint CFG only (cfg_t None)')   # 11-06: causal prefix conditions, floor token     # latent restart: frames over which the jump at the splice is blended out
        if t_int is not None:
            self.l_r = int(np.ceil((t_int + react_delay) * 50 / STRIDE - 1e-9)) if restart == 'latent' else \
                int(np.ceil(t_int / (CHUNK_FRAMES / 50.) - 1e-9)) * CHUNK

    def due(self):
        """apply the interruption before the next tick? (latent restart: once latent l_r + 1 is committed, i.e. frames < 4 l_r
        have been emitted; chunk restart: at the first chunk starting >= t_i)"""
        if self.t_int is None or self.applied: return False
        if self.restart == 'latent': return len(self.lat) >= self.l_r + 1
        return self.k * CHUNK >= self.l_r

    def _apply(self):
        l_r = self.l_r; o = l_r % CHUNK if self.restart == 'latent' else 0
        n50 = int(round(self.t_int * 50))
        if self.causal is not None and self.prefix_npz is not None: raise ValueError('--prefix-feats is the engine-protocol path, not psync')
        if self.causal is not None and self.audio == 'silence': raise ValueError('psync: --audio silence not implemented (no silent PCM)')
        if self.prefix_npz is not None:   # 11-06: feat89 / CAM++ recomputed from the audio before t_i only (prefix_feats.py)
            pf = np.load(self.prefix_npz)
            fz = torch.as_tensor((pf['feat'].astype(np.float32) - self.fm_) / self.fs_, device=self.dev).float()
            if fz.shape[0] != n50: raise ValueError(f'prefix feat {fz.shape[0]} frames != {n50}')
            f = self.feat.clone(); f[:n50] = fz; self.feat = f   # frames >= t_i: handled by --audio below
            self.cond['spk'] = torch.as_tensor(pf['spk'], device=self.dev).float()
            self.cond['spk_valid'] = torch.as_tensor(float(pf['spk_valid']), device=self.dev)
        if self.floor:                    # floor state from the first post-interruption chunk on (models trained with cfg.floor)
            self.floor_k0, self.floor_val = None, int(self.floor)
        # audio after t_i (absolute frames), then the clock shift by o latents
        if self.audio == 'end':
            self.feat = self.feat[:n50]
        elif self.audio == 'silence':
            T = max(self.n_total, self.feat.shape[0])
            f = torch.zeros(T, self.feat.shape[1], device=self.dev); f[:min(n50, self.feat.shape[0])] = self.feat[:n50]
            f[n50:] = self.sil; self.feat = f
        elif self.audio == 'off':
            self.audio_off = True
        self.cond['tmask'] = self.cond['tmask'] & torch.as_tensor(self.keep_tok, device=self.dev)
        frames = list(self.frames)
        if self.restart == 'latent':
            frames = frames[:STRIDE * l_r]                     # emitted frames before the restart latent: kept as played
            self.lat = self.lat[:l_r]
        self.applied = True; self.k_int = self.k
        if self.retract: self._targets(frames)                 # needs the executed frames (absolute) before the shift
        if o:
            sh = o * STRIDE; ds = sh / 50.
            self.prefix = frames[:sh]; frames = frames[sh:]
            self.feat = self.feat[sh:]
            self.cond['ttime'] = self.cond['ttime'] - ds
            if self.arr_text is not None: self.arr_text = self.arr_text - ds
            if self.arr_lm is not None: self.arr_lm = self.arr_lm - ds
            self.lat = self.lat[o:]
        if self.causal is not None:       # psync: inputs of the audio before t_i only, on the re-based clock
            sh = o * STRIDE
            if self.text_mode in ('voiced', 'asr'): self._wt = self._wt - sh / 50.
            self.causal = PostInterrupt(self.causal, self.fresh_feat(), n50, sh, self.feat.shape[0])
        self.o = o
        self.frames = frames; self.n_emitted = len(frames)
        if self.floor: self.floor_k0 = self.k                   # chunk index in the (possibly re-based) clock
        if self.splice and self.restart == 'latent' and len(frames) >= 2:
            # the first new frame should continue the executed motion: x[-1] + (x[-1] - x[-2]); the difference is removed
            # with a weight falling smoothly from 1 to 0 over `splice` frames (only frames emitted after the restart change)
            self._sp = dict(i0=len(frames), pred=frames[-1] + (frames[-1] - frames[-2]), d=None)
        self.k = len(self.lat) // CHUNK
        assert len(self.lat) == self.k * CHUNK

    @torch.no_grad()
    def _targets(self, frames):
        """normalised target latents {absolute latent index: z} from the restart frame on"""
        F = torch.stack(frames).float()                         # [n, 39] physical, n = 4 l_r
        q0 = F[-1]; v0 = (F[-1] - F[-2]) * 50. if len(F) > 1 else torch.zeros_like(q0)
        n = 64 * 2; t = torch.arange(1, n + 1, device=self.dev).float()[:, None] / 50.
        rest = torch.as_tensor(self.rest, device=self.dev).float()
        tr = self.tr if self.retract not in ('tr', 'tr2', 'tr2h') else 1e-3     # tr*: target = the settled rest pose
        wn = 4.744 / max(tr, 1e-3); e = q0 - rest
        traj = q0[None].repeat(n, 1)
        up = (rest + (e + (v0 + wn * e) * t) * torch.exp(-wn * t))
        traj[:, UPPER] = up[:, UPPER]; traj[:, 6:10] = 0.
        back = min(len(F) // STRIDE, 16) * STRIDE               # 1.3 s of executed motion as encoder context
        seq = torch.cat([F[len(F) - back:], traj], 0)
        mu, _ = self.vae.encode(((seq - self.qm) / self.qs)[None])
        z = ((mu - self.m.lat_mean) / self.m.lat_std)[0]
        l0 = len(F) // STRIDE - back // STRIDE                  # first encoded latent (F = the frames emitted so far)
        t_settle = self.t_int + (self.tr if self.retract == 'tr' else self.tr - 0.08 - 1e-6 if self.retract in ('tr2', 'tr2h', 'tr2hd') else 0.)
        self.tgt = {l0 + i: z[i] for i in range(z.shape[0]) if l0 + i >= self.l_r and (l0 + i) * STRIDE / 50. >= t_settle - 1e-9}

    def _positions(self):
        """target-chunk positions (0..CHUNK-1) replaced in the chunk about to be generated, with their target latents"""
        a0 = self.o + self.k * CHUNK                            # absolute index of the chunk's first latent
        sel = dict(end=[CHUNK - 1], mid=[1, CHUNK - 1], all=list(range(CHUNK)), tr=list(range(CHUNK)), ramp=list(range(CHUNK)),
                   tr2=list(range(CHUNK)), tr2h=list(range(CHUNK)), tr2hd=list(range(CHUNK)))[self.retract]
        if self.retract in ('tr2', 'tr2h', 'tr2hd'):
            # 11-06: only the two latents ending at t_i + T_r (weights w/2, w: a short ramp onto the rest pose); tr2h also pins the
            # last latent of every later chunk to the rest pose (stay down)
            T = self.t_int + self.tr; out = {}
            for i in sel:
                a = a0 + i; ta = a * STRIDE / 50.
                if a not in self.tgt: continue
                if abs(ta - T) < 0.04: out[i] = (self.tgt[a], self.w)
                elif abs(ta - (T - 0.08)) < 0.04: out[i] = (self.tgt[a], 0.5 * self.w)
                elif self.retract in ('tr2h', 'tr2hd') and ta > T and i == CHUNK - 1: out[i] = (self.tgt[a], self.w)
            return out
        def wt(i):
            if self.retract != 'ramp': return self.w
            u = min(max(((a0 + i) * STRIDE / 50. - self.t_int) / self.tr, 0.), 1.)
            return self.w0 + (self.w - self.w0) * u * u * (3 - 2 * u)   # smoothstep w0 -> w over T_r
        return {i: (self.tgt[a0 + i], wt(i)) for i in sel if (a0 + i) in self.tgt}

    def _wrapped(self, pos):
        real = self.m.flow_velocity; ctx = self.m.cfg.ctx; eps = {}; done = {}
        def fv(xt, t, known, *a, **k):
            for i, (z, w) in pos.items():
                p = ctx + i; tv = float(t[0, p])
                if done.get(i) == tv: continue                    # CFG branches of the same step: replace once
                done[i] = tv
                if tv == 0.: eps[i] = xt[0, p].clone()            # first call of this flow run: the initial noise
                else: xt[0, p] = w * (tv * z + (1 - tv) * eps[i]) + (1 - w) * xt[0, p]
            return real(xt, t, known, *a, **k)
        return fv

    def _emit(self, final=False):
        if self._pend:                                          # committed value of the replaced positions
            base = len(self.lat) - CHUNK
            for i, (z, w) in self._pend.items(): self.lat[base + i] = w * z + (1 - w) * self.lat[base + i]
            self._pend = None
        fr = super()._emit(final)
        sp = self._sp
        if sp is not None and len(self.frames) > sp['i0']:
            if sp['d'] is None: sp['d'] = self.frames[sp['i0']] - sp['pred']
            for j in range(sp['i0'], min(len(self.frames), sp['i0'] + self.splice)):
                if j < self.n_emitted - len(fr): continue                  # corrected when it was emitted
                u = (j - sp['i0']) / self.splice; wgt = 0.5 * (1 + np.cos(np.pi * u))
                self.frames[j] = self.frames[j] - wgt * sp['d']
            if len(self.frames) >= sp['i0'] + self.splice: self._sp = None
        return fr

    _pend = None

    def _guided(self):
        """11-06 yield guidance: the planner runs a second time with the audio-stream bit of every window chunk set to
        'still streaming' (stream state - audio bit); on every flow step the CONDITIONAL velocity becomes
        v(audio ended) + s * (v(audio ended) - v(still streaming)) (the unconditional CFG branch is unchanged), i.e. CFG
        on the audio-end state. Called only for post-interruption ticks."""
        m, s = self.m, self.guide
        real_plan, real_fv = m.plan, m.flow_velocity; st = dict(n_plan=0, n_fv=0, h_spk=None)
        per_step = 2 if self.cfg_w != 1 else 1
        def plan(*a, **kw):
            h = real_plan(*a, **kw)
            st['n_plan'] += 1
            if st['n_plan'] == 1 and kw.get('sstate') is not None:            # the conditional planner call of the tick
                ss = kw['sstate']; st['h_spk'] = real_plan(*a, **dict(kw, sstate=ss - (ss % 2)))
            return h
        def fv(xt, t, known, h, *a, **kw):
            v = real_fv(xt, t, known, h, *a, **kw)
            cond = st['n_fv'] % per_step == 0; st['n_fv'] += 1
            if not cond or st['h_spk'] is None: return v
            hs = st['h_spk'][:, self._W - 1 - LOOK]
            v_spk = real_fv(xt, t, known, hs, *a, **kw)
            return v + s * (v - v_spk)
        return plan, fv

    def _capture(self):
        """Run the companion (reference model) up to its first Local-Flow call of this tick on the SAME committed latents
        and conditions; return the arguments of its conditional and unconditional velocity calls (everything after x, t,
        known). The companion never commits anything."""
        c = self.comp; c.lat = list(self.lat); c.k = self.k
        real = c.m.flow_velocity; cap = []
        class _Cap(Exception): pass
        def fv(xt, t, known, *a, **kw):
            cap.append((a, kw))
            if len(cap) == (2 if c.cfg_w != 1 else 1): raise _Cap()
            return real(xt, t, known, *a, **kw)
        c.m.flow_velocity = fv
        try: c.tick(None, None, None)
        except _Cap: pass
        finally: del c.m.flow_velocity
        return cap, real

    def _diffguided(self):
        """11-06 model-difference guidance: v = v_self + s (v_self - v_ref) on the CFG-combined velocities; CFG is linear, so
        each branch (conditional / unconditional) is replaced by (1 + s) v_self - s v_ref of that branch."""
        cap, real_ref = self._capture(); s = self.diff; real = self.m.flow_velocity
        per = len(cap); st = dict(n=0)
        def fv(xt, t, known, *a, **kw):
            v = real(xt, t, known, *a, **kw)
            ra, rk = cap[st['n'] % per]; st['n'] += 1
            return (1 + s) * v - s * real_ref(xt, t, known, *ra, **rk)
        return fv

    rec = None; plan_noise = 0.; _pg = None

    def tick(self, *a, **kw):
        """11-07 end head: with rec (a list) every post-interruption tick appends the conditional planner output of its chunk
        (read-only); plan_noise > 0 adds N(0, plan_noise) to the history latents of that planner call (label collection)."""
        if self.due(): self._apply()
        if not (self.applied and self.rec is not None): return self._tick_r(*a, **kw)
        real = self.m.plan; st = dict(n=0)
        def plan(lat, *pa, **pk):
            st['n'] += 1
            if st['n'] == 1 and self.plan_noise:
                if self._pg is None: self._pg = torch.Generator(device=self.dev).manual_seed(int(self.seed_) + 1)
                lat = lat + self.plan_noise * torch.randn(lat.shape, device=self.dev, generator=self._pg)
            h = real(lat, *pa, **pk)
            if st['n'] == 1: self.rec.append(dict(k=self.k, h=h[0, self._W - 1 - LOOK].float().cpu().numpy()))
            return h
        self.m.plan = plan
        try: return self._tick_r(*a, **kw)
        finally: del self.m.plan

    def _tick_r(self, *a, **kw):
        if self.applied and self.diff:
            if self.retract or self.guide: raise ValueError('--diff-guide is exclusive with --retract / --yield-guide')
            self.m.flow_velocity = self._diffguided()
            try: return super().tick(*a, **kw)
            finally: del self.m.flow_velocity
        if self.applied and self.guide:
            if self.retract: raise ValueError('--yield-guide and --retract are exclusive')
            self.m.plan, self.m.flow_velocity = self._guided()
            try: return super().tick(*a, **kw)
            finally: del self.m.plan; del self.m.flow_velocity
        if self.applied and self.retract and self.tgt:
            pos = self._positions()
            if pos:
                self._pend = pos; self.m.flow_velocity = self._wrapped(pos)
                try: return super().tick(*a, **kw)
                finally: del self.m.flow_velocity; self._pend = None
        return super().tick(*a, **kw)

    def all_frames(self):
        return self.prefix + list(self.frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--vae', required=True); ap.add_argument('--stats', required=True)
    ap.add_argument('--view', required=True); ap.add_argument('--text-cache', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='test'); ap.add_argument('--interrupts', default=None); ap.add_argument('--words', default=None)
    ap.add_argument('--stems', default=None); ap.add_argument('--no-interrupt', action='store_true')
    ap.add_argument('--audio', default='end', choices=['end', 'silence', 'off']); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--every', type=int, default=1, help='use every n-th clip of the interrupt list (dev screening subsets)')
    ap.add_argument('--restart', default='latent', choices=['latent', 'chunk']); ap.add_argument('--react-delay', type=float, default=0.04)
    ap.add_argument('--retract', default=None, choices=[None, 'end', 'mid', 'all', 'tr', 'ramp', 'tr2', 'tr2h', 'tr2hd'])
    ap.add_argument('--rest', default='R_sil'); ap.add_argument('--human', default=None, help='human_ref json (rest poses)')
    ap.add_argument('--tr', type=float, default=1.0); ap.add_argument('--w', type=float, default=1.0)
    ap.add_argument('--w0', type=float, default=0., help='ramp: weight at t_i (rises to --w at t_i + T_r)')
    ap.add_argument('--prefix-feats', default=None, help='prefix_feats.py dir: causal feat89 / CAM++ of the audio before t_i')
    ap.add_argument('--yield-guide', type=float, default=0., help='yield guidance scale s after the interruption (0 = off)')
    ap.add_argument('--diff-guide', type=float, default=0., help='model-difference guidance s: v + s (v - v_ref) after the interruption')
    ap.add_argument('--ref-ckpt', default=None, help='reference model of --diff-guide (e.g. the sf0 checkpoint)')
    ap.add_argument('--floor', type=int, default=0, help='floor state after the interruption (models with cfg.floor; 0 = off)')
    ap.add_argument('--splice', type=int, default=0, help='latent restart: blend the jump at the splice out over n 50 Hz frames (0 = off)')
    ap.add_argument('--seed-mode', default='run', choices=['run', 'per_clip'], help="clip noise seed: run (all clips share the run seed) | per_clip (seed + crc32(stem))")
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int, default=10); ap.add_argument('--cfg-w', type=float, default=2.5)
    ap.add_argument('--shard', default='0/1'); ap.add_argument('--protocol', default='engine', choices=['offline', 'engine', 'psync'])
    ap.add_argument('--text-causal', default=None); ap.add_argument('--x2-rate', type=float, default=15.0); ap.add_argument('--x2-lag', type=float, default=0.15)
    ap.add_argument('--wav-dir', default=None); ap.add_argument('--words-dir', default=None); ap.add_argument('--pstats', default=None)
    ap.add_argument('--campplus', default='/home/xr-25b7/bulk/models-ro/campplus'); ap.add_argument('--asr-lag', type=float, default=0.5)
    ap.add_argument('--record-plan', action='store_true', help='end head: save the post-interruption conditional planner outputs (<name>.plan.npz)')
    ap.add_argument('--plan-noise', type=float, default=0., help='end-head label collection: N(0, s) on the planner history latents after the interruption')
    ap.add_argument('--spk-refresh', default='tick', choices=['tick', 'grades']); ap.add_argument('--causal-feat', default='prefix', choices=['prefix', 'inc'])
    a = ap.parse_args()
    if EMBODIMENT != 'g1': raise SystemExit('run with SCMG_J=39')
    eng = a.protocol in ('engine', 'psync'); cin = a.protocol == 'psync'
    tm = {'engine': 'x2', 'psync': 'voiced'}.get(a.protocol, 'final'); hz = '2' if eng else 'inf'
    if cin and not (a.wav_dir and a.words_dir and a.pstats): raise SystemExit('psync needs --wav-dir, --words-dir and --pstats')
    if cin and (a.prefix_feats or a.diff_guide or a.floor): raise SystemExit('psync: --prefix-feats / --diff-guide / --floor not supported')
    H = INF if hz == 'inf' else int(hz) * U80; dt = eng
    os.environ.pop('SCMG_TEXT_CAUSAL', None)
    if a.text_causal: os.environ['SCMG_TEXT_CAUSAL'] = a.text_causal
    if tm in ('x2', 'voiced') and not a.text_causal: raise SystemExit(f'{a.protocol} protocol needs --text-causal')
    proto = None if not eng else dict(protocol=a.protocol, text_mode=tm, horizon_80ms=hz, x2_rate=a.x2_rate, x2_lag=a.x2_lag,
                                      deploy_timing=dt, text_causal=a.text_causal,
                                      **({} if not cin else dict(causal_inputs=True, asr_lag=a.asr_lag, pstats=a.pstats, words_dir=a.words_dir)))
    dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stats = json.loads(Path(a.stats).read_text())
    m, vae = load(a.ckpt, a.vae, dev)
    mref = None
    if a.diff_guide:
        mref, _ = load(a.ref_ckpt, a.vae, dev)
        if not (torch.equal(mref.lat_mean, m.lat_mean) and torch.equal(mref.lat_std, m.lat_std)): raise ValueError('latent normalisation differs')
    view = View(a.view, a.split, a.text_cache, stats)
    if cin:
        from scmg import causal_feat as CF
        tokz = CF.e5_tokenizer(); enc = CF.speaker_encoder(a.campplus); pst = CF.load_pstats(a.pstats)
    qm = np.asarray(stats['q_mean'], np.float32)
    fm, fs = np.asarray(stats['f_mean'], np.float32), np.asarray(stats['f_std'], np.float32)
    idx = {c['items'][0]['id']: i for i, c in enumerate(view.clips)}
    if a.no_interrupt:
        jobs = [(s, None) for s in Path(a.stems).read_text().split()]
    else:
        rows = json.loads(Path(a.interrupts).read_text())['interrupts']
        keep = sorted({r['stem'] for r in rows})[::a.every]; ks = set(keep)
        jobs = [(r['stem'], r) for r in rows if r['stem'] in ks]
        words = {r['stem']: r['words'] for r in json.loads(Path(a.words).read_text())} if a.words else {}   # psync: word times from --words-dir
    rest39 = None
    if a.retract:
        Hh = json.loads(Path(a.human).read_text()); rest39 = np.zeros(39, np.float32); rest39[10:] = Hh['rest'][a.rest]['joints']
    if a.limit: jobs = jobs[:a.limit]
    si, sn = (int(x) for x in a.shard.split('/')); jobs = jobs[si::sn]
    t0 = time.time(); log = []
    for k, (stem, r) in enumerate(jobs):
        o = out / (f'{stem}.pt' if r is None else f"{stem}@{r['frac']:.1f}.pt")
        if o.exists(): continue
        ci = idx[stem]; c = view.clips[ci]; it = c['items'][0]
        L = c['frames']; n30 = int(it['echog']['frames30']); nch = -(-L // CHUNK_FRAMES)
        ds = WindowDataset(view, nch * CHUNK_FRAMES, max_tokens=256)
        s = ds.sample(ci, 0, np.random.default_rng(ci))
        s['ref'] = np.zeros_like(s['ref']); s['ref_valid'] = np.float32(0.)
        cond = {kk: s[kk] for kk in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm',
                                     'titime', 'tseg', 'tiseg', 'tiend', 'text_c') if kk in s}
        kw = {} if proto is None else dict(text_mode=tm, x2_rate=a.x2_rate, x2_lag=a.x2_lag, deploy_timing=dt)
        mkw = {}; cw = None
        if cin:                                                  # = infer_test.py --protocol psync (same calls, same order)
            cw = CF.clip_words(Path(a.words_dir) / f'{stem}.json', L)
            kn = view.text['index'].get(stem, [0, 0])[1] if not it.get('text_null') else 0
            if int(np.asarray(s['tmask']).sum()) != kn: raise ValueError(f'{stem}: tmask rows != cache rows')
            cond.update(CF.text_extras(it, cw, tokz, kn, len(s['tmask'])))
            mkw['asr_lag'] = a.asr_lag
            pcm16, wave = CF.clip_audio(Path(a.wav_dir) / f'{stem}.wav')
            mk = lambda: CF.CausalInputs(pcm16, wave, cw, it['lang'], L, pst, view.fm, view.fs, enc, Tpad=len(s['feat']), spk=True,
                                         spk_refresh=a.spk_refresh, feat_mode=a.causal_feat)
            mkw['causal_inputs'] = mk()
            if r is not None: mkw['fresh_feat'] = mk
        ikw = {}
        if r is not None:
            t_i = r['t_i']; n30o = r['f_i'] + 64 + 2
            n_run = -(-int(np.ceil((n30o - 1) / 30. * 50 + 2)) // CHUNK_FRAMES)
            if cin:                                              # psync: tokens whose words all ended by t_i (voiced rule, no look-ahead)
                keep = CF.token_visible(CF.visible_words(np.asarray(cond['wtime'], np.float64).reshape(-1, 2), t_i, 0., 'voiced',
                                                         a.x2_rate, a.x2_lag, a.asr_lag), np.asarray(cond['tword'], np.int64)) & np.asarray(s['tmask'], bool)
            else:
                keep = spoken_tokens(s['ttime'], s['tmask'], words[stem], t_i)
            ikw = dict(t_int=t_i, keep_tok=keep, audio=a.audio, fm=fm, fs=fs, n_total=n_run * CHUNK_FRAMES, restart=a.restart,
                       react_delay=a.react_delay, retract=a.retract, rest=rest39, tr=a.tr, w=a.w, w0=a.w0, splice=a.splice, floor=a.floor, guide=a.yield_guide,
                       prefix=(f"{a.prefix_feats}/{stem}@{r['frac']:.1f}.npz" if a.prefix_feats else None))
        else:
            n_run = nch
        comp = None
        if mref is not None and r is not None:      # companion: same inputs / interruption handling, reference model, no guidance
            comp = RetractStreamer(mref, vae, s['feat'], {k_: (v_.copy() if hasattr(v_, 'copy') else v_) for k_, v_ in cond.items()}, stats,
                                   H=H, steps=a.steps, seed=clip_seed(a.seed, a.seed_mode, stem), standby=qm, ctx_t=mref.ctx_t, cfg_w=a.cfg_w, cfg_t=None, noise_scale=1.0,
                                   **kw, **dict(ikw, retract=None, guide=0., floor=0))
        st = RetractStreamer(m, vae, s['feat'], cond, stats, H=H, steps=a.steps, seed=clip_seed(a.seed, a.seed_mode, stem), standby=qm, ctx_t=m.ctx_t,
                             cfg_w=a.cfg_w, cfg_t=None, noise_scale=1.0, comp=comp, diff=a.diff_guide if r is not None else 0., **kw, **ikw, **mkw)
        if a.record_plan and r is not None: st.rec = []; st.plan_noise = a.plan_noise; st.seed_ = clip_seed(a.seed, a.seed_mode, stem)
        if r is None:
            for i in range(n_run):                               # == scmg.infer.generate (robot=None)
                st.tick(None, None, None)
                if i == n_run - 1: st.flush()
        else:                                                    # until n_run chunks of absolute time are covered
            while (st.o + st.k * CHUNK) < n_run * CHUNK: st.tick(None, None, None)
            st.flush()
        cmd = torch.stack(st.all_frames()).cpu().numpy()
        if r is None:
            if cmd.shape[0] < L or cmd.shape[1] != J or not np.isfinite(cmd).all(): raise ValueError(f'{stem}: generated {cmd.shape}')
            q50 = cmd[:L].astype(np.float64); r30 = g1.to30(q50, n30).astype(np.float32)
            gen = dict(ckpt=str(a.ckpt), seed=a.seed, **({'seed_mode': a.seed_mode, 'clip_seed': clip_seed(a.seed, a.seed_mode, stem)} if a.seed_mode != 'run' else {}), steps=a.steps, cfg_w=a.cfg_w, cfg_t=None, frames50=L, noise_scale=1.0, no_spk=False,
                       smooth=None, smooth_legs=None, resample='scmg/g1.py to30 (cubic / heading spline / RotationSpline)',
                       **({} if not cin else dict(spk_refresh=a.spk_refresh, causal_feat=a.causal_feat)),
                       **({} if proto is None else dict(stream=proto)))
            meta = dict(real_num_frames=n30)
        else:
            q50 = cmd.astype(np.float64); r30 = g1.to30(q50, n30o).astype(np.float32)
            gen = dict(ckpt=str(a.ckpt), seed=a.seed, seed_mode=a.seed_mode, clip_seed=clip_seed(a.seed, a.seed_mode, stem), steps=a.steps, cfg_w=a.cfg_w, stream=proto, interrupt=dict(
                t_i=t_i, f_i=r['f_i'], frac=r['frac'], audio=a.audio, restart=a.restart, react_delay=a.react_delay, l_r=st.l_r,
                o=st.o, k_int=st.k_int, retract=a.retract, rest=a.rest, tr=a.tr, w=a.w, w0=a.w0, splice=a.splice, floor=a.floor, prefix_feats=a.prefix_feats, yield_guide=a.yield_guide, diff_guide=a.diff_guide, ref_ckpt=a.ref_ckpt, n_targets=len(st.tgt or {}),
                tokens_kept=int(keep.sum()), tokens_total=int(np.asarray(s['tmask']).sum())),
                       **({} if not cin else dict(spk_refresh=a.spk_refresh, causal_feat=a.causal_feat, post_feat_calls=sorted(set(st.causal.calls)) if st.applied else None)))
            meta = dict(real_num_frames=n30o, clip_frames30=n30)
            np.save(o.with_suffix('.q50.npy'), cmd.astype(np.float32))
            if st.rec is not None:                               # chunk k of the (re-based) clock ends at ((k + 1) CHUNK + o) latents
                ks = np.array([x['k'] for x in st.rec], np.int64)
                np.savez(o.with_suffix('.plan.npz'), k=ks, h=np.stack([x['h'] for x in st.rec]).astype(np.float32), o=st.o, t_i=t_i,
                         t_end=((ks + 1) * CHUNK + st.o) * STRIDE / 50., plan_noise=a.plan_noise)
        torch.save(dict(robot_repr=torch.from_numpy(r30), fps=30.0, representation_units='physical',
                        representation_schema='projecthermes-g1-39d-standard-v2', source=stem, generator=gen, **meta), o.with_suffix('.pt.tmp'))
        os.replace(o.with_suffix('.pt.tmp'), o)
        log.append(dict(stem=stem, frac=None if r is None else r['frac'], l_r=st.l_r, o=st.o))
        if k % 50 == 0: print(f'{k + 1}/{len(jobs)} {o.name} l_r={st.l_r} o={st.o} {time.time() - t0:.0f}s', flush=True)
    with (out / f'GEN_LOG.{si}of{sn}.jsonl').open('a') as f:
        for x in log: f.write(json.dumps(x) + '\n')
    print('RETRACT_DONE', len(jobs), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

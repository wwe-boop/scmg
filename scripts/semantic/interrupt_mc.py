"""Interruption test of the mask-conditioned Local Flow (11-07 item 5b, coordinator): at the interruption t_i the rest pose is
placed as FLAGGED KNOWN latents from t_i + T_r on (Streamer inpaint weight 1 -> cfg.mask_cond f_mask core) and the model
generates the bridge itself. Continuation = the B3c protocol of scmg-wt-floor/scripts/interrupt/infer_retract.py, re-implemented
here on this branch's Streamer (engine protocol, chunk restart: chunks starting before t_i are unchanged; audio ends at t_i; only
tokens of words spoken by t_i stay visible; per_clip seed). Optional --prefix-feats: feat89 / CAM++ from the audio before t_i.
Rest target (as infer_retract _targets 'tr'): the last emitted frame with waist + arms = the human rest pose (--rest, human_ref
json) and base velocity / yaw-rate channels 0, held; encoded by the VAE after 1.28 s of the executed motion, normalised with the
generator's latent statistics.
  --tr none : no insertion (the model's own B3c continuation, the same-model reference row)
  --no-interrupt --stems S : full-clip generation (the uninterrupted reference the evaluator needs)
Outputs as infer_retract.py: <out>/<stem>@<frac>.pt (30 fps, frames 0 .. f_i + 66) + .q50.npy; <stem>.pt for --no-interrupt.
"""
import argparse, json, os, sys, time, zlib
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg.common import J, CHUNK, CHUNK_FRAMES, STRIDE, EMBODIMENT, U80
from scmg.data import View, WindowDataset
from scmg.evaluate import load
from scmg.infer import Streamer
from scmg.model import INF
from scmg import g1

UPPER = list(range(10 + 12, 10 + 29))


def clip_seed(seed, mode, stem):
    if mode == 'run': return seed
    return (seed + zlib.crc32(stem.encode())) % (2 ** 63)


def spoken_tokens(ttime, tmask, words, t_i):
    """bool [K]: token visible after the interruption (its word ended by t_i) -- same rule as infer_interrupt.py."""
    keep = np.zeros(len(ttime), bool); prev = True
    for k in range(len(ttime)):
        if not tmask[k]: continue
        t = ttime[k]
        if not np.isfinite(t): keep[k] = prev; continue
        w = [e for s, e in words if s - 0.02 <= t < e]
        prev = keep[k] = (w[0] <= t_i + 1e-6) if w else (t < t_i)
    return keep


class IntStreamer(Streamer):
    def __init__(self, *a, t_int=None, keep_tok=None, tr=None, rest=None, prefix=None, fm=None, fs=None, **kw):
        super().__init__(*a, **kw)
        self.t_int, self.keep_tok, self.tr, self.rest, self.prefix_npz, self.fm_, self.fs_ = t_int, keep_tok, tr, rest, prefix, fm, fs
        self.applied = False; self.k_int = None
        self.l_r = None if t_int is None else int(np.ceil(t_int / (CHUNK_FRAMES / 50.) - 1e-9)) * CHUNK

    def tick(self, *a, **kw):
        if self.t_int is not None and not self.applied and self.k * CHUNK >= self.l_r: self._apply()
        return super().tick(*a, **kw)

    @torch.no_grad()
    def _apply(self):
        n50 = int(round(self.t_int * 50))
        if self.prefix_npz is not None:
            pf = np.load(self.prefix_npz)
            fz = torch.as_tensor((pf['feat'].astype(np.float32) - self.fm_) / self.fs_, device=self.dev).float()
            f = self.feat.clone(); f[:n50] = fz[:n50]; self.feat = f
            self.cond['spk'] = torch.as_tensor(pf['spk'], device=self.dev).float()
            self.cond['spk_valid'] = torch.as_tensor(float(pf['spk_valid']), device=self.dev)
        self.feat = self.feat[:n50]                                            # audio ends at t_i
        self.cond['tmask'] = self.cond['tmask'] & torch.as_tensor(self.keep_tok, device=self.dev)
        self.applied = True; self.k_int = self.k
        if self.tr is None: return
        F = torch.stack(self.frames).float(); q0 = F[-1]
        tgt = q0.clone(); tgt[UPPER] = torch.as_tensor(self.rest, device=self.dev).float()[UPPER]; tgt[6:10] = 0.
        hist = F[-min(64, len(F)):]; nr = 64 * 3
        x = torch.cat([hist, tgt[None].expand(nr, -1)], 0)
        x = torch.cat([x, x[-1:].expand((-len(x)) % 4, -1)], 0)
        mu, _ = self.vae.encode(((x - self.qm) / self.qs)[None])
        lat = ((mu[0] - self.m.lat_mean.reshape(-1)) / self.m.lat_std.reshape(-1)).float()
        # emitted frames lag the committed latents by one latent: frame f belongs to latent f // 4, the last emitted frame to
        # latent len(lat) - 2 -> latent j of this encoding sits at global latent len(lat) - 1 - len(hist) / 4 + j
        g0 = len(self.lat) - 1 - len(hist) // STRIDE
        n_lat = g0 + len(lat) + 64
        L = torch.zeros(n_lat, lat.shape[1], device=self.dev); w = torch.zeros(n_lat, device=self.dev)
        L[g0:g0 + len(lat)] = lat
        l0 = int(np.ceil((self.t_int + self.tr) * 50 / STRIDE - 1e-9))
        L[g0 + len(lat):] = lat[-1]; w[l0:] = 1.0
        self.inpaint = dict(lat=L, w=w, g=torch.Generator(device=self.dev).manual_seed(7919))


def main():
    ap = argparse.ArgumentParser()
    for k in ('ckpt', 'vae', 'stats', 'view', 'text-cache', 'out'): ap.add_argument('--' + k, required=True)
    ap.add_argument('--split', default='val'); ap.add_argument('--interrupts'); ap.add_argument('--words'); ap.add_argument('--stems')
    ap.add_argument('--no-interrupt', action='store_true'); ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--tr', default='none', help="seconds after t_i from which the rest pose is known, or 'none'")
    ap.add_argument('--human', default=None); ap.add_argument('--rest', default='R_sil'); ap.add_argument('--prefix-feats', default=None)
    ap.add_argument('--text-causal', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seed-mode', default='per_clip', choices=['run', 'per_clip']); ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--cfg-w', type=float, default=2.5); ap.add_argument('--shard', default='0/1'); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    if EMBODIMENT != 'g1': raise SystemExit('run with SCMG_J=39')
    os.environ['SCMG_TEXT_CAUSAL'] = a.text_causal
    H = 2 * U80; kw = dict(text_mode='x2', x2_rate=15.0, x2_lag=0.15, deploy_timing=True)
    proto = dict(protocol='engine', text_mode='x2', horizon_80ms='2', x2_rate=15.0, x2_lag=0.15, deploy_timing=True, text_causal=a.text_causal)
    dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stats = json.loads(Path(a.stats).read_text()); m, vae = load(a.ckpt, a.vae, dev)
    view = View(a.view, a.split, a.text_cache, stats)
    qm = np.asarray(stats['q_mean'], np.float32); fm, fs = np.asarray(stats['f_mean'], np.float32), np.asarray(stats['f_std'], np.float32)
    idx = {c['items'][0]['id']: i for i, c in enumerate(view.clips)}
    tr = None if a.tr == 'none' else float(a.tr)
    rest39 = None
    if tr is not None:
        Hh = json.loads(Path(a.human).read_text()); rest39 = np.zeros(39, np.float32); rest39[10:] = Hh['rest'][a.rest]['joints']
    if a.no_interrupt: jobs = [(s, None) for s in Path(a.stems).read_text().split()]
    else:
        rows = json.loads(Path(a.interrupts).read_text())['interrupts']
        keep_st = set(sorted({r['stem'] for r in rows})[::a.every]); jobs = [(r['stem'], r) for r in rows if r['stem'] in keep_st]
        words = {r['stem']: r['words'] for r in json.loads(Path(a.words).read_text()) if r['stem'] in keep_st}
    if a.limit: jobs = jobs[:a.limit]
    si, sn = (int(x) for x in a.shard.split('/')); jobs = jobs[si::sn]; t0 = time.time()
    for k, (stem, r) in enumerate(jobs):
        o = out / (f'{stem}.pt' if r is None else f"{stem}@{r['frac']:.1f}.pt")
        if o.exists(): continue
        ci = idx[stem]; c = view.clips[ci]; it = c['items'][0]
        L = c['frames']; n30 = int(it['echog']['frames30']); nch = -(-L // CHUNK_FRAMES)
        s = WindowDataset(view, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
        s['ref'] = np.zeros_like(s['ref']); s['ref_valid'] = np.float32(0.)
        cond = {kk: s[kk] for kk in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm',
                                     'titime', 'tseg', 'tiseg', 'tiend', 'text_c') if kk in s}
        ikw = {}
        if r is not None:
            t_i = r['t_i']; n30o = r['f_i'] + 64 + 2; n_run = -(-int(np.ceil((n30o - 1) / 30. * 50 + 2)) // CHUNK_FRAMES)
            ikw = dict(t_int=t_i, keep_tok=spoken_tokens(s['ttime'], s['tmask'], words[stem], t_i), tr=tr, rest=rest39, fm=fm, fs=fs,
                       prefix=(f"{a.prefix_feats}/{stem}@{r['frac']:.1f}.npz" if a.prefix_feats else None))
        else: n_run = nch
        st = IntStreamer(m, vae, s['feat'], cond, stats, H=H, steps=a.steps, seed=clip_seed(a.seed, a.seed_mode, stem), standby=qm,
                         ctx_t=m.ctx_t, cfg_w=a.cfg_w, noise_scale=1.0, **kw, **ikw)
        with torch.no_grad():
            for i in range(n_run):
                st.tick(None, None, None)
            st.flush()
        cmd = torch.stack(st.frames).cpu().numpy()
        if r is None:
            q50 = cmd[:L].astype(np.float64); r30 = g1.to30(q50, n30).astype(np.float32); meta = dict(real_num_frames=n30)
            gen = dict(ckpt=a.ckpt, seed=a.seed, seed_mode=a.seed_mode, clip_seed=clip_seed(a.seed, a.seed_mode, stem), stream=proto)
        else:
            q50 = cmd.astype(np.float64); r30 = g1.to30(q50, n30o).astype(np.float32); meta = dict(real_num_frames=n30o, clip_frames30=n30)
            gen = dict(ckpt=a.ckpt, seed=a.seed, seed_mode=a.seed_mode, clip_seed=clip_seed(a.seed, a.seed_mode, stem), stream=proto,
                       interrupt=dict(t_i=t_i, f_i=r['f_i'], frac=r['frac'], audio='end', restart='chunk', l_r=st.l_r, o=0, k_int=st.k_int,
                                      retract='mask_rest' if tr is not None else None, rest=a.rest, tr=tr, prefix_feats=a.prefix_feats))
            np.save(o.with_suffix('.q50.npy'), cmd.astype(np.float32))
        torch.save(dict(robot_repr=torch.from_numpy(r30), fps=30.0, representation_units='physical',
                        representation_schema='projecthermes-g1-39d-standard-v2', source=stem, generator=gen, **meta), o.with_suffix('.pt.tmp'))
        os.replace(o.with_suffix('.pt.tmp'), o)
        if k % 50 == 0: print(f'{k + 1}/{len(jobs)} {o.name} {time.time() - t0:.0f}s', flush=True)
    print('INTERRUPT_MC_DONE', len(jobs), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

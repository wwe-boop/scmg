"""B3 of the G1 interruption benchmark (11-05): SCMG keeps streaming after an interruption at t_i.

Same model / sampler / conditioning as scripts/echog_g1/infer_test.py (identical code path; see --no-interrupt), plus an
interruption event handled at chunk granularity (scmg/infer.py Streamer, 320 ms chunks, committed motion is never changed):
  * chunks whose start time is < t_i were generated before the interruption arrived: unchanged (they play out, so the
    reaction cannot start before the next chunk boundary: 0-320 ms + the decoder's 80 ms look-ahead);
  * from the first chunk starting at >= t_i on, the streamer sees
      audio : --audio end     -> the audio stream ends at t_i (feat truncated: frames >= t_i do not exist, exactly the
                                 end-of-utterance convention of the engine; the stream state marks the audio as complete)
              --audio silence -> frames >= t_i = feat89 of digital silence (log-mel log(1e-6), prosody 0, speech flag 0)
      text  : only tokens of words already spoken (word end <= t_i, ECHO-G word annotations; untimed tokens follow the
              previous timed token) stay visible (tmask), later words are withdrawn;
  * generation continues until t_i + 2.13 s (+ margin) is covered; output = 30 fps robot_repr from frame 0 to
    f_i + 64 (+2) frames (<stem>@<frac>.pt) and the 50 Hz frames (.q50.npy).
--no-interrupt: the unmodified full-clip generation, written exactly as infer_test.py writes it (<stem>.pt), for the
bitwise identity check against infer_test.py.
    SCMG_J=39 python scripts/interrupt/infer_interrupt.py --ckpt C --vae V --stats S --view VIEW/test --text-cache T
        --text-causal TC --protocol engine --interrupts interrupts_test.json --words utterances.json --out D [--shard i/n]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg.common import J, CHUNK_FRAMES, EMBODIMENT, U80
from scmg.data import View, WindowDataset
from scmg.evaluate import load
from scmg.infer import Streamer
from scmg.model import INF
from scmg import g1

SIL_RAW = np.concatenate([np.full(80, np.log(1e-6)), np.zeros(9)]).astype(np.float32)   # feat89 of digital silence


class InterruptStreamer(Streamer):
    def __init__(self, *args, t_int=None, keep_tok=None, audio='end', fm=None, fs=None, n_total=None, **kw):
        super().__init__(*args, **kw)
        self.t_int, self.keep_tok, self.audio, self.applied = t_int, keep_tok, audio, False
        # silence row, normalised like every feat row (the speech flag column has f_mean 0 / f_std 1: stays raw 0)
        self.sil = None if fm is None else torch.as_tensor((SIL_RAW - fm) / fs, device=self.dev).float()
        self.n_total = n_total
        self.k_int = None

    def _apply(self):
        n50 = int(round(self.t_int * 50))
        if self.audio == 'end':
            self.feat = self.feat[:n50]
        else:
            T = max(self.n_total, self.feat.shape[0])
            f = torch.zeros(T, self.feat.shape[1], device=self.dev); f[:min(n50, self.feat.shape[0])] = self.feat[:n50]
            f[n50:] = self.sil; self.feat = f
        self.cond['tmask'] = self.cond['tmask'] & torch.as_tensor(self.keep_tok, device=self.dev)
        self.applied = True; self.k_int = self.k

    def tick(self, *a, **kw):
        if self.t_int is not None and not self.applied and self.k * CHUNK_FRAMES / 50. >= self.t_int - 1e-9: self._apply()
        return super().tick(*a, **kw)


def spoken_tokens(ttime, tmask, words, t_i):
    """bool [K]: token visible after the interruption (its word ended by t_i)."""
    keep = np.zeros(len(ttime), bool); prev = True
    for k in range(len(ttime)):
        if not tmask[k]: continue
        t = ttime[k]
        if not np.isfinite(t): keep[k] = prev; continue
        w = [e for s, e in words if s - 0.02 <= t < e]
        prev = keep[k] = (w[0] <= t_i + 1e-6) if w else (t < t_i)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--vae', required=True); ap.add_argument('--stats', required=True)
    ap.add_argument('--view', required=True); ap.add_argument('--text-cache', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='test'); ap.add_argument('--interrupts', default=None); ap.add_argument('--words', default=None)
    ap.add_argument('--stems', default=None, help='--no-interrupt: stem list'); ap.add_argument('--no-interrupt', action='store_true')
    ap.add_argument('--audio', default='end', choices=['end', 'silence']); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int, default=10); ap.add_argument('--cfg-w', type=float, default=2.5)
    ap.add_argument('--shard', default='0/1'); ap.add_argument('--protocol', default='engine', choices=['offline', 'engine'])
    ap.add_argument('--text-causal', default=None); ap.add_argument('--x2-rate', type=float, default=15.0); ap.add_argument('--x2-lag', type=float, default=0.15)
    a = ap.parse_args()
    if EMBODIMENT != 'g1': raise SystemExit('run with SCMG_J=39')
    eng = a.protocol == 'engine'; tm = 'x2' if eng else 'final'; hz = '2' if eng else 'inf'
    H = INF if hz == 'inf' else int(hz) * U80; dt = eng
    os.environ.pop('SCMG_TEXT_CAUSAL', None)
    if a.text_causal: os.environ['SCMG_TEXT_CAUSAL'] = a.text_causal
    if tm == 'x2' and not a.text_causal: raise SystemExit('engine protocol needs --text-causal')
    proto = None if not eng else dict(protocol=a.protocol, text_mode=tm, horizon_80ms=hz, x2_rate=a.x2_rate, x2_lag=a.x2_lag,
                                      deploy_timing=dt, text_causal=a.text_causal)
    dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stats = json.loads(Path(a.stats).read_text())
    m, vae = load(a.ckpt, a.vae, dev)
    view = View(a.view, a.split, a.text_cache, stats)
    qm = np.asarray(stats['q_mean'], np.float32)
    fm, fs = np.asarray(stats['f_mean'], np.float32), np.asarray(stats['f_std'], np.float32)
    idx = {c['items'][0]['id']: i for i, c in enumerate(view.clips)}
    if a.no_interrupt:
        jobs = [(s, None) for s in Path(a.stems).read_text().split()]
    else:
        jobs = [(r['stem'], r) for r in json.loads(Path(a.interrupts).read_text())['interrupts']]
        words = {r['stem']: r['words'] for r in json.loads(Path(a.words).read_text())}
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
        ikw = {}
        if r is not None:
            t_i = r['t_i']; n30o = r['f_i'] + 64 + 2
            n_run = -(-int(np.ceil((n30o - 1) / 30. * 50 + 2)) // CHUNK_FRAMES)          # chunks covering n30o frames @30
            keep = spoken_tokens(s['ttime'], s['tmask'], words[stem], t_i)
            ikw = dict(t_int=t_i, keep_tok=keep, audio=a.audio, fm=fm, fs=fs, n_total=n_run * CHUNK_FRAMES)
        else:
            n_run = nch
        st = InterruptStreamer(m, vae, s['feat'], cond, stats, H=H, steps=a.steps, seed=a.seed, standby=qm, ctx_t=m.ctx_t,
                               cfg_w=a.cfg_w, cfg_t=None, noise_scale=1.0, **kw, **ikw)
        for i in range(n_run):                                   # == scmg.infer.generate (robot=None)
            st.tick(None, None, None)
            if i == n_run - 1: st.flush()
        cmd = torch.stack(st.frames).cpu().numpy()
        if r is None:
            if cmd.shape[0] < L or cmd.shape[1] != J or not np.isfinite(cmd).all(): raise ValueError(f'{stem}: generated {cmd.shape}')
            q50 = cmd[:L].astype(np.float64); r30 = g1.to30(q50, n30).astype(np.float32)
            gen = dict(ckpt=str(a.ckpt), seed=a.seed, steps=a.steps, cfg_w=a.cfg_w, cfg_t=None, frames50=L, noise_scale=1.0, no_spk=False,
                       smooth=None, smooth_legs=None, resample='scmg/g1.py to30 (cubic / heading spline / RotationSpline)',
                       **({} if proto is None else dict(stream=proto)))
            meta = dict(real_num_frames=n30)
        else:
            q50 = cmd.astype(np.float64); r30 = g1.to30(q50, n30o).astype(np.float32)
            gen = dict(ckpt=str(a.ckpt), seed=a.seed, steps=a.steps, cfg_w=a.cfg_w, stream=proto, interrupt=dict(
                t_i=t_i, f_i=r['f_i'], frac=r['frac'], audio=a.audio, k_int=st.k_int, chunks=n_run,
                tokens_kept=int(keep.sum()), tokens_total=int(np.asarray(s['tmask']).sum())))
            meta = dict(real_num_frames=n30o, clip_frames30=n30)
            np.save(o.with_suffix('.q50.npy'), cmd.astype(np.float32))
        torch.save(dict(robot_repr=torch.from_numpy(r30), fps=30.0, representation_units='physical',
                        representation_schema='projecthermes-g1-39d-standard-v2', source=stem, generator=gen, **meta), o.with_suffix('.pt.tmp'))
        os.replace(o.with_suffix('.pt.tmp'), o)
        log.append(dict(stem=stem, frac=None if r is None else r['frac'], k_int=st.k_int, chunks=n_run))
        if k % 20 == 0: print(f'{k + 1}/{len(jobs)} {o.name} chunks={n_run} k_int={st.k_int} {time.time() - t0:.0f}s', flush=True)
    with (out / f'GEN_LOG.{si}of{sn}.jsonl').open('a') as f:
        for x in log: f.write(json.dumps(x) + '\n')
    print('INTERRUPT_DONE', len(jobs), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

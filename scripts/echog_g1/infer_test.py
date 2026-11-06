"""SCMG on the ECHO-G benchmark clips (val_common = echog-g1-v1/test): generate 50 Hz motion offline (full audio + text
context, as the ECHO-G benchmark), convert to 30 FPS with exactly the official aligned frame count, and write one
evaluator-ready file per stem: <out>/<stem>.pt = {robot_repr [n30, 39] float32 physical, real_num_frames, fps 30.0,
representation_units 'physical', representation_schema}. Run with SCMG_J=39.
Conditioning: audio feat89 + e5 word tokens (+ token times) + CAM++ vector of the clip's own audio; NO reference motion
(ref_valid = 0: val_common speakers are absent from train_drop, and test GT must not leak into a condition).
    SCMG_J=39 python scripts/echog_g1/infer_test.py --ckpt base/best.pt --vae vae/final.pt --stats STATS --view VIEW/test
        --text-cache TEXT_TEST --out preds/seed_000 [--stems list.txt] [--n 20] [--seed 0] [--steps 10] [--cfg-w 1]
Protocols (11-04): --protocol offline (default, unchanged: final text, H = inf) | engine = the EX002 engine protocol of
scripts/baseline_eval.sh / scmg.evaluate: text streams word by word and EARLY (x2: rate 15 tok/s, lag 0.15 s) with the
prefix-causal e5 features (--text-causal = the G1 causal cache of the SAME text cache, e.g. text-e5-causal-echog-g1-v1-test),
deploy timing (token times only once voiced), audio look-ahead H = 2 x 80 ms = 160 ms past the chunk start. Individual
switches (--text-mode / --horizon / --x2-rate / --x2-lag / --deploy-timing) override the protocol preset.
11-05 causal protocols (design v5 section 3): --protocol psync = engine but text_mode voiced (a word is visible once it has
been spoken within the arrived audio, = ECHO-G stream_core voiced) | pasr = voiced + 0.5 s recognition lag; both with causal
inputs by default (--causal-inputs: feat89 recomputed every tick from the arrived PCM prefix with fixed prosody statistics
--pstats and committed beats, CAM++ of the PCM prefix, empty embedding < 1 s; scmg/causal_feat.py). They need --wav-dir,
--words-dir (official word times) and --pstats. --no-causal-inputs = the whole-clip feat89 / speaker vector of the view
(leakage control); --causal-inputs also works with the engine protocol.
"""
import zlib
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg.common import J, CHUNK_FRAMES, EMBODIMENT, U80
from scmg.data import View, WindowDataset
from scmg.evaluate import load
from scmg.infer import generate
from scmg.model import INF
from scmg import g1


def clip_seed(seed, mode, stem):
    """Noise seed of one clip. 'run' = the run seed for every clip (old behaviour: all clips share the noise draws);
    'per_clip' = seed + crc32(stem) (independent draws per clip, reproducible, independent of shard / clip order)."""
    if mode == 'run': return seed
    if mode == 'per_clip': return (seed + zlib.crc32(stem.encode())) % (2 ** 63)
    raise ValueError(f'seed mode {mode}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--vae', required=True); ap.add_argument('--stats', required=True)
    ap.add_argument('--view', required=True); ap.add_argument('--text-cache', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--stems', default=None); ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--seed-mode', default='run', choices=['run', 'per_clip'],
                    help="run (old default): every clip of the run uses --seed, so all clips share the same noise draws; "
                         "per_clip (11-06 protocol for all new SCMG evaluations): clip seed = seed + crc32(stem)")
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int, default=10); ap.add_argument('--cfg-w', type=float, default=1.0); ap.add_argument('--cfg-t', type=float, default=None, help='separate TEXT guidance (Streamer cfg_t: v_u + w (v_a - v_u) + t (v_at - v_a)); None = joint CFG')
    ap.add_argument('--split', default='test', help="view split: 'test' (val_common view) | 'val' (held-out dev, for choosing inference settings)")
    ap.add_argument('--shard', default='0/1', help='i/n: this process handles clips i, i+n, ...')
    ap.add_argument('--save50', action='store_true', help='also keep the 50 Hz output (<stem>.q50.npy)')
    ap.add_argument('--protocol', default='offline', choices=['offline', 'engine', 'psync', 'pasr'])
    ap.add_argument('--text-mode', default=None, choices=['final', 'x2', 'voiced', 'asr'], help='override of the protocol preset')
    ap.add_argument('--causal-inputs', action=argparse.BooleanOptionalAction, default=None,
                    help='feat89 / CAM++ from the arrived PCM prefix every tick (default: on for psync / pasr)')
    ap.add_argument('--wav-dir', default=None, help='causal inputs: <stem>.wav (16 kHz, the view audio)')
    ap.add_argument('--words-dir', default=None, help='voiced / asr text and causal feat89: <stem>.json official word times')
    ap.add_argument('--pstats', default=None, help='causal feat89: fixed prosody statistics JSON (scmg.causal_feat pstats)')
    ap.add_argument('--campplus', default='/home/xr-25b7/bulk/models-ro/campplus'); ap.add_argument('--asr-lag', type=float, default=0.5)
    ap.add_argument('--horizon', default=None, help="audio look-ahead in 80 ms units past the chunk start, or 'inf'")
    ap.add_argument('--x2-rate', type=float, default=15.0); ap.add_argument('--x2-lag', type=float, default=0.15)
    ap.add_argument('--deploy-timing', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--text-causal', default=None, help='prefix-causal e5 cache row-aligned with --text-cache (needed by x2)')
    # 11-04 G1 metric work (inference-only probes, chosen on the dev split): speaker conditioning off (the learned null,
    # drop_spk 0.2 in training) for unseen test speakers; Savitzky-Golay smoothing of the 50 Hz joint channels
    ap.add_argument('--noise-scale', type=float, default=1.0, help='scale of the initial flow noise (1 = standard)')
    ap.add_argument('--sde-tmax', type=float, default=1.0, help='with --sde-eta: stochastic steps only while t < sde_tmax')
    ap.add_argument('--sde-eta', type=float, default=0.0, help='stochastic sampler gamma: marginal-preserving flow SDE dx = (v - gamma n0) dt + sqrt(2 gamma (1-t)) dW (0 = Euler ODE, the default path)')
    ap.add_argument('--no-spk', action='store_true', help='use the learned null speaker embedding (spk_valid = 0)')
    ap.add_argument('--smooth', default=None, help="'w,o': SG filter on the 29 joint channels at 50 Hz (root channels untouched)")
    ap.add_argument('--smooth-legs', default=None, help="'w,o': SG filter on the 12 leg joints only (after --smooth)")
    # 11-05 input-modality ablation (defaults = the unchanged path, bit-identical): which speech-side inputs every branch sees
    ap.add_argument('--modality', default='both', choices=['both', 'audio', 'text'],
                    help="both = audio + text (default); audio = no text tokens in any branch (tmask / tidm = 0, as a drop_text "
                         "sample); text = null audio in every branch (Streamer audio_off, as a drop_audio sample; the audio "
                         "still sets the length, the clock and the word times). CFG: cond = the kept modality, uncond = both "
                         "dropped (unchanged), same cfg_w")
    ap.add_argument('--text-swap', action='store_true',
                    help='text sensitivity probe: the text (tokens, features, times) of a donor clip -- the next clip in '
                         'length order of the stem list, times rescaled to this clip -- with this clip\'s audio / speaker')
    ap.add_argument('--text-swap-map', default=None, help='with --text-swap: JSON {stem: donor stem} (e.g. shared with the ECHO-G '
                    'probe scripts/echog_g1/echog_textuse.py) instead of the length-order neighbour')
    ap.add_argument('--causal-feat', default='prefix', choices=['prefix', 'inc'],
                    help="causal feat89: 'prefix' = recomputed from the arrived PCM every tick (default); 'inc' = incremental "
                         "(causal_feat.IncFeat89, bitwise equal, scripts/tests/test_causal_inc.py)")
    ap.add_argument('--diag-whole-spk', action='store_true', help='diagnostic: causal inputs with the whole-clip speaker vector')
    ap.add_argument('--spk-refresh', default='tick', choices=['tick', 'grades'],
                    help="causal speaker vector: 'tick' = CAM++ of the prefix every tick (default); 'grades' = refreshed only at the "
                         "1 / 2 / 4 / 8 s / clip-end grades, the training rule of StageCfg causal_spk")
    a = ap.parse_args()
    if EMBODIMENT != 'g1': raise SystemExit('run with SCMG_J=39')
    eng = a.protocol in ('engine', 'psync', 'pasr')
    tm = a.text_mode or {'engine': 'x2', 'psync': 'voiced', 'pasr': 'asr'}.get(a.protocol, 'final')
    cin = a.causal_inputs if a.causal_inputs is not None else a.protocol in ('psync', 'pasr')
    hz = a.horizon or ('2' if eng else 'inf')
    H = INF if hz == 'inf' else int(hz) * U80
    dt = a.deploy_timing if a.deploy_timing is not None else eng
    # the causal cache comes ONLY from --text-causal: an inherited SCMG_TEXT_CAUSAL (e.g. the EX002 v36_1 cache exported by
    # bagger_env.sh / r4_lane.sh) must never be picked up silently
    os.environ.pop('SCMG_TEXT_CAUSAL', None)
    if a.text_causal: os.environ['SCMG_TEXT_CAUSAL'] = a.text_causal
    if tm in ('x2', 'voiced', 'asr') and not a.text_causal: raise SystemExit(f'--text-mode {tm} needs --text-causal (prefix-causal e5 of the same rows)')
    if tm in ('voiced', 'asr') and not a.words_dir: raise SystemExit(f'--text-mode {tm} needs --words-dir')
    if cin and not (a.wav_dir and a.words_dir and a.pstats): raise SystemExit('--causal-inputs needs --wav-dir, --words-dir and --pstats')
    if cin and hz == 'inf': raise SystemExit('--causal-inputs needs a streaming horizon')
    if a.modality != 'both' and a.cfg_t is not None: raise SystemExit('--cfg-t (separate text guidance) needs --modality both')
    proto = None if (tm, hz, dt) == ('final', 'inf', False) else dict(protocol=a.protocol, text_mode=tm, horizon_80ms=hz,
        x2_rate=a.x2_rate, x2_lag=a.x2_lag, deploy_timing=dt, text_causal=a.text_causal,
        **({} if not (cin or tm in ('voiced', 'asr')) else dict(causal_inputs=bool(cin), asr_lag=a.asr_lag, pstats=a.pstats, words_dir=a.words_dir)))
    dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stats = json.loads(Path(a.stats).read_text())
    m, vae = load(a.ckpt, a.vae, dev)
    view = View(a.view, a.split, a.text_cache, stats)
    qm, qs = np.asarray(stats['q_mean'], np.float32), np.asarray(stats['q_std'], np.float32)
    if cin or tm in ('voiced', 'asr'):
        from scmg import causal_feat as CF
        tokz = CF.e5_tokenizer() if tm in ('voiced', 'asr') else None
        enc = CF.speaker_encoder(a.campplus) if (cin and not a.no_spk) else None
        pst = CF.load_pstats(a.pstats) if cin else None
    want = Path(a.stems).read_text().split() if a.stems else None
    idx = {c['items'][0]['id']: i for i, c in enumerate(view.clips)}
    stems = want if want else [c['items'][0]['id'] for c in view.clips]
    if a.n: stems = stems[:a.n]
    donor = {}
    if a.text_swap:                              # derangement over the WHOLE list (before sharding): neighbour in length order
        order = sorted(stems, key=lambda x: (view.clips[idx[x]]['frames'], x))
        donor = {x: order[(i + 1) % len(order)] for i, x in enumerate(order)}
        if a.text_swap_map: donor = {x: json.loads(Path(a.text_swap_map).read_text())[x] for x in stems}
    si, sn = (int(x) for x in a.shard.split('/'))
    stems = stems[si::sn]
    t0 = time.time(); log = []
    ticks = []                                   # 11-05: per-chunk compute time (s), Streamer.timing (cuda-synchronised)
    for k, stem in enumerate(stems):
        o = out / f'{stem}.pt'
        if o.exists(): continue
        ci = idx[stem]; c = view.clips[ci]; it = c['items'][0]
        if len(c['items']) != 1: raise ValueError(f'{stem}: one item per clip expected')
        L = c['frames']; n30 = int(it['echog']['frames30']); nch = -(-L // CHUNK_FRAMES)
        ds = WindowDataset(view, nch * CHUNK_FRAMES, max_tokens=256)
        s = ds.sample(ci, 0, np.random.default_rng(ci))
        s['ref'] = np.zeros_like(s['ref']); s['ref_valid'] = np.float32(0.)
        if a.no_spk: s['spk'] = np.zeros_like(s['spk']); s['spk_valid'] = np.float32(0.)
        if a.text_swap:
            dc = idx[donor[stem]]; Ld = view.clips[dc]['frames']
            sd = WindowDataset(view, -(-Ld // CHUNK_FRAMES) * CHUNK_FRAMES, max_tokens=256).sample(dc, 0, np.random.default_rng(dc))
            for kk in ('text', 'ttime', 'tmask', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c'):
                if kk in s or kk in sd:
                    if (kk in s) != (kk in sd) or np.shape(s[kk]) != np.shape(sd[kk]): raise ValueError(f'{stem}: text-swap {kk} mismatch')
                    v = np.array(sd[kk])
                    if kk in ('ttime', 'titime', 'tseg', 'tiseg', 'tiend'): v = v * (L / Ld)      # NaN (untimed) stays NaN
                    s[kk] = v.astype(np.asarray(sd[kk]).dtype)
        if a.modality == 'audio':                # = drop_text in training: no text token is valid in any branch
            for kk in ('tmask', 'tidm'):
                if kk in s: s[kk] = np.zeros_like(s[kk])
        cond = {kk: s[kk] for kk in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm',
                                     'titime', 'tseg', 'tiseg', 'tiend', 'text_c') if kk in s}
        kw = {} if proto is None else dict(text_mode=tm, x2_rate=a.x2_rate, x2_lag=a.x2_lag, deploy_timing=dt)
        mkw = dict(audio_off=True) if a.modality == 'text' else {}     # passed only when set (default call unchanged)
        if cin or tm in ('voiced', 'asr'):
            if a.text_swap: raise SystemExit('--text-swap is not supported with causal inputs / voiced text')
            words = CF.clip_words(Path(a.words_dir) / f'{stem}.json', L)
            if tm in ('voiced', 'asr'):
                kn = view.text['index'].get(stem, [0, 0])[1] if not it.get('text_null') else 0
                if int(np.asarray(s['tmask']).sum()) != kn: raise ValueError(f'{stem}: tmask rows != cache rows')
                cond.update(CF.text_extras(it, words, tokz, kn, len(s['tmask'])))
                mkw['asr_lag'] = a.asr_lag
            if cin:
                pcm16, wave = CF.clip_audio(Path(a.wav_dir) / f'{stem}.wav')
                mkw['causal_inputs'] = CF.CausalInputs(pcm16, wave, words, it['lang'], L, pst, view.fm, view.fs, enc,
                                                       Tpad=len(s['feat']), spk=not a.no_spk, spk_refresh=a.spk_refresh,
                                                       feat_mode=a.causal_feat)
                if a.diag_whole_spk:                     # diagnostic split of the causal-input cost: causal feat89, whole-clip speaker
                    _sv = (np.asarray(s['spk'], np.float32), float(s['spk_valid'])); mkw['causal_inputs'].spk = lambda Tk, _sv=_sv: _sv
        cmd, _, tk = generate(m, vae, s['feat'], cond, stats, nch, H=H, steps=a.steps, seed=clip_seed(a.seed, a.seed_mode, stem), standby=qm, ctx_t=m.ctx_t,
                              cfg_w=a.cfg_w, cfg_t=a.cfg_t, noise_scale=a.noise_scale, **({'sde_eta': a.sde_eta, 'sde_tmax': a.sde_tmax} if a.sde_eta else {}), **kw, **mkw)
        if cmd.shape[0] < L or cmd.shape[1] != J or not np.isfinite(cmd).all(): raise ValueError(f'{stem}: generated {cmd.shape}')
        if k > 0: ticks.extend(float(x) for x in tk)           # first clip excluded (CUDA warm-up)
        q50 = cmd[:L].astype(np.float64)
        for opt, ch in ((a.smooth, g1.JOINTS), (a.smooth_legs, g1.LEGS)):
            if opt:
                from scipy.signal import savgol_filter
                sg_w, sg_o = (int(x) for x in opt.split(','))       # not w/o: `o` is the output path
                if len(q50) > sg_w: q50[:, ch] = savgol_filter(q50[:, ch], sg_w, sg_o, axis=0, mode='interp')
        r30 = g1.to30(q50, n30).astype(np.float32)
        torch.save(dict(robot_repr=torch.from_numpy(r30), real_num_frames=n30, fps=30.0, representation_units='physical',
                        representation_schema='projecthermes-g1-39d-standard-v2', source=stem,
                        generator=dict(ckpt=str(a.ckpt), seed=a.seed, **({'seed_mode': a.seed_mode, 'clip_seed': clip_seed(a.seed, a.seed_mode, stem)} if a.seed_mode != 'run' else {}), steps=a.steps, cfg_w=a.cfg_w, cfg_t=a.cfg_t, frames50=L, noise_scale=a.noise_scale, **({'sde_eta': a.sde_eta, 'sde_tmax': a.sde_tmax} if a.sde_eta else {}), no_spk=a.no_spk, **({'diag_whole_spk': True} if a.diag_whole_spk else {}),
                                       **({'spk_refresh': a.spk_refresh} if a.spk_refresh != 'tick' else {}),
                                       **({'causal_feat': a.causal_feat} if a.causal_feat != 'prefix' else {}), smooth=a.smooth, smooth_legs=a.smooth_legs,
                                       **({} if (a.modality, a.text_swap) == ('both', False) else
                                          dict(modality=a.modality, text_swap=donor.get(stem))),
                                       resample='scmg/g1.py to30 (cubic / heading spline / RotationSpline)',
                                       **({} if proto is None else dict(stream=proto)))), o.with_suffix('.pt.tmp'))
        os.replace(o.with_suffix('.pt.tmp'), o)      # atomic: an interrupted run never leaves a truncated <stem>.pt that a resume skips
        if a.save50: np.save(out / f'{stem}.q50.npy', q50.astype(np.float32))
        log.append(dict(stem=stem, frames50=L, frames30=n30, tick_ms_p50=float(np.median(tk[1:]) * 1e3) if len(tk) > 1 else None))
        if k % 20 == 0: print(f'{k + 1}/{len(stems)} {stem} L50={L} n30={n30} {time.time() - t0:.0f}s', flush=True)
    with (out / f'GEN_LOG.{si}of{sn}.jsonl').open('a') as f:
        for r in log: f.write(json.dumps(r) + '\n')
    if ticks:
        tt = np.asarray(ticks) * 1000.
        (out / f'TIMING_{si}of{sn}.json').write_text(json.dumps(dict(
            n_ticks=int(len(tt)), chunk_ms=CHUNK_FRAMES * 20, ms_mean=float(tt.mean()), ms_p50=float(np.percentile(tt, 50)),
            ms_p95=float(np.percentile(tt, 95)), ms_max=float(tt.max()), rtf_p95=float(np.percentile(tt, 95) / (CHUNK_FRAMES * 20)),
            protocol=a.protocol, horizon_80ms=hz, steps=a.steps, cfg_w=a.cfg_w, cfg_t=a.cfg_t, gpu=torch.cuda.get_device_name(0)), indent=1))
    print('INFER_DONE', len(stems), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

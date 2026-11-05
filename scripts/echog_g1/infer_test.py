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
"""
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


CTL_MODES = {
    'none': 'no token (the control-token model without control)',
    'gt': 'every dev token: BEAT2-annotation tokens (values from the GT motion) + lexical tokens (oracle)',
    'ann': 'annotation tokens only (oracle)', 'lex': 'lexical tokens only (from text: deployable)',
    'auto': 'lexical tokens + a semantic-class token (no values) at every lexicon word by scmg.ctl.LEX2SEM (from text: deployable)',
    'gt_novals': 'gt with every value masked (class + time only)',
    'point': 'POINT tokens only, target + hand (amplitude masked): the spatial benchmark input',
    'shuffle': 'gt, POINT target / hand from another dev POINT token (seeded derangement), amplitude bins random',
    'amp0': 'gt, every amplitude bin forced to 0 (small)', 'amp2': 'gt, every amplitude bin forced to 2 (large)',
    'shift': 'gt, annotation tokens moved +1.0 s (t, a, b, apex; dropped past the clip end - 0.3 s)',
    'pred': 'semantic-class tokens (no values) predicted from the prefix-causal text by g1_sem_predictor.py (deployable)',
    'predlex': 'pred + the lexical tokens (deployable)'}


# 11-06 guard: process-global switches read by scmg at import / inference time. A non-default value silently changes every
# output, so it must be declared on the command line (--allow-env); the values are recorded in every output's metadata.
GUARD_ENV = {'SCMG_SEM_SCALE': '1', 'SCMG_BF16': '0', 'SCMG_SEED0': None, 'SCMG_STRIDE': '4'}


def env_guard(allow):
    vals = {k: os.environ.get(k) for k in GUARD_ENV}
    def default(k, v):
        if v is None or v == GUARD_ENV[k]: return True
        if k == 'SCMG_SEM_SCALE':
            try: return float(v) == 1.0
            except ValueError: return False
        return False
    bad = {k: v for k, v in vals.items() if not default(k, v)}
    if bad and not allow:
        raise SystemExit(f'non-default global switches in the environment {bad}: pass --allow-env to run with them (recorded in the outputs)')
    return vals


def ctl_plan(d, mode, stems, dur, pred=None):
    """stem -> list of token dicts (scmg/ctl.py token format) for the chosen mode (deterministic, before sharding)."""
    import copy
    from scmg import ctl
    raw = json.loads((Path(d) / 'CTL.json').read_text())
    if any(t['split'] != 'dev' for x in stems for t in raw.get(x, [])): raise SystemExit('control tokens of a train item requested')
    ann = lambda t: ctl.group_of(t['type']) == 0
    out = {}
    rng = np.random.default_rng(20261105)
    pts = [(x, i) for x in stems for i, t in enumerate(raw.get(x, [])) if t['type'] == ctl.POINT]
    perm = rng.permutation(len(pts)); donor = {pts[perm[i]]: pts[perm[(i + 1) % len(pts)]] for i in range(len(pts))}
    if mode in ('pred', 'predlex'):
        if not pred: raise SystemExit('--ctl-mode pred / predlex needs --ctl-pred')
        P = json.loads(Path(pred).read_text())
    for x in stems:
        ts = copy.deepcopy(raw.get(x, []))
        if mode == 'none': ts = []
        elif mode in ('pred', 'predlex'):
            ts = copy.deepcopy(P.get(x, [])) + ([t for t in ts if not ann(t)] if mode == 'predlex' else [])
        elif mode == 'ann': ts = [t for t in ts if ann(t)]
        elif mode == 'lex': ts = [t for t in ts if not ann(t)]
        elif mode == 'auto':
            ts = [t for t in ts if not ann(t)]
            ts += [dict(type=ctl.TYPE_ID[ctl.LEX2SEM[t['cls'][4:]]], cls=ctl.LEX2SEM[t['cls'][4:]], t=t['t'], a=t['a'], b=t['b'],
                        split='dev', auto_from=t['cls']) for t in ts]
        elif mode == 'gt_novals': ts = [dict(t, target=None, az=None, hand=None, amp_bin=None) for t in ts]
        elif mode == 'point': ts = [dict(t, amp_bin=None) for t in ts if t['type'] == ctl.POINT]
        elif mode == 'shuffle':
            for i, t in enumerate(ts):
                if t['type'] == ctl.POINT:
                    dx, di = donor[(x, i)]; dt_ = raw[dx][di]; t.update(target=dt_['target'], hand=dt_['hand'], shuffled_from=f'{dx}:{di}')
                if ann(t): t['amp_bin'] = int(rng.integers(3))
        elif mode in ('amp0', 'amp2'):
            for t in ts:
                if ann(t): t['amp_bin'] = int(mode[-1])
        elif mode == 'shift':
            new = []
            for t in ts:
                if ann(t):
                    if t['t'] + 1.0 >= dur[x] - 0.3: continue
                    t = dict(t, t=t['t'] + 1.0, a=t['a'] + 1.0, b=min(dur[x], t['b'] + 1.0), apex=t.get('apex', 0) + 1.0, shifted=1.0)
                new.append(t)
            ts = new
        out[x] = sorted(ts, key=lambda t: t['t'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--vae', required=True); ap.add_argument('--stats', required=True)
    ap.add_argument('--view', required=True); ap.add_argument('--text-cache', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--stems', default=None); ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int, default=10); ap.add_argument('--cfg-w', type=float, default=1.0); ap.add_argument('--cfg-t', type=float, default=None, help='separate TEXT guidance (Streamer cfg_t: v_u + w (v_a - v_u) + t (v_at - v_a)); None = joint CFG')
    ap.add_argument('--split', default='test', help="view split: 'test' (val_common view) | 'val' (held-out dev, for choosing inference settings)")
    ap.add_argument('--shard', default='0/1', help='i/n: this process handles clips i, i+n, ...')
    ap.add_argument('--save50', action='store_true', help='also keep the 50 Hz output (<stem>.q50.npy)')
    ap.add_argument('--protocol', default='offline', choices=['offline', 'engine'])
    ap.add_argument('--text-mode', default=None, choices=['final', 'x2'], help='override of the protocol preset')
    ap.add_argument('--horizon', default=None, help="audio look-ahead in 80 ms units past the chunk start, or 'inf'")
    ap.add_argument('--x2-rate', type=float, default=15.0); ap.add_argument('--x2-lag', type=float, default=0.15)
    ap.add_argument('--deploy-timing', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--text-causal', default=None, help='prefix-causal e5 cache row-aligned with --text-cache (needed by x2)')
    # 11-04 G1 metric work (inference-only probes, chosen on the dev split): speaker conditioning off (the learned null,
    # drop_spk 0.2 in training) for unseen test speakers; Savitzky-Golay smoothing of the 50 Hz joint channels
    ap.add_argument('--noise-scale', type=float, default=1.0, help='scale of the initial flow noise (1 = standard)')
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
    # 11-05 explicit control tokens (models trained with cfg ctl; default = no tokens, the unchanged call)
    ap.add_argument('--ctl-tokens', default=None, help='CTL.json dir of scripts/semantic/g1_ctl_tokens.py (dev split tokens)')
    ap.add_argument('--ctl-mode', default='gt', choices=list(CTL_MODES), help='which tokens are given (see CTL_MODES)')
    ap.add_argument('--cfg-ctl', type=float, default=None, help='separate control-token guidance (Streamer cfg_ctl); None = joint CFG')
    ap.add_argument('--ctl-pred', default=None, help='PRED.json of scripts/semantic/g1_sem_predictor.py (text-only predicted semantic tokens; modes pred / predlex)')
    ap.add_argument('--ret', default=None, help='retrieval dir:query file (scripts/semantic/g1_ret_bank.py, e.g. <dir>:RET_dev-auto.json); model with cfg ret')
    ap.add_argument('--allow-env', action='store_true', help='allow non-default SCMG_SEM_SCALE / SCMG_BF16 / SCMG_SEED0 / SCMG_STRIDE')
    a = ap.parse_args()
    genv = env_guard(a.allow_env)
    if EMBODIMENT != 'g1': raise SystemExit('run with SCMG_J=39')
    if a.ctl_tokens: a.save50 = True                # the control metrics are measured on the 50 Hz output
    eng = a.protocol == 'engine'
    tm = a.text_mode or ('x2' if eng else 'final')
    hz = a.horizon or ('2' if eng else 'inf')
    H = INF if hz == 'inf' else int(hz) * U80
    dt = a.deploy_timing if a.deploy_timing is not None else eng
    # the causal cache comes ONLY from --text-causal: an inherited SCMG_TEXT_CAUSAL (e.g. the EX002 v36_1 cache exported by
    # bagger_env.sh / r4_lane.sh) must never be picked up silently
    os.environ.pop('SCMG_TEXT_CAUSAL', None)
    if a.text_causal: os.environ['SCMG_TEXT_CAUSAL'] = a.text_causal
    if tm == 'x2' and not a.text_causal: raise SystemExit('--text-mode x2 needs --text-causal (prefix-causal e5 of the same rows)')
    if a.modality != 'both' and a.cfg_t is not None: raise SystemExit('--cfg-t (separate text guidance) needs --modality both')
    proto = None if (tm, hz, dt) == ('final', 'inf', False) else dict(protocol=a.protocol, text_mode=tm, horizon_80ms=hz,
        x2_rate=a.x2_rate, x2_lag=a.x2_lag, deploy_timing=dt, text_causal=a.text_causal)
    dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stats = json.loads(Path(a.stats).read_text())
    m, vae = load(a.ckpt, a.vae, dev)
    view = View(a.view, a.split, a.text_cache, stats)
    qm, qs = np.asarray(stats['q_mean'], np.float32), np.asarray(stats['q_std'], np.float32)
    want = Path(a.stems).read_text().split() if a.stems else None
    idx = {c['items'][0]['id']: i for i, c in enumerate(view.clips)}
    stems = want if want else [c['items'][0]['id'] for c in view.clips]
    if a.n: stems = stems[:a.n]
    donor = {}
    if a.text_swap:                              # derangement over the WHOLE list (before sharding): neighbour in length order
        order = sorted(stems, key=lambda x: (view.clips[idx[x]]['frames'], x))
        donor = {x: order[(i + 1) % len(order)] for i, x in enumerate(order)}
        if a.text_swap_map: donor = {x: json.loads(Path(a.text_swap_map).read_text())[x] for x in stems}
    retq = None
    if a.ret:
        rd, rq = a.ret.split(':', 1); a.save50 = True
        rlat = np.load(Path(rd) / 'BANK.npz')['lat'].astype(np.float32); retq = json.loads((Path(rd) / rq).read_text())
        if any(not x.startswith(('english_2_scott', 'english_6_carla')) for x in retq): print('note: retrieval queries outside the dev speakers')
    ctl_toks = ctl_plan(a.ctl_tokens, a.ctl_mode, stems, {x: view.clips[idx[x]]['frames'] / 50. for x in stems}, a.ctl_pred) if a.ctl_tokens else {}
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
        rq_ = (retq or {}).get(stem, [])
        if rq_:
            cond.update(ret_t=np.array([x['t'] for x in rq_], np.float32), ret_lat=np.stack([rlat[x['bank']] for x in rq_]))
            if a.cfg_ctl is not None: mkw['cfg_ctl'] = a.cfg_ctl
        used = ctl_toks.get(stem, [])
        if used:
            from scmg.ctl import token_value
            tv = [token_value(t) for t in used]
            cond.update(ctl_t=np.array([t['t'] for t in used], np.float32), ctl_type=np.array([x[0] for x in tv], np.int64),
                        ctl_v=np.stack([x[1] for x in tv]), ctl_vm=np.stack([x[2] for x in tv]))
            if a.cfg_ctl is not None: mkw['cfg_ctl'] = a.cfg_ctl
        cmd, _, tk = generate(m, vae, s['feat'], cond, stats, nch, H=H, steps=a.steps, seed=a.seed, standby=qm, ctx_t=m.ctx_t,
                              cfg_w=a.cfg_w, cfg_t=a.cfg_t, noise_scale=a.noise_scale, **kw, **mkw)
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
                        generator=dict(ckpt=str(a.ckpt), seed=a.seed, steps=a.steps, cfg_w=a.cfg_w, cfg_t=a.cfg_t, frames50=L, noise_scale=a.noise_scale, no_spk=a.no_spk, smooth=a.smooth, smooth_legs=a.smooth_legs,
                                       **({} if (a.modality, a.text_swap) == ('both', False) else
                                          dict(modality=a.modality, text_swap=donor.get(stem))),
                                       resample='scmg/g1.py to30 (cubic / heading spline / RotationSpline)', env=genv,
                                       **({} if proto is None else dict(stream=proto)),
                                       **({} if not a.ctl_tokens else dict(ctl_mode=a.ctl_mode, cfg_ctl=a.cfg_ctl, ctl=used)),
                                       **({} if not a.ret else dict(ret=a.ret, ret_used=rq_)))), o.with_suffix('.pt.tmp'))
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

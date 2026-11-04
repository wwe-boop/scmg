"""SCMG on the ECHO-G benchmark clips (val_common = echog-g1-v1/test): generate 50 Hz motion offline (full audio + text
context, as the ECHO-G benchmark), convert to 30 FPS with exactly the official aligned frame count, and write one
evaluator-ready file per stem: <out>/<stem>.pt = {robot_repr [n30, 39] float32 physical, real_num_frames, fps 30.0,
representation_units 'physical', representation_schema}. Run with SCMG_J=39.
Conditioning: audio feat89 + e5 word tokens (+ token times) + CAM++ vector of the clip's own audio; NO reference motion
(ref_valid = 0: val_common speakers are absent from train_drop, and test GT must not leak into a condition).
    SCMG_J=39 python scripts/echog_g1/infer_test.py --ckpt base/best.pt --vae vae/final.pt --stats STATS --view VIEW/test
        --text-cache TEXT_TEST --out preds/seed_000 [--stems list.txt] [--n 20] [--seed 0] [--steps 10] [--cfg-w 1]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg.common import J, CHUNK_FRAMES, EMBODIMENT
from scmg.data import View, WindowDataset
from scmg.evaluate import load
from scmg.infer import generate
from scmg.model import INF
from scmg import g1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--vae', required=True); ap.add_argument('--stats', required=True)
    ap.add_argument('--view', required=True); ap.add_argument('--text-cache', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--stems', default=None); ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int, default=10); ap.add_argument('--cfg-w', type=float, default=1.0)
    ap.add_argument('--shard', default='0/1', help='i/n: this process handles clips i, i+n, ...')
    ap.add_argument('--save50', action='store_true', help='also keep the 50 Hz output (<stem>.q50.npy)')
    a = ap.parse_args()
    if EMBODIMENT != 'g1': raise SystemExit('run with SCMG_J=39')
    dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stats = json.loads(Path(a.stats).read_text())
    m, vae = load(a.ckpt, a.vae, dev)
    view = View(a.view, 'test', a.text_cache, stats)
    qm, qs = np.asarray(stats['q_mean'], np.float32), np.asarray(stats['q_std'], np.float32)
    want = Path(a.stems).read_text().split() if a.stems else None
    idx = {c['items'][0]['id']: i for i, c in enumerate(view.clips)}
    stems = want if want else [c['items'][0]['id'] for c in view.clips]
    if a.n: stems = stems[:a.n]
    si, sn = (int(x) for x in a.shard.split('/'))
    stems = stems[si::sn]
    t0 = time.time(); log = []
    for k, stem in enumerate(stems):
        o = out / f'{stem}.pt'
        if o.exists(): continue
        ci = idx[stem]; c = view.clips[ci]; it = c['items'][0]
        if len(c['items']) != 1: raise ValueError(f'{stem}: one item per clip expected')
        L = c['frames']; n30 = int(it['echog']['frames30']); nch = -(-L // CHUNK_FRAMES)
        ds = WindowDataset(view, nch * CHUNK_FRAMES, max_tokens=256)
        s = ds.sample(ci, 0, np.random.default_rng(ci))
        s['ref'] = np.zeros_like(s['ref']); s['ref_valid'] = np.float32(0.)
        cond = {kk: s[kk] for kk in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm',
                                     'titime', 'tseg', 'tiseg', 'tiend', 'text_c') if kk in s}
        cmd, _, tm = generate(m, vae, s['feat'], cond, stats, nch, H=INF, steps=a.steps, seed=a.seed, standby=qm, ctx_t=m.ctx_t,
                              cfg_w=a.cfg_w)
        if cmd.shape[0] < L or cmd.shape[1] != J or not np.isfinite(cmd).all(): raise ValueError(f'{stem}: generated {cmd.shape}')
        q50 = cmd[:L].astype(np.float64)
        r30 = g1.to30(q50, n30).astype(np.float32)
        torch.save(dict(robot_repr=torch.from_numpy(r30), real_num_frames=n30, fps=30.0, representation_units='physical',
                        representation_schema='projecthermes-g1-39d-standard-v2', source=stem,
                        generator=dict(ckpt=str(a.ckpt), seed=a.seed, steps=a.steps, cfg_w=a.cfg_w, frames50=L,
                                       resample='scmg/g1.py to30 (cubic / heading spline / RotationSpline)')), o)
        if a.save50: np.save(out / f'{stem}.q50.npy', q50.astype(np.float32))
        log.append(dict(stem=stem, frames50=L, frames30=n30, tick_ms_p50=float(np.median(tm[1:]) * 1e3) if len(tm) > 1 else None))
        if k % 20 == 0: print(f'{k + 1}/{len(stems)} {stem} L50={L} n30={n30} {time.time() - t0:.0f}s', flush=True)
    with (out / f'GEN_LOG.{si}of{sn}.jsonl').open('a') as f:
        for r in log: f.write(json.dumps(r) + '\n')
    print('INFER_DONE', len(stems), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

"""Edge port step 0: where does one deploy tick (scmg.deploy_stream / Streamer.tick, CFG 2.5, 10 steps, H=2) spend its
time, and which network modules must become TensorRT engines on Thor.

Wraps every model / VAE method the Streamer calls with a CUDA-synchronised timer, runs N ticks of a growing utterance
(real feat89 features of a wav when given, else random), and prints per-tick calls / ms per module, parameter counts
per top-level submodule, and the feat89 CPU time per tick (1 thread, prefix recompute as in deploy_stream).

usage: python -m scmg.edge.profile_tick --ckpt final.pt --vae vae.pt --stats STATS.json [--wav x.wav --text ...]
       [--ticks 20] [--threads 1] [--dtype fp32|fp16] [--out profile.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

METHODS = ['audio_tokens', 'text_tokens', 'token_seconds', 'global_cond', 'plan', 'mt_memory', 'planner_memory', 'sem_h',
           'local_audio', 'flow_velocity', 'state_ad', 'style_enc', 'style_head', 'chunk_audio_mask', 'chunk_text_mask']


class Timer:
    def __init__(self):
        self.ms = collections.defaultdict(float); self.n = collections.defaultdict(int); self.on = True

    def wrap(self, obj, name, label):
        f = getattr(obj, name, None)
        if f is None or not callable(f):
            return
        def g(*a, **k):
            if not self.on:
                return f(*a, **k)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            r = f(*a, **k)
            torch.cuda.synchronize()
            self.ms[label] += 1000 * (time.perf_counter() - t0); self.n[label] += 1
            return r
        setattr(obj, name, g)


def params(mod):
    return sum(p.numel() for p in mod.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--vae', required=True); ap.add_argument('--stats', required=True)
    ap.add_argument('--wav', default=None); ap.add_argument('--text', default='请大家看左边这台白色的机器人，它正在向您挥手致意。')
    ap.add_argument('--ticks', type=int, default=20); ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--feat89', default='/mnt/cpfs/zbl-cpfs-new/USERS/zehan/unified-train-A-20261025')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, a.feat89)
    from scmg.evaluate import load
    from scmg.infer import Streamer
    from scmg.common import U80
    stats = json.loads(Path(a.stats).read_text())
    m, vae = load(a.ckpt, a.vae, 'cuda')
    qm = np.asarray(stats['q_mean'], np.float32); fm = np.asarray(stats['f_mean'], np.float32); fs = np.asarray(stats['f_std'], np.float32)
    report = dict(params={n: params(c) for n, c in m.named_children()}, vae_params=params(vae),
                  base_cfg={k: v for k, v in vars(m.cfg).items() if isinstance(v, (int, float, bool, str))} if hasattr(m, 'cfg') else None)

    T_total = a.ticks * 16 + 8
    feats = None; tfeat = []
    if a.wav:
        import wave
        import feat89 as F
        w = wave.open(a.wav); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), '<i2')
        if sr != 16000:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, 16000); x = np.clip(np.round(resample_poly(x.astype(np.float32), 16000 // g, sr // g)), -32768, 32767).astype(np.int16)
        T_total = min(T_total, len(x) // 320)
        def feat(T):
            from scipy.signal import resample_poly
            p24 = resample_poly(x[:T * 320].astype(np.float32) / 32768., 3, 2)
            return np.concatenate([F.gl_build.logmel(p24, T, F.mel()), np.zeros((T, 9), np.float32)], 1).astype(np.float32)
        feats = feat
    tm = Timer()
    for n in METHODS:
        tm.wrap(m, n, n)
    tm.wrap(vae, 'decode', 'vae.decode'); tm.wrap(vae, 'encode', 'vae.encode')
    K = 12
    cond = dict(text_c=torch.randn(K, 768, device='cuda') * .1, ttime=torch.full((K,), float('nan'), device='cuda'),
                tmask=torch.ones(K, dtype=torch.bool, device='cuda'), spk=torch.zeros(192, device='cuda'),
                spk_valid=torch.tensor(0., device='cuda'), source=torch.tensor(5, device='cuda'),
                ref=torch.zeros(128, 34, device='cuda'), ref_valid=torch.tensor(0., device='cuda'))
    s = Streamer(m, vae, torch.zeros(1, 89, device='cuda'), cond, stats, H=2 * U80, steps=10, standby=qm, ctx_t=m.ctx_t,
                 cfg_w=2.5, text_mode='x2', x2_rate=15., x2_lag=.15, deploy_timing=True, trim_start=True, cfg_null_fix=True)
    rng = np.random.default_rng(0)
    tick_ms = []
    for k in range(a.ticks):
        T = min(T_total, 16 * k + 8)
        t0 = time.perf_counter()
        f = feats(T) if feats else rng.standard_normal((T, 89)).astype(np.float32)
        tfeat.append(1000 * (time.perf_counter() - t0))
        s.feat = torch.as_tensor((f - fm) / fs, device='cuda').float()
        if k == 0:
            tm.on = False; s.tick(); torch.cuda.synchronize(); tm.on = True        # warm-up tick not counted
            continue
        torch.cuda.synchronize(); t0 = time.perf_counter()
        s.tick(); torch.cuda.synchronize()
        tick_ms.append(1000 * (time.perf_counter() - t0))
    n = len(tick_ms)
    report['tick_ms'] = dict(p50=float(np.median(tick_ms)), p95=float(np.percentile(tick_ms, 95)), n=n)
    report['modules_per_tick'] = {k: dict(calls=tm.n[k] / n, ms=tm.ms[k] / n) for k in sorted(tm.ms, key=lambda k: -tm.ms[k])}
    report['covered_ms'] = sum(v['ms'] for v in report['modules_per_tick'].values())
    report['feat_ms'] = dict(p50=float(np.median(tfeat[1:])), max=float(np.max(tfeat)), last_T=T, threads=a.threads, real=bool(feats))
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()

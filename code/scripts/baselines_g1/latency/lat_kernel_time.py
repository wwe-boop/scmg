"""GPU kernel time vs wall time of SCMG Streamer ticks (benchmark settings): torch.profiler sums the CUDA kernel time of
each tick; wall - kernel = host-side launch / Python overhead (the part that depends on the host CPU, not the GPU).
  SCMG_J=39 python lat_kernel_time.py --protocol offline|engine --n 6 --out DIR"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lat_common as LC  # noqa: E402
import lat_scmg as LS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--protocol', default='offline', choices=['offline', 'engine'])
    ap.add_argument('--n', type=int, default=6); ap.add_argument('--out', required=True); a = ap.parse_args()
    eng = a.protocol == 'engine'
    os.environ.pop('SCMG_TEXT_CAUSAL', None)
    if eng: os.environ['SCMG_TEXT_CAUSAL'] = LS.TEXT_CAUSAL
    from torch.profiler import profile, ProfilerActivity
    from scmg.common import CHUNK_FRAMES, U80
    from scmg.data import View, WindowDataset
    from scmg.evaluate import load
    from scmg.infer import Streamer
    from scmg.model import INF
    dev = 'cuda'; stats = json.loads(Path(LS.STATS).read_text()); qm = np.asarray(stats['q_mean'], np.float32)
    m, vae = load(LS.CKPT, LS.VAE, dev); view = View(LS.VIEW_TEST, 'test', LS.TEXT_TEST, stats)
    idx = {c['items'][0]['id']: i for i, c in enumerate(view.clips)}
    H = 2 * U80 if eng else INF; recs = []
    for k, stem in enumerate(LC.stems50()[:a.n]):
        ci = idx[stem]; c = view.clips[ci]; L = c['frames']; nch = -(-L // CHUNK_FRAMES)
        s = WindowDataset(view, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
        s['ref'] = np.zeros_like(s['ref']); s['ref_valid'] = np.float32(0.)
        cond = {kk: s[kk] for kk in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm',
                                     'titime', 'tseg', 'tiseg', 'tiend', 'text_c') if kk in s}
        kw = dict(text_mode='x2', x2_rate=15.0, x2_lag=0.15, deploy_timing=True) if eng else {}
        S = Streamer(m, vae, s['feat'], cond, stats, H=H, steps=10, seed=0, standby=qm, ctx_t=m.ctx_t, cfg_w=2.5, cfg_t=None, noise_scale=1.0, **kw)
        for kk in range(nch):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            if k >= 1 and kk % 3 == 1:
                with profile(activities=[ProfilerActivity.CUDA]) as p:
                    S.tick(arrived_s=1e9); torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                kern = sum(e.device_time_total if hasattr(e, 'device_time_total') else e.cuda_time_total for e in p.key_averages()) / 1e6
                nk = sum(e.count for e in p.key_averages())
                recs.append(dict(stem=stem, k=kk, wall_ms=wall * 1e3, kernel_ms=kern * 1e3, kernels=nk))
            else:
                S.tick(arrived_s=1e9); torch.cuda.synchronize()
        print(stem, len(recs), flush=True)
    w = np.array([r['wall_ms'] for r in recs]); kt = np.array([r['kernel_ms'] for r in recs]); nk = np.array([r['kernels'] for r in recs])
    res = dict(protocol=a.protocol, ticks=len(recs), wall_ms_p50=float(np.median(w)), kernel_ms_p50=float(np.median(kt)),
               kernels_per_tick_p50=float(np.median(nk)), wall_rtf_p50=float(np.median(w) / 320), kernel_rtf_p50=float(np.median(kt) / 320),
               gpu=torch.cuda.get_device_name(0), host=LC.host_snapshot(), note='profiled ticks carry profiler overhead in wall_ms; kernel_ms is the GPU-busy lower bound')
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / f'kernel_time_{a.protocol}.json').write_text(json.dumps(dict(summary=res, rows=recs), indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

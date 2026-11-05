"""Does the known-prefix condition work? (11-05 boundary-jerk diagnosis)
For val windows (GT prefix = teacher forcing, no cold start) of a StreamSGDiT checkpoint:
  A  flow-matching error per offset j = f - c from the cut (j = 0 is the first generated frame) at fixed flow times t:
     a model that uses the prefix predicts the frames next to the cut far better than distant ones, at every t.
  B  teacher-forced block sampling (known = GT prefix, the trainer's window, official Euler): boundary step
     |x[c] - x[c-1]| (generated first frame vs last known GT frame) and the in-block steps |x[f] - x[f-1]|, f in (c, f1),
     compared with GT; ratio_boundary = mean boundary step / mean in-block step (GT ~ 1).
    PYTHONPATH=<official>/src python scripts/echog_stream/diag_prefix.py --checkpoint ema.pt [--config yaml] --data-root R --n 64
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from echo_g.data import load_motion_stats
from stream_core import StreamConfig, collate_windows, load_stream_model, sample_window, sample_window_euler
from train_stream import StreamDataset

ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--config', type=Path)
ap.add_argument('--data-root', type=Path, required=True); ap.add_argument('--n', type=int, default=64)
ap.add_argument('--known-xt', choices=['clean', 'noise']); ap.add_argument('--steps', type=int, default=8); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
dev = torch.device('cuda'); torch.backends.cuda.matmul.allow_tf32 = False
model, config, mean, std, sc, meta = load_stream_model(a.checkpoint, dev, a.config)
fps = float(config.data.fps)
if a.known_xt: sc.known_xt = a.known_xt
m_, s_ = load_motion_stats(a.data_root / config.data.stats_file, config.model.motion_dim)
ds = StreamDataset(a.data_root, config.data, config.model, 'val', m_, s_)
scw = StreamConfig.from_dict(dict(sc.to_dict(), p_cold=0.0, p_ahead_more=0.0))
rng = np.random.default_rng(a.seed)
ws = [sample_window(ds[i % len(ds)], rng, scw, fps) for i in range(a.n)]
ws = [w for w in ws if w['meta']['c'] > 0]
b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate_windows(ws).items()}
c_rel = torch.tensor([w['meta']['c'] - w['meta']['ws'] for w in ws], device=dev)
f1_rel = torch.tensor([w['meta']['f1'] - w['meta']['ws'] for w in ws], device=dev)
T = b['motion'].shape[1]; pos = torch.arange(T, device=dev)[None]; off = pos - c_rel[:, None]
gen = (off >= 0) & b['motion_mask']
print(f'{a.checkpoint}  step {meta.get("global_step")}  known_xt {sc.known_xt}  windows {len(ws)}')
g = torch.Generator(device=dev).manual_seed(1)
with torch.no_grad():
    print('A  flow error (MSE, normalised) by offset j from the cut; columns j = 0 1 2 4 8 16 30+')
    for t in (0.0, 0.25, 0.5, 0.75, 0.875):
        noise = torch.randn(b['motion'].shape, generator=g, device=dev)
        x1 = b['motion']; km = b['known_mask']
        xt = t * x1 + (1 - t) * noise
        if sc.known_xt == 'clean': xt = torch.where(km[..., None], x1, xt)
        v = model(xt, torch.full((len(ws),), t, device=dev), b['audio'], b['text'], b['motion_mask'], b['text_padding_mask'],
                  b['time_distance'], known=x1, known_mask=km)
        e = (v - (x1 - noise)).square().mean(-1)
        row = []
        for j in (0, 1, 2, 4, 8, 16):
            mk = gen & (off == j); row.append(e[mk].mean().item() if mk.any() else float('nan'))
        mk = gen & (off >= 30); row.append(e[mk].mean().item() if mk.any() else float('nan'))
        print(f'   t={t:5.3f} ' + ' '.join(f'{x:7.3f}' for x in row))
    noise = torch.randn(b['motion'].shape, generator=g, device=dev)
    y = sample_window_euler(model, {k: b[k] for k in ('audio', 'text', 'text_padding_mask', 'time_distance', 'motion_mask')},
                            noise, b['motion'], b['known_mask'], a.steps, sc.known_xt)
    sd = std.to(dev); mn = mean.to(dev)
    for name, z in (('generated', y), ('GT', b['motion'])):
        x = z * sd + mn; bnd, inb = [], []
        for i in range(len(ws)):
            c, f1 = int(c_rel[i]), int(f1_rel[i])
            bnd.append((x[i, c] - (b['motion'][i, c - 1] * sd + mn)).abs().mean().item())
            if f1 - c > 1: inb.append((x[i, c + 1:f1] - x[i, c:f1 - 1]).abs().mean().item())
        print(f'B  {name:9s} boundary step {np.mean(bnd):.4f}  in-block step {np.mean(inb):.4f}  ratio {np.mean(bnd) / np.mean(inb):.2f}')

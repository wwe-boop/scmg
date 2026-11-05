"""End-to-end check that the known-prefix condition is wired so that it CAN be learned (11-05, boundary-jerk diagnosis).
Synthetic task where the prefix is the only information: each clip is a constant pose c ~ N(0, I) (no audio, no text),
windows from the real trainer (stream_core.sample_window, p_cold = 0), real loss (stream_flow_loss) and real sampler
(sample_window_euler).  A tiny StreamSGDiT trained a few hundred steps must continue the prefix: generated frames ~ c,
in particular the first generated frame (no boundary jump); with the prefix hidden at sampling (known_mask = 0) it cannot.
Run for known_xt clean (default) and noise.  python scripts/echog_stream/test_prefix_learn.py  (PYTHONPATH=<official>/src)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from echo_g.config import ModelConfig
from stream_core import StreamConfig, StreamSGDiT, collate_windows, sample_window, sample_window_euler, stream_flow_loss

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); fps = 30.0; D = 6; F = 96
mc = ModelConfig(motion_dim=D, audio_dim=8, text_dim=8, hidden_dim=64, num_layers=2, num_heads=4, feedforward_dim=128, max_t=128)


def clip(rng):
    c = torch.as_tensor(rng.standard_normal(D), dtype=torch.float32)
    tt = np.array([[100.0, 100.1]])                                       # never arrives -> official null text
    return dict(frames=F, motion=c[None].repeat(F, 1), audio=torch.zeros(F, 8), text=torch.zeros(1, 8),
                token_centers=torch.tensor([100.05]), token_times=tt, stem='syn', has_timing=True), c


def batch(rng, sc, n):
    ws, cs = [], []
    for _ in range(n):
        s, c = clip(rng); ws.append(sample_window(s, rng, sc, fps)); cs.append(c)
    b = collate_windows(ws)
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}, torch.stack(cs).to(dev)


ok_all = True
for known_xt, fill in (('clean', 'zero'), ('noise', 'zero'), ('clean', 'hold')):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    sc = StreamConfig(p_cold=0.0, ctx_s=1.0, tail_max_s=0.32, known_xt=known_xt, known_fill=fill)
    m = StreamSGDiT(mc).to(dev); m.known_fill = fill; opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for step in range(600):
        b, _ = batch(rng, sc, 32)
        loss = stream_flow_loss(m, b, 0.0, 0.5, known_xt)[0]
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); b, cs = batch(np.random.default_rng(123), sc, 64)
    cond = {k: b[k] for k in ('audio', 'text', 'text_padding_mask', 'time_distance', 'motion_mask')}
    noise = torch.randn(b['motion'].shape, device=dev)
    res = {}
    for name, km in (('prefix', b['known_mask']), ('hidden', torch.zeros_like(b['known_mask']))):
        y = sample_window_euler(m, cond, noise, b['motion'], km, 8, known_xt)
        gen = (~b['known_mask']) & b['motion_mask']
        first = torch.zeros_like(gen)
        for i, meta in enumerate(b['meta']): first[i, meta['c'] - meta['ws']] = True
        err = (y - cs[:, None]).abs().mean(-1)
        res[name] = (err[gen].mean().item(), err[first].mean().item())
    good = res['prefix'][0] < 0.15 and res['prefix'][1] < 0.15 and res['hidden'][0] > 0.5
    ok_all &= good
    print(f"{'PASS' if good else 'FAIL'} known_xt={known_xt} fill={fill}: final loss {loss.item():.4f} | |gen - prefix pose| with prefix: all {res['prefix'][0]:.3f} "
          f"first frame {res['prefix'][1]:.3f} | prefix hidden: all {res['hidden'][0]:.3f} (pose std 1)")
print('ALL_OK' if ok_all else 'SOME_FAILED')

"""Edge port E1: export the per-tick network modules to fixed-shape ONNX and check onnxruntime against torch.

1. capture: run a few deploy ticks (same Streamer settings as scmg.deploy_stream) and record the exact inputs of every
   call of the exported modules (tick 2 = steady state; shapes and None-ness);
2. export: wrap each module call with its captured None pattern as an nn.Module with tensor-only inputs, export with
   torch.onnx (opset 17, fixed shapes, fp32);
3. check: onnxruntime (CUDA EP, fp32) on the captured inputs vs torch: max |diff| and relative error per output.

Modules: flow_velocity (also as a CFG pair: batch 2 = cond + uncond in one call), plan, audio_tokens, local_audio,
vae.decode.  usage: python -m scmg.edge.export_onnx --ckpt .. --vae .. --stats .. --wav .. --out DIR [--only flow_velocity]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODS = ['flow_velocity', 'plan', 'audio_tokens', 'local_audio']


def flatten(x, path='', out=None):
    """args/kwargs pytree -> {path: tensor}, plus the structure with tensors replaced by their path."""
    out = {} if out is None else out
    if torch.is_tensor(x):
        out[path] = x.detach().clone(); return ('T', path), out
    if isinstance(x, (list, tuple)):
        items = [flatten(v, f'{path}.{i}', out)[0] for i, v in enumerate(x)]
        return (type(x).__name__, items), out
    if isinstance(x, dict):
        return ('dict', {k: flatten(v, f'{path}.{k}', out)[0] for k, v in x.items()}), out
    return ('C', x), out


def rebuild(spec, tensors):
    kind, v = spec
    if kind == 'T':
        return tensors[v]
    if kind in ('list', 'tuple'):
        r = [rebuild(s, tensors) for s in v]
        return r if kind == 'list' else tuple(r)
    if kind == 'dict':
        return {k: rebuild(s, tensors) for k, s in v.items()}
    return v


class Capture:
    def __init__(self, obj, name, want_tick):
        self.obj, self.name, self.want = obj, name, want_tick
        self.f = getattr(obj, name); self.tick = 0; self.calls = []
        setattr(obj, name, self)

    def __call__(self, *a, **k):
        r = self.f(*a, **k)
        if self.tick == self.want:
            spec, ts = flatten((a, k))
            out_spec, outs = flatten(r)
            self.calls.append(dict(spec=spec, tensors=ts, out_spec=out_spec, outs=outs))
        return r

    def restore(self):
        setattr(self.obj, self.name, self.f)


class Wrap(torch.nn.Module):
    """Tensor-only forward for one captured call pattern: inputs = the call's tensors in key order."""

    def __init__(self, fn, spec, keys, out_spec):
        super().__init__()
        self.fn, self.spec, self.keys, self.out_spec = fn, spec, keys, out_spec

    def forward(self, *xs):
        a, k = rebuild(self.spec, dict(zip(self.keys, xs)))
        r = self.fn(*a, **k)
        _, outs = flatten(r)
        return tuple(outs.values())


def export_one(name, fn, call, out_dir, ort, provider):
    keys = list(call['tensors'])
    xs = [call['tensors'][k] for k in keys]
    w = Wrap(fn, call['spec'], keys, call['out_spec']).eval()
    with torch.inference_mode():
        ref = w(*xs)
    path = out_dir / f'{name}.onnx'
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(w, tuple(xs), str(path), opset_version=17, dynamo=False,
                          input_names=[f'in{i}' for i in range(len(xs))], output_names=[f'out{i}' for i in range(len(ref))])
    exp_s = time.time() - t0
    feed = {}
    sess = ort.InferenceSession(str(path), providers=[provider])
    names = [i.name for i in sess.get_inputs()]                     # unused / constant-folded inputs are dropped
    for i, x in enumerate(xs):
        if f'in{i}' in names:
            feed[f'in{i}'] = x.cpu().numpy()
    t0 = time.perf_counter(); outs = sess.run(None, feed); ort_ms = 1000 * (time.perf_counter() - t0)
    for _ in range(3):
        t0 = time.perf_counter(); sess.run(None, feed); ort_ms = min(ort_ms, 1000 * (time.perf_counter() - t0))
    errs = []
    for r, o in zip(ref, outs):
        r = r.float().cpu().numpy(); o = np.asarray(o, np.float32)
        d = np.abs(r - o); errs.append(dict(shape=list(r.shape), max_abs=float(d.max()) if d.size else 0.,
                                            rel=float(d.max() / (np.abs(r).max() + 1e-9)) if d.size else 0.))
    return dict(module=name, inputs={k: [list(x.shape), str(x.dtype)] for k, x in zip(keys, xs)}, onnx=str(path),
                onnx_mb=round(path.stat().st_size / 2 ** 20, 1) + sum(p.stat().st_size for p in out_dir.glob(f'{name}.onnx.*')) / 2 ** 20,
                export_s=round(exp_s, 1), ort_ms=round(ort_ms, 3), provider=provider, outputs=errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--vae', required=True); ap.add_argument('--stats', required=True)
    ap.add_argument('--wav', required=True); ap.add_argument('--out', required=True); ap.add_argument('--tick', type=int, default=3)
    ap.add_argument('--only', default=None); ap.add_argument('--provider', default='CPUExecutionProvider')
    ap.add_argument('--feat89', default='/mnt/cpfs/zbl-cpfs-new/USERS/zehan/unified-train-A-20261025')
    a = ap.parse_args()
    import onnxruntime as ort
    sys.path.insert(0, a.feat89)
    import wave
    import feat89 as F
    from scipy.signal import resample_poly
    from scmg.evaluate import load
    from scmg.infer import Streamer
    from scmg.common import U80
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads(Path(a.stats).read_text())
    m, vae = load(a.ckpt, a.vae, 'cuda')
    for p in list(m.parameters()) + list(vae.parameters()):
        p.requires_grad_(False)                 # traced as constants (frozen inference weights)
    qm = np.asarray(stats['q_mean'], np.float32); fm = np.asarray(stats['f_mean'], np.float32); fs = np.asarray(stats['f_std'], np.float32)
    w = wave.open(a.wav); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float32) / 32768
    if sr != 16000:
        from math import gcd
        g = gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    feat = lambda T: np.concatenate([F.gl_build.logmel(resample_poly(x[:T * 320], 3, 2), T, F.mel()), np.zeros((T, 9), np.float32)], 1)
    K = 12
    g = torch.Generator(device='cuda').manual_seed(0)
    cond = dict(text_c=torch.randn(K, 768, device='cuda', generator=g) * .1, ttime=torch.full((K,), float('nan'), device='cuda'),
                tmask=torch.ones(K, dtype=torch.bool, device='cuda'), spk=torch.zeros(192, device='cuda'),
                spk_valid=torch.tensor(0., device='cuda'), source=torch.tensor(5, device='cuda'),
                ref=torch.zeros(128, 34, device='cuda'), ref_valid=torch.tensor(0., device='cuda'))
    s = Streamer(m, vae, torch.zeros(1, 89, device='cuda'), cond, stats, H=2 * U80, steps=10, standby=qm, ctx_t=m.ctx_t,
                 cfg_w=2.5, text_mode='x2', x2_rate=15., x2_lag=.15, deploy_timing=True, trim_start=True, cfg_null_fix=True)
    mods = [a.only] if a.only else MODS
    caps = {n: Capture(m, n, a.tick) for n in mods}
    for k in range(a.tick + 1):
        for c in caps.values():
            c.tick = k
        T = 16 * k + 8
        s.feat = torch.as_tensor((feat(T) - fm) / fs, device='cuda').float()
        with torch.inference_mode():
            s.tick()
    for c in caps.values():
        c.restore()
    m.cpu(); vae.cpu()                          # export on CPU: tracing records some shape-derived tensors on the CPU
    def to_cpu(d):
        return {k: v.cpu() for k, v in d.items()}
    for c in caps.values():
        for call in c.calls:
            call['tensors'] = to_cpu(call['tensors'])
    rep = []
    for n, c in caps.items():
        if not c.calls:
            rep.append(dict(module=n, error='not called')); continue
        call = c.calls[0]
        try:
            r = export_one(n, getattr(m, n), call, out_dir, ort, a.provider)
        except Exception as e:
            r = dict(module=n, error=f'{type(e).__name__}: {str(e)[:400]}')
        r['calls_in_tick'] = len(c.calls)
        rep.append(r); print(json.dumps(r, ensure_ascii=False)[:1500], flush=True)
    (out_dir / 'EXPORT.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()

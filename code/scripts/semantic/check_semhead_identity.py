"""Zero-init identity check of the semantic head switches (11-05).

Builds, from ONE checkpoint (hos-base), (a) the control model = checkpoint + adsc (what the control lane starts from) and
(b) the semantic model = checkpoint + adsc + sem_head + sem_cond (train.py fine-tune surgery: build_model with the new
switches -> zero-initialised stream_emb / sem_emb), then generates the same dev clips offline (H = inf) and engine-streaming
(x2 text, H = 160 ms) with the same seed and requires BITWISE identical outputs.  --smoke additionally runs one inference
of a trained smoke checkpoint (finite, right shape) and reports how far it moved from the start.
    python scripts/semantic/check_semhead_identity.py --init <hos-base best.pt> --vae <vae> --view <view> --stats <json>
        --text-cache <dir> [--smoke <final.pt>] [--n 2]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg.common import CHUNK_FRAMES, U80, P
from scmg.data import View, WindowDataset
from scmg.evaluate import load
from scmg.infer import generate
from scmg.model import INF
from scmg.train import build_model
from scmg.vae import MotionVAE

KEYS = ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c')


def surgery(ck, switches, dev):
    m, _ = build_model(dict(ck['base_cfg']), dict(ck['extra_cfg']), {**ck['extra_cfg'], **switches}, ck['ema'])
    for k, v in switches.items(): setattr(m.cfg, k, v)
    m.add_modules(); m.to(dev).eval()
    sc = ck.get('sc') or {}
    m.hist_measured = bool(sc.get('state_hist_measured', False))
    m.ctx_t = float(sc.get('ctx_infer_t', 0.9)) if sc.get('ctx_noise_p', 0) > 0 else 1.0
    return m


def run(m, vae, view, stats, cis, proto):
    outs = []
    for ci in cis:
        L = view.clips[ci]['frames']; nch = min(L // CHUNK_FRAMES, 25)
        s = WindowDataset(view, nch * CHUNK_FRAMES, max_tokens=256, max_ids=160).sample(ci, 0, np.random.default_rng(ci))
        cond = {k: s[k] for k in KEYS if k in s}
        kw = dict(H=INF) if proto == 'offline' else dict(H=2 * U80, text_mode='x2', x2_rate=15.0, x2_lag=0.15, deploy_timing=True)
        with torch.no_grad():
            y = generate(m, vae, s['feat'], cond, stats, nch, steps=10, seed=0, ctx_t=m.ctx_t, cfg_w=2.5, **kw)[0]
        outs.append(np.asarray(y, np.float32))
    return outs


def main():
    ap = argparse.ArgumentParser()
    for k in ('init', 'vae', 'view', 'stats', 'text_cache'): ap.add_argument('--' + k.replace('_', '-'), required=True)
    ap.add_argument('--smoke'); ap.add_argument('--n', type=int, default=2)
    a = ap.parse_args(); dev = 'cuda'
    stats = json.loads(Path(P(a.stats)).read_text())
    view = View(a.view, 'val', a.text_cache, stats, mmap=True)
    cis = list(range(a.n))
    vck = torch.load(P(a.vae), map_location='cpu')
    vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema']); vae.eval()
    ck = torch.load(a.init, map_location='cpu')
    ctrl = surgery(ck, {'adsc': True}, dev)
    sem = surgery(ck, {'adsc': True, 'sem_head': True, 'sem_cond': True}, dev)
    assert float(sem.sem_emb.weight.abs().max()) == 0.0, 'sem_emb not zero-initialised'
    ok = True; rep = {}
    for proto in ('offline', 'engine'):
        ya, yb = run(ctrl, vae, view, stats, cis, proto), run(sem, vae, view, stats, cis, proto)
        same = all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(ya, yb))
        rep[proto] = dict(bitwise_identical=same, max_abs_diff=float(max(np.abs(x - y).max() for x, y in zip(ya, yb))),
                          frames=[len(x) for x in ya])
        ok &= same
    print('IDENTITY', json.dumps(rep), 'PASS' if ok else 'FAIL', flush=True)
    if a.smoke:
        m, _ = load(a.smoke, a.vae, dev)
        ys = run(m, vae, view, stats, cis[:1], 'offline')
        g = torch.load(a.smoke, map_location='cpu')
        e = g['ema']; moved = float(e['sem_emb.weight'].abs().max()) if 'sem_emb.weight' in e else None
        print('SMOKE_INFER', json.dumps(dict(shape=list(ys[0].shape), finite=bool(np.isfinite(ys[0]).all()),
                                             diff_vs_start=float(np.abs(ys[0] - run(ctrl, vae, view, stats, cis[:1], 'offline')[0]).mean()),
                                             sem_emb_absmax=moved, extra_cfg=g['extra_cfg'])), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

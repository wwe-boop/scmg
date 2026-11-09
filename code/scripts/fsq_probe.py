"""Does the FSQ skeleton block the text?  (11-04 quick check, inference only)

For every VoxCPM planner call made while generating a clip with its own text (windowed LM text = honest protocol), the
same call is repeated with the text masked out (tidm = 0) and everything else identical (audio, motion history, speaker).
The relative change caused by the text is measured at each stage of the planner:
    zpre  = fsq_in(LM state)            (before quantisation)
    q     = FSQ(zpre) / tanh / identity (the skeleton the rest of the model sees)
    base  = fsq_out(q)
    h     = base + RALM residual        (the plan handed to the local flow)
plus FSQ saturation (share of skeleton entries at the two extreme levels) and the share of skeleton codes the text flips.
Rows without any visible text (CFG null rows, windows before the first word) are skipped.

    python scripts/fsq_probe.py <manifest dir> <out.jsonl> label=ckpt[@cfg] ... [--n 20]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scmg.common import CHUNK_FRAMES, U80
from scmg.evaluate import load
from scmg.token_cache import checkpoint_config
from scmg.infer import generate
from scmg.model import INF

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
ap = argparse.ArgumentParser(); ap.add_argument('man'); ap.add_argument('out'); ap.add_argument('models', nargs='+')
ap.add_argument('--n', type=int, default=20); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--vae', default='/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/trained/vae-r3/final.pt')
a = ap.parse_args()
st = json.load(open(D + '/STATS_v36_1.json'))
M = json.load(open(Path(a.man) / 'MANIFEST.json')); ents = M['entries'][:a.n]
CK = ('text', 'ttime', 'tmask', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid')


def rel(x, y, rows):
    d = (x - y)[rows].float().flatten(1).norm(dim=1); n = x[rows].float().flatten(1).norm(dim=1).clamp_min(1e-8)
    return (d / n).tolist()


out = open(a.out, 'a')
for spec in a.models:
    lab, ck = spec.split('=', 1); w = 2.0
    if '@' in ck: ck, w = ck.split('@'); w = float(w)
    cfg = checkpoint_config(ck, 'fsq_probe.py')
    if cfg.planner != 'voxcpm': raise ValueError(f'{lab}: not a VoxCPM planner')
    m, vae = load(ck, a.vae, 'cuda'); V = m.vox; mode = V.ab['fsq']
    acc = dict(zpre=[], q=[], base=[], h=[], code_flip=[], sat=[], calls=0, rows=0)
    orig = V.forward

    def probe(lat, pooled_audio, tids, tidm, g, *args, **kw):
        h, q = orig(lat, pooled_audio, tids, tidm, g, *args, **kw)
        zp = V._aux['fsq_pre']; acc['calls'] += 1
        rows = tidm.bool().any(1)
        if rows.any():
            with torch.no_grad():
                h0, q0 = orig(lat, pooled_audio, tids, torch.zeros_like(tidm), g, *args, **kw); zp0 = V._aux['fsq_pre']
                b, b0 = V.fsq_out(q), V.fsq_out(q0)
                acc['zpre'] += rel(zp, zp0, rows); acc['q'] += rel(q, q0, rows); acc['base'] += rel(b, b0, rows); acc['h'] += rel(h, h0, rows)
                if mode is True:
                    half = V.fsq.half; lev = torch.round(q * half)
                    acc['code_flip'].append(float((lev != torch.round(q0 * half))[rows].float().mean()))
                    acc['sat'].append(float((lev.abs() >= half)[rows].float().mean()))
                elif mode == 'tanh':
                    acc['sat'].append(float((q.abs() > 0.99)[rows].float().mean()))
            V._aux['fsq_pre'] = zp
            acc['rows'] += int(rows.sum())
        return h, q

    V.forward = probe
    for e in ents:
        z = np.load(Path(a.man) / f"cond_{e['ci']}.npz"); s = {k: z[k] for k in z.files}
        c = {k: s[k] for k in CK if k in s}; nch = len(s['feat']) // CHUNK_FRAMES
        generate(m, vae, s['feat'], c, st, nch, H=INF, steps=10, seed=a.seed, ctx_t=m.ctx_t, cfg_w=w,
                 cfg_text=getattr(m.cfg, 'vox_text', True), window_lm_text=True, trim_start=True, cfg_null_fix=True)
    V.forward = orig
    md = lambda k: float(np.median(acc[k])) if acc[k] else None
    r = dict(label=lab, ckpt=ck, fsq=str(mode), lm_init=getattr(m.cfg, 'vox_lm_init', 'pretrained'), clips=len(ents), calls=acc['calls'],
             text_rows=acc['rows'], text_effect_median=dict(zpre=md('zpre'), q=md('q'), base=md('base'), h=md('h')),
             code_flip_rate=md('code_flip'), saturation=md('sat'))
    print(json.dumps(r)); out.write(json.dumps(r) + '\n'); out.flush()
    del m, vae; torch.cuda.empty_cache()

"""Does the model USE the text?  (LM text windowed as in training: code-review R1 fix, patch 09)  python scripts/text_probe.py <out.json> <label=ckpt[@cfg_w]> ... [--n 20]
Same audio, same seed: generate with (a) the clip's own text, (b) text removed (all text masks 0), (c) the text of ANOTHER
clip (times rescaled to this clip's length).  Compare the change of the arm joints against the change caused by the
generation seed alone (a vs a with seed 1):  text_effect = |a - b| / |a - a_seed1|.  ~0 -> text ignored;  ~1 -> text matters as
much as sampling noise;  >1 -> text drives the motion.  Also reports the speed ratio of each variant."""
import argparse, json
import numpy as np, torch
from scmg.common import CHUNK_FRAMES
from scmg.data import View, WindowDataset, VIEW, clip_lm_capacity
from scmg.evaluate import load
from scmg.infer import generate
from scmg.model import INF

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('models', nargs='+'); ap.add_argument('--n', type=int, default=20)
ap.add_argument('--vae', default='/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/trained/vae-r3/final.pt')
a = ap.parse_args()
st = json.load(open(D + '/STATS_v36_1.json'))
v = View(VIEW, 'val', D + '/text-e5-v36_1', st, mmap=True, tok_cache=D + '/tok-qwen25-v36_1'); v.glitch = None
TK = ('text', 'ttime', 'tmask', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c')
CK = TK + ('spk', 'spk_valid', 'source', 'ref', 'ref_valid')
A = slice(4, 18)
clips = [ci for ci in range(len(v.clips)) if v.clips[ci]['frames'] // CHUNK_FRAMES >= 3][:a.n]
S = {}
for ci in clips:
    L = v.clips[ci]['frames']; nch = L // CHUNK_FRAMES
    S[ci] = (WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=256, max_ids=clip_lm_capacity(v, ci)).sample(ci, 0, np.random.default_rng(ci)), nch)


def notext(c):
    c = dict(c)
    for k in ('tmask', 'tidm'):
        if k in c: c[k] = torch.zeros_like(torch.as_tensor(c[k])) if torch.is_tensor(c[k]) else np.zeros_like(c[k])
    return c


def swap(c, d, ratio):
    c = dict(c)
    for k in TK:
        if k in d: c[k] = d[k]
    for k in ('ttime', 'titime', 'tiend'):
        if k in c: c[k] = c[k] * ratio
    return c


res = {}
for spec in a.models:
    lab, ck = spec.split('=', 1); w = 2.0
    if '@' in ck: ck, w = ck.split('@'); w = float(w)
    m, vae = load(ck, a.vae, 'cuda')
    cfg_text = m.cfg.planner != 'voxcpm' or getattr(m.cfg, 'vox_text', True)
    run = lambda s, cond, nch, seed: generate(m, vae, s['feat'], cond, st, nch, H=INF, steps=10, seed=seed, ctx_t=m.ctx_t,
                                              cfg_w=w, cfg_text=cfg_text, window_lm_text=m.cfg.planner == 'voxcpm')[0]
    rows = []
    for i, ci in enumerate(clips):
        s, nch = S[ci]; cond = {k: s[k] for k in CK if k in s}
        o, oj = clips[(i + 7) % len(clips)], None
        so, _ = S[o]; dj = {k: so[k] for k in TK if k in so}
        ratio = v.clips[ci]['frames'] / v.clips[o]['frames']
        g0 = run(s, cond, nch, 0); g1 = run(s, cond, nch, 1)
        gn = run(s, notext(cond), nch, 0); gs = run(s, swap(cond, dj, ratio), nch, 0)
        n = min(len(g0), len(g1), len(gn), len(gs))
        d = lambda x, y: float(np.abs(x[:n, A] - y[:n, A]).mean())
        sp = lambda x: float(np.abs(np.diff(x[:n, A], axis=0)).mean())
        rows.append(dict(ci=ci, seed=d(g0, g1), notext=d(g0, gn), swap=d(g0, gs), sp0=sp(g0), spn=sp(gn), sps=sp(gs)))
        print(lab, rows[-1], flush=True)
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    res[lab] = dict(ckpt=ck, cfg_w=w, clips=len(rows), d_seed=mean('seed'), d_notext=mean('notext'), d_swap=mean('swap'),
                    notext_over_seed=mean('notext') / mean('seed'), swap_over_seed=mean('swap') / mean('seed'),
                    speed_notext_over_own=mean('spn') / mean('sp0'), speed_swap_over_own=mean('sps') / mean('sp0'), rows=rows)
    print(lab, {k: round(x, 3) for k, x in res[lab].items() if isinstance(x, float)}, flush=True)
    del m, vae; torch.cuda.empty_cache()
    open(a.out, 'w').write(json.dumps(res, indent=1))

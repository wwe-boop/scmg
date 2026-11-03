"""Generate on the semantic-recognizer manifest (held-out semsynth clips with intent events) for text-use evaluation.
    python scripts/gen_semantic.py <manifest dir> <out root> <label=ckpt[@cfg_w]> ... [--variants own,notext,swap] [--n 200]
For each model and variant writes <out root>/<label>/<variant>/gen_<ci>.npy ([T,34] rad, 50 Hz); then score with
    scripts/semantic_eval.py score --gen-dir <dir> --manifest <manifest dir>/MANIFEST.json
own = the clip's own text; notext = all text masks 0; swap = text of another manifest clip (times rescaled).
Same seed (0) for all variants; LM text windowed as in training (VoxCPM planner)."""
import argparse, json
from pathlib import Path
import numpy as np, torch
from scmg.common import CHUNK_FRAMES
from scmg.evaluate import load
from scmg.infer import generate
from scmg.model import INF

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
ap = argparse.ArgumentParser(); ap.add_argument('man'); ap.add_argument('out'); ap.add_argument('models', nargs='+')
ap.add_argument('--variants', default='own,notext,swap'); ap.add_argument('--n', type=int, default=200)
ap.add_argument('--cfg-t', type=float, default=None)
ap.add_argument('--vae', default='/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/trained/vae-r3/final.pt')
a = ap.parse_args()
st = json.load(open(D + '/STATS_v36_1.json'))
M = json.load(open(Path(a.man) / 'MANIFEST.json'))
ents = M['entries'][:a.n]
TK = ('text', 'ttime', 'tmask', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c')
CK = TK + ('spk', 'spk_valid', 'source', 'ref', 'ref_valid')
S = {}
for e in ents:
    z = np.load(Path(a.man) / f"cond_{e['ci']}.npz"); S[e['ci']] = {k: z[k] for k in z.files}
cis = [e['ci'] for e in ents]


def variant(ci, v):
    s = S[ci]; c = {k: s[k] for k in CK if k in s}
    if v == 'notext':
        for k in ('tmask', 'tidm'):
            if k in c: c[k] = np.zeros_like(c[k])
    elif v == 'swap':
        o = cis[(cis.index(ci) + 7) % len(cis)]; so = S[o]
        r = len(s['feat']) / len(so['feat'])
        for k in TK:
            if k in so: c[k] = so[k]
        for k in ('ttime', 'titime', 'tiend'):
            if k in c: c[k] = c[k] * r
    return c


for spec in a.models:
    lab, ck = spec.split('=', 1); w = 2.0
    if '@' in ck: ck, w = ck.split('@'); w = float(w)
    m, vae = load(ck, a.vae, 'cuda')
    cfg_text = m.cfg.planner != 'voxcpm' or getattr(m.cfg, 'vox_text', True)
    for v in a.variants.split(','):
        od = Path(a.out) / lab / v; od.mkdir(parents=True, exist_ok=True)
        for ci in cis:
            f = od / f'gen_{ci}.npy'
            if f.exists(): continue
            s = S[ci]; nch = len(s['feat']) // CHUNK_FRAMES
            y = generate(m, vae, s['feat'], variant(ci, v), st, nch, H=INF, steps=10, seed=0, ctx_t=m.ctx_t, cfg_w=w,
                         cfg_text=cfg_text, cfg_t=a.cfg_t, window_lm_text=m.cfg.planner == 'voxcpm')[0]
            np.save(f, y[:nch * CHUNK_FRAMES].astype(np.float32))
        print(lab, v, 'done', flush=True)
    del m, vae; torch.cuda.empty_cache()

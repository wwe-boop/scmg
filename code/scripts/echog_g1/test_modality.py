"""11-05 input-modality switch of the Streamer (scmg/infer.py audio_off) and infer_test.py (--modality, --text-swap).  SCMG_J=39 python scripts/echog_g1/test_modality.py <ckpt> <vae> <view> <text cache> [split]
 a default Streamer == explicit audio_off=False (bitwise)
 b text-only (audio_off): output does not depend on the audio features at all (feat replaced by noise -> bitwise equal)
 c audio-only (tmask = 0): output does not depend on the text at all (text features / times replaced -> bitwise equal)
 d both single-modality outputs differ from the default (each modality is actually used)
 e audio_off + tmask 0 (nothing visible) makes the conditional branch identical to the CFG null branch: cfg_w 2.5 == cfg_w 1
   (so single-modality CFG = kept modality vs both dropped, with the unchanged null branch)
The infer_test.py end-to-end parity (HEAD vs working tree, default arguments) is run by modality_ablation.sh check.
Prints ALL_OK / SOME_FAIL."""
import json, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg.common import CHUNK_FRAMES, U80
from scmg.data import View, WindowDataset
from scmg.evaluate import load
from scmg.infer import generate
from scmg.model import INF

CK, VAE, VIEW, TEXT = sys.argv[1:5]; SPLIT = sys.argv[5] if len(sys.argv) > 5 else 'val'
STATS = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/STATS_echog_g1_v1.json'
st = json.loads(Path(STATS).read_text())
v = View(VIEW, SPLIT, TEXT, st)
ci = 0; L = v.clips[ci]['frames']; nch = min(12, -(-L // CHUNK_FRAMES))
s = WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
s['ref'] = np.zeros_like(s['ref']); s['ref_valid'] = np.float32(0.)
KEYS = ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend')
cond0 = {k: np.array(s[k]) for k in KEYS if k in s}
m, vae = load(CK, VAE, 'cuda')
qm = np.asarray(st['q_mean'], np.float32)


def gen(feat=None, cond=None, H=INF, cfg_w=2.5, **kw):
    y, _, _ = generate(m, vae, s['feat'] if feat is None else feat, cond0 if cond is None else cond, st, nch, H=H, steps=4, seed=0,
                       standby=qm, ctx_t=m.ctx_t, cfg_w=cfg_w, **kw)
    return y


ok = True
def check(n, c, msg=''):
    global ok; ok &= bool(c); print(f'{n:64s} {"ok" if c else "FAIL"} {msg}', flush=True)
d = lambda x, y: float(np.abs(x - y).max())

y0 = gen()
check('a default == audio_off=False', np.array_equal(y0, gen(audio_off=False)))
rng = np.random.default_rng(1)
feat_n = rng.standard_normal(s['feat'].shape).astype(np.float32)
yt = gen(audio_off=True); yt2 = gen(feat=feat_n, audio_off=True)
check('b text-only ignores the audio features (noise feat)', np.array_equal(yt, yt2), f'max|d| {d(yt, yt2):.2e}')
ya_c = dict(cond0); ya_c['tmask'] = np.zeros_like(ya_c['tmask'])
yb_c = dict(ya_c); yb_c['text'] = rng.standard_normal(ya_c['text'].shape).astype(ya_c['text'].dtype)
yb_c['ttime'] = np.where(np.isfinite(ya_c['ttime']), ya_c['ttime'] * 0.5, ya_c['ttime']).astype(ya_c['ttime'].dtype)
ya, yb = gen(cond=ya_c), gen(cond=yb_c)
check('c audio-only ignores the text (random features, other times)', np.array_equal(ya, yb), f'max|d| {d(ya, yb):.2e}')
check('d text-only differs from both', not np.array_equal(yt, y0), f'max|d| {d(yt, y0):.3f}')
check('d audio-only differs from both', not np.array_equal(ya, y0), f'max|d| {d(ya, y0):.3f}')
H2 = 2 * U80
ytE, ytE2 = gen(H=H2, audio_off=True), gen(H=H2, feat=feat_n, audio_off=True)
check('b streaming H: text-only ignores the audio features', np.array_equal(ytE, ytE2), f'max|d| {d(ytE, ytE2):.2e}')
yn1, yn25 = gen(cond=ya_c, audio_off=True, cfg_w=1.0), gen(cond=ya_c, audio_off=True, cfg_w=2.5)
check('e nothing visible: cond == uncond (CFG 2.5 == CFG 1)', d(yn1, yn25) < 1e-4, f'max|d| {d(yn1, yn25):.2e}')
print('ALL_OK' if ok else 'SOME_FAIL')

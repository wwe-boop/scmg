"""Generate on the semantic-recognizer manifest (held-out semsynth clips with intent events) for text-use evaluation.
    python scripts/gen_semantic.py <manifest dir> <out root> <label=ckpt[@cfg_w]> ... [--variants own,notext,swap] [--n 200]
For each model and variant writes <out root>/<label>/<variant>/gen_<ci>.npy ([T,34] rad, 50 Hz); then score with
    scripts/semantic_eval.py score --gen-dir <dir> --manifest <manifest dir>/MANIFEST.json
own = the clip's own text; notext = all text masks 0; swap = text of another manifest clip (times rescaled);
gnull = own text but the Local Flow gets the null g_k (Semantic Gesture Head models only).
Same seed (0) for all variants; LM text windowed as in training (VoxCPM planner)."""
import argparse, json, os
from pathlib import Path
import numpy as np, torch
from scmg.common import CHUNK_FRAMES
from scmg.evaluate import load
from scmg.infer import generate
from scmg.model import INF
from scmg.data import View, WindowDataset, VIEW
from scmg.token_cache import checkpoint_config, resolve_token_cache, validate_manifest_tokens, validate_manifest_cache

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
ap = argparse.ArgumentParser(); ap.add_argument('man'); ap.add_argument('out'); ap.add_argument('models', nargs='+')
ap.add_argument('--variants', default='own,notext,swap'); ap.add_argument('--n', type=int, default=200)
ap.add_argument('--cfg-t', type=float, default=None)
ap.add_argument('--vae', default='/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/trained/vae-r3/final.pt')
ap.add_argument('--tok-cache')
ap.add_argument('--text-mode', choices=('final', 'x2', 'segment'), default='final')
ap.add_argument('--x2-lag', type=float, default=0.15); ap.add_argument('--x2-rate', type=float, default=15.0)
ap.add_argument('--deploy-timing', action='store_true')
ap.add_argument('--window-lm-text', action=argparse.BooleanOptionalAction, default=None)
ap.add_argument('--trim-start', action='store_true'); ap.add_argument('--cfg-null-fix', action='store_true')
a = ap.parse_args()
if a.text_mode == 'x2':
    causal = os.environ.get('SCMG_TEXT_CAUSAL')
    if not causal or not (Path(causal) / 'TEXT_FEAT.npy').is_file():
        raise ValueError('x2 requires causal E5: set SCMG_TEXT_CAUSAL=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/text-e5-causal-v36_1')
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
        for k in ('ttime', 'titime', 'tseg', 'tiseg', 'tiend'):        # 11-03 review N6: segment starts too (window text)
            if k in c: c[k] = c[k] * r
    return c


for spec in a.models:
    lab, ck = spec.split('=', 1); w = 2.0
    if '@' in ck: ck, w = ck.split('@'); w = float(w)
    cfg = checkpoint_config(ck, 'gen_semantic.py')
    tok = resolve_token_cache(cfg, a.tok_cache)
    if tok: validate_manifest_cache(M, tok)
    # Check legacy manifest tokens against the selected cache; attach causal E5 from its matching rows.
    if tok or a.text_mode == 'x2':
        view = View(M.get('view', VIEW), M.get('view_split', 'train'), D + '/text-e5-v36_1', st, mmap=True, tok_cache=tok)
        for e in ents:
            ci = e['ci']; saved = S[ci]
            if [it['id'] for it in view.clips[ci]['items']] != e['item_ids']:
                raise ValueError(f'manifest / View clip mismatch at ci={ci}')
            sample = WindowDataset(view, len(saved['feat']), max_tokens=len(saved['text']),
                                   max_ids=len(saved['tids'])).sample(ci, e.get('start_frame', 0), np.random.default_rng(ci))
            if tok: validate_manifest_tokens(saved, sample, ci)
            if a.text_mode == 'x2':
                if not np.array_equal(saved['text'], sample['text']):
                    raise ValueError(f'manifest / causal E5 row mismatch at ci={ci}')
                saved['text_c'] = sample['text_c']
        del view
    m, vae = load(ck, a.vae, 'cuda')
    cfg_text = m.cfg.planner != 'voxcpm' or getattr(m.cfg, 'vox_text', True)
    semc = bool(getattr(m.cfg, 'sem_head', False) and getattr(m.cfg, 'sem_cond', False))
    for v in a.variants.split(','):
        if v == 'gnull' and not semc: continue          # own text, Local Flow given the null g_k: how much the flow uses g
        gk = dict(g_override=torch.zeros(m.cfg.n_sem, device='cuda')) if v == 'gnull' else {}
        od = Path(a.out) / lab / v; od.mkdir(parents=True, exist_ok=True)
        for ci in cis:
            f = od / f'gen_{ci}.npy'
            s = S[ci]; nch = len(s['feat']) // CHUNK_FRAMES
            if f.exists():
                try:
                    old = np.load(f, allow_pickle=False)
                    if old.shape == (nch * CHUNK_FRAMES, 34) and np.isfinite(old).all(): continue
                except (OSError, ValueError): pass
            y = generate(m, vae, s['feat'], variant(ci, v), st, nch, H=INF, steps=10, seed=0, ctx_t=m.ctx_t, cfg_w=w,
                         cfg_text=cfg_text, cfg_t=a.cfg_t, window_lm_text=(m.cfg.planner == 'voxcpm' if a.window_lm_text is None else a.window_lm_text),
                         text_mode=a.text_mode, x2_lag=a.x2_lag, x2_rate=a.x2_rate, deploy_timing=a.deploy_timing,
                         trim_start=a.trim_start, cfg_null_fix=a.cfg_null_fix, **gk)[0]
            import tempfile
            with tempfile.NamedTemporaryFile(dir=od, prefix=f.name + '.pending-', suffix='.npy', delete=False) as tmp:
                np.save(tmp, y[:nch * CHUNK_FRAMES].astype(np.float32))
            check = np.load(tmp.name)
            if check.shape != (nch * CHUNK_FRAMES, 34) or not np.isfinite(check).all():
                raise ValueError(f'invalid generated artifact: {tmp.name}')
            os.replace(tmp.name, f)
        print(lab, v, 'done', flush=True)
    del m, vae; torch.cuda.empty_cache()

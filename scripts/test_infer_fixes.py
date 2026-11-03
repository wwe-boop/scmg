"""11-03 review N1/N3 inference switches. python scripts/test_infer_fixes.py <vae.pt> <orig_code_dir> <vox ckpt>
 a default (trim_start=False, cfg_null_fix=False) is bit-identical to the pre-change streamer
 b trim_start: windows at the clip start have W = k + 1 + LOOK chunks (no pre-clip pseudo chunks), full WIN afterwards
 c cfg_null_fix changes the CFG output (null branch keeps the visibility mask); both switches run end to end
Prints ALL_OK / SOME_FAIL."""
import json, os, subprocess, sys
import numpy as np, torch
TEST_TMP = os.environ.get('SCMG_TEST_TMP', '/tmp/claude-1000')
os.makedirs(TEST_TMP, exist_ok=True)
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'; VAE, ORIG, CK = sys.argv[1:4]
CHILD = os.environ.get('IF_CHILD')
def run(**kw):
    from scmg.data import View, WindowDataset
    from scmg.evaluate import load
    from scmg.infer import generate, Streamer
    st = json.load(open(D + '/STATS_v36_1.json'))
    v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', tok_cache=D + '/tok-qwen25-v36_1')
    s = WindowDataset(v, 22 * 16, max_tokens=256).sample(0, 0, np.random.default_rng(0))
    m, vae = load(CK, VAE, 'cuda')
    cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend') if k in s}
    S = Streamer(m, vae, s['feat'], cond, st, H=10 ** 9, steps=4, seed=0, standby=st['q_mean'], ctx_t=m.ctx_t, cfg_w=2.0, **kw)
    Ws = []
    for i in range(20):
        S.tick(None, None, None); Ws.append(getattr(S, '_W', 16))
    return torch.stack(S.frames).cpu().numpy(), Ws
if CHILD:
    y, _ = run(); np.save(os.environ['IF_OUT'], y); sys.exit(0)
ok = True
def check(n, c, msg=''):
    global ok; ok &= bool(c); print(f'{n:58s} {"ok" if c else "FAIL"} {msg}', flush=True)
out = os.path.join(TEST_TMP, 'if_ref.npy'); os.makedirs(TEST_TMP, exist_ok=True)
r = subprocess.run([sys.executable, '-c', f'import sys; sys.argv={sys.argv[:4]!r}; exec(open({os.path.abspath(__file__)!r}).read())'],
                   env=dict(os.environ, IF_CHILD='1', IF_OUT=out, PYTHONPATH=ORIG), cwd=ORIG)
y0, W0 = run()
check('a default streamer bit-identical to pre-change', r.returncode == 0 and np.array_equal(y0, np.load(out)), f'W {W0[:3]}')
y1, W1 = run(trim_start=True)
check('b trim_start windows grow from the clip start', W1[:14] == [3 + k for k in range(14)] and set(W1[14:]) == {16}, f'W {W1}')
check('b ... and change the start of the output', not np.array_equal(y0, y1), f'max|d| first 1 s {np.abs(y0[:50] - y1[:50]).max():.3f}')
y2, _ = run(cfg_null_fix=True)
check('c cfg_null_fix changes the CFG output', not np.array_equal(y0, y2) and np.isfinite(y2).all(), f'max|d| {np.abs(y0 - y2).max():.3f}')
y3, _ = run(trim_start=True, cfg_null_fix=True); check('c both switches together run', np.isfinite(y3).all())
print('ALL_OK' if ok else 'SOME_FAIL')

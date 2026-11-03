"""TTS hidden-state lane acceptance (11-03, Claude). usage: python scripts/test_ttsh.py <vae.pt> [orig_code_dir]
 A  tts_rows alignment on synthetic items (row r = window frames [4r, 4r+4); codec k covers item frames [4k, 4k+4))
 B  default OFF is bit-identical to the pre-change code (orig_code_dir, e.g. ~/bulk/scmg-ttsh-orig): loss + streaming output
 C  tts_h ON at zero init == the same weights without tts_h (loss, audio tokens, streaming output)
 D  gradient reaches tts_in; E  streaming H=0: rows not yet arrived do not change emitted frames, arrived rows do
 F  real export (tts-hidden-v36_1): row mask coverage on train/val windows, one training step finite
Prints ALL_OK / SOME_FAIL."""
import json, os, subprocess, sys
import numpy as np, torch
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'; TTS = D + '/tts-hidden-v36_1'; dev = 'cuda'
VAE = sys.argv[1]; ORIG = sys.argv[2] if len(sys.argv) > 2 else None
CHILD = os.environ.get('TTSH_CHILD')            # 'ref': run B's reference computation with whatever scmg is on PYTHONPATH


def setup():
    from scmg.data import View, WindowDataset, collate
    from scmg.vae import MotionVAE
    st = json.load(open(D + '/STATS_v36_1.json'))
    v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1')
    ds = WindowDataset(v, 256, max_tokens=128)
    b = {k: x.to(dev) for k, x in collate([ds.sample(i, 0, np.random.default_rng(i)) for i in (0, 3)]).items()}
    vck = torch.load(VAE, map_location='cpu'); vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema']); vae.eval()
    stats = dict(qm=torch.tensor(st['q_mean'], device=dev), qs=torch.tensor(st['q_std'], device=dev))
    return st, v, ds, b, vae, stats


def ref_outputs(m, vae, b, st, ds, stats, extra_cond=None):
    """loss (fixed RNG), streaming output (H=4 and H=0) of model m on fixed inputs."""
    from scmg.losses import StageCfg, step_losses
    from scmg.infer import generate
    m.train(); torch.manual_seed(123); np.random.seed(123)
    out = step_losses(m, vae, b, StageCfg(), stats)
    loss = (out['loss'] if isinstance(out, dict) else out[0]).detach().float().cpu().numpy()
    m.eval(); s = ds.sample(0, 0, np.random.default_rng(0))
    cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid')}
    if extra_cond: cond.update(extra_cond)
    ys = [generate(m, vae, s['feat'][:192], cond, st, 12, steps=2, standby=st['q_mean'], H=h, seed=0)[0] for h in (4, 0)]
    return loss, ys


def base_model(vae, **kw):
    from scmg.model import SCMG, Config
    torch.manual_seed(0)
    return SCMG(Config(z=vae.z, **kw)).to(dev)


if CHILD == 'ref':
    st, v, ds, b, vae, stats = setup()
    m = base_model(vae)
    loss, ys = ref_outputs(m, vae, b, st, ds, stats)
    np.savez(os.environ['TTSH_OUT'], loss=loss, y4=ys[0], y0=ys[1]); sys.exit(0)

ok = True
def check(name, c, msg=''):
    global ok; ok &= bool(c); print(f'{name:44s} {"ok" if c else "FAIL"} {msg}', flush=True)

# ---------------------------------------------------------------- A
from scmg.data import tts_rows, TTS_DIM
def fake(n_codec, tag):
    a = np.zeros((n_codec, TTS_DIM), np.float16); a[:, 0] = np.arange(n_codec); a[:, 1] = tag; return a
arrs = {'a': fake(10, 1), 'b': fake(6, 2)}
items = [dict(id='a', offset=100, frames=40), dict(id='b', offset=142, frames=22), dict(id='c', offset=170, frames=8)]  # c: no export
# window starts at clip frame s=4 (clip_start=100 -> item a starts at window frame -4); 64 frames -> 16 rows
h, m_ = tts_rows(items, 100, 4, 64, 16, lambda i: arrs.get(i))
exp_k = {r: (4 + 4 * r) // 4 for r in range(9)}                  # rows 0..8 -> item a codec 1..9 (frames 4..39 of a)
okA = all(m_[r] == 1 and h[r, 1] == 1 and h[r, 0] == exp_k[r] for r in range(9))
# item b starts at clip frame 142 = window frame 38 (mid-row 9: row 9 first frame = 40 = b frame 2 -> codec 0);
# rows 9..14 cover window frames 36..60 ; b frames [0,22) -> window [38,60) -> rows whose first frame in [38,60): 10..14 (first frames 40..56)
okA &= all(m_[r] == 1 and h[r, 1] == 2 and h[r, 0] == (4 + 4 * r - 42) // 4 for r in range(10, 15))
okA &= m_[9] == 0 and m_[15] == 0                              # row 9 first frame 40? -> recheck below
check('A tts_rows alignment', okA, f'mask={m_.astype(int).tolist()} k={h[:, 0].astype(int).tolist()}')
# lag property: a row never uses a codec frame that starts after the row's first frame
lagok = True
for r in range(16):
    if m_[r]:
        it = items[0] if h[r, 1] == 1 else items[1]
        cstart = it['offset'] - 100 + 4 * int(h[r, 0])          # clip frame where that codec frame starts
        lagok &= cstart <= 4 + 4 * r < cstart + 4
check('A codec frame contains row first frame', lagok)

# ---------------------------------------------------------------- B
st, v, ds, b, vae, stats = setup()
if ORIG:
    out = '/tmp/claude-1000/ttsh_ref.npz'; os.makedirs('/tmp/claude-1000', exist_ok=True)
    env = dict(os.environ, TTSH_CHILD='ref', TTSH_OUT=out, PYTHONPATH=ORIG)
    r = subprocess.run([sys.executable, '-c', f'import sys; sys.argv={sys.argv[:2]!r}; exec(open({os.path.abspath(__file__)!r}).read())'],
                       env=env, cwd=ORIG)
    if r.returncode == 0:
        ref = np.load(out)
        m = base_model(vae); loss, ys = ref_outputs(m, vae, b, st, ds, stats)
        check('B default OFF == pre-change (loss)', np.array_equal(loss, ref['loss']), f'{float(loss):.6f} vs {float(ref["loss"]):.6f}')
        check('B default OFF == pre-change (stream H=4)', np.array_equal(ys[0], ref['y4']), f'max|d|={np.abs(ys[0] - ref["y4"]).max():.2e}')
        check('B default OFF == pre-change (stream H=0)', np.array_equal(ys[1], ref['y0']), f'max|d|={np.abs(ys[1] - ref["y0"]).max():.2e}')
        del m
    else:
        check('B reference run', False, f'rc={r.returncode}')

# ---------------------------------------------------------------- C / D / E (synthetic hidden rows, real audio)
from scmg.losses import StageCfg, step_losses
from scmg.infer import generate
g = torch.Generator().manual_seed(7)
B_, T = b['feat'].shape[:2]
b2 = dict(b, tts_h=torch.randn(B_, T // 4, TTS_DIM, generator=g).half().to(dev), tts_hm=torch.ones(B_, T // 4, device=dev))
s0 = ds.sample(0, 0, np.random.default_rng(0)); F = 192
th = torch.randn(F // 4, TTS_DIM, generator=g).half().numpy(); thm = np.ones(F // 4, np.float32)
m0 = base_model(vae); sd = m0.state_dict()
m1 = base_model(vae, tts_h=True); missing, unexpected = m1.load_state_dict(sd, strict=False)
check('C only tts_in is new', all(k.startswith('tts_in.') for k in missing) and not unexpected, f'missing={missing}')
m0.eval(); m1.eval()                                             # dropout off for the token comparison
with torch.no_grad():
    causal = torch.zeros(B_, dtype=torch.bool, device=dev)
    a0 = m0.audio(b['feat'], causal=causal); a1 = m1.audio_tokens(b['feat'], causal, b2['tts_h'], b2['tts_hm'])
check('C zero-init audio tokens identical', torch.equal(a0, a1))
l0, y0 = ref_outputs(m0, vae, b, st, ds, stats)
l1, y1 = ref_outputs(m1, vae, b2, st, ds, stats, extra_cond=dict(tts_h=th, tts_hm=thm))
check('C zero-init loss identical', np.array_equal(l0, l1), f'{float(l0):.6f} vs {float(l1):.6f}')
check('C zero-init stream identical', np.array_equal(y0[0], y1[0]) and np.array_equal(y0[1], y1[1]))
# D
m1.train(); m1.zero_grad(); torch.manual_seed(1)
out = step_losses(m1, vae, b2, StageCfg(), stats); loss = out['loss'] if isinstance(out, dict) else out[0]; loss.backward()
gn = float(m1.tts_in[1].weight.grad.norm()) if m1.tts_in[1].weight.grad is not None else 0.
check('D gradient reaches tts_in', gn > 0, f'|g|={gn:.2e}')
# E: give tts_in real weights, then perturb rows of frames >= cut
with torch.no_grad(): m1.tts_in[1].weight.normal_(0, 0.02)
m1.eval()
cond = {k: s0[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid')}
def run(hrows, H):
    return generate(m1, vae, s0['feat'][:F], dict(cond, tts_h=hrows, tts_hm=thm), st, 12, steps=2, standby=st['q_mean'], H=H, seed=0)[0]
cut = 96                                                         # frame; rows >= cut // 4 perturbed
th2 = th.copy(); th2[cut // 4:] = np.random.default_rng(3).standard_normal(th2[cut // 4:].shape).astype(np.float16)
ya, yb = run(th, 0), run(th2, 0)
n = min(len(ya), len(yb))
d_before = np.abs(ya[:min(cut, n)] - yb[:min(cut, n)]).max(); d_after = np.abs(ya[cut:n] - yb[cut:n]).max() if n > cut else 0
check('E H=0 future rows do not leak', d_before == 0, f'max|d| frames<{cut}: {d_before:.2e}')
check('E H=0 arrived rows are used', d_after > 0, f'max|d| frames>={cut}: {d_after:.2e}')
th3 = th.copy(); th3[:4] = 0
check('E early rows change output', np.abs(run(th3, 0) - ya).max() > 0)
del m0, m1; torch.cuda.empty_cache()

# ---------------------------------------------------------------- F real export
from scmg.data import View, WindowDataset, collate
for split in ('val', 'train'):
    vv = View(split=split, stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', tts_hidden=TTS)
    dd = WindowDataset(vv, 256, max_tokens=128); rng = np.random.default_rng(0)
    idx = rng.choice(len(vv.clips), 24, replace=False)
    ss = [dd.sample(int(i), 0, np.random.default_rng(int(i))) for i in idx]
    cov = np.mean([x['tts_hm'].mean() for x in ss]); finite = all(np.isfinite(x['tts_h'].astype(np.float32)).all() for x in ss)
    print(f'F {split}: export index {len(vv.tts["index"])} items, row coverage on 24 windows {cov:.3f}, finite={finite}')
    check(f'F {split} rows present and finite', cov > 0.5 and finite, f'coverage={cov:.3f}')
bb = {k: x.to(dev) for k, x in collate(ss[:2]).items()}
m = base_model(vae, tts_h=True).train()
with torch.autocast('cuda', torch.bfloat16):
    out = step_losses(m, vae, bb, StageCfg(), stats)
loss = out['loss'] if isinstance(out, dict) else out[0]; loss.backward()
check('F real-export training step finite', bool(torch.isfinite(loss)), f'loss={float(loss):.3f}')
print('ALL_OK' if ok else 'SOME_FAIL')

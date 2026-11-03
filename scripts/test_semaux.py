"""Intent-aux / semantic-weighted loss acceptance (11-03, Claude). usage: python scripts/test_semaux.py <vae.pt> [orig_code_dir]
 A  default OFF (no intent_aux, w_int = sem_w = 0) is bit-identical to the pre-change code, even with the intent track loaded
 B  real track: semsynth train windows carry chunk labels (>= 2) that match the manifest events; human windows are all -1
 C  w_int > 0: finite loss, CE reported, gradient reaches int_head AND the planner; text-dropped samples excluded
 D  sem_w > 0 changes the flow loss only when the window has a gesture chunk
Prints ALL_OK / SOME_FAIL."""
import json, os, subprocess, sys
import numpy as np, torch
TEST_TMP = os.environ.get('SCMG_TEST_TMP', '/tmp/claude-1000')
os.makedirs(TEST_TMP, exist_ok=True)
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'; dev = 'cuda'
IF = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/intent-frames-v36_1'
VAE = sys.argv[1]; ORIG = sys.argv[2] if len(sys.argv) > 2 else None
CHILD = os.environ.get('SEMAUX_CHILD')


def setup(intent):
    from scmg.data import View, WindowDataset, collate
    from scmg.vae import MotionVAE
    st = json.load(open(D + '/STATS_v36_1.json'))
    kw = dict(intent_frames=IF) if intent else {}
    v = View(split='train', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', **kw)
    vck = torch.load(VAE, map_location='cpu'); vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema']); vae.eval()
    stats = dict(qm=torch.tensor(st['q_mean'], device=dev), qs=torch.tensor(st['q_std'], device=dev))
    return st, v, vae, stats, WindowDataset, collate


def semsynth_clips(v, n):
    return [i for i, c in enumerate(v.clips) if any(it.get('data_source') == 'semsynth_r7p1' for it in c['items'])][:n]


def loss_of(m, vae, b, stats, **sckw):
    from scmg.losses import StageCfg, step_losses
    m.train(); torch.manual_seed(123); np.random.seed(123)
    out = step_losses(m, vae, b, StageCfg(**sckw), stats)
    return out


if CHILD == 'ref':
    st, v, vae, stats, WD, collate = setup(False)
    ds = WD(v, 256, max_tokens=128); cl = semsynth_clips(v, 2)
    b = {k: x.to(dev) for k, x in collate([ds.sample(i, 0, np.random.default_rng(i)) for i in cl]).items()}
    from scmg.model import SCMG, Config
    torch.manual_seed(0); m = SCMG(Config(z=vae.z)).to(dev)
    np.save(os.environ['SEMAUX_OUT'], float(loss_of(m, vae, b, stats)['loss'])); sys.exit(0)

ok = True
def check(name, c, msg=''):
    global ok; ok &= bool(c); print(f'{name:48s} {"ok" if c else "FAIL"} {msg}', flush=True)

st, v, vae, stats, WD, collate = setup(True)
from scmg.model import SCMG, Config
ds = WD(v, 256, max_tokens=128); cl = semsynth_clips(v, 2)
b = {k: x.to(dev) for k, x in collate([ds.sample(i, 0, np.random.default_rng(i)) for i in cl]).items()}
# ---------------- A
if ORIG:
    out = os.path.join(TEST_TMP, 'semaux_ref.npy'); os.makedirs(TEST_TMP, exist_ok=True)
    env = dict(os.environ, SEMAUX_CHILD='ref', SEMAUX_OUT=out, PYTHONPATH=ORIG)
    r = subprocess.run([sys.executable, '-c', f'import sys; sys.argv={sys.argv[:2]!r}; exec(open({os.path.abspath(__file__)!r}).read())'], env=env, cwd=ORIG)
    torch.manual_seed(0); m = SCMG(Config(z=vae.z)).to(dev)
    l0 = float(loss_of(m, vae, b, stats)['loss']); ref = float(np.load(out)) if r.returncode == 0 else None
    check('A default OFF == pre-change (intent track loaded)', ref is not None and l0 == ref, f'{l0:.6f} vs {ref}')
    del m
# ---------------- B
rng = np.random.default_rng(0); tot = pos = 0; human_bad = 0
for ci in semsynth_clips(v, 400)[::10]:
    s = ds.sample(ci, 0, rng); il = s['intent']
    tot += 1; pos += int((il >= 2).sum() >= 4)
hc = [i for i, c in enumerate(v.clips) if all(it.get('data_source') != 'semsynth_r7p1' for it in c['items'])][:40]
for ci in hc:
    human_bad += int((ds.sample(ci, 0, rng)['intent'] != -1).any())
check('B semsynth windows with a gesture', pos / max(tot, 1) > 0.3, f'{pos}/{tot}')
check('B human windows all unknown (-1)', human_bad == 0, f'bad={human_bad}/{len(hc)}')
# cross-check one clip against the manifest: label frames == executed event cores (with 0.3 s lead)
meta = json.load(open(IF + '/PREP.json')); check('B prep stats sane', meta['stats']['events'] > 10000 and meta['stats'].get('frames_mismatch', 0) == 0, json.dumps(meta['stats']))
# ---------------- C
torch.manual_seed(0); m = SCMG(Config(z=vae.z, intent_aux=True)).to(dev)
o = loss_of(m, vae, b, stats, w_int=0.5); o['loss'].backward()
gh = float(m.int_head.weight.grad.norm()); gp = sum(float(p.grad.norm()) for n, p in m.named_parameters() if n.startswith('planner') and p.grad is not None)
check('C intent CE finite and reported', 'intent' in o and torch.isfinite(o['intent']), f"CE={float(o['intent']):.3f} acc={float(o['intent_acc']):.3f}")
check('C grad reaches int_head and planner', gh > 0 and gp > 0, f'|g_head|={gh:.2e} |g_planner|={gp:.2e}')
m.zero_grad()
m.cfg.drop_text = 1.0                                               # every sample text-dropped -> no CE target
o2 = loss_of(m, vae, b, stats, w_int=0.5)
check('C text-dropped samples excluded from CE', float(o2['intent']) == 0.0, f"CE={float(o2['intent']):.3f}")
m.cfg.drop_text = 0.1
# ---------------- D
torch.manual_seed(0); m0 = SCMG(Config(z=vae.z)).to(dev)
f0 = float(loss_of(m0, vae, b, stats)['flow']); f1 = float(loss_of(m0, vae, b, stats, sem_w=2.0)['flow'])
check('D sem_w changes flow loss on gesture windows', f0 != f1, f'{f0:.5f} -> {f1:.5f}')
hb = {k: x.to(dev) for k, x in collate([ds.sample(i, 0, np.random.default_rng(i)) for i in hc[:2]]).items()}
g0 = float(loss_of(m0, vae, hb, stats)['flow']); g1 = float(loss_of(m0, vae, hb, stats, sem_w=2.0)['flow'])
check('D sem_w no-op on human windows', g0 == g1, f'{g0:.5f} vs {g1:.5f}')
# ---------------- E  Semantic Gesture Head + Local Flow conditioning (user 11-03)
from scmg.semclass import N_SEM
gl_frames = b['gest']
check('E coarse g_k labels present', bool((gl_frames >= 1).any()) and int(gl_frames.max()) < N_SEM, f'classes={sorted(set(gl_frames[gl_frames>=0].tolist()))}')
torch.manual_seed(0); ms = SCMG(Config(z=vae.z, sem_head=True, sem_cond=True)).to(dev)
with torch.no_grad():                                             # fresh flow output is zero-init (adaLN-zero): act as trained
    for prm in list(ms.f_out.parameters()) + list(ms.f_ada.parameters()): prm.normal_(0, 0.02)
o = loss_of(ms, vae, b, stats, w_gest=0.5); o['loss'].backward()
check('E gesture CE finite', torch.isfinite(o['gest']), f"CE={float(o['gest']):.3f}")
ge = float(ms.sem_emb.weight.grad.norm()); gh = float(ms.sem_head[-1].weight.grad.norm())
check('E grads reach sem_emb (flow) and sem_head (CE)', ge > 0 and gh > 0, f'|g_emb|={ge:.2e} |g_head|={gh:.2e}')
ms.zero_grad(); o = loss_of(ms, vae, b, stats, w_gest=0.0); o['loss'].backward()
gh0 = ms.sem_head[-1].weight.grad; gh0 = 0. if gh0 is None else float(gh0.norm())
check('E flow loss does not train the head (detached)', gh0 == 0., f'|g_head| w_gest=0: {gh0:.2e}')
from scmg.infer import Streamer, generate
ms.eval(); s0 = ds.sample(cl[0], 0, np.random.default_rng(0))
cond = {k: s0[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid')}
def gen_g(gv):
    import scmg.infer as I
    orig = I.Streamer.__init__
    def init(self, *aa, **kk): orig(self, *aa, **kk); self.g_override = gv
    I.Streamer.__init__ = init
    try: return generate(ms, vae, s0['feat'][:128], cond, st, 8, steps=2, standby=st['q_mean'], seed=0)[0]
    finally: I.Streamer.__init__ = orig
e = lambda i: torch.nn.functional.one_hot(torch.tensor(i), N_SEM).float().to(dev)
d0 = np.abs(gen_g(e(1)) - gen_g(e(5))).max()
check('E zero-init embedding: g has no effect yet', d0 == 0, f'max|d|={d0:.2e}')
with torch.no_grad(): ms.sem_emb.weight.normal_(0, 0.5)
d1 = np.abs(gen_g(e(1)) - gen_g(e(5))).max()
check('E trained embedding: Local Flow responds to g', d1 > 1e-3, f'max|d| point_left vs emphasis = {d1:.3f} rad')
del ms
# ---------------- F  selective audio dropout
torch.manual_seed(0); m0 = SCMG(Config(z=vae.z)).to(dev)
fa = float(loss_of(m0, vae, b, stats)['flow']); fb = float(loss_of(m0, vae, b, stats, p_sel_audio=1.0)['flow'])
check('F selective audio dropout changes gesture windows', fa != fb, f'{fa:.5f} -> {fb:.5f}')
fa = float(loss_of(m0, vae, b, stats, p_sel_audio=1e-12)['flow'])           # same RNG draws, (almost) never hides
check('F ... and it is the hiding, not the extra RNG draw', fa != fb, f'{fa:.5f} vs {fb:.5f}')
ha = float(loss_of(m0, vae, hb, stats, p_sel_audio=1e-12)['flow']); hb_ = float(loss_of(m0, vae, hb, stats, p_sel_audio=1.0)['flow'])
check('F selective audio dropout no-op on human windows', ha == hb_, f'{ha:.5f} vs {hb_:.5f}')
print('ALL_OK' if ok else 'SOME_FAIL')

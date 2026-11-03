"""Text-path acceptance (11-03, Claude). usage: python scripts/test_textpath.py <vae.pt> <orig_code_dir> <saturated vox ckpt>
 A  default OFF (no new cfg / sc switches) is bit-identical to the pre-change code (vox planner, loss)
 B  word-level intent labels (tilab) exist on semsynth windows and are -1 elsewhere
 C  switching the text path on a trained (FSQ-saturated) checkpoint keeps the plan identical (zero init) ...
 D  ... and once x_out is non-zero the plan h depends on the text (own vs no-text) although the FSQ codes do not
 E  w_word / w_slot / fsq_sat / text_focus: finite losses, gradients reach word_head, slot_cls, x_attn and the LM
Prints ALL_OK / SOME_FAIL."""
import json, os, subprocess, sys
import numpy as np, torch
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'; dev = 'cuda'
IF = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/intent-frames-v36_1'
TI = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/intent-qwen25-v36_1'
VAE, ORIG, SAT = sys.argv[1], sys.argv[2], sys.argv[3]
CHILD = os.environ.get('TP_CHILD')


def setup(extra):
    from scmg.data import View, WindowDataset, collate
    from scmg.vae import MotionVAE
    st = json.load(open(D + '/STATS_v36_1.json'))
    v = View(split='train', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', tok_cache=D + '/tok-qwen25-v36_1', **extra)
    ds = WindowDataset(v, 256, max_tokens=128)
    cl = [i for i, c in enumerate(v.clips) if any(it.get('data_source') == 'semsynth_r7p1' for it in c['items'])][:4]
    b = {k: x.to(dev) for k, x in collate([ds.sample(i, 0, np.random.default_rng(i)) for i in cl]).items()}
    vck = torch.load(VAE, map_location='cpu'); vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema']); vae.eval()
    stats = dict(qm=torch.tensor(st['q_mean'], device=dev), qs=torch.tensor(st['q_std'], device=dev))
    return st, v, ds, b, vae, stats


def loss_of(m, vae, b, stats, **sckw):
    from scmg.losses import StageCfg, step_losses
    m.train(); torch.manual_seed(123); np.random.seed(123)
    return step_losses(m, vae, b, StageCfg(**sckw), stats)


if CHILD == 'ref':
    st, v, ds, b, vae, stats = setup({})
    from scmg.model import SCMG, Config
    torch.manual_seed(0); m = SCMG(Config(z=vae.z, planner='voxcpm', vox_lm_init='random')).to(dev)
    np.save(os.environ['TP_OUT'], float(loss_of(m, vae, b, stats)['loss'])); sys.exit(0)

ok = True
def check(name, c, msg=''):
    global ok; ok &= bool(c); print(f'{name:56s} {"ok" if c else "FAIL"} {msg}', flush=True)

st, v, ds, b, vae, stats = setup(dict(intent_frames=IF, tok_intent=TI))
from scmg.model import SCMG, Config
# ---------------- A
out = '/tmp/claude-1000/tp_ref.npy'; os.makedirs('/tmp/claude-1000', exist_ok=True)
r = subprocess.run([sys.executable, '-c', f'import sys; sys.argv={sys.argv[:4]!r}; exec(open({os.path.abspath(__file__)!r}).read())'],
                   env=dict(os.environ, TP_CHILD='ref', TP_OUT=out, PYTHONPATH=ORIG), cwd=ORIG)
torch.manual_seed(0); m = SCMG(Config(z=vae.z, planner='voxcpm', vox_lm_init='random')).to(dev)
l0 = float(loss_of(m, vae, b, stats)['loss']); ref = float(np.load(out)) if r.returncode == 0 else None
check('A default OFF == pre-change (vox planner, tilab+intent loaded)', ref is not None and l0 == ref, f'{l0:.6f} vs {ref}')
del m
# ---------------- B
tl = b['tilab']; check('B word labels present on semsynth windows', bool((tl >= 2).any()), f'labelled tokens {int((tl >= 0).sum())}, intents {int((tl >= 2).sum())}')
# ---------------- C / D on the saturated checkpoint
from scmg.evaluate import load
import scmg.train as T
ck = torch.load(SAT, map_location='cpu', weights_only=False)
m, _ = T.build_model(ck['base_cfg'], {}, ck['extra_cfg'], ck['ema'])
for k, val in dict(vox_text_xattn=True, vox_word_aux=True).items(): setattr(m.cfg, k, val)
m.add_modules(); m.lat_mean.copy_(m.lat_mean); m.to(dev).eval()
m0, _ = load(SAT, VAE, dev); m0.eval()
def plan_h(mm, bb):
    rec = {}
    hk = mm.p_aux.register_forward_hook(lambda mod, i, o: rec.__setitem__('h', i[0].detach().float()))
    mm.cfg.drop_text = mm.cfg.drop_audio = 0.
    loss_of(mm, vae, bb, stats, self_force_p=0.0, p_noctx=0.0); mm.eval(); hk.remove(); return rec['h']
with torch.no_grad():
    h0 = plan_h(m0, b); h1 = plan_h(m, b)
check('C zero-init text path keeps the plan identical', torch.equal(h0, h1), f'max|d|={float((h0 - h1).abs().max()):.2e}')
with torch.no_grad():
    m.vox.x_out.weight.normal_(0, 0.02)
    bn = dict(b, tidm=torch.zeros_like(b['tidm']))
    ha, hb = plan_h(m, b), plan_h(m, bn); hc, hd = plan_h(m0, b), plan_h(m0, bn)
check('D old model: plan ignores text (saturated FSQ)', float((hc - hd).abs().max()) < 1e-6, f'max|d|={float((hc - hd).abs().max()):.2e}')
check('D text path: plan depends on text', float((ha - hb).abs().max()) > 1e-4, f'max|d|={float((ha - hb).abs().max()):.2e}')
del m0
# ---------------- E losses on a fresh model with every route on
torch.manual_seed(0)
m = SCMG(Config(z=vae.z, planner='voxcpm', vox_lm_init='random', vox_text_xattn=True, vox_slots=8, vox_word_aux=True)).to(dev)
o = loss_of(m, vae, b, stats, w_word=0.3, w_slot=0.3, fsq_sat_w=0.1)
o['loss'].backward()
for k in ('word', 'slot', 'fsq_sat'):
    check(f'E {k} loss finite', k in o and torch.isfinite(o[k]), f'{float(o[k]):.4f}' if k in o else 'missing')
g = lambda mod: sum(float(p.grad.norm()) for p in mod.parameters() if p.grad is not None)
check('E grads: word_head, slot_cls, slot_attn, LM', g(m.vox.word_head) > 0 and g(m.vox.slot_cls) > 0 and g(m.vox.slot_attn) > 0 and g(m.vox.lm) > 0,
      f'{g(m.vox.word_head):.2e} {g(m.vox.slot_cls):.2e} {g(m.vox.slot_attn):.2e} {g(m.vox.lm):.2e}')
check('E x_out receives gradient (zero-init route trains)', g(m.vox.x_out) > 0, f'{g(m.vox.x_out):.2e}')
m.zero_grad(); o2 = loss_of(m, vae, b, stats, text_focus_p=1.0)
check('E text-focused batch trains', torch.isfinite(o2['loss']), f"loss {float(o2['loss']):.4f} vs joint {float(o['loss']):.4f}")
print('ALL_OK' if ok else 'SOME_FAIL')

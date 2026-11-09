"""11-03 review fixes. usage: python scripts/test_review_fixes.py <vae.pt> <saturated vox ckpt>
 1 fine-tune: --cfg adding vox_prefsq_norm/xattn builds + trains from an old checkpoint; a build-time switch is refused
 3 xattn/LM positions follow the visible tokens: leading masked padding does not change the plan
 5 slot loss + continuity loss together (the shadowed 'known' crashed)    6 word labels from another token cache are refused
 7 xattn + slots: chunks read slots AND tokens                            8 gest_grad_scale shrinks the CE gradient into h
 9 FSQ saturation bound default 1.2
Prints ALL_OK / SOME_FAIL."""
import json, os, subprocess, sys, tempfile
import numpy as np, torch
TEST_TMP = os.environ.get('SCMG_TEST_TMP', '/tmp/claude-1000')
os.makedirs(TEST_TMP, exist_ok=True)
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'; B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; dev = 'cuda'
VAE, SAT = sys.argv[1], sys.argv[2]
ok = True
def check(name, c, msg=''):
    global ok; ok &= bool(c); print(f'{name:60s} {"ok" if c else "FAIL"} {msg}', flush=True)
from scmg.data import View, WindowDataset, collate
from scmg.losses import StageCfg, step_losses
from scmg.model import SCMG, Config
from scmg.vae import MotionVAE
st = json.load(open(D + '/STATS_v36_1.json'))
# ---- 1
out = tempfile.mkdtemp(prefix='ft_', dir=TEST_TMP)
base = [sys.executable, '-m', 'scmg.train', '--init', SAT, '--stage', 'base', '--steps', '3', '--warmup', '1', '--batch', '2', '--workers', '1',
        '--mmap', '--eval-every', '1000', '--snap-every', '0', '--stats', D + '/STATS_v36_1.json', '--text-cache', D + '/text-e5-v36_1',
        '--tok-cache', D + '/tok-qwen25-v36_1', '--vae', VAE]
r = subprocess.run(base + ['--out', out + '/a', '--cfg', '{"vox_prefsq_norm": true, "vox_text_xattn": true}'], capture_output=True, text=True)
ck = torch.load(out + '/a/final.pt', map_location='cpu', weights_only=False) if r.returncode == 0 else None
check('1 fine-tune adds prefsq_norm + xattn and trains', ck is not None and any(k.startswith('vox.prefsq_norm') for k in ck['ema'])
      and ck['base_cfg'].get('vox_prefsq_norm') is True, r.stderr[-300:] if r.returncode else '')
if ck is not None:
    from scmg.evaluate import load
    try: load(out + '/a/final.pt', VAE, dev); check('1 fine-tuned checkpoint reloads with its new config', True)
    except Exception as e: check('1 fine-tuned checkpoint reloads with its new config', False, str(e)[:200])
r2 = subprocess.run(base + ['--out', out + '/b', '--cfg', '{"vox_fsq": false}'], capture_output=True, text=True)
check('1 build-time switch on a fine-tune is refused', r2.returncode != 0 and 'build-time switches' in (r2.stderr + r2.stdout), (r2.stderr + r2.stdout)[-160:])
# ---- data
v = View(split='train', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', tok_cache=D + '/tok-qwen25-v36_1',
         intent_frames=B + '/models/intent-frames-v36_1', tok_intent=B + '/models/intent-qwen25-v36_1')
ds = WindowDataset(v, 256, max_tokens=128)
cl = [i for i, c in enumerate(v.clips) if any(it.get('data_source') == 'semsynth_r7p1' for it in c['items'])][:4]
b = {k: x.to(dev) for k, x in collate([ds.sample(i, 0, np.random.default_rng(i)) for i in cl]).items()}
vck = torch.load(VAE, map_location='cpu'); vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema']); vae.eval()
stats = dict(qm=torch.tensor(st['q_mean'], device=dev), qs=torch.tensor(st['q_std'], device=dev))
# ---- 3
torch.manual_seed(0); m = SCMG(Config(z=vae.z, planner='voxcpm', vox_lm_init='random', vox_text_xattn=True)).to(dev).eval()
with torch.no_grad(): m.vox.x_out.weight.normal_(0, 0.02)
P = 40; bp = dict(b)
for k in ('tids', 'tidm'):
    pad = torch.zeros(b[k].shape[0], P, dtype=b[k].dtype, device=dev); bp[k] = torch.cat([pad, b[k]], 1)
def plan(mm, bb):
    rec = {}; hk = mm.p_aux.register_forward_hook(lambda mod, i, o: rec.__setitem__('h', i[0].detach().float()))
    mm.cfg.drop_text = mm.cfg.drop_audio = 0.; torch.manual_seed(1)
    step_losses(mm, vae, bb, StageCfg(self_force_p=0.0, p_noctx=0.0), stats); hk.remove(); return rec['h']
with torch.no_grad(): h1, h2 = plan(m, b), plan(m, bp)
check('3 leading masked padding does not change the plan', float((h1 - h2).abs().max()) < 2e-3, f'max|d|={float((h1 - h2).abs().max()):.2e}')
# ---- 5 + 7
torch.manual_seed(0); m = SCMG(Config(z=vae.z, planner='voxcpm', vox_lm_init='random', vox_text_xattn=True, vox_slots=8)).to(dev)
lens = {}
hk = m.vox.x_attn.register_forward_hook(lambda mod, i, o: lens.__setitem__('kv', i[1].shape[1]))
try:
    o = step_losses(m, vae, b, StageCfg(w_slot=0.3, w_cont=0.5, self_force_p=0.0), stats); o['loss'].backward()
    check('5 slot loss + continuity loss together', torch.isfinite(o['loss']), f"slot {float(o['slot']):.3f}")
except Exception as e: check('5 slot loss + continuity loss together', False, str(e)[:200])
hk.remove(); check('7 xattn + slots: chunks read slots and tokens', lens.get('kv') == 8 + b['tids'].shape[1], f"memory {lens.get('kv')}")
# ---- 6
try:
    View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', tok_cache=B + '/models/tok-qwen3-0.6b-v36_1',
         tok_intent=B + '/models/intent-qwen25-v36_1'); check('6 word labels from another token cache are refused', False)
except ValueError as e: check('6 word labels from another token cache are refused', 'token cache' in str(e) or 'rows' in str(e), str(e)[:100])
# ---- 8
def hgrad(scale):
    torch.manual_seed(0); mm = SCMG(Config(z=vae.z, sem_head=True, sem_cond=True)).to(dev).train()
    hold = {}
    def keep(mod, i, o):
        i[0].retain_grad(); hold['h'] = i[0]
    hk = mm.p_aux.register_forward_hook(keep)
    torch.manual_seed(2); o = step_losses(mm, vae, b, StageCfg(w_gest=50.0, gest_grad_scale=scale, self_force_p=0.0, w_aux=0.0), stats)
    (o['gest'] * 50.0).backward(); hk.remove()
    return float(hold['h'].grad.norm()) if hold['h'].grad is not None else 0., float(mm.sem_head[-1].weight.grad.norm())
g1, hd1 = hgrad(1.0); g3, hd3 = hgrad(0.3)
check('8 gest_grad_scale 0.3 shrinks the CE gradient into h, head unchanged', abs(g3 / max(g1, 1e-12) - 0.3) < 0.02 and abs(hd3 - hd1) < 1e-4 * max(hd1, 1),
      f'|g_h| {g1:.3e} -> {g3:.3e}; |g_head| {hd1:.3e} / {hd3:.3e}')
# ---- 9
check('9 FSQ saturation bound default 1.2', StageCfg().fsq_sat_bound == 1.2)
print('ALL_OK' if ok else 'SOME_FAIL')

"""11-05 input-modality training switches (StageCfg force_drop_text / force_drop_audio).
    python scripts/tests/test_force_drop.py [<HEAD code dir>]      (exit 1 on failure)
 a default StageCfg == explicit force_drop_* = False: every loss term bitwise equal (same seeds)
 b (with a HEAD checkout) default losses bitwise equal to the pre-change code
 c force_drop_text: the text never reaches the model -- losses unchanged when batch text / text_c / times are replaced,
   and the text projection (text_in, text_time) gets an exactly zero gradient
 d force_drop_audio: the audio encoder gets an exactly zero gradient and the conditioning paths ignore the audio (losses
   unchanged when the audio features are replaced, with the target-side audio uses -- rhythm / beat terms -- switched off)
 e both switches change the losses (each switch is live)"""
import dataclasses, json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve()
CODE = os.environ.get('FD_CODE', str(HERE.parents[2]))
sys.path.insert(0, CODE); sys.path.insert(0, str(HERE.parent))
import torch
from test_flow_text_xattn import synth_batch, dev
from scmg.model import SCMG, Config
from scmg.vae import MotionVAE
from scmg.losses import StageCfg, step_losses

SC = dict(p_stream=0.67, p_async=1.0, p_async_seg=0.0, self_force_p=0.0)   # the sf0 lane


def run(m, vae, b, seed=5, **kw):
    fields = {f.name for f in dataclasses.fields(StageCfg)}
    sc = StageCfg(**{k: v for k, v in {**SC, **kw}.items() if k in fields})
    st = dict(qm=torch.zeros(b['q'].shape[-1], device=dev), qs=torch.ones(b['q'].shape[-1], device=dev))
    m.zero_grad(set_to_none=True); torch.manual_seed(seed)
    out = step_losses(m, vae, b, sc, st)
    out['loss'].backward()
    return {k: float(v.detach()) for k, v in out.items() if torch.is_tensor(v) and v.numel() == 1}


def build():
    torch.manual_seed(0); m = SCMG(Config(z=16, adsc=True)).to(dev).train()
    with torch.no_grad():                                   # non-zero zero-init projections so every path carries signal
        g = torch.Generator(device='cpu').manual_seed(3)
        for n, p in m.named_parameters():
            if p.abs().sum() == 0 and p.dim() >= 1: p.copy_(0.02 * torch.randn(p.shape, generator=g))
    torch.manual_seed(1); vae = MotionVAE(z=16).to(dev).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    return m, vae


if os.environ.get('FD_CHILD'):
    m, vae = build(); print('LOSSES ' + json.dumps(run(m, vae, synth_batch()))); sys.exit(0)

ok = True
def check(n, c, msg=''):
    global ok; ok &= bool(c); print(f'{n:66s} {"ok" if c else "FAIL"} {msg}', flush=True)
grad0 = lambda m, pre: all(p.grad is None or float(p.grad.abs().max()) == 0. for n, p in m.named_parameters() if n.startswith(pre))
gradnz = lambda m, pre: any(p.grad is not None and float(p.grad.abs().max()) > 0. for n, p in m.named_parameters() if n.startswith(pre))

m, vae = build(); b = synth_batch()
L0 = run(m, vae, b)
check('a default == explicit force_drop_* False', L0 == run(m, vae, b, force_drop_text=False, force_drop_audio=False), json.dumps(L0)[:120])
if len(sys.argv) > 1:
    r = subprocess.run([sys.executable, str(HERE)], env=dict(os.environ, FD_CHILD='1', FD_CODE=sys.argv[1], PYTHONPATH=sys.argv[1]),
                       capture_output=True, text=True, cwd=sys.argv[1])
    old = [l for l in r.stdout.splitlines() if l.startswith('LOSSES ')]
    check('b default losses bitwise equal to the HEAD code', r.returncode == 0 and old and json.loads(old[0][7:]) == L0,
          (old[0][:120] if old else r.stderr[-300:]))
check('a ... text_in and audio encoder do get gradient by default', gradnz(m, 'text_in') and gradnz(m, 'audio.'))

g = torch.Generator(device='cpu').manual_seed(9)
bt = dict(b, text=torch.randn(b['text'].shape, generator=g).to(dev), text_c=torch.randn(b['text_c'].shape, generator=g).to(dev),
          ttime=b['ttime'] * 0.7)
Lt = run(m, vae, b, force_drop_text=True)
check('c force_drop_text: text_in / text_time gradient exactly zero', grad0(m, 'text_in') and grad0(m, 'text_time'))
check('c force_drop_text: losses unchanged when the text is replaced', Lt == run(m, vae, bt, force_drop_text=True))
check('e force_drop_text changes the losses', Lt != L0)

La = run(m, vae, b, force_drop_audio=True)
check('d force_drop_audio: audio encoder gradient exactly zero', grad0(m, 'audio.'))
check('e force_drop_audio changes the losses', La != L0)
# the conditioning paths: audio features replaced -> only target-side audio terms (rhythm / beat weights) may change
ba = dict(b, feat=torch.randn(b['feat'].shape, generator=g).to(dev))
La2 = run(m, vae, ba, force_drop_audio=True)
same = {k for k in La if La[k] == La2.get(k)}; diff = sorted(set(La) - same)
cond_terms = [k for k in diff if k in ('flow', 'loss_flow', 'fm', 'aux', 'plan')]
check('d force_drop_audio: flow / planner terms ignore the audio features', not cond_terms, f'changed: {diff}')
print('ALL_OK' if ok else 'SOME_FAIL'); sys.exit(0 if ok else 1)

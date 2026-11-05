"""11-06 floor-state token (Config.floor, batch 'floor' / 'aend', Streamer.floor_k0) -- bit-identity of the default paths.
    python scripts/tests/test_floor.py [<HEAD code dir>]      (exit 1 on failure)
 a (with a HEAD checkout) default model + default batch: every loss term bitwise equal to the pre-change code
 b floor module added (zero-init): losses with floor all 0 / aend = clip length bitwise equal to (a); its row 0 gets an
   exactly zero gradient (padding_idx), so training can never move the floor-0 function
 c trained-looking floor weights (rows 1, 2 random): floor all 0 still bitwise equal to (a); planner output with stream
   states < 6 bitwise equal to the model without the module
 d floor 1 / 2 on some chunks changes the losses (the token is live); aend < clip length changes them too (audio end)"""
import dataclasses, json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve()
CODE = os.environ.get('FL_CODE', str(HERE.parents[2]))
sys.path.insert(0, CODE); sys.path.insert(0, str(HERE.parent))
import torch
from test_flow_text_xattn import synth_batch, dev
from scmg.model import SCMG, Config
from scmg.vae import MotionVAE
from scmg.losses import StageCfg, step_losses

SC = dict(p_stream=0.67, p_async=1.0, p_async_seg=0.0, self_force_p=0.0)   # the sf0 lane


def run(m, vae, b, seed=5):
    fields = {f.name for f in dataclasses.fields(StageCfg)}
    sc = StageCfg(**{k: v for k, v in SC.items() if k in fields})
    st = dict(qm=torch.zeros(b['q'].shape[-1], device=dev), qs=torch.ones(b['q'].shape[-1], device=dev))
    m.zero_grad(set_to_none=True); torch.manual_seed(seed)
    out = step_losses(m, vae, b, sc, st)
    out['loss'].backward()
    return {k: float(v.detach()) for k, v in out.items() if torch.is_tensor(v) and v.numel() == 1}


def build(floor=False):
    torch.manual_seed(0); m = SCMG(Config(z=16, adsc=True)).to(dev).train()
    with torch.no_grad():
        g = torch.Generator(device='cpu').manual_seed(3)
        for n, p in m.named_parameters():
            if p.abs().sum() == 0 and p.dim() >= 1: p.copy_(0.02 * torch.randn(p.shape, generator=g))
    if floor:
        m.cfg.floor = True; m.add_modules(); m.to(dev).train()
    torch.manual_seed(1); vae = MotionVAE(z=16).to(dev).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    return m, vae


if os.environ.get('FL_CHILD'):
    m, vae = build(); print('LOSSES ' + json.dumps(run(m, vae, synth_batch()))); sys.exit(0)

ok = True
def check(n, c, msg=''):
    global ok; ok &= bool(c); print(f'{n:70s} {"ok" if c else "FAIL"} {msg}', flush=True)

m0, vae = build(); b = synth_batch(); B, T = b['q'].shape[:2]
L0 = run(m0, vae, b)
if len(sys.argv) > 1:
    r = subprocess.run([sys.executable, str(HERE)], env=dict(os.environ, FL_CHILD='1', FL_CODE=sys.argv[1], PYTHONPATH=sys.argv[1]),
                       capture_output=True, text=True, cwd=sys.argv[1])
    old = [l for l in r.stdout.splitlines() if l.startswith('LOSSES ')]
    check('a default losses bitwise equal to the HEAD code', r.returncode == 0 and old and json.loads(old[0][7:]) == L0,
          (old[0][:100] if old else r.stderr[-300:]))
m1, _ = build(floor=True)
z = torch.zeros(B, T, dtype=torch.long, device=dev); full = torch.full((B,), T, dtype=torch.long, device=dev)
bf = dict(b, floor=z, aend=full)
L1 = run(m1, vae, bf)
check('b floor module + floor 0 + aend = T: losses bitwise equal to (a)', L1 == L0)
check('b floor row 0 gradient exactly zero', m1.floor_emb.weight.grad is None or float(m1.floor_emb.weight.grad[0].abs().max()) == 0.)
with torch.no_grad():
    m1.floor_emb.weight[1:] = 0.05 * torch.randn(2, m1.cfg.d, generator=torch.Generator(device='cpu').manual_seed(7)).to(dev)
check('c trained floor rows, floor all 0: losses bitwise equal to (a)', run(m1, vae, bf) == L0)
fl = z.clone(); fl[:, 160:] = 2; fl[0, 96:160] = 1
Ld = run(m1, vae, dict(bf, floor=fl))
check('d floor 1 / 2 changes the losses', Ld != L0, f"loss {L0['loss']:.6f} -> {Ld['loss']:.6f}")
ae = full.clone(); ae[:] = 150
check('d aend < T changes the losses', run(m1, vae, dict(bf, aend=ae)) != L0)
# planner directly: stream states < 6 -> identical output with and without the module
m0.eval(); m1.eval()
with torch.no_grad():
    K, N, d = 16, 64, m0.cfg.d; g = torch.Generator(device='cpu').manual_seed(11)
    lat = torch.randn(1, K, 4, 16, generator=g).to(dev); a = torch.randn(1, N, d, generator=g).to(dev)
    amask = torch.ones(1, K, N, dtype=torch.bool, device=dev); text = torch.randn(1, 8, d, generator=g).to(dev)
    tmask = torch.ones(1, 8, dtype=torch.bool, device=dev); gg = torch.randn(1, d, generator=g).to(dev)
    ss = torch.randint(0, 6, (1, K), generator=g).to(dev)
    h0 = m0.plan(lat, a, amask, text, tmask, gg, sstate=ss); h1 = m1.plan(lat, a, amask, text, tmask, gg, sstate=ss)
    h2 = m1.plan(lat, a, amask, text, tmask, gg, sstate=ss + 6 * (torch.arange(K, device=dev) >= 8).long()[None])
check('c planner: states < 6 bitwise equal without / with the floor module', torch.equal(h0, h1))
check('c planner: causal -- chunks before the first floor chunk unchanged', torch.equal(h1[:, :8], h2[:, :8]) and not torch.equal(h1[:, 8:], h2[:, 8:]))
print('ALL_OK' if ok else 'SOME_FAIL'); sys.exit(0 if ok else 1)

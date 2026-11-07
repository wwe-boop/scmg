"""Helper of test_causal_train.py: one training-loss evaluation of a fixed batch with the code on PYTHONPATH (the current
tree or an exported older commit), as scmg.train builds a base-stage fine-tune (--init, --cfg).
    python _causal_loss_probe.py <batch.pt> <out.pt> <init> <vae> <stats> <cfg json> <sc json> [drop key,key]
Writes {loss terms, grad of every parameter (flattened, float32)} so the caller compares bitwise."""
import json, os, sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')      # deterministic cuBLAS (before CUDA initialises)
import torch
from scmg.train import build_model, validate_additions
from scmg.losses import StageCfg
from scmg.vae import MotionVAE

bp, op, init, vae_p, stats_p, cfg_j, sc_j = sys.argv[1:8]
drop = sys.argv[8].split(',') if len(sys.argv) > 8 and sys.argv[8] else []
dev = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)   # math attention: deterministic backward
vck = torch.load(vae_p, map_location='cpu', weights_only=False)
vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema']); vae.eval().requires_grad_(False)
prev = torch.load(init, map_location='cpu', weights_only=False)
cfg = json.loads(cfg_j); extra = dict(prev['extra_cfg']); extra.update({k: v for k, v in cfg.items() if k in extra})
validate_additions({**prev['base_cfg'], **prev['extra_cfg']}, cfg)
m, _ = build_model(dict(prev['base_cfg']), dict(prev['extra_cfg']), dict(prev['extra_cfg']), prev['ema'])
for k, v in cfg.items(): setattr(m.cfg, k, v)
m.add_modules(); m.to(dev).train()
stats = json.loads(open(stats_p).read())
st = dict(qm=torch.tensor(stats['q_mean'], device=dev), qs=torch.tensor(stats['q_std'], device=dev), fm_raw=stats['f_mean'], fs_raw=stats['f_std'])
sc = StageCfg(**json.loads(sc_j))
b = {k: v.to(dev) for k, v in torch.load(bp, weights_only=False).items() if k not in drop}
torch.manual_seed(0); torch.cuda.manual_seed_all(0)
with torch.autocast('cuda', torch.bfloat16):
    l = m(vae, b, sc, st)
l['loss'].backward()
g = {n: p.grad.detach().float().flatten().cpu() for n, p in m.named_parameters() if p.grad is not None}
torch.save(dict(loss={k: v.detach().float().cpu() for k, v in l.items()}, grad=g), op)
print('PROBE', {k: round(float(v), 6) for k, v in l.items()}, 'grads', len(g), flush=True)

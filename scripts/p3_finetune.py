"""Phase 2 (physics-feedback fine-tuning): LoRA on the LOCAL FLOW only, everything else frozen.
    python scripts/p3_finetune.py <base ckpt> <vae> <collect dir> <out.pt> [steps] [tau] [w_ref] [w_gt]
Per collected window (p3_collect.py) and chunk k>=1:
  history = the COMMITTED (generated, physics-reranked) latents -> planner + local-flow context (on-policy history)
  targets = the N candidate chunks; weight = softmax(-(score - mean)/(std * tau)) over ELIGIBLE candidates
            (eligible = not slower than 0.8 x the median candidate: physics may not be bought by shrinking gestures)
  loss = sum_n w_n ||v(x_t) - (x1_n - z)||^2                      advantage-weighted flow matching
       + w_ref ||v - v_ref||^2                                    reference-policy constraint (frozen copy)
       + w_gt  * flow matching on ground-truth train windows       keeps the data distribution (amplitude, semantics)
LoRA (r=16) is merged into the flow weights at the end -> a plain checkpoint loadable by evaluate.load."""
import copy, glob, json, sys, time
from pathlib import Path
import numpy as np, torch
from scmg.common import CHUNK
from scmg.data import View, WindowDataset, collate
from scmg.evaluate import load
from scmg.losses import StageCfg, step_losses
from scmg.model import INF
from scmg.nn import apply_lora, LoRALinear

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
base_ck, vae_p, cdir, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 2000; TAU = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
W_REF = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5; W_GT = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
BW, BG = 8, 32                                                # collected windows / ground-truth windows per step
dev = 'cuda'; torch.manual_seed(0)
st = json.load(open(D + '/STATS_v36_1.json'))
v = View(split='train', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', tok_cache=D + '/tok-qwen25-v36_1')
ds = WindowDataset(v, 256, max_tokens=128)
m, vae = load(base_ck, vae_p, dev); ref = copy.deepcopy(m).eval().requires_grad_(False)
for p in m.parameters(): p.requires_grad_(False)
n_lora = apply_lora(m.flow, r=16); m.to(dev)
params = [p for n, p in m.named_parameters() if n.endswith('.A') or n.endswith('.B')]
for p in params: p.requires_grad_(True)
opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
files = sorted(glob.glob(f'{cdir}/w*.npz')); W = [dict(np.load(f)) for f in files]
print(f'LoRA layers {n_lora}, trainable {sum(p.numel() for p in params) / 1e6:.2f}M, collected windows {len(W)}', flush=True)
stats = dict(qm=torch.tensor(st['q_mean'], device=dev), qs=torch.tensor(st['q_std'], device=dev))
gt_iter = iter(torch.utils.data.DataLoader(ds, BG, num_workers=4, collate_fn=collate))
rng = np.random.default_rng(0); log = []; t0 = time.time()
cfg = m.cfg; L = cfg.ctx + CHUNK


def physics_batch(sel):
    s = [ds.sample(int(W[i]['ci']), int(W[i]['start']), np.random.default_rng(int(W[i]['ci']))) for i in sel]
    b = {k: x.to(dev) for k, x in collate(s).items()}
    B = len(sel); K = 16; Nl = K * CHUNK
    hist = torch.as_tensor(np.stack([W[i]['committed'][:Nl] for i in sel]), device=dev).view(B, K, CHUNK, -1)
    cand = torch.as_tensor(np.stack([W[i]['cand_lat'] for i in sel]), device=dev)            # [B, K, N, CHUNK, z]
    tot = torch.as_tensor(np.stack([W[i]['total'] for i in sel]), device=dev)                # [B, K, N]
    el = torch.as_tensor(np.stack([W[i]['eligible'] for i in sel]), device=dev)
    N = cand.shape[2]
    with torch.no_grad():                                                                    # frozen conditioning path
        a = m.audio(b['feat'], causal=torch.zeros(B, dtype=torch.bool, device=dev))
        H = torch.full((B,), INF, device=dev); amask = m.chunk_audio_mask(K, Nl, H, dev)
        text, tmask = m.text_tokens(b['text'], b['ttime'], b['tmask'])
        g = m.global_cond(b['spk'], b['spk_valid'], b['source'])
        h = m.plan(hist, a, amask, text, tmask, g, tids=b['tids'], tidm=b['tidm'])            # on-policy history
        aw, awm = m.local_audio(a, H, K)
    # advantage weights per (window, chunk) over eligible candidates
    mu = (tot * el).sum(-1, keepdim=True) / el.sum(-1, keepdim=True).clamp(min=1)
    sd = (((tot - mu) ** 2) * el).sum(-1, keepdim=True).div(el.sum(-1, keepdim=True).clamp(min=1)).sqrt().clamp(min=1e-3)
    wts = torch.softmax(torch.where(el, -(tot - mu) / (sd * TAU), torch.full_like(tot, -1e9)), -1)   # [B, K, N]
    ks = torch.arange(1, K, device=dev)                                                      # chunks with history
    ctx = hist[:, ks - 1][:, :, -cfg.ctx:]                                                    # [B, K-1, ctx, z]
    x1 = torch.cat([ctx[:, :, None].expand(-1, -1, N, -1, -1), cand[:, ks]], 3)              # [B, K-1, N, L, z]
    M = B * (K - 1) * N
    x1 = x1.reshape(M, L, -1)
    rep = lambda x: x[:, ks][:, :, None].expand(-1, -1, N, *x.shape[2:]).reshape(M, *x.shape[2:])
    hk, awk, awmk = rep(h), rep(aw), rep(awm); gk = g[:, None, None].expand(B, K - 1, N, -1).reshape(M, -1)
    known = torch.zeros(M, L, dtype=torch.bool, device=dev); known[:, :cfg.ctx] = True
    t = torch.sigmoid(torch.randn(M, device=dev))[:, None].expand(M, L).clone(); t[known] = 1.
    z = torch.randn_like(x1); xt = torch.where(known[..., None], x1, t[..., None] * x1 + (1 - t[..., None]) * z)
    with torch.autocast('cuda', torch.bfloat16):
        vel = m.flow_velocity(xt, t, known, hk, gk, awk, awmk)
        with torch.no_grad(): vref = ref.flow_velocity(xt, t, known, hk, gk, awk, awmk)
    err = ((vel.float() - (x1 - z)) ** 2)[:, cfg.ctx:].mean((1, 2))                          # [M]
    reg = ((vel.float() - vref.float()) ** 2)[:, cfg.ctx:].mean((1, 2))
    w = wts[:, ks].reshape(M)
    return (w * err).sum() / (B * (K - 1)), reg.mean(), float((w * err).sum() / (B * (K - 1)))


m.train()
for step in range(1, STEPS + 1):
    sel = rng.choice(len(W), BW, replace=len(W) < BW)
    l_phys, l_ref, _ = physics_batch(sel)
    gb = {k: x.to(dev) for k, x in next(gt_iter).items()}
    with torch.autocast('cuda', torch.bfloat16):
        l_gt = step_losses(m, vae, gb, StageCfg(self_force_p=0.0, ctx_noise_p=0.0), stats)['flow']
    loss = l_phys + W_REF * l_ref + W_GT * l_gt
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    if step % 25 == 0:
        rec = dict(step=step, phys=float(l_phys), ref=float(l_ref), gt=float(l_gt), loss=float(loss), s=round(time.time() - t0))
        log.append(rec); print(json.dumps(rec), flush=True)
# merge LoRA into the base Linear weights -> same structure as the base checkpoint
with torch.no_grad():
    for parent in list(m.flow.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                child.base.weight += (child.B @ child.A) * child.scale; setattr(parent, name, child.base)
ck = torch.load(base_ck, map_location='cpu', weights_only=False)
ck['ema'] = {k: x.detach().cpu() for k, x in m.state_dict().items()}; ck['model'] = ck['ema']
ck['phase2'] = dict(base=base_ck, collect=cdir, windows=len(W), steps=STEPS, tau=TAU, w_ref=W_REF, w_gt=W_GT, log=log[-5:])
torch.save(ck, out); print('saved', out)

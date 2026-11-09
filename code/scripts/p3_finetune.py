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


def tscore(w, target=None):
    """Target score of every candidate: re-weighted from the saved terms (v2 collections) when target is given."""
    if target is None or 'terms' not in w: return w['total']
    names = [str(x) for x in w['term_names']]; t = w['terms']
    gap = np.abs(np.log(np.maximum(w['speed'], 1e-4) / np.maximum(w['gt_speed'][:, None], 1e-4)))
    return sum(v_ * (gap if k == 'speed_gap' else t[..., names.index(k)]) for k, v_ in target.items()).astype(np.float32)

@torch.no_grad()
def physics_conditions(m, b, hist):
    """Offline/text-final conditioning, matching Streamer._tick for the SAME window/history.

    Collection uses H=INF and no measured state. No condition dropout or posterior style is used.
    """
    B, K = hist.shape[:2]; dev = hist.device
    a = m.audio(b['feat'], causal=torch.zeros(B, dtype=torch.bool, device=dev))
    H = torch.full((B,), INF, device=dev)
    amask = m.chunk_audio_mask(K, a.shape[1], H, dev)
    text, tmask = m.text_tokens(b['text'], b['ttime'], b['tmask'])
    g = m.global_cond(b['spk'], b['spk_valid'], b['source'])
    sref = None
    if m.cfg.style:
        sref = torch.where((b['ref_valid'] > .5)[:, None], m.style_enc(b['ref']), m.style_null[None])
    mt = mtm = mf = pm = None
    # Streamer supplies final text/audio state for ADSC models even in offline mode.
    tvis = lvis = sstate = None
    if hasattr(m, 'stream_emb'):
        tvis = tmask[:, None].expand(-1, K, -1)
        if 'tidm' in b: lvis = b['tidm'][:, None].expand(-1, K, -1)
        sstate = torch.full((B, K), 5, dtype=torch.long, device=dev)
    if m.cfg.mtc:
        mt, mtm = m.mt_memory(text, tmask, a, amask, g, sref, hist, None, K, tvis=tvis)
        mf, pm = m.planner_memory(mt, mtm)
    h = m.plan(hist, a, amask, text, tmask, g, mf, pm, b.get('tids'), b.get('tidm'), sref,
               tvis=tvis, lvis=lvis, sstate=sstate)
    aw, awm = m.local_audio(a, H, K)
    return dict(h=h, g=g, aw=aw, aw_mask=awm, mt=mt, mt_mask=mtm, sref=sref)


@torch.no_grad()
def physics_flow_conditions(m, cond, ks, temperature=1.0, generator=None):
    """Select chunks in batch-major order; use the same style prior and MTC mask as Streamer."""
    B = cond['h'].shape[0]; C = len(ks); L = m.cfg.ctx + CHUNK
    pick = lambda x: x[:, ks].flatten(0, 1)
    h = pick(cond['h'])
    g = cond['g'][:, None].expand(B, C, -1).flatten(0, 1)
    style_tok = None
    if m.cfg.style:
        sref = cond['sref'][:, None].expand(B, C, -1).flatten(0, 1)
        style_tok, _, _ = m.style_head(sref, h, temperature=temperature, generator=generator)
    mt = mtm = None
    if m.cfg.mtc:
        mt = pick(cond['mt']); mtm = pick(cond['mt_mask'])[:, None].expand(-1, L, -1)
    return dict(h=h, g=g, aw=pick(cond['aw']), aw_mask=pick(cond['aw_mask']),
                style_tok=style_tok, mt=mt, mt_mask=mtm)


def physics_batch(m, ref, b, hist, cand, tot, el, tau=0.5, mode='awr', margin=0.5):
    cfg = m.cfg; dev = hist.device
    B, K = hist.shape[:2]; N = cand.shape[2]; L = cfg.ctx + CHUNK
    cond = physics_conditions(m, b, hist)
    # advantage weights per (window, chunk) over eligible candidates
    mu = (tot * el).sum(-1, keepdim=True) / el.sum(-1, keepdim=True).clamp(min=1)
    sd = (((tot - mu) ** 2) * el).sum(-1, keepdim=True).div(el.sum(-1, keepdim=True).clamp(min=1)).sqrt().clamp(min=1e-3)
    wts = torch.softmax(torch.where(el, -(tot - mu) / (sd * tau), torch.full_like(tot, -1e9)), -1)   # [B, K, N]
    if mode == 'rft':      # rejection-sampling fine-tuning: only the best eligible candidate, only where it clearly beats the group
        best = torch.where(el, tot, torch.full_like(tot, float('inf'))).argmin(-1, keepdim=True)
        gain = (mu - tot.gather(-1, best)) / sd
        wts = torch.zeros_like(tot).scatter(-1, best, 1.) * (gain > margin).float()
    ks = torch.arange(1, K, device=dev)                                                      # chunks with history
    ctx = hist[:, ks - 1][:, :, -cfg.ctx:]                                                    # [B, K-1, ctx, z]
    x1 = torch.cat([ctx[:, :, None].expand(-1, -1, N, -1, -1), cand[:, ks]], 3)              # [B, K-1, N, L, z]
    M = B * (K - 1) * N
    x1 = x1.reshape(M, L, -1)
    fc = physics_flow_conditions(m, cond, ks)
    # One prior style draw per window/chunk, shared by its candidates and the reference flow.
    fc = {k: (None if v is None else v.reshape(B, K - 1, *v.shape[1:])[:, :, None]
              .expand(B, K - 1, N, *v.shape[1:]).reshape(M, *v.shape[1:])) for k, v in fc.items()}
    known = torch.zeros(M, L, dtype=torch.bool, device=dev); known[:, :cfg.ctx] = True
    t = torch.sigmoid(torch.randn(M, device=dev))[:, None].expand(M, L).clone(); t[known] = 1.
    z = torch.randn_like(x1); xt = torch.where(known[..., None], x1, t[..., None] * x1 + (1 - t[..., None]) * z)
    with torch.autocast('cuda', torch.bfloat16):
        vel = m.flow_velocity(xt, t, known, **fc)
        with torch.no_grad(): vref = ref.flow_velocity(xt, t, known, **fc)
    err = ((vel.float() - (x1 - z)) ** 2)[:, cfg.ctx:].mean((1, 2))                          # [M]
    reg = ((vel.float() - vref.float()) ** 2)[:, cfg.ctx:].mean((1, 2))
    w = wts[:, ks].reshape(M)
    return (w * err).sum() / (B * (K - 1)), reg.mean(), float((w * err).sum() / (B * (K - 1)))


def main():
    D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
    base_ck, vae_p, cdir, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 2000; TAU = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
    W_REF = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5; W_GT = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
    BW, BG = 8, 32                                                # collected windows / ground-truth windows per step
    import os
    MODE = os.environ.get('P3_MODE', 'awr'); MARGIN = float(os.environ.get('P3_MARGIN', '0.5'))
    P3_TARGET = json.loads(os.environ['P3_TARGET']) if os.environ.get('P3_TARGET') else None

    dev = 'cuda'; torch.manual_seed(0)
    st = json.load(open(D + '/STATS_v36_1.json'))
    v = View(split='train', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', tok_cache=D + '/tok-qwen25-v36_1')
    ds = WindowDataset(v, 256, max_tokens=128)
    m, vae = load(base_ck, vae_p, dev); ref = copy.deepcopy(m).eval().requires_grad_(False)
    for p in m.parameters(): p.requires_grad_(False)
    # 11-02 review fix: PEFT checkpoints already carry LoRA (planner/flow/audio). Only the FLOW LoRA is trained: reuse it when
    # present (and save without merging, so the checkpoint keeps its structure), add + merge only for bare base models.
    HAD_LORA = any(isinstance(x, LoRALinear) for x in m.flow.modules())
    n_lora = 0 if HAD_LORA else apply_lora(m.flow, r=16); m.to(dev)
    params = [p for n, p in m.named_parameters() if n.startswith('flow.') and (n.endswith('.A') or n.endswith('.B'))]
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
    files = sorted(glob.glob(f'{cdir}/w*.npz')); W = [dict(np.load(f)) for f in files]
    print(f'LoRA layers {n_lora}, trainable {sum(p.numel() for p in params) / 1e6:.2f}M, collected windows {len(W)}', flush=True)
    stats = dict(qm=torch.tensor(st['q_mean'], device=dev), qs=torch.tensor(st['q_std'], device=dev))
    gt_iter = iter(torch.utils.data.DataLoader(ds, BG, num_workers=4, collate_fn=collate))
    rng = np.random.default_rng(0); log = []; t0 = time.time()
    cfg = m.cfg; L = cfg.ctx + CHUNK

    m.train()
    for step in range(1, STEPS + 1):
        sel = rng.choice(len(W), BW, replace=len(W) < BW)
        samples = [ds.sample(int(W[i]['ci']), int(W[i]['start']), np.random.default_rng(int(W[i]['ci']))) for i in sel]
        for sample, i in zip(samples, sel):
            # New collections retain the actual reference; legacy collections keep their previous resampling behavior.
            if 'ref' in W[i]: sample['ref'], sample['ref_valid'] = W[i]['ref'], W[i]['ref_valid']
        b = {k: x.to(dev) for k, x in collate(samples).items()}
        hist = torch.as_tensor(np.stack([W[i]['committed'][:16 * CHUNK] for i in sel]), device=dev).view(len(sel), 16, CHUNK, -1)
        cand = torch.as_tensor(np.stack([W[i]['cand_lat'] for i in sel]), device=dev)
        tot = torch.as_tensor(np.stack([tscore(W[i], P3_TARGET) for i in sel]), device=dev)
        el = torch.as_tensor(np.stack([W[i]['eligible'] for i in sel]), device=dev)
        l_phys, l_ref, _ = physics_batch(m, ref, b, hist, cand, tot, el, TAU, MODE, MARGIN)
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
        for parent in (list(m.flow.modules()) if not HAD_LORA else []):
            for name, child in list(parent.named_children()):
                if isinstance(child, LoRALinear):
                    child.base.weight += (child.B @ child.A) * child.scale; setattr(parent, name, child.base)
    ck = torch.load(base_ck, map_location='cpu', weights_only=False)
    ck['ema'] = {k: x.detach().cpu() for k, x in m.state_dict().items()}; ck['model'] = ck['ema']
    ck['phase2'] = dict(base=base_ck, collect=cdir, windows=len(W), steps=STEPS, tau=TAU, w_ref=W_REF, w_gt=W_GT, mode=MODE,
                       margin=MARGIN, target=P3_TARGET, log=log[-5:])
    torch.save(ck, out); print('saved', out)


if __name__ == '__main__':
    main()

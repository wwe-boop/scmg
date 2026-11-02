"""Progressive training of SCMG.

  vae     MotionVAE on joint windows; writes vae.pt (+ latent mean/std)                         [stage 0]
  base    planner + local flow + audio encoder + condition projections, full-context audio      [base model]
  peft    + multi-token conditioning + style distribution head + LoRA; ONLY new params train   [proposal stage 1]
  causal  + chunk-aware causal training + masked out-painting + continuity/rhythm losses        [proposal stage 2]
  state   + Robot State Adapter with simulated execution deviations                             [proposal stage 3]
  (--unfreeze-flow: additionally release the local flow backbone, "limited unfreezing")

    torchrun --nproc_per_node=G -m scmg.train --stage base --out <dir> --vae <vae.pt> --text-cache <dir> --stats <json>
"""
import argparse, copy, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist

from .common import CHUNK_FRAMES, STRIDE
from .data import View, WindowDataset, collate, fixed_windows, VIEW
from .model import SCMG, Config
from .losses import step_losses, StageCfg
from dataclasses import replace
from .vae import MotionVAE

STAGES = {
    'base':   dict(cfg={}, sc=dict()),
    'peft':   dict(cfg=dict(mtc=True, style=True, lora_r=16, vox_ref_gate=True), sc=dict()),
    'causal': dict(cfg=dict(mtc=True, style=True, lora_r=16), sc=dict(p_stream=0.67, outpaint=True, w_cont=1.0, w_rhy=0.1)),
    'state':  dict(cfg=dict(mtc=True, style=True, lora_r=16, state=True),
                   sc=dict(p_stream=0.67, outpaint=True, w_cont=1.0, w_rhy=0.1, p_state=0.5)),
}


def ddp():
    # only a torchrun child is distributed: DLC sets RANK=0 / WORLD_SIZE on EVERY pod, also for single-process
    # launches (job dlc1md6q6j0e0paj failed on that: KeyError LOCAL_RANK)
    if 'LOCAL_RANK' in os.environ and int(os.environ.get('WORLD_SIZE', '1')) > 1:
        dist.init_process_group('nccl'); r = dist.get_rank(); torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        return r, dist.get_world_size()
    torch.cuda.set_device(0)
    return 0, 1


def log(path, rec):
    with open(path, 'a') as f: f.write(json.dumps(rec) + '\n')


def overfit_clips(view, n):
    """Restrict `view` IN PLACE to a deterministic subset: the first n BEAT2 train clips of >= 8 s (real mocap, text
    and speaker available); the per-speaker index used for reference-motion sampling is rebuilt."""
    from collections import defaultdict
    c = [x for x in view.clips if x['frames'] >= 400 and str(x['source']).startswith('beat2')]
    view.clips = sorted(c, key=lambda x: x['clip'])[:n]
    view.by_spk = defaultdict(list)
    for i, x in enumerate(view.clips):
        if x['speaker']: view.by_spk[x['speaker']].append(i)
    return view.clips


def build_model(base_cfg, prev_extra, new_extra, prev_state=None):
    """Base-config model -> remember base parameter ids -> rebuild the previous stage's modules -> load its weights
    -> add this stage's modules. LoRA keeps the wrapped Linear parameters, so base ids stay valid through surgery."""
    m = SCMG(Config(**base_cfg))
    base_ids = {id(p) for p in m.parameters()}
    for k, v in prev_extra.items(): setattr(m.cfg, k, v)
    m.add_modules()
    missing = []
    if prev_state is not None:
        missing, unexpected = m.load_state_dict(prev_state, strict=False)
        if unexpected or missing: raise RuntimeError(f'checkpoint mismatch: missing {missing[:5]} unexpected {unexpected[:5]}')
    for k, v in new_extra.items(): setattr(m.cfg, k, v)
    m.add_modules()
    return m, base_ids



@torch.no_grad()
def fixed_dev(model, vae, batches, stats, seed=777):
    """Same random draws in all three branches; token-weighted sums across batches.

    Explicit opt-in preserves legacy training/default dev. No training RNG is consumed.
    """
    dev = next(model.parameters()).device
    sc = StageCfg(p_stream=0., p_async=0., outpaint=False, ctx_noise_p=0., self_force_p=0.,
                  p_noctx=0., sigma_tf=0., w_aux=0., w_cont=0., w_rhy=0., w_kl=0.)
    sums = {mode: {} for mode in ('full', 'no_audio', 'no_text')}
    states = [(m, m.training) for parent in (model, vae) for m in parent.modules()]
    try:
        model.eval(); vae.eval()
        with torch.random.fork_rng(devices=[dev.index or 0] if dev.type == 'cuda' else []):
            for i, b in enumerate(batches):
                b = {k: v.to(dev) for k, v in b.items()}
                for mode in sums:
                    torch.manual_seed(seed + i)
                    with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == 'cuda'):
                        vals = step_losses(model, vae, b, sc, stats, diagnostic=mode, diagnostic_seed=seed + i)
                    for k, v in vals.items():
                        if k.endswith(('_sum', '_count')): sums[mode][k] = sums[mode].get(k, 0.) + float(v)
    finally:
        for m, training in states: m.training = training
    out = {}
    for mode, vals in sums.items():
        if not vals or vals['flow_count'] == 0: raise ValueError('empty dev supervision')
        out[f'dev_{mode}_flow'] = vals['flow_sum'] / vals['flow_count']
        for bi in range(4):
            count = vals[f't{bi}_count']
            out[f'dev_{mode}_t{bi}'] = vals[f't{bi}_sum'] / count if count else None
            out[f'dev_{mode}_t{bi}_count'] = count
    out['dev_flow'] = out['dev_full_flow']
    out['delta_audio'] = out['dev_no_audio_flow'] - out['dev_flow']
    out['delta_text'] = out['dev_no_text_flow'] - out['dev_flow']
    return out


def generation_key(result):
    metrics = list(result['metrics'].values())
    key = (float(np.mean([m['FKD'] for m in metrics])),
           float(np.mean([abs(m['speed_ratio_macro'] - 1.) for m in metrics])))
    if not all(math.isfinite(v) for v in key): raise ValueError('nonfinite generation selection metric')
    return key


def make_gen_evaluator(stats, device):
    """Reuse eval-v2 proxy with exactly 12 fixed dev recordings (round-robin strata)."""
    import pickle
    from eval2.gen_proxy import ProxyEvaluator
    from eval2.data import ROOT, FGD
    from eval2.metrics import kin_feats, speed
    from eval2.batch_stream import feature_windows
    from eval2.fgd_model import Embedder
    pack = pickle.load(open(ROOT / 'pack.pkl', 'rb'))
    strata = {}
    for row in pack: strata.setdefault((row['speaker'], row['language']), []).append(row)
    groups = [sorted(rows, key=lambda r: (len(r['gt']), str(r['ci']))) for _, rows in sorted(strata.items())]
    selected = []
    while len(selected) < 12:
        progressed = False
        for rows in groups:
            if rows and len(selected) < 12: selected.append(rows.pop(0)); progressed = True
        if not progressed: raise ValueError('need 12 distinct dev recordings')
    obj = ProxyEvaluator.__new__(ProxyEvaluator)
    obj.pack = sorted(selected, key=lambda s: str(s['ci'])); obj.stats = stats; obj.device = torch.device(device)
    base = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture/fgd-ex002-v2')
    obj.embs = [Embedder(base / f'human_seed{i}/best.pt', device) for i in (0, 1)]
    obj.gt_fkd = np.concatenate([kin_feats(s['gt']) for s in obj.pack])
    obj.gt_emb = [np.concatenate([e.features(s['gt']) for s in obj.pack]) for e in obj.embs]
    obj.gt_speed = [speed(s['gt'])['mean_rad_s'] for s in obj.pack]
    obj.onsets = json.loads((FGD / 'AUDIO_ONSETS.json').read_text())
    obj.windows = {'offline': feature_windows(obj.pack, stats, 'offline')[0],
                   'H0': feature_windows(obj.pack, stats, 'offline_H4', 0)[0]}
    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['vae'] + list(STAGES))
    ap.add_argument('--out', required=True); ap.add_argument('--view', default=VIEW)
    ap.add_argument('--stats', required=True); ap.add_argument('--text-cache', default=None)
    ap.add_argument('--vae', default=None); ap.add_argument('--init', default=None, help='previous stage checkpoint')
    ap.add_argument('--steps', type=int, default=100000); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--warmup', type=int, default=2000)
    ap.add_argument('--frames', type=int, default=256); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--ema', type=float, default=0.999); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-every', type=int, default=2000); ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--unfreeze-flow', action='store_true')
    ap.add_argument('--cfg', default='{}', help='JSON overrides of the model Config (e.g. ablations)')
    ap.add_argument('--sc', default='{}', help='JSON overrides of the StageCfg')
    ap.add_argument('--semsynth-share', type=float, default=0.4)
    ap.add_argument('--snap-every', type=int, default=10000, help='EMA snapshot for mid-training evaluation')
    ap.add_argument('--deriv-ramp', type=int, default=3000, help='vae: steps to ramp in derivative losses, noise, KL')
    ap.add_argument('--tok-cache', default=None, help='LM token ids (voxcpm planner)')
    ap.add_argument('--overfit', type=int, default=0, help='train AND monitor on the first N long train clips only (sanity check)')
    ap.add_argument('--mmap', action='store_true', help='memory-map the arrays instead of loading them (smoke tests)')
    ap.add_argument('--max-hours', type=float, default=0, help='stop cleanly after this wall time (0 = no limit)')
    ap.add_argument('--fixed-dev', action='store_true', help='r8 fixed dev and paired condition information gain')
    ap.add_argument('--select', choices=['flow', 'gen'], default='flow')
    ap.add_argument('--gen-cfg', type=float, default=2.5)
    a = ap.parse_args()
    if a.select == 'gen':
        if a.stage == 'vae' or a.overfit: raise ValueError('gen selection requires held-out dev and SCMG')
        a.fixed_dev = True
    rank, world = ddp(); dev = torch.device('cuda')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed + rank); np.random.seed(a.seed + rank)
    stats = json.loads(Path(a.stats).read_text())
    view = View(a.view, 'train', a.text_cache, stats, mmap=a.mmap, tok_cache=a.tok_cache)
    dview = View(a.view, 'val', a.text_cache, stats, mmap=a.mmap, tok_cache=a.tok_cache)
    if a.overfit:
        overfit_clips(view, a.overfit); dview = view
    ds = WindowDataset(view, a.frames, seed=a.seed * 1000 + rank, semsynth_share=a.semsynth_share)
    dl = iter(torch.utils.data.DataLoader(ds, a.batch, num_workers=a.workers, collate_fn=collate, pin_memory=True,
                                          persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None))
    dev_ds = WindowDataset(dview, a.frames, seed=12345, fixed=fixed_windows(dview, a.frames))
    st = dict(qm=torch.tensor(stats['q_mean'], device=dev), qs=torch.tensor(stats['q_std'], device=dev))
    if rank == 0: (out / 'ARGS.json').write_text(json.dumps(vars(a), indent=1))

    # ------------------------------------------------------------ model
    if a.stage == 'vae':
        model = MotionVAE().to(dev); base_ids = set(); vae = None; sc = None
    else:
        vck = torch.load(a.vae, map_location='cpu')
        vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema']); vae.eval().requires_grad_(False)
        spec = STAGES[a.stage]
        base_cfg = dict(z=vae.z)
        prev = torch.load(a.init, map_location='cpu') if a.init else None
        if prev is not None: base_cfg = prev['base_cfg']
        base_cfg.update(json.loads(a.cfg) if a.stage == 'base' else {})
        extra = dict(prev['extra_cfg']) if prev is not None else {}
        extra.update(spec['cfg']); extra.update(json.loads(a.cfg) if a.stage != 'base' else {})
        if json.loads(a.sc).get('p_hist_drop', 0) > 0: extra['hist_null'] = True
        prev_extra = dict(prev['extra_cfg']) if prev is not None else {}
        model, base_ids = build_model(base_cfg, prev_extra, extra, prev['ema'] if prev is not None else None)
        if prev is None:
            model.lat_mean.copy_(torch.tensor(vck['lat_mean'])); model.lat_std.copy_(torch.tensor(vck['lat_std']))
        elif rank == 0:
            print(f'init from {a.init} (stage {prev.get("stage")}); added modules: {sorted(set(extra) - set(prev_extra))}')
        model.to(dev)
        sc = StageCfg(**{**spec['sc'], **json.loads(a.sc)})
        if a.stage != 'base':
            for p in model.parameters(): p.requires_grad_(id(p) not in base_ids)
            if a.unfreeze_flow:
                for n, p in model.named_parameters():
                    if n.startswith(('flow.', 'f_')): p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params); n_all = sum(p.numel() for p in model.parameters())
    if rank == 0: print(f'stage {a.stage}: trainable {n_train / 1e6:.2f}M / {n_all / 1e6:.2f}M', flush=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    net = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True) if world > 1 else model
    lm_ids = {id(p) for n, p in model.named_parameters() if n.startswith('vox.lm.')}
    lm_scale = getattr(getattr(model, 'cfg', None), 'lm_lr_scale', 1.0)
    groups = [dict(params=[p for p in params if id(p) not in lm_ids], scale=1.0)]
    if any(id(p) in lm_ids for p in params): groups.append(dict(params=[p for p in params if id(p) in lm_ids], scale=lm_scale))
    opt = torch.optim.AdamW([dict(params=g_['params'], scale=g_['scale']) for g_ in groups], lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    sched = lambda s: min(1., (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1., s / a.steps))))
    step = 0; gn_hist = []
    ck = out / 'resume.pt'
    if ck.exists():
        r = torch.load(ck, map_location='cpu')
        model.load_state_dict(r['model']); ema.load_state_dict(r['ema']); opt.load_state_dict(r['opt']); step = r['step']
        gn_hist = r.get('gn_hist', [])
        if rank == 0: print('resumed at', step, flush=True)

    def save(path, final=False):
        if rank: return
        if a.stage == 'vae':
            d = dict(cfg=dict(z=model.z, stride=model.stride), model=model.state_dict(), ema=ema.state_dict(), step=step)
            if final: d.update(latent_stats(ema))
        else:
            d = dict(base_cfg=base_cfg, extra_cfg=extra, ema=ema.state_dict(), model=model.state_dict(), step=step,
                     stage=a.stage, sc=vars(sc))
        if not final: d.update(opt=opt.state_dict(), gn_hist=gn_hist[-500:])
        tmp = Path(str(path) + '.tmp'); torch.save(d, tmp); tmp.replace(path)

    def latent_stats(v):
        mus = []
        it = iter(torch.utils.data.DataLoader(ds, 64, num_workers=2, collate_fn=collate))
        with torch.no_grad():
            for _ in range(60):
                b = next(it); m = b['mask'].to(dev).view(b['mask'].shape[0], -1, STRIDE).min(-1).values > .5
                mu, _ = v.encode(b['q'].to(dev)); mus.append(mu[m].cpu())
        mu = torch.cat(mus)
        return dict(lat_mean=mu.mean(0).tolist(), lat_std=mu.std(0).clamp(min=1e-3).tolist())

    def losses(m, b):
        if a.stage == 'vae':
            _, l = m(b['q'], b['mask'], deriv=min(1., step / a.deriv_ramp))
            return l
        return m(vae, b, sc, st, ema_model=ema) if sc.w_mg else m(vae, b, sc, st)

    @torch.no_grad()
    def evaluate():
        if a.fixed_dev and a.stage != 'vae':
            dl_ = torch.utils.data.DataLoader(dev_ds, min(a.batch, 64), num_workers=0, collate_fn=collate)
            return fixed_dev(ema, vae, dl_, st)
        g = torch.random.get_rng_state(); gc = torch.cuda.get_rng_state(); torch.manual_seed(777)
        acc = {}; n = 0
        dl_ = torch.utils.data.DataLoader(dev_ds, 64, num_workers=2, collate_fn=collate)
        for b in dl_:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast('cuda', torch.bfloat16):
                l = losses(ema, b)
            for k, v in l.items(): acc[k] = acc.get(k, 0.) + float(v)
            n += 1
        torch.random.set_rng_state(g); torch.cuda.set_rng_state(gc)
        return {f'dev_{k}': v / max(n, 1) for k, v in acc.items()}

    t0 = time.time(); tl = time.time(); agg = {}; skipped = 0
    proxy = make_gen_evaluator(stats, dev) if a.select == 'gen' and rank == 0 else None
    best_key = 'dev_rec' if a.stage == 'vae' else ('gen_fkd' if a.select == 'gen' else 'dev_flow')
    best = json.loads((out / 'BEST.json').read_text()) if (out / 'BEST.json').exists() else dict(value=1e9, step=0)
    model.train()
    while step < a.steps:
        b = {k: v.to(dev, non_blocking=True) for k, v in next(dl).items()}
        for gp in opt.param_groups: gp['lr'] = a.lr * gp.get('scale', 1.0) * sched(step)
        with torch.autocast('cuda', torch.bfloat16):
            l = losses(net, b)
        loss = l['loss']
        opt.zero_grad(set_to_none=True)
        fin = torch.isfinite(loss.detach()).float()
        if world > 1: dist.all_reduce(fin, op=dist.ReduceOp.MIN)     # all ranks skip together (no allreduce hang)
        if fin.item() < 1:
            skipped += 1; step += 1; continue
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1e9))
        med = float(np.median(gn_hist[-200:])) if len(gn_hist) >= 50 else None
        if not math.isfinite(gn) or (med is not None and gn > 20 * med):
            skipped += 1                                         # gradient spike: skip the update, keep the step count
        else:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm.detach(), 1 - a.ema)
                for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)
        gn_hist.append(gn if math.isfinite(gn) else 1e9); gn_hist = gn_hist[-1000:]
        for k, v in l.items(): agg[k] = agg.get(k, 0.) + float(v)
        agg['gn'] = agg.get('gn', 0.) + gn; agg['_n'] = agg.get('_n', 0) + 1
        step += 1
        if rank == 0 and step % a.log_every == 0:
            n = agg.pop('_n'); rec = {k: v / n for k, v in agg.items()}
            rec.update(step=step, lr=a.lr * sched(step), skipped=skipped, s_per_step=(time.time() - tl) / a.log_every,
                       hours=(time.time() - t0) / 3600)
            log(out / 'train.jsonl', rec); print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
            agg = {}; tl = time.time()
            if rec['loss'] > 50 and step > 5000:
                (out / 'DIVERGED.json').write_text(json.dumps(rec)); raise SystemExit(3)
        if rank == 0 and step % a.eval_every == 0:
            model.eval(); ev = evaluate(); model.train(); ev['step'] = step
            if proxy is not None:
                from eval2.gen_proxy import gen_proxy
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    gr = gen_proxy(ema, vae, evaluator=proxy, cfg=a.gen_cfg, steps=10, seed=777, ctx_t=sc.ctx_infer_t)
                ev['gen_fkd'], ev['gen_speed_error'] = generation_key(gr)
                log(out / 'gen.jsonl', dict(step=step, **gr))
            log(out / 'dev.jsonl', ev); print('DEV', json.dumps(ev), flush=True)
            if (ev.get(best_key, 1e9) < best['value'] or (a.select == 'gen' and ev.get(best_key) == best['value'] and ev['gen_speed_error'] < best.get('speed_error', float('inf')))):              # model selection on the never-trained dev set
                best = dict(value=ev[best_key], speed_error=ev.get('gen_speed_error', 0.), step=step, key=best_key); save(out / 'best.pt', final=True)
                (out / 'BEST.json').write_text(json.dumps(best))
        if step % a.save_every == 0: save(ck)
        if a.snap_every and step % a.snap_every == 0: save(out / f'snap_{step:06d}.pt', final=True)
        if a.max_hours and (time.time() - t0) / 3600 > a.max_hours: break
    save(ck); save(out / 'final.pt', final=True)
    if rank == 0 and not (out / 'best.pt').exists():                # no dev evaluation happened (tiny runs)
        import shutil; shutil.copy(out / 'final.pt', out / 'best.pt')
    if rank == 0: (out / 'DONE.json').write_text(json.dumps(dict(step=step, skipped=skipped, hours=(time.time() - t0) / 3600)))
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__':
    main()

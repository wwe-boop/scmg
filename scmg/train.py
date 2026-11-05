"""Progressive training of SCMG.

  vae     MotionVAE on joint windows; writes vae.pt (+ latent mean/std)                         [stage 0]
  base    planner + local flow + audio encoder + condition projections, full-context audio      [base model]
  peft    + multi-token conditioning + style distribution head + LoRA; ONLY new params train   [proposal stage 1]
  causal  + chunk-aware causal training + masked out-painting + continuity/rhythm losses        [proposal stage 2]
  state   + Robot State Adapter with simulated execution deviations                             [proposal stage 3]
  (--unfreeze-flow: additionally release the local flow backbone, "limited unfreezing")

    torchrun --nproc_per_node=G -m scmg.train --stage base --out <dir> --vae <vae.pt> --text-cache <dir> --stats <json>
"""
import argparse, contextlib, copy, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist

from .common import CHUNK_FRAMES, STRIDE, J
from .data import View, WindowDataset, collate, fixed_windows, VIEW
from .model import SCMG, Config
from .losses import step_losses, StageCfg
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
    c = [x for i, x in enumerate(view.clips) if x['frames'] >= 400 and str(x['source']).startswith('beat2')
         and (view.holdout is None or not view.holdout[i])]
    if view.holdout is not None: view.holdout = np.zeros(min(n, len(c)), dtype=bool)
    view.clips = sorted(c, key=lambda x: x['clip'])[:n]
    view.by_spk = defaultdict(list)
    for i, x in enumerate(view.clips):
        if x['speaker']: view.by_spk[x['speaker']].append(i)
    return view.clips


ADDABLE = {'vox_prefsq_norm', 'vox_text_xattn', 'vox_slots', 'vox_word_aux', 'sem_head', 'sem_cond', 'intent_aux', 'tts_h', 'adsc',
           'flow_text_xattn'}  # adsc: zero-init stream_emb (11-04); flow_text_xattn: zero-gated flow text cross-attn (11-04)
RESIZE_TEXT = {'text_dim'}  # allowed on a fine-tune: ONLY the text input LayerNorm+Linear is re-initialised (see build_model)


def validate_additions(old, changes):
    """Module surgery can add modules or resize the text input, never remove/resize anything else."""
    defaults = {f.name: f.default for f in __import__('dataclasses').fields(Config)}
    for k, value in changes.items():
        before = old.get(k, defaults.get(k))
        if type(value) is type(before) and value == before: continue
        allowed = (k in ADDABLE and ((type(before) is bool and before is False and value is True) or
                   (k == 'vox_slots' and type(before) is int and before == 0 and type(value) is int and value > 0))) or \
                  (k in RESIZE_TEXT and type(before) is int and type(value) is int and before > 0 and value > 0)
        if not allowed:
            raise ValueError(f'--cfg changes build-time switches / unsupported module transition: {k}={before!r} -> {value!r}; '
                             'only false->true, 0->positive slots, text_dim>0, or unchanged values are allowed')


Z_IO = ('post.', 'p_in.', 'p_aux.', 'f_in.', 'f_out.', 'mt_hist.', 'lat_mean', 'lat_std')   # the only z-shaped tensors


def build_model(base_cfg, prev_extra, new_extra, prev_state=None, resize_z=False):
    """Base-config model -> remember base parameter ids -> rebuild the previous stage's modules -> load its weights
    -> add this stage's modules. LoRA keeps the wrapped Linear parameters, so base ids stay valid through surgery.

    text_dim differs from the checkpoint (e.g. r4 e5 768 -> LLM 1024/2560): every other weight is loaded normally and
    only text_in (LayerNorm + Linear) is re-initialised, because its input width changed."""
    if prev_state is not None:
        effective = {**base_cfg, **prev_extra}
        validate_additions(effective, {k: v for k, v in new_extra.items() if k in ADDABLE})
    m = SCMG(Config(**base_cfg))
    base_ids = {id(p) for p in m.parameters()}
    for k, v in prev_extra.items(): setattr(m.cfg, k, v)
    m.add_modules()
    missing = []
    if prev_state is not None:
        state = prev_state
        prev_dim = int(prev_state['text_in.0.weight'].shape[0]) if 'text_in.0.weight' in prev_state else m.cfg.text_dim
        if prev_dim != m.cfg.text_dim:
            state = {k: v for k, v in prev_state.items() if not k.startswith('text_in.')}
            print(f'text_dim {prev_dim} -> {m.cfg.text_dim}: re-initialised text_in (LayerNorm + Linear); '
                  f'all other weights load from the checkpoint', flush=True)
        dropped = []
        if resize_z:
            # 11-04 codec change (VAE latents z=16 -> identity codec z=156): latent input/output layers are re-initialised,
            # everything else loads; any OTHER shape mismatch is an error
            own = m.state_dict()
            dropped = [k for k, v in state.items() if k in own and own[k].shape != v.shape]
            bad = [k for k in dropped if not k.startswith(Z_IO)]
            if bad: raise RuntimeError(f'z resize: unexpected shape mismatch outside the latent I/O layers: {bad[:5]}')
            state = {k: v for k, v in state.items() if k not in dropped}
            print(f'z {int(prev_state["lat_mean"].shape[0])} -> {m.cfg.z}: re-initialised {sorted(dropped)}; all other weights '
                  f'load from the checkpoint', flush=True)
        missing, unexpected = m.load_state_dict(state, strict=False)
        missing = [k for k in missing if k not in dropped]
        if prev_dim != m.cfg.text_dim:
            missing = [k for k in missing if not k.startswith('text_in.')]
            unexpected = [k for k in unexpected if not k.startswith('text_in.')]
        if unexpected or missing: raise RuntimeError(f'checkpoint mismatch: missing {missing[:5]} unexpected {unexpected[:5]}')
    for k, v in new_extra.items(): setattr(m.cfg, k, v)
    m.add_modules()
    return m, base_ids


SELF_FORCE_ARGS = {'p': float, 'steps': int, 'rounds': int, 'cfg': float, 'mix': float, 'noise': float}


def add_self_force_args(ap):
    for name, typ in SELF_FORCE_ARGS.items():
        ap.add_argument('--self-force-' + name, type=typ, default=None,
                        help='override self_force_' + name + ' in --sc (otherwise StageCfg / --sc wins)')


def resolve_self_force_args(a):
    overrides = json.loads(a.sc)
    for name in SELF_FORCE_ARGS:
        key = 'self_force_' + name
        value = getattr(a, key)
        if value is not None: overrides[key] = value
    for name in ('cfg', 'mix', 'noise'):
        key = 'self_force_' + name
        overrides.setdefault(key, getattr(StageCfg(), key))
    resolved = StageCfg(**overrides)
    for name in SELF_FORCE_ARGS:
        key = 'self_force_' + name
        value = getattr(resolved, key)
        if not math.isfinite(value) or value < 0 or (name in ('p', 'mix') and value > 1) or (name in ('steps', 'rounds') and value < 1):
            raise ValueError(f'invalid {key}={value}')
        setattr(a, key, value)             # ARGS.json records the effective values, including --sc overrides
    a.sc = json.dumps(overrides)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['vae'] + list(STAGES))
    ap.add_argument('--out', required=True); ap.add_argument('--view', default=VIEW)
    ap.add_argument('--stats', required=True); ap.add_argument('--text-cache', default=None)
    ap.add_argument('--vae', default=None); ap.add_argument('--init', default=None, help='previous stage checkpoint')
    ap.add_argument('--steps', type=int, default=100000); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--accum', type=int, default=1, help='gradient accumulation micro-batches per step (1 = the original loop)')
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--warmup', type=int, default=2000)
    ap.add_argument('--frames', type=int, default=256); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--ema', type=float, default=0.999); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-every', type=int, default=2000); ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--unfreeze-flow', action='store_true')
    ap.add_argument('--cfg', default='{}', help='JSON overrides of the model Config (e.g. ablations)')
    ap.add_argument('--sc', default='{}', help='JSON overrides of the StageCfg')
    ap.add_argument('--semsynth-share', type=float, default=0.4)
    ap.add_argument('--anchor-share', default=None, help='JSON {"sources": [...], "share": x}: sampling share of measured-motion sources inside the human mass')
    ap.add_argument('--snap-every', type=int, default=10000, help='EMA snapshot for mid-training evaluation')
    ap.add_argument('--deriv-ramp', type=int, default=3000, help='vae: steps to ramp in derivative losses, noise, KL')
    ap.add_argument('--tok-cache', default=None, help='LM token ids (voxcpm planner)')
    ap.add_argument('--overfit', type=int, default=0, help='train AND monitor on the first N long train clips only (sanity check)')
    ap.add_argument('--mmap', action='store_true', help='memory-map the arrays instead of loading them (smoke tests)')
    ap.add_argument('--max-hours', type=float, default=0, help='stop cleanly after this wall time (0 = no limit)')
    ap.add_argument('--tts-hidden', default=None, help='Qwen3-TTS hidden-state export (scmg/tts-hidden-v36_1); needed by cfg tts_h')
    ap.add_argument('--vae-cfg', default='{}', help='stage vae: MotionVAE kwargs (z, c, beta, w_vel, w_acc, w_accmag)')
    ap.add_argument('--holdout', default=None, help='semantic-eval MANIFEST.json: its clips and rhythm variants are never sampled')
    ap.add_argument('--tok-intent', default=None, help='word-level intent labels row-aligned with --tok-cache (models/intent-qwen25-v36_1)')
    ap.add_argument('--intent-frames', default=None, help='frame-level intent track (scmg.prep_intent_frames); used by sc w_int / sem_w')
    ap.add_argument('--gest-frames', default=None, help='direct coarse g_k frame track (scripts/semantic/g1_gest_frames.py); train split only')
    add_self_force_args(ap)
    ap.add_argument('--bagger-dir', default=None, help='colon-separated cumulative BAgger collection directories')
    ap.add_argument('--bagger-share', type=float, default=0.)
    ap.add_argument('--causal-dir', default=None, help='11-05 训练匹配: scripts/echog_g1/precompute_causal.py output (merged) '
                    'for the train view; needed by --sc causal_front / causal_spk / causal_text')
    a = ap.parse_args()
    resolve_self_force_args(a)
    if a.bagger_share > 0 and a.stage != 'base':
        raise ValueError('BAgger currently supports the base fine-tuning stage only')
    rank, world = ddp(); dev = torch.device('cuda')
    if a.bagger_share > 0 and os.environ.get('BAGGER_MEMORY_FRACTION'):
        torch.cuda.set_per_process_memory_fraction(float(os.environ['BAGGER_MEMORY_FRACTION']))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed + rank); np.random.seed(a.seed + rank)
    stats = json.loads(Path(a.stats).read_text())
    view = View(a.view, 'train', a.text_cache, stats, mmap=a.mmap, tok_cache=a.tok_cache, tts_hidden=a.tts_hidden,
                intent_frames=a.intent_frames, holdout=a.holdout, tok_intent=a.tok_intent, bagger_dir=a.bagger_dir,
                gest_frames=a.gest_frames)
    dview = View(a.view, 'val', a.text_cache, stats, mmap=a.mmap, tok_cache=a.tok_cache, tts_hidden=a.tts_hidden)
    if a.overfit:
        overfit_clips(view, a.overfit); dview = view
    if a.anchor_share: view.anchor = json.loads(a.anchor_share)
    if a.stage != 'vae':
        _sc0 = StageCfg(**{**STAGES[a.stage]['sc'], **json.loads(a.sc)})
        _cz = _sc0.causal_front or _sc0.causal_spk or _sc0.causal_text
        if _cz and not a.causal_dir: raise SystemExit('--sc causal_front / causal_spk / causal_text need --causal-dir')
        if a.causal_dir:
            cov = view.load_causal(a.causal_dir, _sc0.horizons)
            if cov < 0.99: raise SystemExit(f'--causal-dir covers only {cov:.1%} of the train clips')
    ds = WindowDataset(view, a.frames, seed=a.seed * 1000 + rank, semsynth_share=a.semsynth_share, bagger_share=a.bagger_share)
    dl = iter(torch.utils.data.DataLoader(ds, a.batch, num_workers=a.workers, collate_fn=collate, pin_memory=True,
                                          persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None))
    dev_ds = WindowDataset(dview, a.frames, seed=12345, fixed=fixed_windows(dview, a.frames))
    st = dict(qm=torch.tensor(stats['q_mean'], device=dev), qs=torch.tensor(stats['q_std'], device=dev))
    if rank == 0: (out / 'ARGS.json').write_text(json.dumps(vars(a), indent=1))

    # ------------------------------------------------------------ model
    if a.stage == 'vae':
        model = MotionVAE(**json.loads(a.vae_cfg)).to(dev); base_ids = set(); vae = None; sc = None
    else:
        vck = torch.load(a.vae, map_location='cpu')
        vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema']); vae.eval().requires_grad_(False)
        spec = STAGES[a.stage]
        base_cfg = dict(z=vae.z)
        prev = torch.load(a.init, map_location='cpu') if a.init else None
        if prev is not None: base_cfg = dict(prev['base_cfg'])            # copy: never mutate the checkpoint's own config
        base_cfg.update(json.loads(a.cfg) if a.stage == 'base' else {})
        resize_z = prev is not None and int(prev['base_cfg']['z']) != int(vae.z)
        if resize_z:
            if a.stage != 'base': raise SystemExit(f'codec z {prev["base_cfg"]["z"]} -> {vae.z} is only supported on a base-stage fine-tune')
            base_cfg['z'] = int(vae.z)
        extra = dict(prev['extra_cfg']) if prev is not None else {}
        extra.update(spec['cfg']); extra.update(json.loads(a.cfg) if a.stage != 'base' else {})
        prev_extra = dict(prev['extra_cfg']) if prev is not None else {}
        if a.stage == 'base' and prev is not None and json.loads(a.cfg):
            # only switches whose modules add_modules() can create (zero-init / unit-gain) may be changed on a fine-tune;
            # anything else is a build-time switch (vox_fsq, vox_lm_init, d, ...) that would be silently ignored -> refuse
            validate_additions({**prev['base_cfg'], **prev_extra}, json.loads(a.cfg))
            # base-stage fine-tune from a checkpoint that switches NEW optional modules on (11-03): build + load with the
            # checkpoint's own config, then turn the new switches on and add their (zero-initialised) modules.
            # text_dim is special: it must be set while building, so text_in is re-initialised before loading.
            build_cfg = dict(prev['base_cfg'])
            if 'text_dim' in base_cfg: build_cfg['text_dim'] = base_cfg['text_dim']
            if resize_z: build_cfg['z'] = int(vae.z)
            model, base_ids = build_model(build_cfg, prev_extra, extra, prev['ema'], resize_z=resize_z)
            for k, v in json.loads(a.cfg).items(): setattr(model.cfg, k, v)
            model.add_modules(); base_ids = set()
            extra.update({k: v for k, v in json.loads(a.cfg).items() if k in extra})
        else:
            model, base_ids = build_model(base_cfg, prev_extra, extra, prev['ema'] if prev is not None else None, resize_z=resize_z)
        if prev is None or resize_z:
            model.lat_mean.copy_(torch.tensor(vck['lat_mean'])); model.lat_std.copy_(torch.tensor(vck['lat_std']))
        elif rank == 0:
            print(f'init from {a.init} (stage {prev.get("stage")}); added modules: {sorted(set(extra) - set(prev_extra))}')
        if a.bagger_share > 0:
            from .bagger import expand_source
            expand_source(model)
            base_cfg['n_source'] = model.cfg.n_source
        model.to(dev)
        if view.text is not None and int(model.cfg.text_dim) != view.text_dim:
            raise SystemExit(f'text cache dim {view.text_dim} != model cfg.text_dim {model.cfg.text_dim}; '
                             f'pass --cfg \'{{"text_dim": {view.text_dim}}}\' (fine-tune re-initialises only text_in)')
        if getattr(model.cfg, 'tts_h', False) and not a.tts_hidden:
            raise SystemExit('cfg tts_h needs --tts-hidden <dir>')
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
            d = dict(cfg=dict(z=model.z, stride=model.stride, **({'c': model.c} if model.c != 384 else {}),
                              **({'dim_mask': list(model.dim_mask)} if model.dim_mask else {}),
                              **({'j': J} if J != 34 else {})), model=model.state_dict(),
                     ema=ema.state_dict(), step=step, vae_cfg=json.loads(a.vae_cfg))
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

    # 11-03 review: the dev loss that picks best.pt is teacher-forced and clean -- no self-forcing, text-focused batches or
    # selective audio dropout (with self_force_p = 1 the old dev loss was itself self-forced, unlike the other lanes)
    sc_dev = __import__('dataclasses').replace(sc, self_force_p=0., text_focus_p=0., p_sel_audio=0.) if sc is not None else None

    def losses(m, b, dev_eval=False):
        if a.stage == 'vae':
            _, l = m(b['q'], b['mask'], deriv=min(1., step / a.deriv_ramp))
            return l
        return m(vae, b, sc_dev if dev_eval else sc, st)

    @torch.no_grad()
    def evaluate():
        g = torch.random.get_rng_state(); gc = torch.cuda.get_rng_state(); torch.manual_seed(777)
        acc = {}; n = 0
        dl_ = torch.utils.data.DataLoader(dev_ds, 64, num_workers=2, collate_fn=collate)
        for b in dl_:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast('cuda', torch.bfloat16):
                l = losses(ema, b, dev_eval=True)
            for k, v in l.items(): acc[k] = acc.get(k, 0.) + float(v)
            n += 1
        torch.random.set_rng_state(g); torch.cuda.set_rng_state(gc)
        return {f'dev_{k}': v / max(n, 1) for k, v in acc.items()}

    t0 = time.time(); tl = time.time(); agg = {}; skipped = 0
    best_key = 'dev_rec' if a.stage == 'vae' else 'dev_flow'
    best = json.loads((out / 'BEST.json').read_text()) if (out / 'BEST.json').exists() else dict(value=1e9, step=0)
    model.train()
    while step < a.steps:
        for gp in opt.param_groups: gp['lr'] = a.lr * gp.get('scale', 1.0) * sched(step)
        if a.accum == 1:
            b = {k: v.to(dev, non_blocking=True) for k, v in next(dl).items()}
            with torch.autocast('cuda', torch.bfloat16):
                l = losses(net, b)
            loss = l['loss']
            opt.zero_grad(set_to_none=True)
            fin = torch.isfinite(loss.detach()).float()
            if world > 1: dist.all_reduce(fin, op=dist.ReduceOp.MIN)     # all ranks skip together (no allreduce hang)
            if fin.item() < 1:
                skipped += 1; step += 1; continue
            loss.backward()
        else:
            # 11-04: gradient accumulation (e.g. 2 x 5090: batch 256 x accum 2 x 2 GPUs = the 1024 of the A800 recipe). Losses
            # are averaged over the micro-batches; DDP syncs gradients only on the last one; a non-finite micro-batch on any
            # rank skips the whole step on all ranks.
            opt.zero_grad(set_to_none=True); lsum = {}; ok = True
            for ai in range(a.accum):
                b = {k: v.to(dev, non_blocking=True) for k, v in next(dl).items()}
                sync = (ai == a.accum - 1) or world == 1
                with (contextlib.nullcontext() if sync else net.no_sync()):
                    with torch.autocast('cuda', torch.bfloat16):
                        l = losses(net, b)
                    fin = torch.isfinite(l['loss'].detach()).float()
                    if world > 1: dist.all_reduce(fin, op=dist.ReduceOp.MIN)
                    if fin.item() < 1: ok = False; break
                    (l['loss'] / a.accum).backward()
                for k, v in l.items(): lsum[k] = lsum.get(k, 0.) + float(v) / a.accum
            if not ok:
                opt.zero_grad(set_to_none=True); skipped += 1; step += 1; continue
            l = lsum
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
            log(out / 'dev.jsonl', ev); print('DEV', json.dumps(ev), flush=True)
            if ev.get(best_key, 1e9) < best['value']:              # model selection on the never-trained dev set
                best = dict(value=ev[best_key], step=step, key=best_key); save(out / 'best.pt', final=True)
                (out / 'BEST.json').write_text(json.dumps(best))
        if step % a.save_every == 0: save(ck)
        if a.snap_every and step % a.snap_every == 0: save(out / f'snap_{step:06d}.pt', final=True)
        if a.max_hours and (time.time() - t0) / 3600 > a.max_hours: break
    save(ck); save(out / 'final.pt', final=True)
    if rank == 0 and not (out / 'best.pt').exists():                # no dev evaluation happened (tiny runs)
        import shutil; shutil.copy(out / 'final.pt', out / 'best.pt')
    if rank == 0:
        done = out / f'DONE.json.pending-{os.getpid()}'
        done.write_text(json.dumps(dict(step=step, skipped=skipped, hours=(time.time() - t0) / 3600)))
        assert 'step' in json.loads(done.read_text())
        os.replace(done, out / 'DONE.json')
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__':
    main()

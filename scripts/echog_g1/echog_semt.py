"""11-07 §7.55 item 1, ECHO-G side: the semantic-first temporal stream as an equivalent module BEFORE the official SGDiT, and a
short offline fine-tune from the official G1 checkpoint (control = the same fine-tune without the module).

SemTimeEG (cf. scmg.model.SemTime): queries = frame-time lattice (sin/cos of the frame index), keys / values = the timed text
tokens (official Qwen features, own LayerNorm + projection to d_s); per head logits q.k / sqrt(dh) - 0.5 ((t_frame - t_word - lag)
/ sigma)^2 with sigma in [0.25, 2] s and lag in +-0.5 s (time_distance = frame time - token centre, the official convention), no
self-attention, 2 layers + MLP; the per-frame semantic state is projected to the DiT width by a ZERO-INITIALISED linear and added to
the DiT input next to the motion / audio projections. Rows without a real text token (the official null text: one zero token, or
condition dropout / CFG unconditional) give exactly 0, so the null branch is the official one.

  train : official RobotSpeechDataset / LengthBucketBatchSampler / collate_motion / flow_matching_loss (condition dropout 0.1,
          temporal weight 0.5), AdamW wd 1e-4, clip 1.0, EMA 0.999 (official), lr --lr with linear warm-up + cosine to 0;
          training stems = official train_drop minus the held-out dev speakers (english_2_scott / english_6_carla, the hos view's
          dev speakers).  NOTE: the official initial checkpoint was trained on ALL of train_drop, so it has seen the dev clips;
          this is identical for control and module lanes (relative comparison only).
  sample: official sampler (echo_g.flow.sample_euler, 8 steps, guidance 1, initial noise stable_seed(stem, seed) per clip) with
          text modes own / notext (official null text) / swap (donor tokens, centres rescaled F / F_donor), as echog_textuse.py.
  test  : zero-init bitwise identity to the official model (with and without text), null text exactly the official, gradients
          reach the module after one step.

    PYTHONPATH=<official>/src python scripts/echog_g1/echog_semt.py train --init best.pt --config sgdit_audio_text.yaml --data-root R --out O [--sem-time]
    PYTHONPATH=<official>/src python scripts/echog_g1/echog_semt.py sample --ckpt O/final.pt --data-root R --stems dev.txt \
        --out D --modes own notext swap --swap-map map.json --seed 0
    PYTHONPATH=<official>/src python scripts/echog_g1/echog_semt.py test --init best.pt --config sgdit_audio_text.yaml --data-root R
"""
import argparse, dataclasses, json, math, os, random, sys, time
from pathlib import Path

import torch
import torch.nn as nn

from echo_g.condition_provenance import load_condition_manifest
from echo_g.data import (LengthBucketBatchSampler, RobotSpeechDataset, collate_motion, decode_condition, load_aligned_lengths,
                         read_stems)
from echo_g.flow import flow_matching_loss, sample_euler
from echo_g.inference import load_model_and_config
from echo_g.model import SpeechGroundedDiT, timestep_embedding
from echo_g.utils import stable_seed

HELD = ('english_2_scott_', 'english_6_carla_')


def sincos(pos, d):
    half = d // 2
    f = torch.exp(-math.log(10_000.0) * torch.arange(half, device=pos.device, dtype=torch.float32) / half)
    a = pos.float()[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1)


class SemTimeEG(nn.Module):
    def __init__(self, text_dim, hidden, d=256, heads=4, layers=2):
        super().__init__()
        self.h, self.d = heads, d
        self.tn = nn.LayerNorm(text_dim); self.tp = nn.Linear(text_dim, d); self.q0 = nn.Linear(d, d)
        self.layers = nn.ModuleList([nn.ModuleDict(dict(
            nq=nn.LayerNorm(d), nk=nn.LayerNorm(d), q=nn.Linear(d, d), k=nn.Linear(d, d), v=nn.Linear(d, d), o=nn.Linear(d, d),
            n2=nn.LayerNorm(d), mlp=nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)))) for _ in range(layers)])
        self.sig_raw = nn.Parameter(torch.zeros(layers, heads)); self.lag_raw = nn.Parameter(torch.zeros(layers, heads))
        self.out_n = nn.LayerNorm(d); self.out = nn.Linear(d, hidden)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, text, text_padding_mask, time_distance):
        B, N, _ = text.shape; T = time_distance.shape[1]; H = self.h; dh = self.d // H
        kp = ~text_padding_mask & (text.abs().amax(-1) > 0)                       # real tokens (the null token is all-zero)
        has = kp.any(-1)                                                          # [B]
        h = self.q0(sincos(torch.arange(T, device=text.device), self.d))[None].expand(B, T, self.d)
        tk = self.tp(self.tn(text.float()))
        dt = time_distance.float()                                                # [B, T, N] seconds
        for li, ly in enumerate(self.layers):
            q = ly['q'](ly['nq'](h)).view(B, T, H, dh); kv = ly['nk'](tk)
            k = ly['k'](kv).view(B, N, H, dh); v = ly['v'](kv).view(B, N, H, dh)
            lg = torch.einsum('bthe,bnhe->bhtn', q.float(), k.float()) / math.sqrt(dh)
            sig = 0.25 + 1.75 * self.sig_raw[li].float().sigmoid(); lag = 0.5 * self.lag_raw[li].float().tanh()
            lg = lg - 0.5 * ((dt[:, None] - lag[None, :, None, None]) / sig[None, :, None, None]).square()
            lg = lg.masked_fill(~kp[:, None, None, :], -1e9)
            w = torch.softmax(lg, -1) * has[:, None, None, None]
            o = torch.einsum('bhtn,bnhe->bthe', w, v.float()).reshape(B, T, self.d)
            h = h + ly['o'](o); h = h + ly['mlp'](ly['n2'](h))
        return self.out(self.out_n(h)) * has[:, None, None].to(h.dtype)


class EGSemDiT(SpeechGroundedDiT):
    """official SpeechGroundedDiT + SemTimeEG added to the DiT input (forward = the official one plus that add)."""

    def __init__(self, config):
        super().__init__(config)
        self.semt = SemTimeEG(config.text_dim, config.hidden_dim)

    def forward(self, noised_motion, flow_time, audio, text_tokens, motion_mask, text_padding_mask, time_distance):
        frames = noised_motion.shape[1]
        if not 1 <= text_tokens.shape[1] <= self.config.max_text_tokens:
            raise ValueError("text token count exceeds model capacity")
        hidden = (self.motion_projection(noised_motion) + self.audio_projection(self.audio_norm(audio))
                  + self.positional_encoding(frames, noised_motion.device, noised_motion.dtype))
        hidden = hidden + self.semt(text_tokens, text_padding_mask, time_distance).to(hidden.dtype)
        condition = self.timestep_mlp(timestep_embedding(flow_time, self.hidden))
        text = self.text_projection(self.text_norm(text_tokens))
        motion_padding_mask = ~motion_mask
        for block in self.blocks:
            hidden = block(hidden, condition, text, motion_padding_mask, text_padding_mask, time_distance)
        shift, scale = self.final_adaptive_norm(condition).chunk(2, dim=-1)
        from echo_g.model import modulate
        return self.output(modulate(self.final_norm(hidden), shift, scale))


class Prosody:
    """§7.55 item 2 on ECHO-G: the 8 prosody columns (onset, prominence, energy, logf0, voiced, beat, beat_left, beat_right; SCMG
    feat89 columns 80:88, normalised) of the SCMG G1 view at 50 Hz, nearest-frame resampled to the ECHO-G 30 fps frames. Replaces
    the Wav2Vec2 (1024) audio features entirely."""
    COLS = slice(80, 88); DIM = 8

    def __init__(self, view):
        import numpy as np
        v = Path(view); self.np = np
        self.items = {it['id']: it for it in json.loads((v / 'train.json').read_text())}
        self.feat = np.load(v / 'train_feat.npy', mmap_mode='r')

    def get(self, stem, F):
        np = self.np; it = self.items[stem]
        idx = np.clip(np.round(np.arange(F) * 50. / 30.).astype(int), 0, it['frames'] - 1) + it['offset']
        return torch.from_numpy(np.asarray(self.feat[idx, self.COLS], np.float32))


class ProsDataset(RobotSpeechDataset):
    def __init__(self, *args, prosody=None, **kw):
        super().__init__(*args, **kw); self.pros = prosody

    def __getitem__(self, index):
        s = super().__getitem__(index); s['audio'] = self.pros.get(s['stem'], int(s['frames'])); return s


def build(init, sem_time, dev, cfg_path=None, audio='w2v'):
    base, config, mean, std, exp = load_model_and_config(Path(init), Path(cfg_path) if cfg_path else None, dev)
    if not sem_time and audio == 'w2v': return base, config, mean, std, exp
    m = EGSemDiT(config.model) if sem_time else SpeechGroundedDiT(config.model)
    with torch.random.fork_rng(devices=[]):   # the new modules' random init does not move the global RNG
        torch.manual_seed(20261107)
        if sem_time: m.semt = SemTimeEG(config.model.text_dim, config.model.hidden_dim)
        if audio == 'prosody':
            m.audio_norm = nn.LayerNorm(Prosody.DIM); m.audio_projection = nn.Linear(Prosody.DIM, config.model.hidden_dim)
    sd = base.state_dict(); new = ('semt.',) + (('audio_norm.', 'audio_projection.') if audio == 'prosody' else ())
    miss, unexp = m.load_state_dict({k: v for k, v in sd.items() if not k.startswith(new)}, strict=False)
    if unexp or any(not k.startswith(new) for k in miss): raise ValueError(f'state mismatch {miss} {unexp}')
    if audio == 'prosody':      # start at the official NULL-audio model: weight 0, bias = official projection of the dropped audio
        with torch.no_grad():
            c0 = base.audio_projection(base.audio_norm(torch.zeros(1, config.model.audio_dim, device=next(base.parameters()).device)))[0]
            m.audio_projection.weight.zero_(); m.audio_projection.bias.copy_(c0.cpu() if not m.audio_projection.bias.is_cuda else c0)
    return m.to(dev), config, mean, std, exp


def load_ft(ckpt, dev):
    c = torch.load(ckpt, map_location='cpu', weights_only=False)
    m, config, mean, std, _ = build(c['init'], c['sem_time'], dev, c['config_path'], c.get('audio', 'w2v'))
    m.load_state_dict(c['ema'], strict=True); m.eval()
    return m, config, c['motion_mean'].float(), c['motion_std'].float(), c


def train(a):
    dev = torch.device('cuda'); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); random.seed(a.seed)
    model, config, mean, std, exp = build(a.init, a.sem_time, dev, a.config, a.audio)
    R = Path(a.data_root)
    stems = [s for s in read_stems(R / config.data.train_split) if not s.startswith(HELD)]
    sp = out / 'train_stems.txt'; sp.write_text('\n'.join(stems) + '\n')
    dcfg = dataclasses.replace(config.data, train_split=str(sp.resolve()))
    if a.audio == 'prosody':
        pros = Prosody(a.view); stems_ok = [s for s in stems if s in pros.items]
        if len(stems_ok) < len(stems): print(f'prosody: {len(stems) - len(stems_ok)} stems without view features dropped')
        sp.write_text('\n'.join(stems_ok) + '\n')
        ds = ProsDataset(R, dcfg, config.model, 'train', mean, std, prosody=pros)
    else: ds = RobotSpeechDataset(R, dcfg, config.model, 'train', mean, std)
    sampler = LengthBucketBatchSampler(ds.lengths, a.batch, a.seed, config.training.bucket_multiplier, shuffle=True, drop_last=True)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate_motion, num_workers=a.workers, pin_memory=True,
                                     persistent_workers=True, prefetch_factor=4)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=config.training.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1., (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    meta = dict(init=str(a.init), config_path=str(a.config), sem_time=a.sem_time, audio=a.audio, view=a.view, data_root=str(R), n_train=len(ds), held_out=list(HELD), steps=a.steps, lr=a.lr,
                warmup=a.warmup, batch=a.batch, accum=a.accum, seed=a.seed, ema_decay=config.training.ema_decay,
                condition_dropout=config.training.condition_dropout, temporal_weight=config.training.temporal_loss_weight)
    (out / 'config.json').write_text(json.dumps(meta, indent=1))
    log = open(out / 'train_log.jsonl', 'a'); step = 0; acc = 0; ep = 0; t0 = time.time(); run = []
    model.train(); opt.zero_grad(set_to_none=True)
    while step < a.steps:
        sampler.set_epoch(ep); ep += 1
        for b in dl:
            b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.bf16):
                tot, fl, tl = flow_matching_loss(model, b, config.training.condition_dropout, config.training.temporal_loss_weight)
            if not torch.isfinite(tot): raise FloatingPointError(f'non-finite loss {b["stems"]}')
            (tot / a.accum).backward(); acc += 1; run.append([tot.item(), fl.item(), tl.item()])
            if acc < a.accum: continue
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            opt.step(); opt.zero_grad(set_to_none=True); sched.step(); acc = 0; step += 1
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point: ema[k].mul_(config.training.ema_decay).add_(v.detach(), alpha=1 - config.training.ema_decay)
                    else: ema[k].copy_(v)
            if step % 25 == 0:
                r = torch.tensor(run).mean(0).tolist(); run = []
                log.write(json.dumps(dict(step=step, loss=r[0], flow=r[1], temporal=r[2], gn=float(gn), lr=sched.get_last_lr()[0],
                                          hours=(time.time() - t0) / 3600)) + '\n'); log.flush()
            if step >= a.steps: break
    pay = dict(ema={k: v.cpu() for k, v in ema.items()}, model={k: v.cpu() for k, v in model.state_dict().items()},
               motion_mean=mean.cpu(), motion_std=std.cpu(), step=step, **meta)
    torch.save(pay, out / 'final.pt.tmp'); os.replace(out / 'final.pt.tmp', out / 'final.pt')
    (out / 'DONE.json').write_text(json.dumps(dict(step=step, hours=(time.time() - t0) / 3600)))
    print('ECHOG_FT_DONE', step, f'{(time.time() - t0) / 3600:.2f}h')


def sample(a):
    dev = torch.device('cuda')
    model, config, mean, std, c = load_ft(a.ckpt, dev); mean, std = mean.to(dev), std.to(dev)
    R = Path(a.data_root); fps = float(config.data.fps)
    lengths = load_aligned_lengths(R / config.data.aligned_lengths_file)
    man = load_condition_manifest(config.data.condition_manifest, config.data.condition_manifest_sha256, R)
    swap = json.loads(Path(a.swap_map).read_text()) if a.swap_map else None
    if 'swap' in a.modes and swap is None: raise SystemExit('--modes swap needs --swap-map')
    stems = read_stems(Path(a.stems))
    pros = Prosody(c['view']) if c.get('audio', 'w2v') == 'prosody' else None

    def cond(stem):
        cp = R / config.data.condition_dir / f'{stem}.pt'; F = min(lengths[stem], config.data.max_frames)
        return decode_condition(torch.load(cp, map_location='cpu', weights_only=True), stem, config.data, config.model, F,
                                condition_path=cp, condition_manifest=man)
    t0 = time.time()
    for n, stem in enumerate(stems):
        cc = cond(stem); F = cc['frames']; audio = (cc['audio'] if pros is None else pros.get(stem, F)).to(dev)
        g = torch.Generator(device=dev); g.manual_seed(stable_seed(stem, a.seed))
        noise = torch.randn(F, model.motion_dim, generator=g, device=dev)
        ft = torch.arange(F, device=dev, dtype=torch.float32) / fps
        for mode in a.modes:
            o = Path(a.out) / f'{mode}_seed_{a.seed:03d}' / f'{stem}.pt'
            if o.exists(): continue
            o.parent.mkdir(parents=True, exist_ok=True); extra = {}
            if mode == 'own': tx, centre = cc['text'].to(dev), cc['token_centers'].to(dev)
            elif mode == 'swap':
                d = cond(swap[stem]); tx = d['text'].to(dev); centre = d['token_centers'].to(dev) * (F / d['frames'])
                extra = dict(text_donor=swap[stem], time_scale=F / d['frames'])
            elif mode == 'notext': tx = centre = None
            else: raise SystemExit(f'mode {mode}')
            if tx is None:
                text = torch.zeros(1, 1, config.model.text_dim, device=dev); tpm = torch.zeros(1, 1, dtype=torch.bool, device=dev)
                td = torch.zeros(1, F, 1, device=dev)
            else:
                text = tx[None]; tpm = torch.zeros(1, len(tx), dtype=torch.bool, device=dev); td = (ft[:, None] - centre[None])[None]
            batch = dict(audio=audio[None], text=text, motion_mask=torch.ones(1, F, dtype=torch.bool, device=dev),
                         text_padding_mask=tpm, time_distance=td)
            y = sample_euler(model, batch, a.steps, a.guidance_scale, initial_noise=noise[None].clone())
            r = (y[0].float() * std + mean).cpu().contiguous()
            if not torch.isfinite(r).all(): raise ValueError(f'{stem}: non-finite')
            torch.save(dict(robot_repr=r, real_num_frames=F, fps=fps, stem=stem, seed=a.seed, representation_units='physical',
                            representation_schema=config.data.representation_schema, source_checkpoint=str(a.ckpt),
                            sem_time=bool(c['sem_time']), text_mode=mode, steps=a.steps, guidance_scale=a.guidance_scale, **extra),
                       o.with_suffix('.tmp'))
            o.with_suffix('.tmp').rename(o)
        if n % 50 == 0: print(f'{n + 1}/{len(stems)} {stem} F={F} {time.time() - t0:.0f}s', flush=True)
    print('ECHOG_SEMT_SAMPLE_DONE', len(stems), f'{time.time() - t0:.0f}s')


def test(a):
    dev = torch.device('cuda'); ok = {}
    base, config, mean, std, _ = build(a.init, False, dev, a.config); semt = build(a.init, True, dev, a.config)[0]
    base.eval(); semt.eval()
    R = Path(a.data_root); stems = [s for s in read_stems(R / config.data.train_split) if not s.startswith(HELD)][:6]
    sp = Path(a.tmp) / 'test_stems.txt'; sp.parent.mkdir(parents=True, exist_ok=True); sp.write_text('\n'.join(stems) + '\n')
    ds = RobotSpeechDataset(R, dataclasses.replace(config.data, train_split=str(sp.resolve())), config.model, 'train', mean, std)
    b = collate_motion([ds[i] for i in range(3)]); b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
    x = torch.randn_like(b['motion']); t = torch.rand(3, device=dev)
    args = (x, t, b['audio'], b['text'], b['motion_mask'], b['text_padding_mask'], b['time_distance'])
    with torch.no_grad():
        ok['zero_init_text'] = torch.equal(base(*args), semt(*args))
        null = torch.ones_like(b['text_padding_mask']); null[:, 0] = False
        nargs = (x, t, torch.zeros_like(b['audio']), torch.zeros_like(b['text']), b['motion_mask'], null, torch.zeros_like(b['time_distance']))
        with torch.no_grad(): semt.semt.out.weight.normal_(0, 0.05)
        ok['null_exact'] = torch.equal(base(*nargs), semt(*nargs))
        ok['active_with_text'] = not torch.equal(base(*args), semt(*args))
        semt.semt.out.weight.zero_()
    semt.train(); torch.manual_seed(0)
    tot, _, _ = flow_matching_loss(semt, b, 0.0, 0.5); tot.backward()
    ok['grad_out'] = float(semt.semt.out.weight.grad.abs().sum()) > 0
    with torch.no_grad(): semt.semt.out.weight -= 1e-2 * semt.semt.out.weight.grad.sign()
    semt.zero_grad(); torch.manual_seed(0); tot, _, _ = flow_matching_loss(semt, b, 0.0, 0.5); tot.backward()
    ok['grad_inner'] = all(float(p.grad.abs().sum()) > 0 for n, p in semt.semt.named_parameters() if n.startswith('layers.0.q.weight') or n == 'tp.weight')
    if a.view:                  # prosody audio: at init exactly the official model with the null (dropped) audio
        pm = build(a.init, False, dev, a.config, 'prosody')[0].eval(); pr = Prosody(a.view)
        pa = collate_motion([dict(ds[i], audio=pr.get(ds[i]['stem'], int(ds[i]['frames']))) for i in range(3)])['audio'].to(dev)
        with torch.no_grad():
            dd = float((pm(x, t, pa, *args[3:]) - base(x, t, torch.zeros_like(b['audio']), *args[3:])).abs().max())
            print('prosody init vs null audio max |diff|', dd)   # not bitwise: the null projection is a GEMM of another shape
            ok['prosody_init_is_null_audio'] = dd < 1e-4
        pm.train(); torch.manual_seed(0)
        tot, _, _ = flow_matching_loss(pm, dict(b, audio=pa), 0.0, 0.5); tot.backward()
        ok['prosody_grad'] = float(pm.audio_projection.weight.grad.abs().sum()) > 0
    print(json.dumps(ok)); print('ALL_OK' if all(ok.values()) else 'SOME_FAIL')


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train'); t.add_argument('--init', required=True); t.add_argument('--data-root', required=True)
    t.add_argument('--config', required=True, help='official yaml, e.g. official/configs/sgdit_audio_text.yaml')
    t.add_argument('--out', required=True); t.add_argument('--sem-time', action='store_true'); t.add_argument('--steps', type=int, default=2000)
    t.add_argument('--lr', type=float, default=1e-4); t.add_argument('--warmup', type=int, default=100)
    t.add_argument('--batch', type=int, default=12); t.add_argument('--accum', type=int, default=4)
    t.add_argument('--workers', type=int, default=8); t.add_argument('--seed', type=int, default=0)
    t.add_argument('--bf16', action='store_true')
    t.add_argument('--audio', default='w2v', choices=['w2v', 'prosody'], help='prosody = §7.55 item 2: Wav2Vec2 replaced by 8 prosody dims')
    t.add_argument('--view', default=None, help='SCMG G1 view with train.json / train_feat.npy (for --audio prosody)')
    s = sub.add_parser('sample'); s.add_argument('--ckpt', required=True); s.add_argument('--data-root', required=True)
    s.add_argument('--stems', required=True); s.add_argument('--out', required=True); s.add_argument('--modes', nargs='+', default=['own'])
    s.add_argument('--swap-map'); s.add_argument('--seed', type=int, default=0); s.add_argument('--steps', type=int, default=8)
    s.add_argument('--guidance-scale', type=float, default=1.0)
    e = sub.add_parser('test'); e.add_argument('--init', required=True); e.add_argument('--data-root', required=True)
    e.add_argument('--config', required=True); e.add_argument('--tmp', default='/tmp/echog_semt_test')
    e.add_argument('--view', default=None, help='also test --audio prosody')
    a = ap.parse_args()
    dict(train=train, sample=sample, test=test)[a.cmd](a)


if __name__ == '__main__':
    main()

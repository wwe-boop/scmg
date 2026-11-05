"""Unit tests of the ECHO-G streaming fine-tune (11-05).  PYTHONPATH=<official>/src python scripts/echog_stream/test_stream.py
    --checkpoint best.pt --config sgdit_audio_text.yaml --cond-dir C --meta M --stems S [--official-pred official_seed_000]
1  zero init: StreamSGDiT(official weights) == official SpeechGroundedDiT, same inputs, bit for bit (with and without known
   frames given), and the official best.pt loads with only known_projection.* missing.
2  degenerate stream == offline: stream_clip with ahead = inf, text final, audio frozen, one block covering the clip, no tail
   == the offline path == echo_g.flow.sample_euler of the official model with the official noise (and the released
   official prediction, if given); also for a model with a NONZERO known_projection (stream path == its offline path).
3  no leakage in training windows: (a) structural checks over 4000 random windows (audio rows only for f / fps < now + ahead,
   visible tokens arrived (x2) and voiced, time_distance = frame time - centre, known = frames < c on the block grid,
   no loss on known frames, window length bounded); (b) perturbing ALL future information (audio frames at / after
   now + ahead, every token not visible) leaves the window tensors bit-identical.
4  no leakage in streaming inference: perturbing audio after T0 and tokens ending after T0 leaves every block decided with
   now + ahead <= T0 bit-identical.
5  lockstep multi-clip batching (--clip-batch) matches the single-clip path (padding masked; float order only).
"""
import argparse, math, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from echo_g.condition_provenance import load_condition_manifest
from echo_g.data import decode_condition, load_aligned_lengths, read_stems
from echo_g.flow import sample_euler
from echo_g.inference import load_model_and_config
from echo_g.utils import stable_seed
from stream_core import (StreamConfig, StreamSGDiT, block_start, frames_before, load_official_into, sample_window,
                         sample_window_euler, visible_tokens, window_conditions, x2_arrival)
from stream_infer import batch1, stream_clip, stream_clips

ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--config', type=Path, required=True)
ap.add_argument('--cond-dir', type=Path, required=True); ap.add_argument('--meta', type=Path, required=True)
ap.add_argument('--stems', type=Path, required=True); ap.add_argument('--official-pred', type=Path)
ap.add_argument('--device', default='cuda')
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = torch.device(a.device); RES = {}


def ok(name, cond, msg=''):
    RES[name] = bool(cond); print(('PASS ' if cond else 'FAIL ') + name + (' | ' + msg if msg else ''), flush=True)


off, config, mean, std, _ = load_model_and_config(a.checkpoint, a.config, dev)
ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
sm = StreamSGDiT(config.model); load_official_into(sm, ck['ema']); sm.to(dev).eval()
fps = float(config.data.fps); D = config.model.motion_dim

# ---- 1 zero init
g = torch.Generator(device=dev).manual_seed(1)
B, T, N = 2, 150, 17
x = torch.randn(B, T, D, generator=g, device=dev); t = torch.rand(B, generator=g, device=dev)
au = torch.randn(B, T, config.model.audio_dim, generator=g, device=dev); tx = torch.randn(B, N, config.model.text_dim, generator=g, device=dev)
mm = torch.ones(B, T, dtype=torch.bool, device=dev); mm[1, 120:] = False
tpm = torch.zeros(B, N, dtype=torch.bool, device=dev); tpm[1, 11:] = True
td = torch.randn(B, T, N, generator=g, device=dev)
kn = torch.randn(B, T, D, generator=g, device=dev); km = torch.rand(B, T, generator=g, device=dev) < 0.4
with torch.no_grad():
    yo = off(x, t, au, tx, mm, tpm, td); y0 = sm(x, t, au, tx, mm, tpm, td); y1 = sm(x, t, au, tx, mm, tpm, td, known=kn, known_mask=km)
ok('1a zero-init forward == official (no known)', torch.equal(yo, y0), f'max|diff| {(yo - y0).abs().max().item():.3g}')
ok('1b zero-init forward == official (random known frames)', torch.equal(yo, y1), f'max|diff| {(yo - y1).abs().max().item():.3g}')
sm.known_fill = 'hold'
with torch.no_grad(): y2 = sm(x, t, au, tx, mm, tpm, td, known=kn, known_mask=km)
sm.known_fill = 'zero'
ok('1b2 zero-init forward == official (random known frames, known_fill=hold)', torch.equal(yo, y2), f'max|diff| {(yo - y2).abs().max().item():.3g}')
ok('1c official weights load with only known_projection new', set(sm.state_dict()) - set(ck['ema']) == {'known_projection.weight', 'known_projection.bias'})

# ---- 2 degenerate stream == offline == official sample
lengths = load_aligned_lengths(a.meta / config.data.aligned_lengths_file)
man = load_condition_manifest(a.meta / config.data.condition_manifest, config.data.condition_manifest_sha256) if config.data.condition_manifest else None
stems = read_stems(a.stems)[:3]
sc = StreamConfig()


def clip(stem):
    cp = a.cond_dir / f'{stem}.pt'; p = torch.load(cp, map_location='cpu', weights_only=True)
    F = min(lengths[stem], config.data.max_frames)
    c = decode_condition(p, stem, config.data, config.model, F, condition_path=cp, condition_manifest=man)
    return F, c['audio'].to(dev), c['text'].to(dev), c['token_centers'].to(dev), torch.as_tensor(p['token_times']).double().numpy()


worst = dict(off_vs_official=0.0, stream_vs_off=0.0, pred=0.0, nz_stream_vs_off=0.0)
nz = StreamSGDiT(config.model); nz.load_state_dict(sm.state_dict()); nz.to(dev).eval()
with torch.no_grad(): nz.known_projection.weight.normal_(0, 0.05); nz.known_projection.bias.normal_(0, 0.05)
for stem in stems:
    F, au_, tx_, ce_, tt_ = clip(stem)
    gg = torch.Generator(device=dev); gg.manual_seed(stable_seed(stem, 0)); noise = torch.randn(F, D, generator=gg, device=dev)
    cond = window_conditions(au_, tx_, tt_, ce_, 0, F, 0.0, math.inf, fps, 'final')
    z = torch.zeros(1, F, D, device=dev); zm = torch.zeros(1, F, dtype=torch.bool, device=dev)
    for mdl, key in ((sm, ''), (nz, 'nz_')):
        yoff = sample_window_euler(mdl, batch1(cond, F, dev), noise[None].clone(), z, zm, config.sampling.integration_steps, 'clean')[0]
        ys, _ = stream_clip(mdl, sc, au_, tx_, tt_, ce_, F, noise, fps, config.sampling.integration_steps, block_s=1e3, ahead=math.inf,
                            ctx_f=10 ** 6, tail_f=0, text_mode='final', dev=dev)
        worst[key + 'stream_vs_off'] = max(worst[key + 'stream_vs_off'], (ys - yoff).abs().max().item())
        if key == '':
            ref = sample_euler(off, batch1(cond, F, dev), config.sampling.integration_steps, 1.0, initial_noise=noise[None].clone())[0]
            worst['off_vs_official'] = max(worst['off_vs_official'], (yoff - ref).abs().max().item())
            if a.official_pred:
                pr = torch.load(a.official_pred / f'{stem}.pt', map_location='cpu', weights_only=True)['robot_repr'].to(dev)
                worst['pred'] = max(worst['pred'], ((yoff * std.to(dev) + mean.to(dev)) - pr).abs().max().item())
ok('2a offline path (zero init) == official sample_euler, official noise', worst['off_vs_official'] == 0, f"max|diff| {worst['off_vs_official']:.3g}")
ok('2b degenerate stream == offline (zero init)', worst['stream_vs_off'] == 0, f"max|diff| {worst['stream_vs_off']:.3g}")
ok('2c degenerate stream == offline (nonzero known_projection)', worst['nz_stream_vs_off'] == 0, f"max|diff| {worst['nz_stream_vs_off']:.3g}")
if a.official_pred:
    ok('2d offline path == released official prediction (physical units; official ran batch 3 with padding)', worst['pred'] < 1e-4, f"max|diff| {worst['pred']:.3g}")

# ---- 3 training windows: no leakage
Fs = 480; Nt = 40; rng0 = np.random.default_rng(0)
starts = np.sort(rng0.uniform(0.3, Fs / fps - 0.8, Nt)); ends = starts + rng0.uniform(0.05, 0.6, Nt)
tt = np.stack([starts, ends], 1)
base = dict(frames=Fs, motion=torch.randn(Fs, D), audio=torch.arange(1, Fs + 1).float()[:, None].repeat(1, 8),
            text=torch.arange(1, Nt + 1).float()[:, None].repeat(1, 6), token_centers=torch.as_tensor(tt.mean(1)).float(),
            token_times=tt, stem='synthetic', has_timing=True)
scs = StreamConfig(); bad = []; n_cold = 0; n_null = 0; lens = []; perturb_ok = True
arr = x2_arrival(tt[:, 0], scs.x2_rate, scs.x2_lag); bs = scs.block_ms / 1000
for i in range(4000):
    w = sample_window(base, np.random.default_rng(i), scs, fps); m = w['meta']
    ws, c, f1, fe, now, ahead, k = m['ws'], m['c'], m['f1'], m['fe'], m['now'], m['ahead'], m['k']
    W = fe - ws; lens.append(W); n_cold += c == 0
    rows = torch.nonzero(w['audio'][:, 0]).flatten() + ws                     # absolute frames that carry audio
    if len(rows) and not (rows.float() / fps < now + ahead).all(): bad.append((i, 'audio future'))
    if len(rows) and not torch.equal(w['audio'][rows - ws, 0], rows.float() + 1): bad.append((i, 'audio misplaced'))
    if len(rows) != max(0, min(fe, frames_before(now + ahead, fps)) - ws): bad.append((i, 'audio missing'))
    if w['text'].abs().sum() == 0: n_null += 1; vis_idx = np.zeros(0, int)
    else:
        vis_idx = (w['text'][:, 0].numpy() - 1).astype(int)
        if not ((arr[vis_idx] <= now + 1e-9).all() and (tt[vis_idx, 1] <= now + ahead + 1e-9).all()): bad.append((i, 'token future'))
        exp_td = torch.arange(ws, fe).float()[:, None] / fps - base['token_centers'][vis_idx][None]
        if not torch.allclose(w['time_distance'], exp_td, atol=1e-5): bad.append((i, 'time distance'))
    vis_all = np.nonzero(visible_tokens(tt, now, ahead, 'voiced', scs.x2_rate, scs.x2_lag))[0]
    if not np.array_equal(vis_idx, vis_all): bad.append((i, 'token set'))
    if c != block_start(k, bs, fps) or abs(now - k * bs) > 1e-12: bad.append((i, 'grid'))
    if not torch.equal(w['known_mask'], torch.arange(ws, fe) < c) or (w['loss_weight'][w['known_mask']] != 0).any() \
            or (w['loss_weight'][~w['known_mask']] <= 0).any(): bad.append((i, 'known/loss'))
    if not torch.equal(w['motion'], base['motion'][ws:fe]): bad.append((i, 'motion'))
    if W > int(round(scs.ctx_s * fps)) + int(math.ceil(bs * fps)) + 1 + int(round(scs.tail_max_s * fps)): bad.append((i, 'length'))   # G1 30 FPS: 11
    if i % 8 == 0:                                                              # (b) perturb every future quantity
        p = dict(base); a2 = base['audio'].clone(); a2[frames_before(now + ahead, fps):] = -7.0; p['audio'] = a2
        t2 = base['text'].clone(); hide = np.setdiff1d(np.arange(Nt), vis_all); t2[hide] = -3.0; p['text'] = t2
        tt2 = tt.copy(); tt2[hide] += 0.5                                     # timing of unseen words must not matter either
        p['token_times'] = tt2; p['token_centers'] = torch.as_tensor(tt2.mean(1)).float()
        w2 = sample_window(p, np.random.default_rng(i), scs, fps)
        same = all(torch.equal(w[k_], w2[k_]) for k_ in ('motion', 'known_mask', 'loss_weight', 'audio', 'text', 'text_padding_mask', 'time_distance'))
        perturb_ok &= same
        if not same: bad.append((i, 'perturbation changed the window'))
ok('3a training windows: no future audio / text, exact time_distance, block grid, no loss on known', not bad,
   f'{len(bad)} bad {bad[:3]}; cold-start {n_cold / 4000:.3f}, null-text {n_null / 4000:.3f}, window len {min(lens)}..{max(lens)} mean {np.mean(lens):.0f}')
ok('3b training windows: perturbing all future information leaves the window bit-identical', perturb_ok)

# ---- 4 streaming inference: no leakage (frozen audio, voiced text)
stem = stems[0]; F, au_, tx_, ce_, tt_ = clip(stem)
gg = torch.Generator(device=dev); gg.manual_seed(stable_seed(stem, 0)); noise = torch.randn(F + 19, D, generator=gg, device=dev)
T0 = 0.5 * F / fps; ahead = 0.16
au2 = au_.clone(); au2[frames_before(T0, fps):] += 5.0
tx2 = tx_.clone(); tx2[torch.as_tensor(tt_[:, 1] > T0, device=dev)] += 5.0
kw = dict(block_s=0.32, ahead=ahead, ctx_f=240, tail_f=19, text_mode='voiced', dev=dev)
nzh = StreamSGDiT(config.model); nzh.load_state_dict(nz.state_dict()); nzh.to(dev).eval(); nzh.known_fill = 'hold'
for mdl, key in ((sm, 'zero-init'), (nz, 'nonzero known_projection'), (nzh, 'nonzero known_projection, hold fill')):
    y1, _ = stream_clip(mdl, sc, au_, tx_, tt_, ce_, F, noise, fps, 8, **kw)
    y2, _ = stream_clip(mdl, sc, au2, tx2, tt_, ce_, F, noise, fps, 8, **kw)
    kmax = max(k for k in range(10 ** 4) if k * 0.32 + ahead <= T0 + 1e-9)
    fcut = block_start(kmax + 1, 0.32, fps)
    ok(f'4 stream inference ({key}): blocks decided by T0 unchanged when the future changes', torch.equal(y1[:fcut], y2[:fcut])
       and not torch.equal(y1, y2), f'{fcut} frames identical of {F}; later frames differ: {not torch.equal(y1[fcut:], y2[fcut:])}')
# ---- 5 lockstep multi-clip batching (padding masked) == single-clip streaming up to float accumulation order
cl = []
for st in stems:
    F, au_, tx_, ce_, tt_ = clip(st); gg = torch.Generator(device=dev); gg.manual_seed(stable_seed(st, 0))
    cl.append(dict(audio=au_, text=tx_, tt=tt_, centres=ce_, F=F, noise=torch.randn(F + 19, D, generator=gg, device=dev), wav=None))
kw5 = dict(block_s=0.32, ahead=0.16, ctx_f=240, tail_f=19, text_mode='voiced', dev=dev)
yb, _ = stream_clips(nz, sc, cl, fps, 8, **kw5)
d5 = max((stream_clips(nz, sc, [c], fps, 8, **kw5)[0][0] - y).abs().max().item() for c, y in zip(cl, yb))
ok('5 lockstep batch of clips == one clip at a time (normalised units)', d5 < 1e-3, f'max|diff| {d5:.3g} over {len(cl)} clips')
print('ALL_OK' if all(RES.values()) else 'SOME_FAILED', sum(RES.values()), '/', len(RES))

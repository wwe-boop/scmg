"""Fine-tune ECHO-G (official SGDiT) into the trained streaming model StreamSGDiT (11-05).

Init from an official-format checkpoint (G1: weights/best.pt; EX002: an EX002-trained SGDiT best.pt), the new
known_projection zero-initialised.  Every training item = one streaming window of one clip (stream_core.sample_window:
cut c on the 320 ms block grid, p_cold c = 0, committed context up to ctx_s, block + random tail generated, audio / text
truncated to what has arrived at now = c / fps + ahead); loss = official flow-matching + temporal loss on the generated
frames only, official condition dropout.  Clips are drawn with probability proportional to their block count (= uniform
over decisions).  Optimiser / EMA / clipping / weight decay as the official config (AdamW, EMA 0.999, clip 1.0, wd 1e-4);
learning rate --lr with a short linear warm-up (fresh optimiser state on a converged model) and cosine decay to 0.
Single-node DDP (torchrun --standalone --nproc_per_node N); only LOCAL_RANK / LOCAL_WORLD_SIZE are used.

    torchrun --standalone --nproc_per_node 4 scripts/echog_stream/train_stream.py --init best.pt --data-root R --out O \
        --profile g1 --steps 20000 --lr 1e-4 --batch 16
Outputs: latest.pt (resume), ema.pt (latest EMA, inference format), ema_step<N>.pt every --snap-every, final.pt,
complete.json, train_log.jsonl.  Inference format = official checkpoint keys + 'stream' (StreamConfig) + 'schema'.
"""
from __future__ import annotations

import argparse, collections, concurrent.futures, contextlib, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'echog_g1'))
from echo_g.config import ExperimentConfig
from echo_g.data import RobotSpeechDataset, load_motion_stats
from echo_g.training import experiment_metadata
from echo_g.utils import atomic_json_save, atomic_torch_save
from stream_core import (STREAM_SCHEMA, StreamConfig, StreamSGDiT, collate_windows, frames_before, load_official_into, n_blocks,
                         sample_window, stream_flow_loss)

PROFILES = {'g1': dict(fps=30, motion_dim=39), 'ex002': dict(fps=50, motion_dim=34)}


class StreamDataset(RobotSpeechDataset):
    """official RobotSpeechDataset item (all official checks, frozen-condition manifest verified at first load) + the raw
    token times (x2 arrival / voiced rule).  cache=True keeps decoded clips in RAM (audio / text as the released FP16,
    lossless; ~0.8 MB per clip) so the condition files are read and verified once per rank."""

    def __init__(self, *args, cache=False, **kw):
        super().__init__(*args, **kw); self.cache = {} if cache else None

    def _load(self, index):
        s = super().__getitem__(index)
        if s.get('has_timing'):
            p = torch.load(self.condition_root / f'{self.stems[index]}.pt', map_location='cpu', weights_only=True)
            s['token_times'] = torch.as_tensor(p['token_times']).double().numpy()
        else:
            s['token_times'] = np.zeros((s['text'].shape[0], 2))
        return s

    def __getitem__(self, index):
        if self.cache is None: return self._load(index)
        c = self.cache.get(index)
        if c is None:
            s = self._load(index); c = dict(s, audio=s['audio'].half(), text=s['text'].half()); self.cache[index] = c
        return dict(c, audio=c['audio'].float(), text=c['text'].float())


class WavStore:
    """stem -> 16 kHz mono waveform exactly as stream_infer loads it (librosa.load sr=16000), cached per rank as int16 when
    that is lossless (PCM16 sources), else float32.  src: a directory of <stem>.wav or a jsonl with stem / audio_path."""

    def __init__(self, src: Path):
        import librosa, threading
        self.librosa = librosa; self.lock = threading.Lock(); self.cache = {}
        if src.is_dir(): self.path = lambda stem: src / f'{stem}.wav'
        else:
            m = {}
            with src.open() as f:
                for line in f:
                    r = json.loads(line); m[r['stem']] = Path(r['audio_path'])
            self.path = m.__getitem__

    def __call__(self, stem):
        c = self.cache.get(stem)
        if c is None:
            x = self.librosa.load(self.path(stem), sr=16_000, mono=True)[0]
            q = np.round(x.astype(np.float64) * 32768.0)
            c = q.astype(np.int16) if (np.abs(q).max(initial=0) <= 32767 and np.array_equal((q / 32768.0).astype(np.float32), x)) else x
            with self.lock: self.cache[stem] = c
        return (c.astype(np.float32) / 32768.0) if c.dtype == np.int16 else c


class TrainPrefixAudio:
    """training windows get the audio stream_infer --audio prefix gives at the same decision: Wav2Vec2 (echog_slide.PrefixAudio, FP32 with
    the PyTorch default TF32 flags of stream_infer) on the waveform that has arrived, wav[: now + ahead], rows ws .. min(fe, frames_before(now + ahead),
    len(features)) - 1 of its features, zeros after (= stream_core.window_conditions on the prefix features)."""

    def __init__(self, model_dir, dev, fps, src_fps, mode='batch16'):
        from echog_slide import PrefixAudio
        self.pa = PrefixAudio(model_dir, dev); self.fps = fps; self.src_fps = src_fps; self.mode = mode
        if mode not in ('exact', 'batch32', 'batch16'): raise ValueError(mode)
        for p in self.pa.model.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def apply(self, b, dev):
        m32, c32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True   # = stream_infer (PyTorch defaults)
        try:
            ts = [m['now'] + m['ahead'] for m in b['meta']]
            if self.mode == 'exact': fs = [self.pa(w, t, self.fps, self.src_fps) if t > 0 else None for w, t in zip(b['wavs'], ts)]
            else: fs = self.pa.batch(b['wavs'], ts, self.fps, self.src_fps, fp16=self.mode == 'batch16')
            for i, (meta, t, f) in enumerate(zip(b['meta'], ts, fs)):
                ws, fe = meta['ws'], meta['fe']
                b['audio'][i].zero_()
                if t <= 0: continue
                f = f.to(dev, b['audio'].dtype)
                hi = min(fe, frames_before(t, self.fps), f.shape[0])
                if hi > ws: b['audio'][i, :hi - ws] = f[ws:hi]
                meta['audio_frames'] = max(0, hi - ws)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = m32, c32
        return b


def attach_wavs(batch, wavs, fps):
    """the arrived waveform of each window (CPU; only wav[: now + ahead] is ever read)"""
    if wavs is not None:
        batch['wavs'] = [wavs(st)[:int(round((m['now'] + m['ahead']) * 16000)) + 1] for st, m in zip(batch['stems'], batch['meta'])]
    return batch


class WindowBatches:
    """bounded ordered thread prefetch (no multiprocessing IPC on CPFS); infinite; deterministic given the seed."""

    def __init__(self, ds, sc, fps, batch, seed, workers, fixed=None, wavs=None):
        self.ds, self.sc, self.fps, self.batch, self.workers, self.wavs = ds, sc, fps, batch, max(1, workers), wavs
        nb = np.array([n_blocks(L, sc.block_ms / 1000, fps) for L in ds.lengths], np.float64)
        self.p = nb / nb.sum(); self.rng = np.random.default_rng(seed); self.fixed = fixed

    def _make(self, idx, seeds):
        return attach_wavs(collate_windows([sample_window(self.ds[i], np.random.default_rng(s), self.sc, self.fps) for i, s in zip(idx, seeds)]),
                           self.wavs, self.fps)

    def _draw(self):
        return self.rng.choice(len(self.p), size=self.batch, p=self.p), self.rng.integers(0, 2 ** 62, size=self.batch)

    def __iter__(self):
        with concurrent.futures.ThreadPoolExecutor(self.workers) as pool:
            q = collections.deque(pool.submit(self._make, *self._draw()) for _ in range(2 * self.workers))
            while True:
                r = q.popleft().result(); q.append(pool.submit(self._make, *self._draw())); yield r


def fixed_val_batches(ds, sc, fps, batch, n_clips, seed=100_003, wavs=None):
    rng = np.random.default_rng(seed); idx = list(range(min(n_clips, len(ds)))); out = []
    for i in range(0, len(idx), batch):
        out.append(attach_wavs(collate_windows([sample_window(ds[j], rng, sc, fps) for j in idx[i:i + batch]]), wavs, fps))
    return out


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def val_loss(model, ema, batches, dev, temporal, known_xt):
    """fixed val windows, fixed noise: (EMA loss, raw-model loss)"""
    def run():
        tot = 0.0
        with torch.random.fork_rng(devices=[dev.index]):
            torch.manual_seed(100_003)
            for b in batches: tot += stream_flow_loss(model, to_dev(b, dev), 0.0, temporal, known_xt)[0].item()
        return tot / max(1, len(batches))
    model.eval(); raw = run()
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema); e = run()
    model.load_state_dict(backup); model.train()
    return e, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', type=Path, required=True, help='official-format SGDiT checkpoint (ema + motion stats + experiment)')
    ap.add_argument('--init-stats', action='store_true', help='normalise with the init checkpoint motion_mean/std instead of the data root stats')
    ap.add_argument('--config', type=Path, help='official yaml; default: the config embedded in --init')
    ap.add_argument('--data-root', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--profile', choices=list(PROFILES), default='g1')
    ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--ema-decay', type=float, help='default: official config (0.999); smoke runs of a few hundred steps need less')
    ap.add_argument('--batch', type=int, default=16, help='windows per GPU per micro-batch'); ap.add_argument('--accum', type=int, default=1)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--seed', type=int, default=20261105)
    ap.add_argument('--log-every', type=int, default=50); ap.add_argument('--save-every', type=int, default=1000)
    ap.add_argument('--snap-every', type=int, default=5000); ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--val-clips', type=int, default=256)
    ap.add_argument('--max-train', type=int, default=0); ap.add_argument('--max-val', type=int, default=0)
    ap.add_argument('--cache', type=int, default=1, help='keep decoded training clips in RAM (FP16 features, ~0.8 MB / clip / rank)')
    ap.add_argument('--tf32', type=int, default=1, help='TF32 matmuls in training (official trains strict FP32; inference stays FP32)')
    ap.add_argument('--stream', default='{}', help='JSON overrides of StreamConfig')
    ap.add_argument('--prefix-audio', type=Path, help='Wav2Vec2 model dir: train on the strictly causal prefix audio of stream_infer '
                    '--audio prefix (re-extracted per window from wav[: now + ahead]) instead of slices of the released full-clip features')
    ap.add_argument('--prefix-audio-mode', default='batch16', choices=['exact', 'batch32', 'batch16'],
                    help='exact: per-window FP32 (bit-identical to stream_infer); batch32 / batch16: one padded forward per batch '
                         '(FP32 / FP16 autocast; relative L2 vs exact ~3e-4 / ~6e-3, vs the frozen slices ~0.5)')
    ap.add_argument('--wav-dir', type=Path, help='with --prefix-audio: directory of <stem>.wav (train + val stems) or a jsonl stem / audio_path')
    a = ap.parse_args()

    dist_on = 'LOCAL_RANK' in os.environ
    rank = int(os.environ.get('LOCAL_RANK', 0)); world = int(os.environ.get('LOCAL_WORLD_SIZE', 1)) if dist_on else 1
    if dist_on:
        os.environ['RANK'] = str(rank); os.environ['WORLD_SIZE'] = str(world)
        dist.init_process_group('nccl', rank=rank, world_size=world)
    torch.cuda.set_device(rank); dev = torch.device('cuda', rank)
    torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32); torch.backends.cudnn.allow_tf32 = bool(a.tf32)

    init = torch.load(a.init, map_location='cpu', weights_only=False)
    if a.config: config = ExperimentConfig.from_yaml(a.config)
    elif 'config' in init['experiment']: config = ExperimentConfig.from_dict(init['experiment']['config'])
    else: raise SystemExit('--init has no embedded config (official release best.pt): pass --config <official yaml>')
    prof = PROFILES[a.profile]
    if config.model.motion_dim != prof['motion_dim'] or config.data.fps != prof['fps']:
        raise SystemExit(f'profile {a.profile} expects {prof}, config has {config.model.motion_dim}D / {config.data.fps} FPS')
    fps = float(config.data.fps)
    sc = StreamConfig.from_dict(json.loads(a.stream))
    if int(round(sc.ctx_s * fps)) + int(round(sc.block_ms / 1000 * fps)) + 1 + int(round(max(sc.tail_max_s, sc.infer_tail_ms / 1000) * fps)) > config.model.max_t:
        raise SystemExit('ctx + block + tail exceeds the model positional capacity')
    if a.init_stats:      # keep the init model's normalisation (fine-tune on a data root whose train statistics differ, 11-06)
        mean, std = init['motion_mean'].float().clone(), init['motion_std'].float().clone()
    else:
        mean, std = load_motion_stats(a.data_root / config.data.stats_file, config.model.motion_dim)
        if not (torch.allclose(mean, init['motion_mean'].float()) and torch.allclose(std, init['motion_std'].float())):
            raise SystemExit('data-root statistics differ from the init checkpoint (pass --init-stats to keep the init normalisation)')
    tr = StreamDataset(a.data_root, config.data, config.model, 'train', mean, std, a.max_train, cache=bool(a.cache))
    va = StreamDataset(a.data_root, config.data, config.model, 'val', mean, std, a.max_val)
    wavs = tpa = None
    if a.prefix_audio:
        if not a.wav_dir: raise SystemExit('--prefix-audio needs --wav-dir')
        from echog_slide import resample_source_fps
        p0 = torch.load(tr.condition_root / f'{tr.stems[0]}.pt', map_location='cpu', weights_only=True)
        src_fps = resample_source_fps(p0)
        wavs = WavStore(a.wav_dir); tpa = TrainPrefixAudio(a.prefix_audio, dev, fps, src_fps, a.prefix_audio_mode)
        for st in (tr.stems[0], va.stems[0]): wavs(st)          # fail early on a missing waveform

    model = StreamSGDiT(config.model); model.known_fill = sc.known_fill
    resume = a.out / 'latest.pt'; step = 0; ck = None
    if resume.exists():
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); step = int(ck['global_step'])
        if StreamConfig.from_dict(ck['stream']) != sc: raise SystemExit(f"--stream differs from the resumed run: {ck['stream']}")
    elif init.get('schema') == STREAM_SCHEMA:      # continue a trained streaming model (its EMA, incl. known_projection)
        model.load_state_dict(init['ema'], strict=True)
    else:
        load_official_into(model, init['ema'])
    model.to(dev)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if ck is not None: ema = {k: v.to(dev) for k, v in ck['ema'].items()}
    tc = config.training
    ema_decay = a.ema_decay if a.ema_decay is not None else tc.ema_decay
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=tc.weight_decay)
    lam = lambda s: min(1.0, (s + 1) / max(1, a.warmup)) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    if ck is not None: opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler'])
    net = DDP(model, device_ids=[rank], broadcast_buffers=False) if world > 1 else model
    torch.manual_seed(a.seed + 7919 * rank + step)
    loader = iter(WindowBatches(tr, sc, fps, a.batch, a.seed + 1009 * rank + 100_000 * step, a.workers, wavs=wavs))
    vb = fixed_val_batches(va, sc, fps, a.batch, a.val_clips, wavs=wavs) if rank == 0 and a.eval_every > 0 else []
    if tpa is not None:
        for x in vb: tpa.apply(x, x['audio'].device); x.pop('wavs')

    exp = experiment_metadata(config, a.data_root, tr, va)
    train_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    exp.update(stream_train=dict(train_args, world_size=world, global_batch=a.batch * a.accum * world, init=str(a.init),
                                 init_schema=init.get('schema'), adapter='StreamSGDiT (official SGDiT + zero-init known_projection)',
                                 train_audio=f'prefix ({a.prefix_audio_mode}: per-window Wav2Vec2 on the arrived waveform)' if tpa else 'frozen (released full-clip features)'))
    if rank == 0:
        a.out.mkdir(parents=True, exist_ok=True); atomic_json_save(exp, a.out / 'experiment.json')
        nparam = sum(p.numel() for p in model.parameters())
        print(json.dumps(dict(event='start', step=step, train=len(tr), val=len(va), params_m=nparam / 1e6, world=world,
                              global_batch=a.batch * a.accum * world, stream=sc.to_dict())), flush=True)

    def snapshot(val):
        return dict(schema=STREAM_SCHEMA, global_step=step, validation=val, ema={k: v.detach().cpu() for k, v in ema.items()},
                    motion_mean=mean, motion_std=std, experiment=exp, stream=sc.to_dict())

    log = open(a.out / 'train_log.jsonl', 'a') if rank == 0 else None
    t0 = time.perf_counter(); tl = t0; sl = step; last_val = float('nan'); peak = 0.0
    if rank == 0 and vb and step == 0:
        e0, r0 = val_loss(model, ema, vb, dev, tc.temporal_loss_weight, sc.known_xt)
        r = dict(step=0, ema_val=e0, model_val=r0, val_windows=sum(len(x['stems']) for x in vb)); print(json.dumps(r), flush=True)
        log.write(json.dumps(r) + '\n'); log.flush()
    net.train(); wait = 0.0; audio_s = 0.0
    while step < a.steps:
        for i in range(a.accum):
            tw = time.perf_counter(); b = to_dev(next(loader), dev); wait += time.perf_counter() - tw
            if tpa is not None:
                ta = time.perf_counter(); tpa.apply(b, dev); audio_s += time.perf_counter() - ta
            with (net.no_sync() if world > 1 and i < a.accum - 1 else contextlib.nullcontext()):
                loss, fl, tl_ = stream_flow_loss(net, b, tc.condition_dropout, tc.temporal_loss_weight, sc.known_xt, sc.known_aug_std)
                if not torch.isfinite(loss): raise FloatingPointError(f'non-finite loss at step {step} stems {b["stems"]}')
                (loss / a.accum).backward()
        g = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.gradient_clip)
        if not torch.isfinite(g): raise FloatingPointError(f'non-finite gradient at step {step}')
        opt.step(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
        with torch.no_grad():
            for k, v in model.state_dict().items(): ema[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)
        if rank == 0 and (step <= 3 or step % a.log_every == 0):
            now = time.perf_counter(); peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30
            r = dict(step=step, loss=loss.item(), flow=fl.item(), temporal=tl_.item(), grad=float(g), lr=sched.get_last_lr()[0],
                     s_per_step=(now - tl) / max(1, step - sl), kp_norm=float(model.known_projection.weight.norm()),
                     data_wait_s=round(wait / max(1, step - sl), 4), prefix_audio_s=round(audio_s / max(1, step - sl), 4),
                     peak_gib=round(peak, 2), wall=round(now - t0, 1))
            tl, sl, wait, audio_s = now, step, 0.0, 0.0; print(json.dumps(r), flush=True); log.write(json.dumps(r) + '\n'); log.flush()
        last = step >= a.steps
        if rank == 0 and vb and (step % a.eval_every == 0 or last):
            last_val, raw_val = val_loss(model, ema, vb, dev, tc.temporal_loss_weight, sc.known_xt)
            r = dict(step=step, ema_val=last_val, model_val=raw_val, val_windows=sum(len(x['stems']) for x in vb)); print(json.dumps(r), flush=True)
            log.write(json.dumps(r) + '\n'); log.flush()
        if step % a.save_every == 0 or last or (a.snap_every and step % a.snap_every == 0):
            if rank == 0:
                snap = snapshot(last_val)
                atomic_torch_save(snap, a.out / 'ema.pt')
                if a.snap_every and step % a.snap_every == 0 and not last: atomic_torch_save(snap, a.out / f'ema_step{step:06d}.pt')
                atomic_torch_save(dict(snap, model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                       optimizer=opt.state_dict(), scheduler=sched.state_dict()), a.out / 'latest.pt')
                if last:
                    atomic_torch_save(snap, a.out / 'final.pt')
                    atomic_json_save(dict(status='complete', global_step=step, final_validation=last_val,
                                          wall_seconds=time.perf_counter() - t0, peak_memory_gib=peak, world_size=world,
                                          global_batch=a.batch * a.accum * world), a.out / 'complete.json')
            if world > 1: dist.barrier()
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__':
    main()

"""Streaming inference of the trained StreamSGDiT (or the official best.pt = zero-init StreamSGDiT), 11-05.

Same protocol as training (stream_core): block k (block_ms, default 320 ms) is decided at now_k = k * block_s; window
[max(0, f0 - ctx), f1 + tail) where [f0, f1) is block k; committed frames [ws, f0) are the known prefix (condition +
clean x_t), the block and the tail are generated from the clip's official noise (stable_seed(stem, seed): frames < F are
exactly the echo_g.inference initial noise, frames past the clip end draw further from the same generator); only block k is
committed.  The tail may run past the clip end (no information there: zero audio, no text) so the window never uses the
clip length.  Conditions at now_k:
  --audio prefix (default, strictly causal): official Wav2Vec2 extraction re-run on the waveform prefix [0, now + ahead)
          (echog_slide.PrefixAudio); frozen: slices of the released full-utterance features (training uses these; leaks the
          future through the bidirectional encoder / utterance normalisation -- measured as a control);
  --text voiced (default) | x2 | final (see stream_core.visible_tokens).
--offline: whole clip in one window, c = 0, frozen audio, final text (checks that the streaming fine-tune kept the offline
ability).  With --ahead-ms inf --text final --audio frozen and --block-ms >= the clip, the stream path reduces to the
offline sample (test_stream.py).
Output <out>/<stem>.pt in the official prediction format (robot_repr [F, D] physical, real_num_frames, fps, seed, ...).
    PYTHONPATH=<official>/src python scripts/echog_stream/stream_infer.py --checkpoint final.pt --cond-dir C --meta R \
        --stems S --out preds/seed_000 --seed 0 --audio prefix --wav-dir R/audio --audio-model wav2vec2-large-xlsr-53-english
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'echog_g1'))
from echo_g.condition_provenance import load_condition_manifest
from echo_g.data import decode_condition, load_aligned_lengths, read_stems
from echo_g.utils import atomic_torch_save, stable_seed
from stream_core import block_start, load_stream_model, sample_window_euler, window_conditions


def batch1(cond, W, dev):
    return dict(audio=cond['audio'][None], text=cond['text'][None], text_padding_mask=cond['text_padding_mask'][None],
                time_distance=cond['time_distance'][None], motion_mask=torch.ones(1, W, dtype=torch.bool, device=dev))


@torch.no_grad()
def stream_clips(model, sc, clips, fps, steps, *, block_s, ahead, ctx_f, tail_f, text_mode, prefix_audio=None, dev):
    """several clips streamed in lockstep (block k of every active clip in one padded batch; padding is masked as in the
    official batched inference). clip = dict(audio, text, tt, centres, F, noise [F + tail_f, D], wav).
    returns normalised motions [F_i, D] + per-block logs. One clip -> no padding, the exact single-clip computation."""
    D = model.motion_dim; outs = [torch.zeros(c['F'], D, device=dev) for c in clips]; logs = [[] for _ in clips]; k = 0
    while True:
        f0 = block_start(k, block_s, fps); act = [i for i, c in enumerate(clips) if f0 < c['F']]
        if not act: break
        now = k * block_s; items = []
        for i in act:
            c = clips[i]; f1 = min(c['F'], block_start(k + 1, block_s, fps)); ws = max(0, f0 - ctx_f); fe = f1 + tail_f
            feats = prefix_audio(c['wav'], now + ahead, fps).to(dev) if prefix_audio is not None else c['audio']
            cond = window_conditions(feats, c['text'], c['tt'], c['centres'], ws, fe, now, ahead, fps, text_mode, sc.x2_rate, sc.x2_lag)
            items.append((i, f1, ws, fe, cond)); logs[i].append((cond['n_visible'], cond['audio_frames']))
        B = len(items); Wm = max(fe - ws for _, _, ws, fe, _ in items); Nm = max(it[4]['text'].shape[0] for it in items)
        c0 = items[0][4]
        bt = dict(audio=torch.zeros(B, Wm, c0['audio'].shape[1], device=dev), text=torch.zeros(B, Nm, c0['text'].shape[1], device=dev),
                  text_padding_mask=torch.ones(B, Nm, dtype=torch.bool, device=dev), time_distance=torch.zeros(B, Wm, Nm, device=dev),
                  motion_mask=torch.zeros(B, Wm, dtype=torch.bool, device=dev))
        nz = torch.zeros(B, Wm, D, device=dev); known = torch.zeros(B, Wm, D, device=dev); km = torch.zeros(B, Wm, dtype=torch.bool, device=dev)
        for r, (i, f1, ws, fe, cond) in enumerate(items):
            W = fe - ws; n = cond['text'].shape[0]
            bt['audio'][r, :W] = cond['audio']; bt['text'][r, :n] = cond['text']; bt['text_padding_mask'][r, :n] = cond['text_padding_mask']
            bt['time_distance'][r, :W, :n] = cond['time_distance']; bt['motion_mask'][r, :W] = True
            nz[r, :W] = clips[i]['noise'][ws:fe]; known[r, :f0 - ws] = outs[i][ws:f0]; km[r, :f0 - ws] = True
        y = sample_window_euler(model, bt, nz, known, km, steps, sc.known_xt)
        for r, (i, f1, ws, fe, cond) in enumerate(items): outs[i][f0:f1] = y[r, f0 - ws:f1 - ws]
        k += 1
    return outs, logs


def stream_clip(model, sc, audio_full, text, tt, centres, F, noise, fps, steps, *, block_s, ahead, ctx_f, tail_f, text_mode,
                prefix_audio=None, wav=None, dev):
    """one clip, block by block. noise: [F + tail_f, D] (absolute frames). returns normalised motion [F, D] + per-block log."""
    o, l = stream_clips(model, sc, [dict(audio=audio_full, text=text, tt=tt, centres=centres, F=F, noise=noise, wav=wav)], fps, steps,
                        block_s=block_s, ahead=ahead, ctx_f=ctx_f, tail_f=tail_f, text_mode=text_mode, prefix_audio=prefix_audio, dev=dev)
    return o[0], l[0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--config', type=Path, help='only for an official checkpoint without embedded config')
    ap.add_argument('--cond-dir', type=Path, required=True)
    ap.add_argument('--meta', type=Path, required=True, help='data root (or official-meta) holding config.data.aligned_lengths_file / condition_manifest')
    ap.add_argument('--stems', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int)
    ap.add_argument('--offline', action='store_true')
    ap.add_argument('--block-ms', type=float); ap.add_argument('--ahead-ms', default=None, help="ms or 'inf'; default: checkpoint stream config")
    ap.add_argument('--ctx-s', type=float); ap.add_argument('--tail-ms', type=float)
    ap.add_argument('--audio', choices=['prefix', 'frozen'], default='prefix')
    ap.add_argument('--wav-dir', type=Path); ap.add_argument('--audio-model', type=Path)
    ap.add_argument('--text', choices=['voiced', 'x2', 'final'], default='voiced')
    ap.add_argument('--clip-batch', type=int, default=1, help='clips streamed in lockstep per padded batch (1 = exact single-clip path)')
    ap.add_argument('--shard', default='0/1'); ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, config, mean, std, sc, meta = load_stream_model(a.checkpoint, dev, a.config)
    if config.data.conditioning != 'audio-text': raise SystemExit('audio-text checkpoint expected')
    fps = float(config.data.fps); mean, std = mean.to(dev), std.to(dev)
    steps = a.steps or config.sampling.integration_steps
    if config.sampling.guidance_scale != 1.0: raise SystemExit('guidance 1 only')
    block_s = (a.block_ms if a.block_ms is not None else sc.block_ms) / 1000.
    ahead = sc.ahead_s if a.ahead_ms is None else (math.inf if a.ahead_ms == 'inf' else float(a.ahead_ms) / 1000.)
    ctx_f = int(round((a.ctx_s if a.ctx_s is not None else sc.ctx_s) * fps))
    tail_f = int(round((a.tail_ms if a.tail_ms is not None else sc.infer_tail_ms) / 1000. * fps))
    lengths = load_aligned_lengths(a.meta / config.data.aligned_lengths_file)
    man = (load_condition_manifest(a.meta / config.data.condition_manifest, config.data.condition_manifest_sha256)
           if config.data.condition_manifest else None)
    pa = None
    if not a.offline and a.audio == 'prefix':
        if not (a.wav_dir and a.audio_model): raise SystemExit('--audio prefix needs --wav-dir and --audio-model')
        import librosa
        from echog_slide import PrefixAudio
        pa = PrefixAudio(a.audio_model, dev)
    stems = read_stems(a.stems); si, sn = (int(x) for x in a.shard.split('/')); stems = stems[si::sn]
    a.out.mkdir(parents=True, exist_ok=True)
    proto = dict(mode='offline' if a.offline else 'stream', block_ms=block_s * 1000, ahead_s=ahead, ctx_frames=ctx_f, tail_frames=tail_f,
                 audio='frozen' if a.offline else a.audio, text='final' if a.offline else a.text, steps=steps,
                 known_xt=sc.known_xt, x2_rate=sc.x2_rate, x2_lag=sc.x2_lag, checkpoint=str(a.checkpoint),
                 checkpoint_step=meta.get('global_step'), stream_trained='stream' in meta)
    t0 = time.time(); log = []; done = 0
    todo = [s_ for s_ in stems if not (a.out / f'{s_}.pt').exists()]
    G = 1 if a.offline else max(1, a.clip_batch)

    def load(stem):
        cp = a.cond_dir / f'{stem}.pt'; payload = torch.load(cp, map_location='cpu', weights_only=True)
        F = min(lengths[stem], config.data.max_frames)
        c = decode_condition(payload, stem, config.data, config.model, F, condition_path=cp, condition_manifest=man)
        g = torch.Generator(device=dev); g.manual_seed(stable_seed(stem, a.seed))
        noise = torch.randn(F, model.motion_dim, generator=g, device=dev)        # == echo_g.inference initial noise of this clip
        if tail_f and not a.offline: noise = torch.cat([noise, torch.randn(tail_f, model.motion_dim, generator=g, device=dev)])
        wav = librosa.load(a.wav_dir / f'{stem}.wav', sr=16_000, mono=True)[0] if pa is not None else None
        return dict(stem=stem, F=F, audio=c['audio'].to(dev), text=c['text'].to(dev), centres=c['token_centers'].to(dev),
                    tt=torch.as_tensor(payload['token_times']).double().numpy(), noise=noise, wav=wav)

    for gi in range(0, len(todo), G):
        tc = time.time(); clips = [load(s_) for s_ in todo[gi:gi + G]]
        if a.offline:
            c = clips[0]; F = c['F']
            cond = window_conditions(c['audio'], c['text'], c['tt'], c['centres'], 0, F, 0.0, math.inf, fps, 'final')
            outs = [sample_window_euler(model, batch1(cond, F, dev), c['noise'][:F][None].clone(), torch.zeros(1, F, model.motion_dim, device=dev),
                                        torch.zeros(1, F, dtype=torch.bool, device=dev), steps, sc.known_xt)[0]]; blogs = [[]]
        else:
            outs, blogs = stream_clips(model, sc, clips, fps, steps, block_s=block_s, ahead=ahead, ctx_f=ctx_f, tail_f=tail_f,
                                       text_mode=a.text, prefix_audio=pa, dev=dev)
        sec = (time.time() - tc) / len(clips)
        for c, outn, blog in zip(clips, outs, blogs):
            r = (outn.float() * std + mean).cpu().contiguous()
            if not torch.isfinite(r).all(): raise ValueError(f"{c['stem']}: non-finite")
            atomic_torch_save(dict(robot_repr=r, real_num_frames=c['F'], fps=fps, stem=c['stem'], seed=a.seed, representation_units='physical',
                                   representation_schema=config.data.representation_schema, streaming=dict(proto, blocks=len(blog)),
                                   source_checkpoint=str(a.checkpoint)), a.out / f"{c['stem']}.pt")
            log.append(dict(stem=c['stem'], frames=c['F'], blocks=len(blog), tokens=len(c['tt']), seconds_group_avg=round(sec, 3),
                            visible_tokens_per_block=[b[0] for b in blog])); done += 1
        print(f'{done}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    with (a.out / f'STREAM_LOG.{si}of{sn}.jsonl').open('a') as f:
        for r in log: f.write(json.dumps(r) + '\n')
    (a.out / 'STREAM_PROTOCOL.json').write_text(json.dumps(proto, indent=1))
    print('STREAM_DONE', done, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()

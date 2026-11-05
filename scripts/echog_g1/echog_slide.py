"""ECHO-G sliding-window streaming baseline (11-04): the official SGDiT (bidirectional, offline) run block by block.

Every block (320 ms, the SCMG chunk; at 30 FPS block k = frames f with k*0.32 <= f/30 < (k+1)*0.32) is decided at its start
time now_k = k * 0.32 s, using ONLY what has arrived by then:
  audio  up to t_a = now_k + ahead (default 160 ms = the SCMG engine look-ahead H = 2 x 80 ms; 'inf' = whole clip);
         window frames at or after t_a get zero audio features (the official null-audio value).
         --audio prefix (default, strictly causal): the official extractor (Wav2Vec2 final hidden state of the 16 kHz waveform,
                 processor normalisation, linear interpolation align_corners=True to round(duration x 30) frames) is re-run on
                 the waveform PREFIX [0, t_a) -- features cannot see later audio.
         --audio frozen: slices of the released full-utterance features. LEAKS the future: Wav2Vec2 self-attention spans the
                 whole utterance, the processor normalises with whole-utterance mean / variance and the 50 -> 30 FPS
                 interpolation grid depends on the total duration (diagnostic only).
  text   --text voiced (default, strictly causal): a token is visible iff it has arrived under the SCMG x2 rule
                 (arrival_j = min(start_j - lag, start_first - lag + j / rate), rate 15 tok/s, lag 0.15 s) by now_k AND its word
                 has been fully voiced (token end <= t_a): ECHO-G needs a token centre (start+end)/2 for its word-time
                 attention, which is only known once the word has ended. The released Qwen3.5-4B features are hidden layer -2
                 of a causal decoder over the transcript without a template, so the prefix of the token sequence equals what
                 the encoder gives on the text prefix (no future words leak through the features).
         --text x2: x2 arrival only, with the TRUE (future) word times of not-yet-voiced tokens -- leaks timing (upper bound).
         --text final: the whole transcript and timing from the start (offline text).
  window frames [max(0, f1 - window), f1) with f1 = end of block k (default window 20 s = the model limit -> the whole prefix);
         regenerated from scratch every block with the clip's official initial noise (stable_seed(stem, seed), sliced), only
         block k is committed; committed frames are never changed.
  --prefix none (default): plain regeneration (the committed past is not a condition of the official model);
  --prefix inpaint: committed frames of the window are imposed along the flow path (x_t = t x_committed + (1 - t) noise) at
         every Euler step, so the new block continues the committed motion (RePaint-style; ECHO-G was not trained for it).
With --ahead inf --text final --audio frozen the last block's window is the whole clip with the official noise and conditions,
i.e. exactly the official offline sample (sanity check against echo_g.inference output).
Output: <out>/<stem>.pt in the official prediction format (robot_repr [F, D] physical, real_num_frames, fps, ...).
EX002 (34D, 50 FPS; 11-05): same code; prefix audio repeats the EX002 condition pipeline (official 30 FPS extraction, then the
30 -> 50 FPS resampling recorded in payload['resampling']); data roots without a frozen condition manifest are accepted.
    PYTHONPATH=<official>/src python scripts/echog_g1/echog_slide.py --checkpoint best.pt --config sgdit_audio_text.yaml
        --cond-dir condition_30fps --meta <official-meta> --wav-dir audio --audio-model wav2vec2-large-xlsr-53-english
        --stems stems.txt --out preds/seed_000 --seed 0
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

from echo_g.conditions import interpolate_features
from echo_g.condition_provenance import load_condition_manifest
from echo_g.data import decode_condition, load_aligned_lengths, read_stems
from echo_g.flow import sample_euler
from echo_g.inference import load_model_and_config
from echo_g.utils import stable_seed


def x2_arrival(start, rate, lag):
    """SCMG scmg.model.x2_arrival for one utterance (all tokens valid, onsets finite)."""
    j = np.arange(len(start), dtype=np.float64)
    return np.minimum(start - lag, start.min() - lag + j / rate) if len(start) else start


class PrefixAudio:
    """Official Wav2Vec2 feature extraction (echo_g.conditions.extract_audio_conditions) on a waveform prefix."""

    def __init__(self, model_dir, device):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.proc = Wav2Vec2Processor.from_pretrained(model_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device).eval(); self.dev = device

    @torch.no_grad()
    def __call__(self, wav, t_end, fps=30.0, src_fps=None):
        """src_fps: frame rate the released features were extracted at before a resampling to fps (EX002: the official
        30 FPS extraction, FP16, then FP32 linear align_corners=True to 50 FPS, FP16 -- payload['resampling']); None or
        == fps: the official direct extraction (G1)."""
        x = wav[:int(round(min(t_end, len(wav) / 16000.) * 16000))]
        dur = len(x) / 16000.
        iv = self.proc(x, sampling_rate=16_000, return_tensors='pt').input_values.to(self.dev)
        h = self.model(iv, output_hidden_states=True).hidden_states[-1].float()
        return self._post(h, dur, fps, src_fps)

    @torch.no_grad()
    def batch(self, wavs, t_ends, fps=30.0, src_fps=None, fp16=True):
        """__call__ for several prefixes in one padded forward (attention mask; per-sample normalisation as the processor
        does it). Not bit-identical to __call__: relative L2 ~3e-4 (FP32) / ~6e-3 (FP16 autocast), measured 11-05."""
        xs = [w[:int(round(min(t, len(w) / 16000.) * 16000))] for w, t in zip(wavs, t_ends)]
        out = [None] * len(xs); idx = [i for i, x in enumerate(xs) if len(x) >= 400]   # shorter than one conv frame: per-sample
        for i in range(len(xs)):
            if i not in idx: out[i] = self(wavs[i], t_ends[i], fps, src_fps)
        if not idx: return out
        iv = self.proc([xs[i] for i in idx], sampling_rate=16_000, return_tensors='pt', padding=True, return_attention_mask=True)
        am = iv.attention_mask.to(self.dev)
        with torch.autocast('cuda', torch.float16, enabled=fp16):
            h = self.model(iv.input_values.to(self.dev), attention_mask=am, output_hidden_states=True).hidden_states[-1].float()
        lens = self.model._get_feat_extract_output_lengths(am.sum(-1)).tolist()
        for r, i in enumerate(idx): out[i] = self._post(h[r:r + 1, :int(lens[r])], len(xs[i]) / 16000., fps, src_fps)
        return out

    @staticmethod
    def _post(h, dur, fps, src_fps):
        if src_fps is None or float(src_fps) == float(fps):
            return interpolate_features(h, int(round(dur * fps))).half().float()   # the official cache stores FP16
        f = interpolate_features(h, int(round(dur * float(src_fps)))).half()         # official 30 FPS cache (FP16)
        n = max(2, int(round(dur * fps)))                                            # EX002: round(duration x 50) frames
        return torch.nn.functional.interpolate(f.float().T[None], size=n, mode='linear', align_corners=True)[0].T.half().float()


def resample_source_fps(payload):
    """source frame rate of resampled released features (EX002 condition_50fps: 30), else None"""
    r = payload.get('resampling')
    return float(r['source_fps']) if isinstance(r, dict) and 'source_fps' in r else None


@torch.no_grad()
def euler_inpaint(model, batch, steps, noise, known, n_known):
    """official echo_g.flow.sample_euler (guidance 1) with the first n_known frames imposed along the flow path."""
    x = noise.clone(); dt = 1.0 / steps
    for i in range(steps):
        t = i * dt
        if n_known: x[:, :n_known] = t * known + (1 - t) * noise[:, :n_known]
        ft = torch.full((x.shape[0],), t, device=x.device, dtype=torch.float32)
        x = x + dt * model(x, ft, batch['audio'], batch['text'], batch['motion_mask'], batch['text_padding_mask'], batch['time_distance'])
    if n_known: x[:, :n_known] = known
    return x


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--config', type=Path, help='default: the config embedded in the checkpoint')
    ap.add_argument('--cond-dir', type=Path, required=True)
    ap.add_argument('--meta', type=Path, required=True, help='official data root part with audit/condition/ (lengths + frozen manifest)')
    ap.add_argument('--stems', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--guidance-scale', type=float, default=1.0)
    ap.add_argument('--block-ms', type=float, default=320.); ap.add_argument('--ahead-ms', default='160')
    ap.add_argument('--window-s', type=float, default=20.)
    ap.add_argument('--tail-ms', type=float, default=0., help='frames appended after the block end with zero audio and no text '
                    '(no information; keeps the committed block away from the sequence end the model was trained on)')
    ap.add_argument('--audio', choices=['prefix', 'frozen'], default='prefix')
    ap.add_argument('--wav-dir', type=Path); ap.add_argument('--audio-model', type=Path)
    ap.add_argument('--text', choices=['voiced', 'x2', 'final'], default='voiced')
    ap.add_argument('--x2-rate', type=float, default=15.); ap.add_argument('--x2-lag', type=float, default=0.15)
    ap.add_argument('--prefix', choices=['none', 'inpaint'], default='none')
    ap.add_argument('--shard', default='0/1'); ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, config, mean, std, _ = load_model_and_config(a.checkpoint, a.config, dev)
    if config.data.conditioning != 'audio-text': raise SystemExit('audio-text checkpoint expected')
    if a.prefix == 'inpaint' and a.guidance_scale != 1: raise SystemExit('--prefix inpaint implements guidance 1 only')
    fps = float(config.data.fps); mean, std = mean.to(dev), std.to(dev)
    lengths = load_aligned_lengths(a.meta / config.data.aligned_lengths_file)
    man = (load_condition_manifest(a.meta / config.data.condition_manifest, config.data.condition_manifest_sha256)
           if config.data.condition_manifest else None)                         # EX002 data roots: no frozen manifest
    pa = None
    if a.audio == 'prefix':
        if not (a.wav_dir and a.audio_model): raise SystemExit('--audio prefix needs --wav-dir and --audio-model')
        import librosa
        pa = PrefixAudio(a.audio_model, dev)
    ahead = math.inf if a.ahead_ms == 'inf' else float(a.ahead_ms) / 1000.
    blk = a.block_ms / 1000.; Wmax = min(int(round(a.window_s * fps)), config.data.max_frames)
    stems = read_stems(a.stems); si, sn = (int(x) for x in a.shard.split('/')); stems = stems[si::sn]
    a.out.mkdir(parents=True, exist_ok=True)
    proto = dict(block_ms=a.block_ms, ahead_ms=a.ahead_ms, window_s=a.window_s, tail_ms=a.tail_ms, audio=a.audio, text=a.text, x2_rate=a.x2_rate,
                 x2_lag=a.x2_lag, prefix=a.prefix, steps=a.steps, guidance_scale=a.guidance_scale)
    t0 = time.time(); log = []
    for n, stem in enumerate(stems):
        o = a.out / f'{stem}.pt'
        if o.exists(): continue
        cp = a.cond_dir / f'{stem}.pt'
        payload = torch.load(cp, map_location='cpu', weights_only=True)
        F = min(lengths[stem], config.data.max_frames)
        c = decode_condition(payload, stem, config.data, config.model, F, condition_path=cp, condition_manifest=man)
        audio_full = c['audio'].to(dev); text = c['text'].to(dev)
        tt = torch.as_tensor(payload['token_times']).double().numpy()          # [N, 2] start / end, s (FP16 as released)
        centre = c['token_centers'].to(dev)                                    # official FP32 centres
        arr = x2_arrival(tt[:, 0], a.x2_rate, a.x2_lag)
        wav = librosa.load(a.wav_dir / f'{stem}.wav', sr=16_000, mono=True)[0] if pa is not None else None
        src_fps = resample_source_fps(payload)
        g = torch.Generator(device=dev); g.manual_seed(stable_seed(stem, a.seed))
        noise = torch.randn(F, model.motion_dim, generator=g, device=dev)    # == echo_g.inference initial noise of this clip
        nT = int(round(a.tail_ms / 1000. * fps))                               # tail frames past the clip end: extra noise
        nz = torch.cat([noise, torch.randn(nT, model.motion_dim, generator=g, device=dev)]) if nT else noise
        out = torch.zeros(F, model.motion_dim, device=dev); nb = 0; vis_hist = []
        k = 0
        while True:
            f0 = int(math.ceil(k * blk * fps - 1e-6)); f1 = min(F, int(math.ceil((k + 1) * blk * fps - 1e-6)))
            if f0 >= F: break
            now = k * blk; ta = now + ahead
            fe = f1 + nT                                                    # window end incl. the empty tail
            ws = max(0, fe - Wmax); W = fe - ws
            if ws > f0: raise ValueError('window shorter than tail + block')
            na = F if ta == math.inf else min(F, int(math.ceil(ta * fps - 1e-6)))      # frames with f / fps < t_a
            if pa is not None:                                            # re-extracted on the arrived waveform prefix
                feats = pa(wav, ta, fps, src_fps).to(dev); na = min(na, len(feats))
            else:
                feats = audio_full
            au = torch.zeros(W, audio_full.shape[1], device=dev)
            lo, hi = ws, min(f1, na)
            if hi > lo: au[:hi - ws] = feats[lo:hi]
            if a.text == 'final': vis = np.ones(len(tt), bool)
            elif a.text == 'x2': vis = arr <= now + 1e-9
            else: vis = (arr <= now + 1e-9) & (tt[:, 1] <= ta + 1e-9)
            idx = torch.as_tensor(np.nonzero(vis)[0], device=dev)
            if len(idx):
                tx = text[idx]; tpm = torch.zeros(1, len(idx), dtype=torch.bool, device=dev)
                ft = torch.arange(ws, fe, device=dev, dtype=torch.float32) / fps
                td = (ft[:, None] - centre[idx][None])[None]
            else:                                                        # nothing arrived: the official null text
                tx = torch.zeros(1, text.shape[1], device=dev); tpm = torch.zeros(1, 1, dtype=torch.bool, device=dev)
                td = torch.zeros(1, W, 1, device=dev)
            batch = dict(audio=au[None], text=tx[None], motion_mask=torch.ones(1, W, dtype=torch.bool, device=dev),
                         text_padding_mask=tpm, time_distance=td)
            if a.prefix == 'inpaint' and f0 > ws:
                y = euler_inpaint(model, batch, a.steps, nz[ws:fe][None], ((out[ws:f0] - mean) / std)[None], f0 - ws)
            else:
                y = sample_euler(model, batch, a.steps, a.guidance_scale, initial_noise=nz[ws:fe][None].clone())
            out[f0:f1] = y[0, f0 - ws:f1 - ws].float() * std + mean
            vis_hist.append(int(vis.sum())); nb += 1; k += 1
        r = out.cpu().contiguous()
        if not torch.isfinite(r).all(): raise ValueError(f'{stem}: non-finite')
        torch.save(dict(robot_repr=r, real_num_frames=F, fps=fps, stem=stem, seed=a.seed, representation_units='physical',
                        representation_schema=config.data.representation_schema, streaming=dict(proto, blocks=nb),
                        source_checkpoint=str(a.checkpoint)), o)
        log.append(dict(stem=stem, frames=F, blocks=nb, tokens=len(tt), visible_tokens_per_block=vis_hist))
        if n % 10 == 0: print(f'{n + 1}/{len(stems)} {stem} F={F} blocks={nb} {time.time() - t0:.0f}s', flush=True)
    with (a.out / f'SLIDE_LOG.{si}of{sn}.jsonl').open('a') as f:
        for r in log: f.write(json.dumps(r) + '\n')
    (a.out / 'SLIDE_PROTOCOL.json').write_text(json.dumps(dict(proto, checkpoint=str(a.checkpoint), config=str(a.config)), indent=1))
    print('SLIDE_DONE', len(stems), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

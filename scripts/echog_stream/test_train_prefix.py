"""train_stream --prefix-audio == stream_infer --audio prefix, window by window (11-05).

For fixed training windows of a few clips: the audio rows TrainPrefixAudio puts in the batch must be bit-identical to
stream_core.window_conditions(echog_slide.PrefixAudio(full waveform loaded as stream_infer loads it, now + ahead), ...)
-- the inference path at the same decision. Also: the waveform cache is lossless, the prefix differs from the released
frozen slice (the leak this removes), and the per-window cost.

    PYTHONPATH=$OFF/src python scripts/echog_stream/test_train_prefix.py --init I [--config Y] --data-root R --wav-dir W \
        --audio-model M --profile g1|ex002 [--split val] [--clips 6]
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'echog_g1'))
import librosa
from echo_g.config import ExperimentConfig
from echo_g.data import load_motion_stats
from echog_slide import PrefixAudio, resample_source_fps
from stream_core import StreamConfig, window_conditions
import train_stream as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', type=Path, required=True); ap.add_argument('--config', type=Path)
    ap.add_argument('--data-root', type=Path, required=True); ap.add_argument('--wav-dir', type=Path, required=True)
    ap.add_argument('--audio-model', type=Path, required=True); ap.add_argument('--profile', default='g1')
    ap.add_argument('--split', default='val'); ap.add_argument('--clips', type=int, default=6); ap.add_argument('--batch', type=int, default=16)
    a = ap.parse_args()
    dev = torch.device('cuda'); ok = True
    init = torch.load(a.init, map_location='cpu', weights_only=False)
    config = ExperimentConfig.from_yaml(a.config) if a.config else ExperimentConfig.from_dict(init['experiment']['config'])
    fps = float(config.data.fps); sc = StreamConfig.from_dict({'p_ahead_more': 0.5})
    mean, std = load_motion_stats(a.data_root / config.data.stats_file, config.model.motion_dim)
    ds = T.StreamDataset(a.data_root, config.data, config.model, a.split, mean, std, a.clips)
    p0 = torch.load(ds.condition_root / f'{ds.stems[0]}.pt', map_location='cpu', weights_only=True); src_fps = resample_source_fps(p0)
    wavs = T.WavStore(a.wav_dir); tpa = T.TrainPrefixAudio(a.audio_model, dev, fps, src_fps, 'exact')
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True   # training flags; apply() must reset them

    # 1 waveform cache lossless vs stream_infer's librosa.load
    for st in ds.stems:
        ref = librosa.load(wavs.path(st), sr=16_000, mono=True)[0]
        got = wavs(st)
        if not (got.dtype == ref.dtype and np.array_equal(got, ref)): ok = False; print('WAV_MISMATCH', st)
    print('cache dtypes', sorted({str(wavs.cache[s].dtype) for s in ds.stems}))

    # 2 training batches vs the inference path
    batches = T.fixed_val_batches(ds, sc, fps, a.batch, len(ds), seed=7, wavs=wavs)
    frozen = [x['audio'].clone() for x in batches]
    pa = PrefixAudio(a.audio_model, dev); n = 0; nd = 0; worst = 0.0; rel = []; t_apply = 0.0
    for x, fz in zip(batches, frozen):
        ta = time.perf_counter(); tpa.apply(x, x['audio'].device); torch.cuda.synchronize(); t_apply += time.perf_counter() - ta
        if torch.backends.cuda.matmul.allow_tf32 is not True or torch.backends.cudnn.allow_tf32 is not True: ok = False; print('TF32_FLAGS_NOT_RESTORED')
        for i, (st, m) in enumerate(zip(x['stems'], x['meta'])):
            j = ds.stems.index(st); s = ds[j]
            full = librosa.load(wavs.path(st), sr=16_000, mono=True)[0]
            m32, c32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True       # stream_infer process defaults
            feats = pa(full, m['now'] + m['ahead'], fps, src_fps).to(dev)
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = m32, c32
            ref = window_conditions(feats, s['text'].to(dev), s['token_times'], s['token_centers'], m['ws'], m['fe'], m['now'], m['ahead'],
                                    fps, 'voiced', sc.x2_rate, sc.x2_lag)['audio'].cpu()
            W = m['fe'] - m['ws']; got = x['audio'][i, :W]
            if not torch.equal(got, ref) or x['audio'][i, W:].abs().sum() != 0:
                nd += 1; worst = max(worst, float((got - ref).abs().max()))
            a_f = m['audio_frames']
            if a_f > 0:
                d = (got[:a_f] - fz[i, :a_f]).norm() / fz[i, :a_f].norm().clamp_min(1e-9); rel.append(float(d))
            n += 1
    if nd: ok = False
    print(json.dumps(dict(windows=n, not_bit_identical=nd, worst_abs=worst, src_fps=src_fps,
                          prefix_vs_frozen_rel_l2_median=float(np.median(rel)) if rel else None,
                          prefix_vs_frozen_rel_l2_p90=float(np.quantile(rel, 0.9)) if rel else None,
                          apply_s_per_window=round(t_apply / max(1, n), 4))))
    # 3 batched modes: close to exact (not bit-identical), far closer than the frozen slices
    for mode, tol in (('batch32', 5e-3), ('batch16', 3e-2)):
        tb = T.TrainPrefixAudio.__new__(T.TrainPrefixAudio); tb.pa, tb.fps, tb.src_fps, tb.mode = tpa.pa, fps, src_fps, mode
        bb = T.fixed_val_batches(ds, sc, fps, a.batch, len(ds), seed=7, wavs=wavs); ex = T.fixed_val_batches(ds, sc, fps, a.batch, len(ds), seed=7, wavs=wavs)
        r = []; tt = 0.0
        for x, y in zip(bb, ex):
            ta = time.perf_counter(); tb.apply(x, x['audio'].device); torch.cuda.synchronize(); tt += time.perf_counter() - ta; tpa.apply(y, y['audio'].device)
            for i, m in enumerate(x['meta']):
                k = m['audio_frames']
                if k > 0: r.append(float((x['audio'][i, :k] - y['audio'][i, :k]).norm() / y['audio'][i, :k].norm().clamp_min(1e-9)))
                if m['audio_frames'] != y['meta'][i]['audio_frames'] or x['audio'][i, k:].abs().sum() != 0: ok = False; print(mode, 'ROW_RANGE_MISMATCH')
        print(json.dumps(dict(mode=mode, rel_l2_vs_exact_max=max(r), median=float(np.median(r)), s_per_window=round(tt / max(1, n), 4))))
        if max(r) > tol: ok = False
    print('ALL_OK' if ok else 'FAILED')


if __name__ == '__main__':
    main()

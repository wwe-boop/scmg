"""Replay recorded TTS streams (scripts/tts_record.py) through the deployment worker scmg/deploy_stream.py, per model.

  PYTHONPATH=<scmg> USE_TF=0 CUDA_VISIBLE_DEVICES=0 <.venv-asr-hf python> scripts/tts_deploy_replay.py <tts dir> <out dir> \
      <label>=<ckpt>[@<original ckpt>] [...]
(<ckpt> may be a local byte copy / EMA-only strip of <original ckpt>; both paths and the loaded file's sha256 are recorded.)
<tts dir>/<id>/{stream.jsonl,text.txt}. The worker's own classes are used unchanged (deploy_stream.Model / E5 / Utt): only
the module constant CKPT is swapped per model. The worker is single-threaded and processes its NDJSON input strictly in
order, so feeding the journaled rows (speak -> audio/anchor in TTS arrival order -> audio_done) reproduces the online
output exactly, independent of wall-clock timing. Config = deploy_stream: CFG 2.5, 10 steps, H=2, trim_start,
cfg_null_fix, text x2 (15 tok/s, lag .15) + deploy_timing, causal e5, speaker null, source 5, Streamer seed 0.
Writes <out>/<label>/<id>.npy (T, 34) rad 50 Hz (deploy_stream.NAMES34 order), <out>/<id>.wav (16 kHz, the exact
decimation the worker uses), <out>/REPLAY.json (frames, tick ms, ckpt sha256, config).
"""
import json, os, sys, wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scmg import deploy_stream as DS  # noqa: E402

sys.path.insert(0, str(DS.FEAT89))


def main():
    import torch
    tts, out = Path(sys.argv[1]), Path(sys.argv[2])
    models = [a.split('=', 1) for a in sys.argv[3:]]
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get('SCMG_TORCH_THREADS', '4')))
    ids = sorted(p.name for p in tts.iterdir() if (p / 'stream.jsonl').exists())
    rec = dict(tts_dir=str(tts), sentences={}, models={})
    for i in ids:  # 16 kHz copy for the video (same decimation + s16 rounding as Utt._feat)
        rows = [json.loads(l) for l in open(tts / i / 'stream.jsonl')]
        x24 = np.concatenate([np.frombuffer(__import__('base64').b64decode(r['pcm']), '<f4') for r in rows if r['type'] == 'audio'])
        x16 = np.clip(np.round(DS._resample_2_3(x24.astype(np.float32)) * 32768.), -32768, 32767).astype(np.int16)
        with wave.open(str(out / f'{i}.wav'), 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(x16.tobytes())
        rec['sentences'][i] = dict(text=(tts / i / 'text.txt').read_text().strip(), samples24=int(len(x24)),
                                   seconds=round(len(x24) / 24000, 3), frames_audio=int(len(x24) // DS.FRAME24))
    e5 = DS.E5()
    for lab, ck in models:
        ck, orig = (ck.split('@', 1) + [None])[:2] if '@' in ck else (ck, ck)
        DS.CKPT = Path(ck)
        model = DS.Model(); model.warm(); torch.cuda.synchronize()
        rec['models'][lab] = dict(ckpt=orig, loaded_file=ck, loaded_sha256=model.sha, vae=str(DS.VAE), text_model=str(DS.TEXT_MODEL),
                                  cfg=model.cfg, sentences={})
        (out / lab).mkdir(exist_ok=True)
        for i in ids:
            got = []
            DS.emit = lambda row, got=got: got.append(row)
            text = rec['sentences'][i]['text']
            u = DS.Utt('u-' + i, text, model, e5)
            for r in (json.loads(l) for l in open(tts / i / 'stream.jsonl')):
                if r['type'] == 'audio':
                    u.add_audio(r['pcm'], int(r['start']), int(r['end'])); u._tick()
                elif r['type'] == 'anchor':
                    u.add_anchor(r['anchor']); u._tick()
                elif r['type'] == 'audio_done':
                    u.finish()
            err = [g for g in got if g.get('type') == 'error']
            assert not err, err
            fr = [g for g in got if g.get('type') == 'frames']
            assert [g['start'] for g in fr] == list(np.cumsum([0] + [len(g['q']) for g in fr[:-1]])), 'frame gap'
            q = np.concatenate([np.asarray(g['q'], np.float32) for g in fr])
            done = [g for g in got if g.get('type') == 'done'][0]
            np.save(out / lab / f'{i}.npy', q)
            rec['models'][lab]['sentences'][i] = dict(frames=int(len(q)), ticks=done['ticks'], tick_ms_p50=done['tick_ms_p50'],
                                                      tick_ms_p95=done['tick_ms_p95'], finite=bool(np.isfinite(q).all()))
            print(lab, i, len(q), 'frames', f"tick p50 {done['tick_ms_p50']:.0f} ms", flush=True)
        del model; torch.cuda.empty_cache()
    rec['names34'] = DS.NAMES34
    (out / 'REPLAY.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

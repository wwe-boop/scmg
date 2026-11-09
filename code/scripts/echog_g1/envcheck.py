"""Interpreter-equivalence check of the per-stem features (build_view.py extract): feat89 [T50, 89] and CAM++ vector from
the same audio + words in THIS interpreter -> <out>/<stem>.npz. Run once with the pipeline env (u3a-venv-asr-hf on DSW;
CAM++ with the speaker-style env) and once locally, then compare.
    envcheck.py <ds dir> <campplus dir|-> <out dir> <stem:n30> ...   (n30 = official aligned_frames)"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/unified-train-A-20261025')
sys.path.insert(1, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/speaker-style-emb-20261030/code')
ds, mdl, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
import soundfile as sf
if mdl != '-':
    from backend import SpeakerEncoder, read_audio
    from audio_window import speech_window
    enc = SpeakerEncoder('campplus', mdl, device='cpu')
else:
    import feat89 as F
for arg in sys.argv[4:]:
    stem, n30 = arg.split(':'); T = int(round(int(n30) * 5 / 3))
    res = {}
    if mdl == '-':
        x, _ = sf.read(str(ds / 'audio' / f'{stem}.wav'), dtype='float32')
        pcm = np.clip(np.round(x * 32768.), -32768, 32767).astype(np.int16)[:T * 320]
        pcm = np.concatenate([pcm, np.zeros(T * 320 - len(pcm), np.int16)])
        W = json.loads((ds / 'annotations/words' / f'{stem}.json').read_text())
        words = [dict(text=str(w['text']).strip(), start_time=float(w['start']), end_time=float(w['end'])) for w in W['words']]
        rec = F.Recording(None, None, 'en' if stem.startswith('english') else 'zh', T, words=[w for w in words if w['text']], pcm_fs=(pcm, 16000))
        res['feat'] = rec.segment(0, T)['feat']
    else:
        w = read_audio(dict(path=str(ds / 'audio' / f'{stem}.wav'))); chunk, _ = speech_window(w, 10.)
        res['spk'] = enc.encode([chunk])[0]
    np.savez(out / f'{stem}.npz', **res); print(stem, {k: v.shape for k, v in res.items()}, flush=True)

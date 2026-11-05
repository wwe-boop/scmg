"""Interruption-causal audio conditions (11-06): feat89 and the CAM++ speaker vector recomputed from the audio BEFORE t_i.

The view's feat89 of a clip is extracted from the whole utterance (prosody z-scored over the clip, beats picked over all
units) and its CAM++ vector from the whole clip audio, so even the frames before t_i carry information about the speech
after t_i. For every interrupt this writes the conditions a deployed system has at t_i:
  feat  [round(50 t_i), 89] = feat89.Recording(words that ENDED by t_i, pcm[0 : t_i]).segment(...) (same code path as
        scripts/echog_g1/build_view.py extract; no complete word -> logmel + zero prosody, as there)
  spk   CAM++ (speaker-style-emb-20261030, same crop rule speech_window(10 s)) of pcm[0 : t_i]; spk_valid 0 if < 0.5 s speech
infer_retract.py --prefix-feats loads them at the interruption (post-interruption ticks then never see audio after t_i).
    <h3 python> scripts/interrupt/prefix_feats.py --interrupts I.json --data <echog ds> --out D [--workers 8]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

U = Path('/mnt/cpfs/zbl-cpfs-new/USERS/zehan')
FEAT89_DIR = U / 'unified-train-A-20261025'; SPK_CODE = U / 'speaker-style-emb-20261030/code'
SPK_MODEL_SHA = 'a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73'
_W = {}


def _init(model_path):
    sys.path.insert(0, str(FEAT89_DIR)); sys.path.insert(1, str(SPK_CODE))
    import feat89 as F
    import soundfile as sf
    from backend import SpeakerEncoder, read_audio
    from audio_window import speech_window
    enc = SpeakerEncoder('campplus', model_path, device='cpu')
    if enc.weight_sha != SPK_MODEL_SHA: raise ValueError('CAM++ weight SHA')
    _W.update(F=F, sf=sf, enc=enc, read_audio=read_audio, speech_window=speech_window)


def one(job):
    stem, frac, t_i, data, out = job
    o = Path(out) / f'{stem}@{frac:.1f}.npz'
    if o.exists(): return dict(stem=stem, frac=frac, cached=True)
    F, sf = _W['F'], _W['sf']; data = Path(data)
    T = int(round(t_i * 50)); ap = data / 'audio' / f'{stem}.wav'
    x, sr = sf.read(str(ap), dtype='float32')
    if sr != 16000: raise ValueError(f'{stem}: {sr} Hz')
    pcm = np.clip(np.round(x * 32768.), -32768, 32767).astype(np.int16)[:T * 320]
    if len(pcm) < T * 320: pcm = np.concatenate([pcm, np.zeros(T * 320 - len(pcm), np.int16)])
    W = json.loads((data / 'annotations/words' / f'{stem}.json').read_text())
    words = [dict(text=str(w['text']).strip(), start_time=float(w['start']), end_time=float(w['end'])) for w in W['words']]
    words = [w for w in words if w['text'] and w['end_time'] >= w['start_time'] and w['end_time'] <= t_i + 1e-6]   # spoken by t_i
    lang = {'english': 'en', 'chinese': 'zh'}[stem.split('_', 1)[0]]
    feat, err = None, None
    if words:
        try: feat = F.Recording(None, None, lang, T, words=words, pcm_fs=(pcm, 16000)).segment(0, T)['feat']
        except (ValueError, IndexError) as e: err = f'{type(e).__name__}: {e}'
    if feat is None:
        from scipy.signal import resample_poly
        logmel = F.gl_build.logmel(resample_poly(pcm.astype(np.float32) / 32768, 3, 2), T, F.mel())
        feat = np.concatenate([logmel, np.zeros((T, 9), np.float32)], 1).astype(np.float16)
    spk, spk_err = np.zeros(192, np.float32), None
    try:
        w = _W['read_audio'](dict(path=str(ap), duration_s=t_i))
        chunk, _ = _W['speech_window'](w, 10.)
        if len(chunk) < 8000: raise ValueError('less than 0.5s audio')
        spk = np.asarray(_W['enc'].encode([chunk])[0], np.float32)
    except Exception as e:
        spk_err = f'{type(e).__name__}: {e}'
    np.savez(o, feat=feat.astype(np.float16), spk=spk, spk_valid=np.float32(spk_err is None))
    return dict(stem=stem, frac=frac, T=T, words=len(words), feat_error=err, spk_error=spk_err)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--interrupts', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--campplus', default='/home/xr-25b7/bulk/models-ro/campplus')
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jobs = [(r['stem'], r['frac'], r['t_i'], a.data, a.out) for r in json.loads(Path(a.interrupts).read_text())['interrupts']]
    import multiprocessing as mp
    errs = []
    with mp.get_context('spawn').Pool(a.workers, initializer=_init, initargs=(a.campplus,)) as pool:
        for k, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            if r.get('feat_error') or r.get('spk_error'): errs.append(r)
            if k % 300 == 0: print(k, len(jobs), flush=True)
    Path(a.out, 'ERRORS.json').write_text(json.dumps(errs, indent=1))
    print('PREFIX_DONE', len(jobs), 'with errors', len(errs))

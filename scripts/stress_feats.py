"""Stress-test feature packs (11-04 Claude; user: stress-test our baseline and external baselines).
    stress_feats.py <out dir> <n_clips_per_lang=6> [min_seconds=30]
For dev clips (val split of data-v36_1, BEAT2 zh 13_lu + en 2_scott) builds perturbed audio and recomputes the 89-d
features with the pinned training extractor (feat89, clip-level prosody scope -- NOT the recording scope of the view, so
every condition is compared with `clean`, recomputed the same way, never with view features).  Conditions:
  clean      original audio
  noise10    + white noise at 10 dB SNR (seeded)
  silence3   3.0 s of silence inserted at the inter-word gap closest to the clip middle (later words shifted)
  rate080    time-stretched to 0.8x speed (slower; librosa phase vocoder, pitch kept), all times x 1.25
  rate125    1.25x speed, all times x 0.8
Per clip/condition: <clip>__<cond>.npz with feat [T,89] float32, frames, knots_old/knots_new (piecewise-linear map of
window seconds old->new, for text-token times), and <clip>__<cond>.wav (16 kHz) for external baselines; INDEX.json."""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf, librosa
sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/unified-train-A-20261025')
import feat89 as F
V = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1')
PARTS = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/parts')
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
NPL = int(sys.argv[2]) if len(sys.argv) > 2 else 6; MINS = float(sys.argv[3]) if len(sys.argv) > 3 else 30.
items = json.loads((V / 'train.json').read_text()); keep = set(json.loads((V / 'SPLIT.json').read_text())['val_ids'])
clips = {}
for it in items:
    if it['id'] in keep: clips.setdefault(it['clip'], []).append(it)
order = []                                   # same clip index as data.View(split='val') (insertion order of train.json)
for c, xs in clips.items(): xs.sort(key=lambda x: int(x['offset'])); order.append((c, xs))
audio = np.load(Path(str(V) + '-audio16') / 'AUDIO16.npy', mmap_mode='r')
pick = {'zh': [], 'en': []}
for ci, (c, xs) in enumerate(order):
    fr = sum(int(x['frames']) for x in xs); lang = xs[0]['lang']
    if fr / 50 >= MINS and len(pick[lang]) < NPL: pick[lang].append((ci, c, xs, fr))


def chars_of(xs, start):
    plain, chars, idx = '', [], 0
    for x in xs:
        al = json.load(open(PARTS / x['part'].split('/')[-1] / x['dir'] / 'alignment.json')) if x.get('dir') else []
        t0 = (int(x['offset']) - start) / 50.
        sep = ' ' if (x['lang'] == 'en' and plain) else ''
        base = len(plain) + len(sep); plain += sep + x['text']
        for a in al:
            chars.append(dict(index=base + int(a['index']), char=a['char'], onset_s=t0 + a['onset_s'], offset_s=t0 + a['offset_s']))
    return plain, chars


def feats(pcm, chars, plain, lang, frames):
    rec = F.Recording.from_clip(pcm.astype(np.float32), 16000, chars, plain, lang, frames)
    return np.asarray(rec.segment(0, frames)['feat'], np.float32)


index = []
for lang in ('zh', 'en'):
    for ci, c, xs, fr in pick[lang]:
        start = int(xs[0]['offset']); pcm = np.asarray(audio[start * 320:(start + fr) * 320], np.float32) / 32768.
        plain, chars = chars_of(xs, start); T = fr / 50.
        conds = {}
        conds['clean'] = (pcm, chars, [0., T], [0., T])
        rng = np.random.default_rng(ci); p = np.mean(pcm ** 2) + 1e-12
        conds['noise10'] = (pcm + rng.normal(0, np.sqrt(p / 10 ** (10 / 10)), pcm.shape).astype(np.float32), chars, [0., T], [0., T])
        mids = [(chars[i]['offset_s'] + chars[i + 1]['onset_s']) / 2 for i in range(len(chars) - 1)] or [T / 2]
        tg = min(mids, key=lambda m: abs(m - T / 2)); g = int(tg * 16000)
        pcm_s = np.concatenate([pcm[:g], np.zeros(3 * 16000, np.float32), pcm[g:]])
        ch_s = [dict(cc, onset_s=cc['onset_s'] + (3. if cc['onset_s'] > tg else 0.), offset_s=cc['offset_s'] + (3. if cc['onset_s'] > tg else 0.)) for cc in chars]
        conds['silence3'] = (pcm_s, ch_s, [0., tg, tg, T], [0., tg, tg + 3., T + 3.])
        for name, r in (('rate080', 0.8), ('rate125', 1.25)):
            st = librosa.effects.time_stretch(pcm, rate=r).astype(np.float32)
            conds[name] = (st, [dict(cc, onset_s=cc['onset_s'] / r, offset_s=cc['offset_s'] / r) for cc in chars], [0., T], [0., T / r])
        for cond, (x, ch, ko, kn) in conds.items():
            frames = int(len(x) / 320) // 16 * 16
            f = feats(x[:frames * 320], ch, plain, lang, frames)
            name = f'{c}__{cond}'
            np.savez(out / f'{name}.npz', feat=f, frames=frames, knots_old=np.array(ko, np.float32), knots_new=np.array(kn, np.float32))
            sf.write(out / f'{name}.wav', x[:frames * 320], 16000, subtype='PCM_16')
            index.append(dict(ci=ci, clip=c, lang=lang, cond=cond, frames=frames, seconds=frames / 50, text=plain[:200], npz=f'{name}.npz', wav=f'{name}.wav'))
            print(name, frames, flush=True)
(out / 'INDEX.json').write_text(json.dumps(index, ensure_ascii=False, indent=1))
print('STRESS_FEATS_DONE', len(index))

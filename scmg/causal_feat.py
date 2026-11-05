"""Causal (prefix) streaming inputs, design v5 section 3 (11-05): block k may only use the audio before
t_k = k * 320 ms + H (H = 160 ms) and the words completed by then.

  feat89   CausalFeat89: the feat89 of the ARRIVED PCM prefix pcm[:t_k] with only the words whose end <= t_k, through the
           same extractor and word path as the G1 view (scripts/echog_g1/build_view.py extract_one: feat89.Recording(words=..,
           pcm_fs=..).segment(0, T); no complete word -> build_view.null_features) and the v2 prosody hooks of
           scripts/feat_causality.py (prominence z-scored with FIXED global statistics of the training split,
           prosody_stats(); committed beats, no global rate cap; `final` = the whole clip has arrived).  The speech flag
           (column 88, first unit onset .. last unit offset) then only spans completed words.
  speaker  SpeakerPrefix: CAM++ (build_view: backend.SpeakerEncoder campplus onnx, SHA-pinned; audio_window.speech_window 10 s
           crop; kaldi fbank; L2) of the arrived prefix; prefix < 1 s (or no energetic speech) -> the EMPTY embedding of
           training: spk = 0, spk_valid = 0 (SCMG.global_cond then uses the learned spk_null, the drop_spk / invalid-speaker
           case of training).  Embeddings are cached by speech-window crop (a prefix > 10 s often keeps the same crop).
  text     visible_words(): the ECHO-G streaming rule (scmg-wt-echog-stream scripts/echog_stream/stream_core.py
           visible_tokens, copied verbatim for 'x2' / 'voiced'; tested against it) on the official word list, plus 'asr'
           (voiced + 0.5 s recognition lag); token_word_span() maps the e5 token rows of the SCMG text cache to words
           (a token is visible iff every word it overlaps is).

Nothing here is used unless the Streamer gets causal_inputs / text_mode voiced|asr (scmg/infer.py); the default paths are
unchanged.  CLI:
    python -m scmg.causal_feat pstats --view V --split train --audio-dir D/audio --words-dir D/annotations/words --out J [--n 400]
"""
import argparse, contextlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMPPLUS = '/home/xr-25b7/bulk/models-ro/campplus'
FPS, SR, HOP = 50, 16000, 320
_MODS = {}


def _load(name, path):
    if name not in _MODS:
        spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod); _MODS[name] = mod
    return _MODS[name]


def build_view():
    """scripts/echog_g1/build_view.py (constants, null_features: the feature path of the G1 view)."""
    return _load('scmg_build_view', ROOT / 'scripts/echog_g1/build_view.py')


def F89():
    bv = build_view()
    if str(bv.FEAT89_DIR) not in sys.path: sys.path.insert(0, str(bv.FEAT89_DIR))
    import feat89
    return feat89


def fc():
    """scripts/feat_causality.py (v2 hooks fixed_features / committed_beats); imports the same feat89 module object."""
    F89()
    return _load('scmg_feat_causality', ROOT / 'scripts/feat_causality.py')


# ------------------------------------------------------------------ clip inputs exactly as build_view.extract_one
def clip_words(words_json, T):
    """Word list of the view: stripped, finite, end >= start; end clamped to T/50, words starting after it dropped."""
    W = json.loads(Path(words_json).read_text())
    words = [dict(text=str(w['text']).strip(), start_time=float(w['start']), end_time=float(w['end'])) for w in W['words']]
    words = [w for w in words if w['text'] and np.isfinite(w['start_time']) and np.isfinite(w['end_time']) and w['end_time'] >= w['start_time']]
    dur = T / 50.
    return [dict(w, end_time=min(w['end_time'], dur)) for w in words if w['start_time'] < dur - 1e-3]


def clip_audio(wav):
    """(pcm16 of the whole file, float32 wave as backend.read_audio) at 16 kHz mono."""
    import soundfile as sf
    x, sr = sf.read(str(wav), dtype='float32', always_2d=True)
    if sr != SR or x.shape[1] != 1: raise ValueError(f'{wav}: {sr} Hz x {x.shape[1]}')
    x = x[:, 0]
    return np.clip(np.round(x * 32768.), -32768, 32767).astype(np.int16), x


def pad_pcm(pcm16, T):
    p = pcm16[:T * HOP]
    return p if len(p) == T * HOP else np.concatenate([p, np.zeros(T * HOP - len(p), np.int16)])


@contextlib.contextmanager
def _hooks(pstats, final):
    """v2 prosody of scripts/feat_causality.py: fixed prominence statistics + committed beats (monkey-patched like its extract)."""
    F, m = F89(), fc()
    old = F.features, F.pick_beats
    F.features = m.fixed_features(pstats); F.pick_beats = m.committed_beats(final)
    try: yield
    finally: F.features, F.pick_beats = old


def _feat(pcm16, words, lang, T):
    """build_view.extract_one's feature step for frames [0, T) (pcm16 already [T*320])."""
    F = F89()
    if words:
        try:
            return np.asarray(F.Recording(None, None, lang, T, words=words, pcm_fs=(pcm16, SR)).segment(0, T)['feat'])
        except ValueError:
            pass
    return build_view().null_features(F, pcm16, T)


def full_feat89(pcm16, words, lang, T):
    """The view's whole-clip feat89 (prosody z-scored over the clip, pick_beats with the global rate cap) [T, 89] f16."""
    return _feat(pad_pcm(pcm16, T), words, lang, T)


def prefix_feat89(pcm16, words, lang, T, Tk, pstats):
    """Causal v2 feat89 of the first Tk frames: pcm[:Tk*320] and the words with end <= Tk/50 only. [Tk, 89] f16."""
    Tk = int(min(Tk, T)); final = Tk >= T
    use = words if final else [w for w in words if w['end_time'] <= Tk / 50.]
    with _hooks(pstats, final):
        return _feat(pad_pcm(pcm16, Tk), use, lang, Tk)


class CausalFeat89:
    def __init__(self, pcm16, words, lang, T, pstats):
        self.pcm, self.words, self.lang, self.T, self.ps = pcm16, words, lang, int(T), list(pstats)
        self.cache = {}

    def prefix(self, Tk):
        Tk = int(min(Tk, self.T))
        if Tk not in self.cache: self.cache[Tk] = prefix_feat89(self.pcm, self.words, self.lang, self.T, Tk, self.ps)
        return self.cache[Tk]


def load_pstats(path):
    d = json.loads(Path(path).read_text())
    return [d[k] for k in ('E_mean', 'E_std', 'logf0_mean', 'logf0_std', 'dur_mean', 'dur_std')]


# ------------------------------------------------------------------ speaker
def speaker_encoder(model_dir=CAMPPLUS):
    bv = build_view()
    if str(bv.SPK_CODE) not in sys.path: sys.path.insert(1, str(bv.SPK_CODE))
    from backend import SpeakerEncoder
    from audio_window import speech_window
    enc = SpeakerEncoder('campplus', model_dir, device='cpu')
    if enc.weight_sha != bv.SPK_MODEL_SHA: raise ValueError('CAM++ weight SHA')
    enc.speech_window = speech_window
    return enc


class SpeakerPrefix:
    """CAM++ of the arrived prefix wave[:n]; < min_s seconds or no speech crop -> (zeros, 0) = training's empty speaker."""

    def __init__(self, wave, enc, min_s=1.0, crop_s=10.):
        self.w, self.enc, self.min_n, self.crop_s = np.asarray(wave, np.float32), enc, int(round(min_s * SR)), crop_s
        self.cache = {}; self.n_encode = 0

    def at(self, n):
        x = self.w[:int(n)]
        if len(x) < self.min_n: return np.zeros(192, np.float32), 0.
        try:
            chunk, crop = self.enc.speech_window(x, self.crop_s)
        except ValueError:
            return np.zeros(192, np.float32), 0.
        key = (round(crop['start_s'] * SR), len(chunk))
        if key not in self.cache:
            try:
                self.cache[key] = (self.enc.encode([chunk])[0].astype(np.float32), 1.); self.n_encode += 1
            except ValueError:
                self.cache[key] = (np.zeros(192, np.float32), 0.)
        return self.cache[key]

    def full(self):
        """the view's embedding: whole file, speech_window 10 s, >= 0.5 s (build_view rule)."""
        chunk, _ = self.enc.speech_window(self.w, self.crop_s)
        if len(chunk) < 8000: raise ValueError('less than 0.5s audio')
        return self.enc.encode([chunk])[0].astype(np.float32)


class CausalInputs:
    """Per-block audio-side conditions for scmg.infer.Streamer(causal_inputs=...). T = clip frames, Tpad = Streamer feat rows
    (>= T, the chunk-padded window). feat(Tk) -> normalised [Tpad, 89] float32: rows [0, min(Tk, T)) = the prefix features,
    zeros after (the Streamer zeroes not-arrived rows anyway; rows >= T are zero as in the whole-clip window); spk(Tk) ->
    (spk [192], spk_valid) of the PCM prefix of min(Tk, T) frames (spk=False: always the empty embedding)."""

    def __init__(self, pcm16, wave, words, lang, T, pstats, fm, fs, enc=None, Tpad=None, spk_min_s=1.0, spk=True):
        self.T = int(T); self.Tpad = int(Tpad or T); self.f = CausalFeat89(pcm16, words, lang, T, pstats)
        if self.Tpad < self.T: raise ValueError('Tpad < T')
        self.fm, self.fs = np.asarray(fm, np.float32), np.asarray(fs, np.float32)
        self.sp = SpeakerPrefix(wave, enc, spk_min_s) if spk else None
        self.calls = []

    def feat(self, Tk):
        Tk = int(min(Tk, self.T)); out = np.zeros((self.Tpad, 89), np.float32)
        out[:Tk] = (np.asarray(self.f.prefix(Tk), np.float32) - self.fm) / self.fs
        self.calls.append(Tk)
        return out

    def spk(self, Tk):
        if self.sp is None: return np.zeros(192, np.float32), 0.
        return self.sp.at(int(min(Tk, self.T)) * HOP)


# ------------------------------------------------------------------ text visibility (word level)
def x2_arrival(start, rate, lag):
    """= scmg-wt-echog-stream scripts/echog_stream/stream_core.py x2_arrival (SCMG scmg.model.x2_arrival, one utterance)."""
    j = np.arange(len(start), dtype=np.float64)
    return np.minimum(start - lag, start.min() - lag + j / rate) if len(start) else start


def visible_words(token_times, now, ahead, mode, rate, lag, asr_lag=0.5):
    """stream_core.visible_tokens (modes final / x2 / voiced, verbatim) + 'asr': voiced with the word end + asr_lag."""
    if mode == 'final': return np.ones(len(token_times), bool)
    arr = x2_arrival(token_times[:, 0].astype(np.float64), rate, lag)
    vis = arr <= now + 1e-9
    if mode == 'voiced': vis &= token_times[:, 1].astype(np.float64) <= now + ahead + 1e-9
    elif mode == 'asr': vis &= token_times[:, 1].astype(np.float64) + asr_lag <= now + ahead + 1e-9
    elif mode != 'x2': raise ValueError(mode)
    return vis


def word_times(words):
    return np.asarray([[w['start_time'], w['end_time']] for w in words], np.float64).reshape(-1, 2)


def token_word_span(text, words, lang, tokenizer, n_tok, max_length=160):
    """[n_tok, 2] int: first / last word index overlapped by each e5 token row of the text cache (prep_text.py
    tokenisation: same call, special tokens removed). A token touching no word character (e.g. a lone '_') gets the
    next word (the previous one at the end)."""
    sep = ' ' if lang == 'en' else ''
    if sep.join(w['text'] for w in words) != text: raise ValueError('item text != joined view words')
    spans, pos = [], 0
    for w in words:
        spans.append((pos, pos + len(w['text']))); pos += len(w['text']) + len(sep)
    enc = tokenizer([text], truncation=True, max_length=max_length, return_offsets_mapping=True)
    ids, offs = enc['input_ids'][0], enc['offset_mapping'][0]
    sel = [j for j in range(len(ids)) if ids[j] not in tokenizer.all_special_ids]
    if len(sel) != n_tok: raise ValueError(f'{len(sel)} tokens != {n_tok} cache rows')
    out = np.zeros((n_tok, 2), np.int64)
    ws = np.array([s for s, _ in spans]); we = np.array([e for _, e in spans])
    for r, j in enumerate(sel):
        s, e = offs[j]
        hit = np.nonzero((ws < e) & (s < we))[0]
        if not len(hit):
            nxt = np.nonzero(ws >= s)[0]
            hit = nxt[:1] if len(nxt) else np.array([len(words) - 1])
        out[r] = hit.min(), hit.max()
    return out


def token_visible(word_vis, span):
    """token visible iff all words of its span are; span rows < 0 = not a token."""
    bad = np.concatenate([[0], np.cumsum(~word_vis)])
    ok = span[:, 0] >= 0
    s0, s1 = np.clip(span[:, 0], 0, None), np.clip(span[:, 1], 0, None)
    return ok & (bad[s1 + 1] - bad[s0] == 0) if len(word_vis) else np.zeros(len(span), bool)


def text_extras(item, words, tokenizer, n_tok, K):
    """cond extras of text_mode voiced / asr for a one-item clip whose n_tok cache rows fill tmask[:n_tok]:
    wtime [n_words, 2] (start, end s), tword [K, 2] (word span of each token row, -1 = no token)."""
    tw = np.full((K, 2), -1, np.int64)
    if n_tok and not item.get('text_null'): tw[:n_tok] = token_word_span(item['text'], words, item['lang'], tokenizer, n_tok)
    return dict(wtime=word_times(words).astype(np.float64), tword=tw)


def e5_tokenizer(path=None):
    from transformers import AutoTokenizer
    from .common import P
    return AutoTokenizer.from_pretrained(P(path or '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/models/scmg/multilingual-e5-base'))


# ------------------------------------------------------------------ fixed prosody statistics of the training split
def prosody_stats(view_dir, split, audio_dir, words_dir, n=400, seed=0):
    """Global syllable statistics (energy dB > -80, log F0 of voiced, duration > 0) over n random clips of the split, the
    same measurement as feat_causality.speaker_stats (gesturelab.prosody.measure via feat89.Recording)."""
    F, m = F89(), fc()
    v = Path(view_dir); items = json.loads((v / 'train.json').read_text()); sp = json.loads((v / 'SPLIT.json').read_text())
    keep = set(sp['train_ids'] if split == 'train' else sp['val_ids'])
    its = sorted([it for it in items if it['id'] in keep and not it.get('text_null')], key=lambda it: it['id'])
    pick = np.random.default_rng(seed).choice(len(its), size=min(n, len(its)), replace=False)
    E, Fv, Dd = [], [], []
    orig = F.features

    def rec(syllables, pcm, fs=24000, cache=None):
        pcm = np.asarray(pcm, np.float32)
        for s in syllables:
            e, f = m.F_measure(s, pcm, fs); E.append(e); Fv.append(f); Dd.append(s.offset_s - s.onset_s)
        return orig(syllables, pcm, fs)
    used = []; t0 = time.time()
    for i in sorted(pick):
        it = its[i]; T = int(it['frames'])
        words = clip_words(Path(words_dir) / f"{it['id']}.json", T)
        pcm16, _ = clip_audio(Path(audio_dir) / f"{it['id']}.wav")
        F.features = rec
        try: F.Recording(None, None, it['lang'], T, words=words, pcm_fs=(pad_pcm(pcm16, T), SR))
        except ValueError: continue
        finally: F.features = orig
        used.append(it['id'])
    E, Fv, Dd = map(np.array, (E, Fv, Dd))
    lf = np.log(Fv[Fv > 0]); Ev = E[E > -80]; Dv = Dd[Dd > 0]
    return dict(E_mean=float(Ev.mean()), E_std=float(Ev.std()), logf0_mean=float(lf.mean()), logf0_std=float(lf.std()),
                dur_mean=float(Dv.mean()), dur_std=float(Dv.std()), syllables=int(len(E)), clips=len(used), view=str(view_dir),
                split=split, n_request=n, seed=seed, rule='numpy default_rng(seed).choice over the split items sorted by id; '
                'E over E > -80 dB, log f0 over voiced, duration over > 0 (scripts/feat_causality.py speaker_stats measurement)',
                seconds=round(time.time() - t0, 1), clip_ids_sha=__import__('hashlib').sha256('\n'.join(used).encode()).hexdigest())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['pstats'])
    ap.add_argument('--view', required=True); ap.add_argument('--split', default='train')
    ap.add_argument('--audio-dir', required=True); ap.add_argument('--words-dir', required=True)
    ap.add_argument('--n', type=int, default=400); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    st = prosody_stats(a.view, a.split, a.audio_dir, a.words_dir, a.n, a.seed)
    Path(a.out).write_text(json.dumps(st, indent=1)); print(json.dumps(st))


if __name__ == '__main__':
    main()

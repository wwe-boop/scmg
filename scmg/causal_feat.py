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

  train    (11-05 "训练匹配") unit_table() / front_prosody(): the whole prosody block 80-88 (frame_prosody columns onset /
           prominence / energy / logf0 / voiced / committed beats + speech flag; fm_build.frame_prosody loaded from its
           SHA-pinned source file) of any stream time Tk from the per-unit table of the whole-clip v2 recording: only the
           units whose WORD has ended by Tk, beats re-picked over them, as prefix_feat89 sees them (mel 0-79 of the prefix
           equals the whole-clip mel row for row). scripts/tests/test_causal_train.py checks it against prefix_feat89 bitwise;
           scmg/data.py + scmg/losses.py (StageCfg causal_front / causal_spk / causal_text) use it.

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
# 11-06 word-ended predicates. Inference text (voiced / asr): end <= t + VOICED_EPS (1e-9, verbatim the ECHO-G stream_core
# rule; word times carry float noise such as 2.7200000000000006 for a word ending on the 2.72 s block boundary). Training
# causal_text compares window-relative float32 times: end_f32 <= t_f32 + WORD_EPS (0.1 ms) absorbs the float32 rounding; word
# times sit on a 10 ms grid, so the two decide every real word identically (all 400 D-spk clips x 4 window starts x 4 horizons
# x 16 ticks = 3.59M token-ticks: 0 disagreements; 821 without the slack). PROSODY (prefix_feat89 / front_prosody / IncFeat89)
# keeps feat89.Recording's own float64 filters (end <= Tk / 50, offset <= dur), the same arithmetic on both sides; a word ending
# within float noise after a block boundary is text-visible at that tick and enters the prosody one tick later, identically
# in training and inference. Tests perturb with the inference predicate (ended()).
VOICED_EPS, WORD_EPS = 1e-9, 1e-4


def ended(end_s, t_s):
    """the inference word-ended predicate of the text rules (numpy / float)"""
    return end_s <= t_s + VOICED_EPS
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
def _hooks(pstats, final, capture=None):
    """v2 prosody of scripts/feat_causality.py: fixed prominence statistics + committed beats (monkey-patched like its extract).
    capture: a list -> the syllables handed to the beat picker are appended (unrounded onset / offset / prominence)."""
    F, m = F89(), fc()
    old = F.features, F.pick_beats
    pick = m.committed_beats(final)
    F.features = m.fixed_features(pstats)
    F.pick_beats = pick if capture is None else (lambda syl, **kw: (capture.extend(syl), pick(syl, **kw))[1])
    try: yield
    finally: F.features, F.pick_beats = old


def _feat(pcm16, words, lang, T, units=False):
    """build_view.extract_one's feature step for frames [0, T) (pcm16 already [T*320]). units=True: (feat, the segment's
    prosody units (Recording.segment()['units'] with char indices back in the recording text; [] on the no-unit
    fallback))."""
    F = F89()
    if words:
        try:
            rec = F.Recording(None, None, lang, T, words=words, pcm_fs=(pcm16, SR)); seg = rec.segment(0, T)
            if not units: return np.asarray(seg['feat'])
            i0 = rec.units_in(0, T)[0]['chars'][0]                     # segment() shifts char indices by the first unit's
            return np.asarray(seg['feat']), [dict(u, chars=[c + i0 for c in u['chars']]) for u in seg['units']]
        except ValueError:
            pass
    f = build_view().null_features(F, pcm16, T)
    return (f, []) if units else f


def full_feat89(pcm16, words, lang, T):
    """The view's whole-clip feat89 (prosody z-scored over the clip, pick_beats with the global rate cap) [T, 89] f16."""
    return _feat(pad_pcm(pcm16, T), words, lang, T)


def prefix_feat89(pcm16, words, lang, T, Tk, pstats, units=False):
    """Causal v2 feat89 of the first Tk frames: pcm[:Tk*320] and the words with end <= Tk/50 only. [Tk, 89] f16.
    units=True: (feat, segment units) -- the per-unit values the prosody columns are drawn from, each with the beat
    picker's syllable (syl_onset / syl_offset / syl_prom: unrounded, as gesturelab.prosody.Syllable)."""
    Tk = int(min(Tk, T)); final = Tk >= T
    use = words if final else [w for w in words if w['end_time'] <= Tk / 50.]
    cap = [] if units else None
    with _hooks(pstats, final, cap):
        out = _feat(pad_pcm(pcm16, Tk), use, lang, Tk, units=units)
    if not units: return out
    f, us = out; syl = {int(x.index): x for x in cap}
    for u in us:
        x = syl[int(u['index'])]; u.update(syl_onset=float(x.onset_s), syl_offset=float(x.offset_s), syl_prom=float(x.prominence))
    return f, us


# ------------------------------------------------------------------ training-time front prosody (训练匹配)
FM_BUILD = '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture/experiments/rsi/motion/fm_build.py'
FM_BUILD_SHA = 'be7540ab3ce4694e439e05af892482d0c1c5a01714e16edc690b60cb2dc8002b'   # = feat89.PINS (C043 build)
FRONT_COLS = tuple(range(80, 89))       # the prosody block of feat89: frame_prosody 0-7 (onset, prominence, energy, logf0,
                                        # voiced, beat strength, beat left, beat right) + the speech flag
UNIT_COLS = ('onset_s', 'offset_s', 'prominence', 'energy_db', 'f0_hz', 'word_end', 'syl_onset', 'syl_offset', 'syl_prom')


def frame_prosody():
    """fm_build.frame_prosody (the function feat89.Recording.segment uses), executed from its SHA-pinned source file
    alone (no gl_build / audio imports: the training data loader needs only numpy)."""
    if 'frame_prosody' not in _MODS:
        import ast, hashlib
        from .common import P
        src = Path(P(FM_BUILD)).read_bytes()
        if hashlib.sha256(src).hexdigest() != FM_BUILD_SHA: raise RuntimeError(f'{FM_BUILD} changed since the C043 build')
        tree = ast.parse(src.decode()); ns = dict(np=np)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) in ('PROSODY', 'FPS') for t in node.targets):
                exec(compile(ast.Module([node], []), FM_BUILD, 'exec'), ns)
            if isinstance(node, ast.FunctionDef) and node.name == 'frame_prosody':
                exec(compile(ast.Module([node], []), FM_BUILD, 'exec'), ns)
        _MODS['frame_prosody'] = ns['frame_prosody']
    return _MODS['frame_prosody']


class _Syl:
    """the attributes of gesturelab.prosody.Syllable the beat picker reads (nucleus_s = the Syllable property)."""
    __slots__ = ('onset_s', 'offset_s', 'prominence')

    def __init__(self, on, off, prom): self.onset_s, self.offset_s, self.prominence = on, off, prom

    @property
    def nucleus_s(self): return self.onset_s + min(.1, .4 * (self.offset_s - self.onset_s))


def committed_beats(final, threshold=.6, min_gap_s=.45):
    """= scripts/feat_causality.py committed_beats (verbatim; scripts/tests/test_causal_train.py checks the source)."""
    def pick(syllables, **kw):
        n = len(syllables); ncand = n if final else max(0, n - 2)
        beats = []
        for i in range(ncand):
            s = syllables[i]; nb = syllables[max(0, i - 2):i] + syllables[i + 1:i + 3]
            if s.prominence < threshold or any(x.prominence > s.prominence for x in nb): continue
            t = s.nucleus_s; st = float(np.clip((s.prominence - threshold) / 1.5 + .4, .3, 1.))
            if beats and t - beats[-1][0] < min_gap_s:
                if s.prominence > beats[-1][2]: beats[-1] = (t, st, s.prominence)
                continue
            beats.append((t, st, s.prominence))
        if not final and beats and ncand < n and syllables[ncand].onset_s - beats[-1][0] < min_gap_s:
            beats = beats[:-1]                                         # could still be replaced by an undecided candidate
        return [(t, s) for t, s, _ in beats]
    return pick


def assign_sides(beats, seed=0):
    """= feat89's human_stage2.assign_sides (verbatim; checked by scripts/tests/test_causal_train.py)."""
    rng = np.random.default_rng(seed); sides = rng.choice(['right', 'left', 'both'], size=len(beats), p=[.55, .3, .15]); out = []
    for (t, s), side in zip(beats, sides): out += [(t, s, 'left'), (t, s, 'right')] if side == 'both' else [(t, s, str(side))]
    return out


def unit_word(onsets, words):
    """index of the view word each unit belongs to, by TIME: the last word whose start <= unit onset (+1e-6). (11-06: the
    former char-index mapping drifted after words whose apostrophe / hyphen the aligner does not emit, e.g. "aren't",
    and gave a unit the end of a later word.)"""
    st = np.asarray([w['start_time'] for w in words], np.float64)
    return np.clip(np.searchsorted(st, np.asarray(onsets, np.float64) + 1e-6, side='right') - 1, 0, len(words) - 1)


def unit_table(units, words, lang):
    """[U, 9] float64 (UNIT_COLS) of the segment units of the whole-clip v2 recording (prefix_feat89(units=True), onset
    order): word_end = end_time of the view word the unit's first CHARACTER index falls in (char positions of the joined
    text). NOTE (11-06): this is deliberately the char-index mapping, not the unit's own word by time: feat89's aligner
    drops apostrophes / hyphens, so after such a word the char indices of later units point one word further -- and
    exactly as late as the prefix recordings admit those units (a prefix ending at the true word lacks them). With this
    mapping front_prosody reproduces prefix_feat89 at every tick (20 D-spk clips, 647 ticks, 0 mismatches); the
    by-time mapping (unit_word) would admit them one word too early (75 / 647 ticks differ)."""
    sep = 1 if lang == 'en' else 0
    starts, pos = [], 0
    for w in words:
        starts.append(pos); pos += len(w['text'].strip()) + sep
    starts = np.asarray(starts)
    out = np.zeros((len(units), len(UNIT_COLS)), np.float64)
    for i, u in enumerate(units):
        ci = int(u['chars'][0]); wi = int(np.searchsorted(starts, ci, side='right') - 1)
        out[i] = (u['onset_s'], u['offset_s'], u['prominence'], u['energy_db'], u['f0_hz'], words[wi]['end_time'],
                  u['syl_onset'], u['syl_offset'], u['syl_prom'])
    return out


def front_prosody(table, Tk, final=False, admitted=False):
    """Raw feat89 columns FRONT_COLS [Tk, 9] float32 of the causal prefix at Tk frames (= prefix_feat89(Tk)[:, 80:89],
    scripts/tests/test_causal_train.py checks it bitwise after the f16 cast), rebuilt from the whole-clip unit table:
    the prefix recording holds the units whose word ended by Tk / 50 (all of them when final, i.e. Tk >= clip frames)
    and whose offset <= Tk / 50 (feat89.Recording); its beats = committed_beats(final) over those syllables in onset
    order, then assign_sides(seed 0), kept when 0 <= t < Tk / 50 (Recording.segment); the segment units additionally
    need onset < Tk / 50 - 1e-6 (units_in); frame_prosody(Tk, units, beats) + the speech flag (first unit onset .. last
    unit offset); no unit -> build_view.null_features: zero prosody, beats and flag."""
    Tk = int(Tk); lim = Tk / 50.
    t = np.asarray(table, np.float64).reshape(-1, len(UNIT_COLS))
    rec = t[:, 1] <= lim
    if not (final or admitted): rec &= t[:, 5] <= lim             # admitted: the table holds only admitted units already
    t = t[rec]
    out = np.zeros((Tk, len(FRONT_COLS)), np.float32)
    seg = t[t[:, 0] < lim - 1e-6]
    if not len(seg): return out
    beats = assign_sides(committed_beats(final)([_Syl(r[6], r[7], r[8]) for r in t]), seed=0)
    bl = [dict(time_s=round(b, 4), strength=st, side=side) for b, st, side in beats if 0 <= b < lim]
    us = [dict(onset_s=float(r[0]), offset_s=float(r[1]), prominence=float(r[2]), energy_db=float(r[3]), f0_hz=float(r[4])) for r in seg]
    out[:, :8] = frame_prosody()(Tk, us, bl)[:, :8]
    ar = np.arange(Tk) / 50
    out[:, 8] = ((ar >= us[0]['onset_s']) & (ar <= us[-1]['offset_s'])).astype(np.float32)
    return out


def stream_lat_grid(K, horizons, chunk, u80):
    """Sorted distinct stream times in latents (k * chunk + h * u80, k < K, h in horizons) of a training window: the
    rows of WindowDataset's cz_pro table (scmg/losses.py looks a chunk's time up in the same list)."""
    return sorted({k * chunk + int(h) * u80 for k in range(K) for h in horizons})


class CausalFeat89:
    def __init__(self, pcm16, words, lang, T, pstats):
        self.pcm, self.words, self.lang, self.T, self.ps = pcm16, words, lang, int(T), list(pstats)
        self.cache = {}

    def prefix(self, Tk):
        Tk = int(min(Tk, self.T))
        if Tk not in self.cache: self.cache[Tk] = prefix_feat89(self.pcm, self.words, self.lang, self.T, Tk, self.ps)
        return self.cache[Tk]


class IncFeat89:
    """Incremental prefix feat89 (11-06; the same values as prefix_feat89, scripts/tests/test_causal_inc.py):
      mel     only the rows not final yet: logmel of the PCM window [r0 - MARGIN, Tk); the last EDGE rows (computed with the
              zero padding past Tk) are recomputed next tick. Equal to the prefix mel up to rare 1-ulp f16 flips (the
              whole-clip view mel the training reads differs from the prefix mel the same way)
      prosody recomputed only when the set of admitted words changes (end <= Tk / 50, all once final): feat89.Recording over
              the admitted words (the prefix_feat89 call, so its alignment / filters are identical) with a MEMOISED
              syllable measurement (fixed statistics: a syllable's energy / f0 / prominence depend only on its own audio),
              then front_prosody of that recording's units (= prefix_feat89's columns 80-88, the bitwise identity of
              test_causal_train b); ticks without a new word reuse it.
    Only pcm[:Tk] and the words ended by Tk are read."""
    MARGIN, EDGE = 25, 2

    def __init__(self, pcm16, words, lang, T, pstats):
        self.pcm, self.words, self.lang, self.T, self.ps = pcm16, words, lang, int(T), list(pstats)
        self.mel = np.zeros((0, 80), np.float32); self.mel_ok = 0
        self.key = None; self.units = []; self.memo = {}
        self.cache = {}

    def _mel(self, Tk):
        from scipy.signal import resample_poly
        F = F89(); a = max(0, self.mel_ok - self.MARGIN)
        p = pad_pcm(self.pcm, Tk)[a * HOP:Tk * HOP]
        m = F.gl_build.logmel(resample_poly(p.astype(np.float32) / 32768., F.gl_build.FS, SR), Tk - a, F.mel())
        out = np.zeros((Tk, 80), np.float32); out[:self.mel_ok] = self.mel[:self.mel_ok]; out[self.mel_ok:] = m[self.mel_ok - a:]
        self.mel = out; self.mel_ok = max(self.mel_ok, Tk - self.EDGE)
        return out

    def _measure(self):
        """fixed_features with a per-syllable memo (key: onset, offset, char); the measurement reads only the syllable."""
        base = fc().fixed_features(self.ps); F = F89(); memo = self.memo

        def features(syllables, pcm, fs=24000, cache=None):
            todo = [s for s in syllables if (s.onset_s, s.offset_s, s.char) not in memo]
            for s, r in zip(todo, base(todo, pcm, fs) if todo else []): memo[(s.onset_s, s.offset_s, s.char)] = r
            return [F.Syllable(s.index, s.char, s.onset_s, s.offset_s, *(lambda r: (r.energy_db, r.f0_hz, r.prominence))(memo[(s.onset_s, s.offset_s, s.char)]))
                    for s in syllables]
        return features

    def _prosody(self, Tk, final):
        use = self.words if final else [w for w in self.words if w['end_time'] <= Tk / 50.]
        key = (len(use), final)
        if key != self.key:
            self.key = key; self.units = []
            if use:
                F, m = F89(), fc(); old = F.features, F.pick_beats; cap = []
                pick = m.committed_beats(final)
                F.features = self._measure(); F.pick_beats = lambda syl, **kw: (cap.extend(syl), pick(syl, **kw))[1]
                try:
                    rec = F.Recording(None, None, self.lang, Tk, words=use, pcm_fs=(pad_pcm(self.pcm, Tk), SR))
                    syl = {int(x.index): x for x in cap}
                    self.units = [(round(u['onset_s'], 4), round(u['offset_s'], 4), u['prominence'], u['energy_db'], u['f0_hz'], 0.,
                                   float(syl[int(u['index'])].onset_s), float(syl[int(u['index'])].offset_s), float(syl[int(u['index'])].prominence))
                                  for u in rec.units]
                except ValueError:
                    self.units = []
                finally:
                    F.features, F.pick_beats = old
        tab = np.asarray(self.units, np.float64).reshape(-1, len(UNIT_COLS))
        return front_prosody(tab, Tk, final=final, admitted=True)   # admission already done by the recording (its word list)

    def prefix(self, Tk):
        Tk = int(min(Tk, self.T))
        if Tk not in self.cache:
            final = Tk >= self.T
            self.cache[Tk] = np.concatenate([self._mel(Tk), self._prosody(Tk, final)], 1).astype(np.float16)
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

    def __init__(self, pcm16, wave, words, lang, T, pstats, fm, fs, enc=None, Tpad=None, spk_min_s=1.0, spk=True, spk_refresh='tick',
                 feat_mode='prefix', spk_empty=None):
        self.T = int(T); self.Tpad = int(Tpad or T)
        if feat_mode not in ('prefix', 'inc'): raise ValueError(feat_mode)
        self.f = (IncFeat89 if feat_mode == 'inc' else CausalFeat89)(pcm16, words, lang, T, pstats)
        if self.Tpad < self.T: raise ValueError('Tpad < T')
        self.fm, self.fs = np.asarray(fm, np.float32), np.asarray(fs, np.float32)
        self.sp = SpeakerPrefix(wave, enc, spk_min_s) if spk else None
        if spk_refresh not in ('tick', 'grades'): raise ValueError(spk_refresh)
        self.spk_refresh = spk_refresh; self.grade_cache = {}
        self.spk_empty = None if spk_empty is None else np.asarray(spk_empty, np.float32)   # 11-06 (B2), StageCfg.spk_empty_mean
        self.calls = []

    def feat(self, Tk):
        Tk = int(min(Tk, self.T)); out = np.zeros((self.Tpad, 89), np.float32)
        out[:Tk] = (np.asarray(self.f.prefix(Tk), np.float32) - self.fm) / self.fs
        self.calls.append(Tk)
        return out

    def spk(self, Tk):
        v, ok = self._spk(Tk)
        if self.spk_empty is not None and ok < .5: return self.spk_empty, 1.       # B2: empty -> mean training embedding
        return v, ok

    def _spk(self, Tk):
        """spk_refresh 'tick': CAM++ of the PCM prefix at every tick. 'grades' (11-05, the training rule of StageCfg
        causal_spk, losses.causal_spk_select): the vector is refreshed only when a precompute grade becomes available --
        prefix 1 / 2 / 4 / 8 s at clip frame min(50 g, T), the whole clip (view rule, SpeakerPrefix.full) at T; the grade
        with the largest availability <= Tk, ties -> the prefix grade (the first); none -> the empty speaker. A grade uses
        only the audio before its availability frame (<= Tk): causal; at most 5 encodes per clip."""
        if self.sp is None: return np.zeros(192, np.float32), 0.
        if self.spk_refresh == 'tick': return self.sp.at(int(min(Tk, self.T)) * HOP)
        Tk = int(min(Tk, self.T)); av = [min(int(50 * g), self.T) for g in (1, 2, 4, 8)] + [self.T]
        cand = [j for j in range(5) if av[j] <= Tk]
        if not cand: return np.zeros(192, np.float32), 0.
        j = max(cand, key=lambda j: (av[j], j < 4))
        if j not in self.grade_cache:
            if j < 4: self.grade_cache[j] = self.sp.at(min(int(round((1, 2, 4, 8)[j] * SR)), self.T * HOP))
            else:
                try: self.grade_cache[j] = (self.sp.full(), 1.)
                except ValueError: self.grade_cache[j] = (np.zeros(192, np.float32), 0.)
        return self.grade_cache[j]


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
    if mode == 'voiced': vis &= ended(token_times[:, 1].astype(np.float64), now + ahead)
    elif mode == 'asr': vis &= ended(token_times[:, 1].astype(np.float64) + asr_lag, now + ahead)
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

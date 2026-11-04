"""Training-view loader (train-v1-20261101/views/data-v36_1 + -audio16 not needed; mel/prosody features are used).

A sample is a WINDOW of `frames` 50 Hz frames cut from one recording ("clip"; the utterance items of a clip are
contiguous in the arrays, verified 0 gaps), so a window may span several utterances. Returned per window:
  q      [T, J]   normalised joints              mask [T]   1 inside the clip
  feat   [T, 89]  normalised mel/prosody (+valid bit kept raw)
  text   [K, 768] frozen e5 token features of every utterance overlapping the window (prep_text.py cache)
  ttime  [K]      token time in s relative to window start (NaN = unknown)
  spk    [192], spk_valid, source        (cond/COND.json + SPK_EMB.npy)
  ref    [R, J]   reference motion: a window of ANOTHER clip of the same speaker (ref_valid = 0 if none)
  tts_h  [T/4, 2048] fp16, tts_hm [T/4]   ONLY with View(tts_hidden=<dir>): Qwen3-TTS Talker last-layer hidden state
                  per 80 ms row of the window (row r = window frames [4r, 4r+4)); see tts_rows()
"""
import json, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from .common import J, FEAT, SPK, TEXT_DIM, STRIDE, CHUNK_FRAMES, P

VIEW = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1'
SEMSYNTH = ('semsynth_r7p1',)
STD_FLOOR = 0.03            # rad; two finger joints are near-constant (std .007/.013) -> avoid z-score blow-up


def compute_stats(view, n=400000, seed=0):
    q = np.load(Path(view) / 'train_q.npy', mmap_mode='r'); f = np.load(Path(view) / 'train_feat.npy', mmap_mode='r')
    idx = np.sort(np.random.default_rng(seed).choice(len(q), min(n, len(q)), replace=False))
    qs, fs = np.asarray(q[idx], np.float32), np.asarray(f[idx], np.float32)
    fstd = fs.std(0); fstd[-1] = 1.; fmean = fs.mean(0); fmean[-1] = 0.        # keep the valid bit raw
    return dict(q_mean=qs.mean(0).tolist(), q_std=np.maximum(qs.std(0), STD_FLOOR).tolist(),
                f_mean=fmean.tolist(), f_std=np.maximum(fstd, 1e-3).tolist())


TTS_DIM = 2048              # Qwen3-TTS Talker hidden size (scmg/tts-hidden-v36_1)


def load_tts_index(root):
    """tts-hidden export index: id -> (relative npy path, codec frames). Items with an export error are absent."""
    idx = {}
    for f in sorted((Path(root) / 'index').glob('worker_*.jsonl')):
        for line in f.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            if 'file' in r and 'error' not in r: idx[r['id']] = (r['file'], int(r['codec']))
    return idx


def tts_rows(items, clip_start, s, n, R, load):
    """TTS hidden states of a window, one row per 80 ms (row r = window frames [4r, 4r+4), window frame 0 = clip frame s).
    h[k] of an item is the Talker state BEFORE emitting codec frame k, i.e. it covers item time [0.08k, 0.08(k+1)) =
    item frames [4k, 4k+4). Row r takes the codec frame that CONTAINS its first frame (clip frame s + 4r): k =
    (s + 4r - i0) // 4. When the item does not start on a 4-frame boundary of the window (item offset mid-codec, or
    STRIDE-2 windows) the row lags its codec frame by 1-3 frames -- never leads it, so the row is available as soon as
    the row's first audio frame is (h[k] exists once codec k-1 has been emitted). Rows past the clip end, of items
    without an export, or beyond an item's codec count stay zero with mask 0.
    items: clip items sorted by offset; load(item_id) -> (array [codec, D] (mmap) or None).
    Returns (h [R, D] float16, m [R] float32)."""
    h = np.zeros((R, TTS_DIM), np.float16); m = np.zeros(R, np.float32)
    rmax = min(R, -(-n // 4))                                  # rows whose first frame is inside the clip
    for it in items:
        i0 = it['offset'] - clip_start; i1 = i0 + it['frames']
        if i1 <= s or i0 >= s + n: continue
        r0 = max(0, -(-(i0 - s) // 4)); r1 = min(rmax, -(-(i1 - s) // 4))   # rows with first frame in [i0, i1)
        if r1 <= r0: continue
        arr = load(it['id'])
        if arr is None: continue
        k0 = (s + 4 * r0 - i0) // 4
        k1 = min(arr.shape[0], k0 + (r1 - r0))
        if k1 <= k0: continue
        h[r0:r0 + (k1 - k0)] = np.asarray(arr[k0:k1], np.float16); m[r0:r0 + (k1 - k0)] = 1
    return h, m


def require_causal_text(text_c, required, context):
    """Reject token streaming without prefix-causal E5, before sampling modes or running the model."""
    if required and text_c is None:
        raise ValueError(f"{context} requires text_c (prefix-causal E5); set SCMG_TEXT_CAUSAL "
                         "to the prep_text --causal cache. Bidirectional E5 cannot be used for token streaming.")


class View:
    """Everything loaded once (RAM, ~8 GB) before DataLoader workers fork (shared copy-on-write)."""

    def __init__(self, view=VIEW, split='train', text_cache=None, stats=None, mmap=False, tok_cache=None, tts_hidden=None,
                 intent_frames=None, holdout=None, tok_intent=None, bagger_dir=None):
        self.bagger_dir = bagger_dir
        v = Path(P(view)); self.root = v
        items = json.loads((v / 'train.json').read_text())
        sp = json.loads((v / 'SPLIT.json').read_text())
        keep = set(sp['train_ids'] if split == 'train' else sp['val_ids'])
        self.items = [it for it in items if it['id'] in keep]
        mode = 'r' if mmap else None
        self.q = np.load(v / 'train_q.npy', mmap_mode=mode)
        # frames inside retargeting jumps (BEAT2 only, scripts/glitch_audit.py; jump frame +-1) are excluded from losses
        gm = Path(P('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/glitch-audit/GLITCH_MASK.npy'))
        self.glitch = np.load(gm, mmap_mode='r') if (gm.exists() and os.environ.get('SCMG_NO_GLITCH') != '1') else None   # data ablation switch
        self.f = np.load(v / 'train_feat.npy', mmap_mode=mode)
        cond = json.loads((v / 'cond' / 'COND.json').read_text())
        self.cond = cond['items']; self.spk_emb = np.load(v / 'cond' / 'SPK_EMB.npy').astype(np.float32)
        self.stats = stats or compute_stats(v)
        self.qm, self.qs = (np.asarray(self.stats[k], np.float32) for k in ('q_mean', 'q_std'))
        self.fm, self.fs = (np.asarray(self.stats[k], np.float32) for k in ('f_mean', 'f_std'))
        clips = defaultdict(list)
        for it in self.items: clips[it['clip']].append(it)
        self.clips = []
        for c, xs in clips.items():
            xs.sort(key=lambda x: x['offset'])
            self.clips.append(dict(clip=c, start=xs[0]['offset'], frames=sum(x['frames'] for x in xs), items=xs,
                                   speaker=xs[0].get('speaker'), source=xs[0].get('data_source')))
        self.by_spk = defaultdict(list)
        for i, c in enumerate(self.clips):
            if c['speaker']: self.by_spk[c['speaker']].append(i)
        self.tok = None
        if tok_cache:                                   # LM token ids for the VoxCPM-style planner (prep_tokens.py)
            tc = Path(P(tok_cache))
            self.tok = dict(ids=np.load(tc / 'TOK_IDS.npy'), index=json.loads((tc / 'TOK_INDEX.json').read_text()))
            if tok_intent:                              # 11-03 word-level intent labels, row-aligned with TOK_IDS (prep_intent.py)
                pj = Path(P(tok_intent)) / 'PREP.json'
                if pj.exists():
                    src = json.loads(pj.read_text()).get('tok_cache')
                    if src and Path(src).name != Path(str(tok_cache)).name:
                        raise ValueError(f'word intent labels were built on {src}, not on the token cache {tok_cache}')
                sh = Path(P(tok_intent)) / 'TOK_IDS.sha256'        # written after a token-by-token rebuild check (11-03 review #6)
                if sh.exists():
                    import hashlib
                    got = hashlib.sha256(np.ascontiguousarray(self.tok['ids']).tobytes()).hexdigest()
                    if got != sh.read_text().split()[0]:
                        raise ValueError(f'word intent labels were verified against another TOK_IDS (sha {got[:12]} != {sh.read_text()[:12]})')
                il = np.load(Path(P(tok_intent)) / 'INTENT_LABEL.npy')
                if len(il) != len(self.tok['ids']): raise ValueError('INTENT_LABEL rows != TOK_IDS rows')
                self.tok['ilab'] = il
        self.text = None; self.text_c = None
        if text_cache:
            tc = Path(P(text_cache))
            self.text = dict(feat=np.load(tc / 'TEXT_FEAT.npy', mmap_mode=mode), time=np.load(tc / 'TEXT_TIME.npy'),
                             index=json.loads((tc / 'TEXT_INDEX.json').read_text()))
            # prefix-causal features of the same rows (prep_text --causal): ADSC token streaming must not see later words
            import os as _os
            cc = _os.environ.get('SCMG_TEXT_CAUSAL')
            self.text_c = np.load(Path(P(cc)) / 'TEXT_FEAT.npy', mmap_mode='r') if cc else None
            if cc:
                cr = Path(P(cc))
                if not json.loads((cr / 'PREP.json').read_text()).get('causal', False):
                    raise ValueError('SCMG_TEXT_CAUSAL must be a prefix-causal E5 cache')
                if (self.text_c.shape != self.text['feat'].shape or
                    json.loads((cr / 'TEXT_INDEX.json').read_text()) != self.text['index'] or
                    not np.array_equal(np.load(cr / 'TEXT_TIME.npy'), self.text['time'], equal_nan=True)):
                    raise ValueError('causal E5 rows / TEXT_INDEX / TEXT_TIME do not match the text cache')
        self.holdout = None                             # semantic-eval clips (+ their rhythm variants) never sampled (11-03)
        if holdout:
            pairs = {e['clip_id'].split('__')[0] for e in json.loads(Path(P(holdout)).read_text())['entries']}
            self.holdout = np.array([any(str(it.get('clip_id', '')).split('__')[0] in pairs for it in c['items'])
                                     for c in self.clips])
        if self.holdout is not None:
            for speaker, candidates in self.by_spk.items():
                self.by_spk[speaker] = [i for i in candidates if not self.holdout[i]]
        self.intent = None                              # frame-level intent track (scmg/prep_intent_frames.py, opt-in 11-03)
        if intent_frames:
            self.intent = np.load(Path(P(intent_frames)) / 'INTENT_FRAMES.npy', mmap_mode='r')
            if len(self.intent) != len(self.q): raise ValueError('INTENT_FRAMES rows != train_q rows')
            from .semclass import coarse_lut
            self.gest_lut = coarse_lut(json.loads((v / 'VOCAB.json').read_text()))
        self.tts = None
        if tts_hidden:                                  # Qwen3-TTS Talker hidden states (opt-in conditioning, 11-03)
            tr = Path(P(tts_hidden))
            self.tts = dict(root=tr, index=load_tts_index(tr))

    def tts_array(self, item_id):
        """[codec, 2048] fp16 memory map of one item's TTS hidden states, or None (no export)."""
        e = self.tts['index'].get(item_id)
        if e is None: return None
        a = np.load(self.tts['root'] / e[0], mmap_mode='r')
        return a[:e[1]]

    def weights(self, semsynth_share=0.4):
        """Clip sampling weights: human / semsynth category shares, frame-proportional inside a category."""
        fr = np.array([c['frames'] for c in self.clips], np.float64)
        syn = np.array([c['source'] in SEMSYNTH for c in self.clips])
        w = np.zeros_like(fr)
        if syn.any(): w[syn] = semsynth_share * fr[syn] / fr[syn].sum()
        hum = 1 - semsynth_share if syn.any() else 1.
        anc = getattr(self, 'anchor', None)   # {'sources': [...], 'share': x}: measured-motion "action anchor" share of the human mass
        if anc:
            a_ = np.array([c['source'] in anc['sources'] for c in self.clips]) & ~syn; o_ = ~syn & ~a_
            if a_.any(): w[a_] = hum * anc['share'] * fr[a_] / fr[a_].sum()
            if o_.any(): w[o_] = hum * (1 - anc['share']) * fr[o_] / fr[o_].sum()
        else: w[~syn] = hum * fr[~syn] / fr[~syn].sum()
        if getattr(self, 'holdout', None) is not None: w = np.where(self.holdout, 0., w)
        return w / w.sum()


class WindowDataset(torch.utils.data.IterableDataset):
    def __init__(self, view: View, frames=256, ref_frames=128, max_tokens=128, seed=0, semsynth_share=0.4,
                 fixed=None, max_ids=160, bagger_share=0.):
        self.v, self.T, self.R, self.K, self.seed = view, frames, ref_frames, max_tokens, seed
        self.w = view.weights(semsynth_share)
        self.fixed = fixed            # list of (clip_idx, start) for deterministic evaluation
        self.KI = max_ids
        self.bagger = None
        if not 0 <= bagger_share <= 1: raise ValueError('bagger_share must be in [0,1]')
        self.bagger_share = bagger_share
        if bagger_share > 0:
            from .bagger import BaggerPool
            self.bagger = BaggerPool(view.bagger_dir, self)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, wi.id if wi else 0])
        if self.fixed is not None:
            for ci, s in self.fixed[(wi.id if wi else 0)::(wi.num_workers if wi else 1)]: yield self.sample(ci, s, rng)
            return
        while True:
            if self.bagger is not None and rng.random() < self.bagger_share:
                yield self.bagger.sample(rng)
                continue
            ci = int(rng.choice(len(self.w), p=self.w))
            L = self.v.clips[ci]['frames']
            s = int(rng.integers(0, max(1, L - self.T + 1))) // STRIDE * STRIDE
            sample = self.sample(ci, s, rng)
            if self.bagger is not None:
                sample['loss_from_chunk'] = np.int64(-1)
                sample['bagger_null'] = np.bool_(False)
            yield sample

    def _window(self, ci, s, T):
        c = self.v.clips[ci]; a = c['start'] + s; n = max(0, min(T, c['frames'] - s))
        q = np.zeros((T, J), np.float32); m = np.zeros(T, np.float32)
        q[:n] = (np.asarray(self.v.q[a:a + n], np.float32) - self.v.qm) / self.v.qs; m[:n] = 1
        if self.v.glitch is not None and n > 0: m[:n] *= 1 - np.asarray(self.v.glitch[a:a + n], np.float32)
        if n < T and n > 0: q[n:] = q[n - 1]                         # hold last pose (masked anyway)
        return q, m, a, n

    def sample(self, ci, s, rng):
        v, T = self.v, self.T
        c = v.clips[ci]
        q, m, a, n = self._window(ci, s, T)
        f = np.zeros((T, FEAT), np.float32)
        f[:n] = (np.asarray(v.f[a:a + n], np.float32) - v.fm) / v.fs
        # text tokens of every utterance overlapping [s, s+n)
        toks, times, segs, toks_c = [], [], [], []      # segs: start time of the TTS segment (utterance) of each token
        if v.text is not None:
            off = c['start']
            for it in c['items']:
                i0, i1 = it['offset'] - off, it['offset'] - off + it['frames']
                if i1 <= s or i0 >= s + n or it.get('text_null'): continue
                e = v.text['index'].get(it['id'])
                if not e: continue
                k0, kn = e
                toks.append(np.asarray(v.text['feat'][k0:k0 + kn], np.float32))
                if v.text_c is not None: toks_c.append(np.asarray(v.text_c[k0:k0 + kn], np.float32))
                times.append(v.text['time'][k0:k0 + kn] + (i0 - s) / 50.)
                segs.append(np.full(kn, (i0 - s) / 50., np.float32))
        K = self.K
        text = np.zeros((K, TEXT_DIM), np.float32); ttime = np.full(K, np.nan, np.float32); tmask = np.zeros(K, bool)
        tseg = np.full(K, np.nan, np.float32)
        text_c = np.zeros((K, TEXT_DIM), np.float32) if v.text_c is not None else None
        if toks:
            tk, tt, ts_ = np.concatenate(toks), np.concatenate(times), np.concatenate(segs)
            tkc = np.concatenate(toks_c) if toks_c else None
            if len(tk) > K:                                            # keep tokens closest to the window
                order = np.argsort(np.nan_to_num(np.abs(tt - T / 100.), nan=1e9))[:K]; order.sort()
                tk, tt, ts_ = tk[order], tt[order], ts_[order]
                if tkc is not None: tkc = tkc[order]
            text[:len(tk)] = tk; ttime[:len(tk)] = tt; tmask[:len(tk)] = True; tseg[:len(tk)] = ts_
            if tkc is not None: text_c[:len(tk)] = tkc
        # LM token ids of the utterances overlapping the window, in order (VoxCPM-style planner)
        tids = np.zeros(self.KI, np.int64); tidm = np.zeros(self.KI, bool); titime = np.full(self.KI, np.nan, np.float32); tiseg = np.full(self.KI, np.nan, np.float32)
        tiend = np.full(self.KI, np.nan, np.float32)
        tilab = np.full(self.KI, -1, np.int64) if (v.tok is not None and 'ilab' in v.tok) else None
        if v.tok is not None:
            seq = []; tsq = []; ssq = []; endsq = []; lsq = []
            for it in c['items']:
                i0 = it['offset'] - c['start']
                if i0 + it['frames'] <= s or i0 >= s + n or it.get('text_null'): continue
                e = v.tok['index'].get(it['id'])
                if e:
                    seq.append(v.tok['ids'][e[0]:e[0] + e[1]])
                    if tilab is not None: lsq.append(v.tok['ilab'][e[0]:e[0] + e[1]])
                    # LM tokens carry no timing: spread them evenly over the utterance (window-relative s) -- used only
                    # for ADSC arrival simulation (text streams ahead of / alongside the speech)
                    tsq.append((i0 - s + (np.arange(e[1]) + 0.5) / max(e[1], 1) * it['frames']) / 50.)
                    ssq.append(np.full(e[1], (i0 - s) / 50., np.float32))
                    endsq.append(np.full(e[1], (i0 + it['frames'] - s) / 50., np.float32))
            if seq:
                seq = np.concatenate(seq)[:self.KI]; tids[:len(seq)] = seq; tidm[:len(seq)] = True
                titime[:len(seq)] = np.concatenate(tsq)[:self.KI]; tiseg[:len(seq)] = np.concatenate(ssq)[:self.KI]
                tiend[:len(seq)] = np.concatenate(endsq)[:self.KI]
                if tilab is not None: tilab[:len(seq)] = np.concatenate(lsq)[:self.KI]
        # speaker / source conditions of the first utterance in the window
        it0 = next((it for it in c['items'] if it['offset'] - c['start'] + it['frames'] > s), c['items'][0])
        row = v.cond.get(str(it0['id']))
        spk = np.zeros(SPK, np.float32); spk_valid = 0.; source = 0
        if row:
            if row[1]: spk = v.spk_emb[row[0]]; spk_valid = 1.
            source = int(row[2])
        # reference motion: another clip of the same speaker
        ref = np.zeros((self.R, J), np.float32); ref_valid = 0.
        cand = v.by_spk.get(c['speaker'], [])
        if getattr(v, 'holdout', None) is not None:
            cand = [j for j in cand if not v.holdout[j]]
        if len(cand) > 1:
            cj = int(cand[rng.integers(len(cand))])
            if cj == ci: cj = int(cand[(cand.index(ci) + 1) % len(cand)])
            Lr = v.clips[cj]['frames']
            if Lr >= self.R // 2:
                rs = int(rng.integers(0, max(1, Lr - self.R + 1)))
                ref, rm, _, _ = self._window(cj, rs, self.R); ref_valid = 1.
        extra = dict(text_c=text_c) if text_c is not None else {}
        if tilab is not None: extra['tilab'] = tilab
        if v.intent is not None:                        # [T] int16: -1 unknown / 0 none / >=2 class (frames past the clip end -1)
            il = np.full(T, -1, np.int16); il[:n] = v.intent[a:a + n]; extra['intent'] = il
            extra['gest'] = np.where(il >= 0, v.gest_lut[np.clip(il, 0, None)], -1).astype(np.int16)   # coarse g_k frame labels
        if v.tts is not None:
            if T % 4: raise ValueError(f'tts_hidden needs windows of a multiple of 4 frames (got {T})')
            extra['tts_h'], extra['tts_hm'] = tts_rows(c['items'], c['start'], s, n, T // 4, v.tts_array)
        return dict(**extra, q=q, mask=m, feat=f, text=text, ttime=ttime, tmask=tmask, spk=spk, spk_valid=np.float32(spk_valid),
                    source=np.int64(source), tids=tids, tidm=tidm, titime=titime, tseg=tseg, tiseg=tiseg, tiend=tiend, ref=ref, ref_valid=np.float32(ref_valid), clip=np.int64(ci), start=np.int64(s))


def collate(batch):
    return {k: torch.as_tensor(np.stack([b[k] for b in batch])) for k in batch[0]}


def fixed_windows(view: View, frames=256, n=None):
    """Deterministic evaluation windows: every clip from its start (dev set: 380 items)."""
    out = [(i, 0) for i in range(len(view.clips))]
    return out[:n] if n else out


def clip_lm_capacity(view, ci):
    """Evaluation can retain the whole clip, then the streamer selects its training window."""
    if view.tok is None: return 160
    return max(160, sum(view.tok['index'].get(it['id'], (0, 0))[1]
                        for it in view.clips[ci]['items'] if not it.get('text_null')))

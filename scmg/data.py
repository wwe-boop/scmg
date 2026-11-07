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
  cz_*   ONLY after View.load_causal(<precompute dir>) (11-05 训练匹配, scripts/echog_g1/precompute_causal.py): see causal_rows()
"""
import json, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from .common import J, FEAT, SPK, TEXT_DIM, STRIDE, CHUNK_FRAMES, P, EMBODIMENT

VIEW = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1'
SEMSYNTH = ('semsynth_r7p1',)
STD_FLOOR = 0.03            # rad; two finger joints are near-constant (std .007/.013) -> avoid z-score blow-up


def compute_stats(view, n=400000, seed=0, rows=None, std_floor=STD_FLOOR):
    """rows: optional array of row indices to sample from (e.g. train-split rows only); default every row (as before)."""
    q = np.load(Path(view) / 'train_q.npy', mmap_mode='r'); f = np.load(Path(view) / 'train_feat.npy', mmap_mode='r')
    pool = np.arange(len(q)) if rows is None else np.asarray(rows)
    idx = np.sort(pool[np.random.default_rng(seed).choice(len(pool), min(n, len(pool)), replace=False)])   # rows=None: as before
    qs, fs = np.asarray(q[idx], np.float32), np.asarray(f[idx], np.float32)
    fstd = fs.std(0); fstd[-1] = 1.; fmean = fs.mean(0); fmean[-1] = 0.        # keep the valid bit raw
    return dict(q_mean=qs.mean(0).tolist(), q_std=np.maximum(qs.std(0), std_floor).tolist(),
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
                 intent_frames=None, holdout=None, tok_intent=None, bagger_dir=None, gest_frames=None, floor_frames=None):
        self.bagger_dir = bagger_dir
        v = Path(P(view)); self.root = v
        items = json.loads((v / 'train.json').read_text())
        sp = json.loads((v / 'SPLIT.json').read_text())
        if split == 'test' and 'test_ids' not in sp: raise ValueError(f'{v}/SPLIT.json has no test_ids')
        keep = set(sp['train_ids'] if split == 'train' else sp['test_ids'] if split == 'test' else sp['val_ids'])
        self.items = [it for it in items if it['id'] in keep]
        mode = 'r' if mmap else None
        self.q = np.load(v / 'train_q.npy', mmap_mode=mode)
        # frames inside retargeting jumps (BEAT2 only, scripts/glitch_audit.py; jump frame +-1) are excluded from losses
        gm = Path(P('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/glitch-audit/GLITCH_MASK.npy'))
        self.glitch = np.load(gm, mmap_mode='r') if (gm.exists() and os.environ.get('SCMG_NO_GLITCH') != '1'
                                                     and EMBODIMENT == 'ex002') else None   # data ablation switch; EX002 rows only
        if self.glitch is not None and len(self.glitch) != len(self.q):
            # the mask is row-aligned with data-v36_1 only: another view (e.g. echog-g1-v1) must not index it (11-04)
            print(f'View {v.name}: GLITCH_MASK rows {len(self.glitch)} != {len(self.q)} view rows -> not used', flush=True)
            self.glitch = None
        # SCMG_EXTRA_MASK=<uint8 npy, bad=1, row-aligned with this view>: more frames excluded from losses (11-04: pose-audit
        # ARM_MASK, arm/palm/self-contact/outlier frames). Explicit request -> a length mismatch is an error, not a skip.
        em = os.environ.get('SCMG_EXTRA_MASK')
        self.extra_mask = np.load(P(em), mmap_mode='r') if em else None
        if self.extra_mask is not None and len(self.extra_mask) != len(self.q):
            raise ValueError(f'SCMG_EXTRA_MASK rows {len(self.extra_mask)} != {len(self.q)} rows of view {v.name}')
        if self.extra_mask is not None:
            print(f'View {v.name} ({split}): SCMG_EXTRA_MASK {em} masks {float(np.asarray(self.extra_mask).mean()):.4f} of all rows', flush=True)
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
        self.text = None; self.text_c = None; self.text_dim = TEXT_DIM; self.text_native_causal = False
        if text_cache:
            tc = Path(P(text_cache))
            feat = np.load(tc / 'TEXT_FEAT.npy', mmap_mode=mode)
            # PREP.json is optional for legacy / ad-hoc caches (old e5 path must keep working); LLM caches always write it.
            prep = json.loads((tc / 'PREP.json').read_text()) if (tc / 'PREP.json').is_file() else {}
            self.text_dim = int(feat.shape[1])
            if 'hidden_size' in prep and int(prep['hidden_size']) != self.text_dim:
                raise ValueError(f'text cache PREP.hidden_size={prep["hidden_size"]} but TEXT_FEAT has {self.text_dim} dims')
            self.text = dict(feat=feat, time=np.load(tc / 'TEXT_TIME.npy'),
                             index=json.loads((tc / 'TEXT_INDEX.json').read_text()))
            # A decoder LLM (prep_text_llm.py) is prefix-causal by construction: the full-sentence feature at token j is
            # exactly what a prefix ending at j produces, so the same cache serves ADSC token streaming.  Flagged in
            # PREP.json; e5 caches keep the old SCMG_TEXT_CAUSAL path below unchanged.
            native = bool(prep.get('causal') and (prep.get('native_causal') or prep.get('encoder') == 'llm'))
            if native:
                self.text_native_causal = True
                self.text_c = feat
            else:
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
        self.gest = None                                # 11-05: direct coarse g_k frame track (G1 BEAT2 annotation), opt-in
        if gest_frames:
            if intent_frames: raise ValueError('gest_frames and intent_frames both define the g_k labels: pass one')
            self.gest = load_gest_frames(gest_frames, v, len(self.q))
        self.floor = None                               # 11-06: floor track of post-speech tail views (build_tail_view.py), opt-in
        if floor_frames:
            fp = Path(P(floor_frames)); self.floor = np.load(fp / 'FLOOR.npy', mmap_mode='r')
            if len(self.floor) != len(self.q): raise ValueError(f'FLOOR rows {len(self.floor)} != {len(self.q)} rows of view {v.name}')
            self.tail_clips = [i for i, c in enumerate(self.clips) if np.asarray(self.floor[c['start']:c['start'] + c['frames']]).max() > 0]
        if EMBODIMENT != 'ex002': self._refuse_foreign_assets(v, holdout, tok_cache, tok_intent, intent_frames)
        self.tts = None
        if tts_hidden:                                  # Qwen3-TTS Talker hidden states (opt-in conditioning, 11-03)
            tr = Path(P(tts_hidden))
            self.tts = dict(root=tr, index=load_tts_index(tr))

    def _refuse_foreign_assets(self, v, holdout, tok_cache, tok_intent, intent_frames):
        """Non-EX002 views (G1, 11-04): the EX002 side assets (semantic holdout manifest, Qwen2.5 token / intent caches,
        frame intent track) are keyed by EX002 item / clip ids. On another view they would silently match nothing (an
        empty holdout, empty LM ids) -> refuse instead of training on a silently changed condition set."""
        if holdout is not None and not self.holdout.any():
            raise ValueError(f'holdout {holdout} matches no clip of {v.name} (EX002 manifest on a {EMBODIMENT} view?)')
        if self.tok is not None:
            cov = np.mean([str(it['id']) in self.tok['index'] for it in self.items]) if self.items else 0.
            if cov < 0.5: raise ValueError(f'token cache {tok_cache} covers {cov:.0%} of {v.name} items (EX002 cache on a {EMBODIMENT} view?)')
        if intent_frames is not None or tok_intent is not None:
            raise ValueError(f'intent tracks / word intent labels exist for EX002 only (view {v.name}, embodiment {EMBODIMENT})')

    def load_causal(self, d, horizons, check_spk=True):
        """Training-matched causal inputs (precompute_causal.py output; 11-05). horizons = StageCfg.horizons (80 ms units):
        the stream times of a window whose prefix prosody columns every sample carries (causal_feat.stream_lat_grid).
        Checks (check_spk, off only for tests on perturbed audio): the 'full' speaker grade of up to 64 covered clips equals
        the view's SPK_EMB row and (with FEAT_PREFIX present) the whole-clip mel equals the view's mel (same audio, same rule)."""
        d = Path(P(d)); meta = json.loads((d / 'META.json').read_text())
        from .causal_feat import UNIT_COLS
        if tuple(meta.get('unit_cols') or ()) != UNIT_COLS: raise ValueError(f'{d}: no / other unit table ({meta.get("unit_cols")})')
        if list(meta['grades_s'][:4]) != [1., 2., 4., 8.] or meta['grades_s'][4] is not None: raise ValueError(f'{d}: grades {meta["grades_s"]}')
        self.causal = dict(dir=str(d), index=json.loads((d / 'INDEX.json').read_text()), spk=np.load(d / 'SPK_PREFIX.npy'),
                           spkv=np.load(d / 'SPK_VALID.npy'), units=np.load(d / 'UNITS.npy'), tend=np.load(d / 'TOKEN_END.npy'),
                           horizons=tuple(int(h) for h in horizons), meta=meta)
        ids = [c['items'][0]['id'] for c in self.clips if len(c['items']) == 1]
        cov = [i for i in ids if i in self.causal['index']]
        nchk = nmel = 0
        F = np.load(d / 'FEAT_PREFIX.npy', mmap_mode='r') if (d / 'FEAT_PREFIX.npy').exists() else None
        for i in cov[:64]:
            row = self.cond.get(str(i)); e = self.causal['index'][i]
            if check_spk and row and row[1]:
                if not np.allclose(self.spk_emb[row[0]], self.causal['spk'][e['clip_row'], 4], atol=1e-5):
                    raise ValueError(f'{d}: full-grade CAM++ of {i} != view SPK_EMB row (other audio / view?)')
                nchk += 1
            if check_spk and F is not None and e['feat'][4][1]:  # the view mel IS the prefix mel (cz_feat columns 0-79)
                c = self.clips[[k for k, x in enumerate(self.clips) if x['items'][0]['id'] == i][0]]
                r0, nf = e['feat'][4]
                if nf != c['frames'] or not np.array_equal(np.asarray(self.f[c['start']:c['start'] + nf, :80]), np.asarray(F[r0:r0 + nf, :80])):
                    raise ValueError(f'{d}: whole-clip mel of {i} != view feat mel')
                nmel += 1
        self.causal['coverage'] = len(cov) / max(1, len(self.clips))
        if (d / 'SPK_MEAN.npy').exists(): self.causal['spk_mean'] = np.load(d / 'SPK_MEAN.npy').astype(np.float32)
        print(f'View {self.root.name}: causal inputs {d} cover {len(cov)} / {len(self.clips)} clips; SPK_EMB check {nchk}, '
              f'mel check {nmel} clips', flush=True)
        return self.causal['coverage']

    def restrict(self, frac=None, n_speakers=None, seed=0):
        """11-07 data ablation: sample only a subset of the train clips (added to the holdout mask; weights() zeroes the rest).
        n_speakers: the first N speakers of a seeded permutation of the sorted speaker list; frac: whole RECORDINGS (clip id
        minus '_utt_NNNN') kept iff crc32(seed:recording) / 2^32 < frac. Returns a summary dict."""
        import zlib, re as _re
        spk = [c['items'][0].get('speaker', '') for c in self.clips]
        keep = np.ones(len(self.clips), bool)
        chosen = None
        if n_speakers:
            allspk = sorted(set(spk)); order = np.random.default_rng(seed).permutation(len(allspk))
            chosen = sorted(allspk[i] for i in order[:int(n_speakers)])
            keep &= np.array([x in chosen for x in spk])
        if frac is not None and frac < 1:
            rec = [_re.sub(r'_utt_\d+$', '', str(c['items'][0]['id'])) for c in self.clips]
            keep &= np.array([zlib.crc32(f'{seed}:{r}'.encode()) / 2 ** 32 < frac for r in rec])
        if not keep.any(): raise ValueError('data restriction leaves no clip')
        self.holdout = ~keep if getattr(self, 'holdout', None) is None else (self.holdout | ~keep)
        fr = np.array([c['frames'] for c in self.clips], np.float64)
        return dict(clips=int(keep.sum()), of=len(keep), frames_share=float(fr[keep].sum() / fr.sum()), speakers=chosen,
                    n_speakers=len(set(np.array(spk)[keep])))

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
                 fixed=None, max_ids=160, bagger_share=0., tail_share=0., p_start0=0.):
        self.v, self.T, self.R, self.K, self.seed = view, frames, ref_frames, max_tokens, seed
        # 11-06 (B1): P(a training window starts at the clip start); 0 = the original uniform start (no extra draw: bit-identical)
        self.p_start0 = float(p_start0)
        self.w = view.weights(semsynth_share)
        self.fixed = fixed            # list of (clip_idx, start) for deterministic evaluation
        self.KI = max_ids
        self.bagger = None
        if not 0 <= bagger_share <= 1: raise ValueError('bagger_share must be in [0,1]')
        self.bagger_share = bagger_share
        self.tail_share = tail_share        # 11-06: share of windows drawn around a speech offset of a tail clip (floor views)
        if tail_share > 0 and (view.floor is None or not view.tail_clips): raise ValueError('tail_share needs a floor view with tails')
        if bagger_share > 0 and getattr(view, 'causal', None) is not None:
            raise ValueError('causal inputs (cz_*) are not produced for BAgger samples')
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
            if self.tail_share > 0 and rng.random() < self.tail_share:
                ci = int(self.v.tail_clips[rng.integers(len(self.v.tail_clips))]); c = self.v.clips[ci]
                fl = np.asarray(self.v.floor[c['start']:c['start'] + c['frames']]); off = int(np.argmax(fl > 0))
                s = max(0, off - int(rng.integers(64, self.T - 32))) // STRIDE * STRIDE      # window covers the speech offset
                yield self.sample(ci, s, rng); continue
            ci = int(rng.choice(len(self.w), p=self.w))
            L = self.v.clips[ci]['frames']
            s = int(rng.integers(0, max(1, L - self.T + 1))) // STRIDE * STRIDE
            if self.p_start0 > 0 and rng.random() < self.p_start0: s = 0      # stream starts seen as often as at inference
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
        if self.v.extra_mask is not None and n > 0: m[:n] *= 1 - np.asarray(self.v.extra_mask[a:a + n], np.float32)
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
        czt = [] if getattr(v, 'causal', None) is not None else None     # 11-05 训练匹配: token end times (cz_tend)
        cz_tend = None
        native_c = v.text_c is not None and v.text_c is v.text['feat']   # LLM cache: text_c IS text
        if v.text is not None:
            off = c['start']
            for it in c['items']:
                i0, i1 = it['offset'] - off, it['offset'] - off + it['frames']
                if i1 <= s or i0 >= s + n or it.get('text_null'): continue
                e = v.text['index'].get(it['id'])
                if not e: continue
                k0, kn = e
                toks.append(np.asarray(v.text['feat'][k0:k0 + kn], np.float32))
                if v.text_c is not None and not native_c:
                    toks_c.append(np.asarray(v.text_c[k0:k0 + kn], np.float32))
                times.append(v.text['time'][k0:k0 + kn] + (i0 - s) / 50.)
                segs.append(np.full(kn, (i0 - s) / 50., np.float32))
                if czt is not None: czt.append(causal_tend(v, it['id'], kn) + (i0 - s) / 50.)
        K = self.K; TD = v.text_dim
        text = np.zeros((K, TD), np.float32); ttime = np.full(K, np.nan, np.float32); tmask = np.zeros(K, bool)
        tseg = np.full(K, np.nan, np.float32)
        text_c = np.zeros((K, TD), np.float32) if (v.text_c is not None and not native_c) else None
        if toks:
            tk, tt, ts_ = np.concatenate(toks), np.concatenate(times), np.concatenate(segs)
            tkc = np.concatenate(toks_c) if toks_c else None
            te_ = np.concatenate(czt) if czt else None
            if len(tk) > K:                                            # keep tokens closest to the window
                order = np.argsort(np.nan_to_num(np.abs(tt - T / 100.), nan=1e9))[:K]; order.sort()
                tk, tt, ts_ = tk[order], tt[order], ts_[order]
                if tkc is not None: tkc = tkc[order]
                if te_ is not None: te_ = te_[order]
            text[:len(tk)] = tk; ttime[:len(tk)] = tt; tmask[:len(tk)] = True; tseg[:len(tk)] = ts_
            if tkc is not None: text_c[:len(tk)] = tkc
            if te_ is not None: cz_tend = np.full(K, np.nan, np.float32); cz_tend[:len(tk)] = te_
        if czt is not None and cz_tend is None: cz_tend = np.full(K, np.nan, np.float32)
        if native_c: text_c = text
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
        if getattr(v, 'gest', None) is not None:      # [T] int16: -1 unknown / 0 none / 1.. coarse class (past the clip end -1)
            gl = np.full(T, -1, np.int16); gl[:n] = v.gest[a:a + n]; extra['gest'] = gl
        if getattr(v, 'floor', None) is not None:       # [T] floor per frame (0 speaking, 1 hold, 2 yield; 0 past the clip end)
            fl = np.zeros(T, np.int64); fl[:n] = v.floor[a:a + n]; extra['floor'] = fl
            extra['aend'] = np.int64(int(np.argmax(fl[:n] > 0)) if (n and fl[:n].max() > 0) else n)   # audio frames of the window
        if v.tts is not None:
            if T % 4: raise ValueError(f'tts_hidden needs windows of a multiple of 4 frames (got {T})')
            extra['tts_h'], extra['tts_hm'] = tts_rows(c['items'], c['start'], s, n, T // 4, v.tts_array)
        if getattr(v, 'causal', None) is not None:
            extra.update(causal_rows(v, c, s, n, T, f)); extra['cz_tend'] = cz_tend
        return dict(**extra, q=q, mask=m, feat=f, text=text, ttime=ttime, tmask=tmask, spk=spk, spk_valid=np.float32(spk_valid),
                    source=np.int64(source), tids=tids, tidm=tidm, titime=titime, tseg=tseg, tiseg=tiseg, tiend=tiend, ref=ref, ref_valid=np.float32(ref_valid), clip=np.int64(ci), start=np.int64(s))


def causal_tend(v, item_id, kn):
    """[kn] token end times (item-relative s) of an item's e5 cache rows (precompute TOKEN_END; NaN = never voiced:
    not in the precompute, or a token count that differs from the text cache)."""
    e = v.causal['index'].get(item_id)
    if e is None or 'tend' not in e or e['tend'][1] != kn: return np.full(kn, np.nan, np.float32)
    r0, nr = e['tend']
    return np.asarray(v.causal['tend'][r0:r0 + nr], np.float32)


def causal_rows(v, c, s, n, T, f):
    """cz_* sample keys (training-matched causal inputs, 11-05; window = clip frames [s, s + n), T rows):
      cz_ok    bool: the clip (one item) is in the precompute
      cz_feat  [T, 89] the window's view feat (its mel columns 0-79 == the prefix mel row for row, checked against the
               precompute's whole-clip grade by load_causal); columns 80-88 are always overwritten from cz_pro
      cz_pro   [S, T, 9] normalised causal_feat.FRONT_COLS (80-88) of the prefix at stream time j (window latents
               stream_lat_grid(K, horizons)[j], clip frame s + 4 * latents capped at the clip end): rows below the
               stream time = causal_feat.front_prosody (bitwise = prefix_feat89 there), rows at / after it 0
      cz_spk [5, 192], cz_spkv [5], cz_spkt [5]: prefix CAM++ grades 1 / 2 / 4 / 8 s / whole clip, valid, and the clip
               frame from which each grade exists at inference (min(50 g, frames); whole clip: frames)
      cz_tend  [Kt] (added by sample()) window-relative end time of the last word of every text row (NaN = never voiced)
    Not covered -> cz_ok False and cz_feat = the view feat, zeros elsewhere (the loss keeps the original inputs)."""
    from .common import CHUNK, U80
    from .causal_feat import FRONT_COLS, front_prosody, stream_lat_grid
    cz = v.causal; it = c['items'][0]; L = c['frames']
    grid = stream_lat_grid(T // CHUNK_FRAMES, cz['horizons'], CHUNK, U80)
    out = dict(**({'cz_spkmean': cz['spk_mean']} if 'spk_mean' in cz else {}), cz_ok=np.bool_(False), cz_feat=f.copy(), cz_pro=np.zeros((len(grid), T, len(FRONT_COLS)), np.float32),
               cz_spk=np.zeros((5, SPK), np.float32), cz_spkv=np.zeros(5, np.float32),
               cz_spkt=np.array([min(50 * g, L) for g in (1, 2, 4, 8)] + [L], np.int64))
    e = cz['index'].get(it['id']) if len(c['items']) == 1 else None
    if e is None or int(e['frames']) != L: return out
    cols = list(FRONT_COLS); fm, fs = v.fm[cols], v.fs[cols]
    u0, un = e['units']; tab = cz['units'][u0:u0 + un]
    cache = {}
    for j, lat in enumerate(grid):
        Tk = min(L, s + lat * STRIDE)                   # inference: CausalInputs.feat(min(Tk, T))
        rows = min(n, Tk - s)
        if rows <= 0: continue
        if Tk not in cache: cache[Tk] = front_prosody(tab, Tk, final=Tk >= L).astype(np.float16).astype(np.float32)
        out['cz_pro'][j, :rows] = (cache[Tk][s:s + rows] - fm) / fs
    out['cz_spk'] = cz['spk'][e['clip_row']].astype(np.float32); out['cz_spkv'] = cz['spkv'][e['clip_row']].astype(np.float32)
    if 'spk_mean' in cz: out['cz_spkmean'] = cz['spk_mean']
    out['cz_ok'] = np.bool_(True)
    return out


def load_gest_frames(d, view_dir, n_rows):
    """GEST_FRAMES.npy (int16 per train_q row: -1 unknown, 0 none, 1..6 coarse g_k class) of scripts/semantic/g1_gest_frames.py.
    PREP.json must name the same row count and the sha256 of THIS view's SPLIT.json (labels are only built for its train_ids)."""
    import hashlib
    d = Path(P(d)); g = np.load(d / 'GEST_FRAMES.npy', mmap_mode='r'); prep = json.loads((d / 'PREP.json').read_text())
    if len(g) != n_rows or prep.get('rows') != n_rows: raise ValueError(f'GEST_FRAMES rows {len(g)} / PREP {prep.get("rows")} != train_q rows {n_rows}')
    sha = hashlib.sha256((Path(view_dir) / 'SPLIT.json').read_bytes()).hexdigest()
    if prep.get('split_sha256') != sha: raise ValueError(f'GEST_FRAMES built for split {prep.get("split_sha256")}, view {view_dir} has {sha}')
    return np.asarray(g)


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

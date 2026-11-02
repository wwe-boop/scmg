"""Training-view loader (train-v1-20261101/views/data-v36_1 + -audio16 not needed; mel/prosody features are used).

A sample is a WINDOW of `frames` 50 Hz frames cut from one recording ("clip"; the utterance items of a clip are
contiguous in the arrays, verified 0 gaps), so a window may span several utterances. Returned per window:
  q      [T, J]   normalised joints              mask [T]   1 inside the clip
  feat   [T, 89]  normalised mel/prosody (+valid bit kept raw)
  text   [K, 768] frozen e5 token features of every utterance overlapping the window (prep_text.py cache)
  ttime  [K]      token time in s relative to window start (NaN = unknown)
  spk    [192], spk_valid, source        (cond/COND.json + SPK_EMB.npy)
  ref    [R, J]   reference motion: a window of ANOTHER clip of the same speaker (ref_valid = 0 if none)
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from .common import J, FEAT, SPK, TEXT_DIM, STRIDE, CHUNK_FRAMES

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


class View:
    """Everything loaded once (RAM, ~8 GB) before DataLoader workers fork (shared copy-on-write)."""

    def __init__(self, view=VIEW, split='train', text_cache=None, stats=None, mmap=False, tok_cache=None):
        v = Path(view); self.root = v
        items = json.loads((v / 'train.json').read_text())
        sp = json.loads((v / 'SPLIT.json').read_text())
        keep = set(sp['train_ids'] if split == 'train' else sp['val_ids'])
        self.items = [it for it in items if it['id'] in keep]
        mode = 'r' if mmap else None
        self.q = np.load(v / 'train_q.npy', mmap_mode=mode)
        # frames inside retargeting jumps (BEAT2 only, scripts/glitch_audit.py; jump frame +-1) are excluded from losses
        gm = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/glitch-audit/GLITCH_MASK.npy')
        self.glitch = np.load(gm, mmap_mode='r') if gm.exists() else None
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
            tc = Path(tok_cache)
            self.tok = dict(ids=np.load(tc / 'TOK_IDS.npy'), index=json.loads((tc / 'TOK_INDEX.json').read_text()))
        self.text = None
        if text_cache:
            tc = Path(text_cache)
            self.text = dict(feat=np.load(tc / 'TEXT_FEAT.npy', mmap_mode=mode), time=np.load(tc / 'TEXT_TIME.npy'),
                             index=json.loads((tc / 'TEXT_INDEX.json').read_text()))

    def weights(self, semsynth_share=0.4):
        """Clip sampling weights: human / semsynth category shares, frame-proportional inside a category."""
        fr = np.array([c['frames'] for c in self.clips], np.float64)
        syn = np.array([c['source'] in SEMSYNTH for c in self.clips])
        w = np.zeros_like(fr)
        if syn.any(): w[syn] = semsynth_share * fr[syn] / fr[syn].sum()
        w[~syn] = (1 - semsynth_share if syn.any() else 1.) * fr[~syn] / fr[~syn].sum()
        return w / w.sum()


class WindowDataset(torch.utils.data.IterableDataset):
    def __init__(self, view: View, frames=256, ref_frames=128, max_tokens=128, seed=0, semsynth_share=0.4,
                 fixed=None, max_ids=160):
        self.v, self.T, self.R, self.K, self.seed = view, frames, ref_frames, max_tokens, seed
        self.w = view.weights(semsynth_share)
        self.fixed = fixed            # list of (clip_idx, start) for deterministic evaluation
        self.KI = max_ids

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, wi.id if wi else 0])
        if self.fixed is not None:
            for ci, s in self.fixed[(wi.id if wi else 0)::(wi.num_workers if wi else 1)]: yield self.sample(ci, s, rng)
            return
        while True:
            ci = int(rng.choice(len(self.w), p=self.w))
            L = self.v.clips[ci]['frames']
            s = int(rng.integers(0, max(1, L - self.T + 1))) // STRIDE * STRIDE
            yield self.sample(ci, s, rng)

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
        toks, times, segs = [], [], []                 # segs: start time of the TTS segment (utterance) of each token
        if v.text is not None:
            off = c['start']
            for it in c['items']:
                i0, i1 = it['offset'] - off, it['offset'] - off + it['frames']
                if i1 <= s or i0 >= s + n or it.get('text_null'): continue
                e = v.text['index'].get(it['id'])
                if not e: continue
                k0, kn = e
                toks.append(np.asarray(v.text['feat'][k0:k0 + kn], np.float32))
                times.append(v.text['time'][k0:k0 + kn] + (i0 - s) / 50.)
                segs.append(np.full(kn, (i0 - s) / 50., np.float32))
        K = self.K
        text = np.zeros((K, TEXT_DIM), np.float32); ttime = np.full(K, np.nan, np.float32); tmask = np.zeros(K, bool)
        tseg = np.full(K, np.nan, np.float32)
        if toks:
            tk, tt, ts_ = np.concatenate(toks), np.concatenate(times), np.concatenate(segs)
            if len(tk) > K:                                            # keep tokens closest to the window
                order = np.argsort(np.nan_to_num(np.abs(tt - T / 100.), nan=1e9))[:K]; order.sort()
                tk, tt, ts_ = tk[order], tt[order], ts_[order]
            text[:len(tk)] = tk; ttime[:len(tk)] = tt; tmask[:len(tk)] = True; tseg[:len(tk)] = ts_
        # LM token ids of the utterances overlapping the window, in order (VoxCPM-style planner)
        tids = np.zeros(self.KI, np.int64); tidm = np.zeros(self.KI, bool); titime = np.full(self.KI, np.nan, np.float32); tiseg = np.full(self.KI, np.nan, np.float32)
        if v.tok is not None:
            seq = []; tsq = []; ssq = []
            for it in c['items']:
                i0 = it['offset'] - c['start']
                if i0 + it['frames'] <= s or i0 >= s + n or it.get('text_null'): continue
                e = v.tok['index'].get(it['id'])
                if e:
                    seq.append(v.tok['ids'][e[0]:e[0] + e[1]])
                    # LM tokens carry no timing: spread them evenly over the utterance (window-relative s) -- used only
                    # for ADSC arrival simulation (text streams ahead of / alongside the speech)
                    tsq.append((i0 - s + (np.arange(e[1]) + 0.5) / max(e[1], 1) * it['frames']) / 50.)
                    ssq.append(np.full(e[1], (i0 - s) / 50., np.float32))
            if seq:
                seq = np.concatenate(seq)[:self.KI]; tids[:len(seq)] = seq; tidm[:len(seq)] = True
                titime[:len(seq)] = np.concatenate(tsq)[:self.KI]; tiseg[:len(seq)] = np.concatenate(ssq)[:self.KI]
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
        if len(cand) > 1:
            cj = int(cand[rng.integers(len(cand))])
            if cj == ci: cj = int(cand[(cand.index(ci) + 1) % len(cand)])
            Lr = v.clips[cj]['frames']
            if Lr >= self.R // 2:
                rs = int(rng.integers(0, max(1, Lr - self.R + 1)))
                ref, rm, _, _ = self._window(cj, rs, self.R); ref_valid = 1.
        return dict(q=q, mask=m, feat=f, text=text, ttime=ttime, tmask=tmask, spk=spk, spk_valid=np.float32(spk_valid),
                    source=np.int64(source), tids=tids, tidm=tidm, titime=titime, tseg=tseg, tiseg=tiseg, ref=ref, ref_valid=np.float32(ref_valid), clip=np.int64(ci), start=np.int64(s))


def collate(batch):
    return {k: torch.as_tensor(np.stack([b[k] for b in batch])) for k in batch[0]}


def fixed_windows(view: View, frames=256, n=None):
    """Deterministic evaluation windows: every clip from its start (dev set: 380 items)."""
    out = [(i, 0) for i in range(len(view.clips))]
    return out[:n] if n else out

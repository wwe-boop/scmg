"""11-08 EX002 duplex ladder L1 (docs/criteria/EX002_DUPLEX_LADDER_1107.md REVISION): data-v36_1-tail = data-v36_1 with post-speech
tails appended to the items they continue, and a YIELD control token at the speech offset.
Tails: the 757 retargeted EX002 tails (ex002-interrupt-1107/tails/seg, frozen release06 retargeter, TRAIN-split speakers minus the
dev speakers), half A = crc32(recording) % 2 == 0 (half B stays the RKD / rest reference, never used here).
Matching: the tail's 25 frames before its offset must occur bitwise (max |dq| == 0) in an item of the same clip, and every item frame
after that point must equal the tail's continuation; the item is then extended with the rest of the tail (GT motion of the
silence); appended feat rows = the raw feat89 of digital silence (log 1e-6 mel, zero prosody / valid); text unchanged (all words end
before the offset). Clips with an extended item are re-laid out contiguously at the end of the arrays (the View needs contiguous
clips); row-aligned side arrays (GLITCH_MASK, INTENT_FRAMES) are extended the same way (tail rows: glitch 0, intent -1).
Leakage asserts (fail the build): every matched item is in the view's TRAIN split; no dev speaker (2_scott, 13_lu); no clip of the
dev interruption benchmark (interrupts_dev_ex002.json) appears in the output train.json.
Writes OUT/ (train.json, train_q.npy, train_feat.npy, GLITCH_MASK.npy, symlinks of the rest), OUT_INTENT/INTENT_FRAMES.npy,
OUT_TOK/ (CTL.json with the YIELD tokens only + PREP.json), BUILD.json (counts used / dropped by reason).
    python scripts/ex002/build_tail_view_ex002.py --out V --out-intent I --out-tok T"""
import argparse, json, os, zlib, glob
from collections import defaultdict, Counter
from pathlib import Path
import hashlib
import numpy as np

B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'
ap = argparse.ArgumentParser()
ap.add_argument('--view', default=D + '/train-v1-20261101/views/data-v36_1')
ap.add_argument('--tails', default=B + '/ex002-interrupt-1107/tails')
ap.add_argument('--intent', default=B + '/models/intent-frames-v36_1')
ap.add_argument('--glitch', default=D + '/scmg/glitch-audit/GLITCH_MASK.npy')
ap.add_argument('--interrupts', default=B + '/ex002-interrupt-1107/interrupts_dev_ex002.json')
ap.add_argument('--out', required=True); ap.add_argument('--out-intent', required=True); ap.add_argument('--out-tok', required=True)
ap.add_argument('--yield-b', type=float, default=1.4, help='YIELD token interval end = t + this (s)')
a = ap.parse_args()
V = Path(a.view); O = Path(a.out); OI = Path(a.out_intent); OT = Path(a.out_tok)
for d in (O, OI, OT): d.mkdir(parents=True, exist_ok=True)
DEV_SPK = {'2_scott', '13_lu'}
items = json.loads((V / 'train.json').read_text()); split = json.loads((V / 'SPLIT.json').read_text())
train_ids = set(split['train_ids']); val_ids = set(split['val_ids'])
q = np.load(V / 'train_q.npy', mmap_mode='r'); f = np.load(V / 'train_feat.npy', mmap_mode='r')
gl = np.load(a.glitch, mmap_mode='r'); it_ = np.load(Path(a.intent) / 'INTENT_FRAMES.npy', mmap_mode='r')
assert len(gl) == len(q) == len(it_) == len(f), 'row-aligned arrays differ in length'
by_clip = defaultdict(list)
for k, x in enumerate(items): by_clip[x['clip']].append(k)
SIL = np.concatenate([np.full(80, np.log(1e-6)), np.zeros(f.shape[1] - 80)]).astype(f.dtype)
idx = [json.loads(l) for fn in sorted(glob.glob(str(Path(a.tails) / 'index.*of28.jsonl'))) for l in open(fn)]
why = Counter(); ext = {}                                  # item index -> (extra q rows, t_off item s)
dev_clips = {r['clip'] for r in json.loads(Path(a.interrupts).read_text())['interrupts']}
idx = [r for r in idx if 'id' in r]                       # rows without an id = whole recordings rejected by the retargeter (counted)
why['recording_rejected_by_retargeter_rows'] = sum(1 for fn in sorted(glob.glob(str(Path(a.tails) / 'index.*of28.jsonl'))) for l in open(fn) if '"id"' not in l)
for r in idx:
    if zlib.crc32(r['recording'].encode()) % 2 != 0: why['half_B'] += 1; continue
    fn = Path(a.tails) / 'seg' / f"{r['id']}.npz"
    if not fn.exists(): why['A_not_retargeted'] += 1; continue
    if r['speaker'] in DEV_SPK: why['A_dev_speaker'] += 1; continue
    clip = 'beat2-%s-%s' % ('zh' if r['lang'] == 'chinese' else 'en', r['recording'])
    if clip not in by_clip: why['A_recording_not_in_view'] += 1; continue
    z = np.load(fn); t = z['q34'].astype(np.float32); off = int(z['off']); pre = t[off - 25:off]
    hit = None
    for k in by_clip[clip]:
        x = items[k]; Q = np.asarray(q[x['offset']:x['offset'] + x['frames']], np.float32)
        for s in range(0, len(Q) - 25 + 1):
            if np.array_equal(Q[s:s + 25], pre): hit = (k, s); break
        if hit: break
    if hit is None: why['A_no_exact_match'] += 1; continue
    k, s = hit; x = items[k]; Q = np.asarray(q[x['offset']:x['offset'] + x['frames']], np.float32)
    after = Q[s + 25:]; cont = t[off:off + len(after)]
    if len(cont) < len(after) or not np.array_equal(after, cont): why['A_item_continues_differently'] += 1; continue
    extra = t[off + len(after):]
    if len(extra) < 25: why['A_tail_too_short'] += 1; continue
    if x['id'] not in train_ids: why['A_item_not_in_train'] += 1; continue          # leakage guard (counted, then asserted below)
    if k in ext: why['A_duplicate_item'] += 1; continue
    ext[k] = (extra, (s + 25) / 50.); why['A_used'] += 1
# leakage asserts
for k in ext:
    x = items[k]
    assert x['id'] in train_ids and x['id'] not in val_ids, f"matched item {x['id']} is not a TRAIN item"
    assert x.get('speaker') not in DEV_SPK, f"matched item {x['id']} is from a dev speaker"
assert why['A_item_not_in_train'] == 0, f"{why['A_item_not_in_train']} matched items outside the train split"
# re-layout the clips holding an extended item at the end of the arrays
clips_x = sorted({items[k]['clip'] for k in ext})
n0 = len(q); add_q, add_f, add_g, add_i = [], [], [], []; pos = n0; new = [dict(x) for x in items]
for c in clips_x:
    for k in sorted(by_clip[c], key=lambda k: items[k]['offset']):
        x = items[k]; r0, r1 = x['offset'], x['offset'] + x['frames']
        qq, ff, gg, ii = (np.asarray(arr[r0:r1]) for arr in (q, f, gl, it_))
        if k in ext:
            e = ext[k][0]; n = len(e)
            qq = np.concatenate([qq, e.astype(qq.dtype)]); ff = np.concatenate([ff, np.repeat(SIL[None], n, 0)])
            gg = np.concatenate([gg, np.zeros(n, gg.dtype)]); ii = np.concatenate([ii, np.full((n,) + ii.shape[1:], -1, ii.dtype)])
        add_q.append(qq); add_f.append(ff); add_g.append(gg); add_i.append(ii)
        new[k]['offset'] = pos; new[k]['frames'] = len(qq); pos += len(qq)
assert not ({x['clip'] for x in new if x['id'] in train_ids} & dev_clips), 'a dev interruption clip is in the training view'
def cat(base, adds, path):
    out = np.lib.format.open_memmap(path, mode='w+', dtype=base.dtype, shape=(pos,) + base.shape[1:])
    for s in range(0, n0, 1 << 20): out[s:min(n0, s + (1 << 20))] = base[s:min(n0, s + (1 << 20))]
    p = n0
    for x in adds: out[p:p + len(x)] = x; p += len(x)
    out.flush(); del out
cat(q, add_q, O / 'train_q.npy'); cat(f, add_f, O / 'train_feat.npy'); cat(gl, add_g, O / 'GLITCH_MASK.npy')
cat(it_, add_i, OI / 'INTENT_FRAMES.npy')
(O / 'train.json').write_text(json.dumps(new))
for n in os.listdir(V):
    if n not in ('train.json', 'train_q.npy', 'train_feat.npy') and not (O / n).exists(): os.symlink(V / n, O / n)
for n in os.listdir(a.intent):
    if n != 'INTENT_FRAMES.npy' and not (OI / n).exists(): os.symlink(Path(a.intent) / n, OI / n)
from importlib import util as _u
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg.ctl import YIELD
tok = {items[k]['id']: [dict(type=int(YIELD), t=float(t0), a=float(t0), b=float(t0 + a.yield_b), split='train')] for k, (e, t0) in ext.items()}
(OT / 'CTL.json').write_text(json.dumps(tok))
(OT / 'PREP.json').write_text(json.dumps(dict(split_sha256=hashlib.sha256((V / 'SPLIT.json').read_bytes()).hexdigest(), view=str(O),
                                              n_items=len(tok), rule='YIELD at the speech offset of every item extended with a half-A tail')))
info = dict(counts=dict(why), n_tails_index=len(idx), extended_items=len(ext), clips_relaid=len(clips_x), rows_added=pos - n0,
            tail_seconds=float(sum(len(e) for e, _ in ext.values()) / 50.), half_rule='crc32(recording) % 2 == 0 -> A (training)')
(O / 'BUILD.json').write_text(json.dumps(info, indent=1)); print('TAIL_VIEW_DONE', json.dumps(info))

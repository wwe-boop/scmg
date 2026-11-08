"""11-08 audit A3 (refine-logs/EVAL_AUDIT_1108.md): ONE script asserting that no dev / test recording reaches any EX002 training input.
Held-out sets come from IDS ONLY (no motion / audio of the sealed test split is opened):
  manifests v3_5 (BEAT2 zh / en / additional) and v3_6 (bilibili, embody_*, semsynth_r7p1, SPEED_BEAT2_V35): every record whose split is
  not 'train' -> held-out recording (clip_id before '#' / '__'); dev speakers = speakers of the dev records + the fixed {2_scott, 13_lu};
  + the base view's val_ids, the dev interruption benchmark clips, the fresh dev-speaker tails.
Training inputs checked (every input of the current EX002 training commands: r4 base, enc / sfscope rows, L1, control-token module):
  views data-v36_1, data-v36_1-tail (SPLIT train_ids + the train.json rows they select), control tokens tokens/ + tokens-duplex/ (CTL.json
  items + event split), intent frames (PREP split sha), BAgger data (every sample's clip / item_ids), the human tail index (half A used for
  the tail view), B's libraries (HOLD_STATS / HOLD_POOL / HAND_LIBRARY / TARGET_POOLS / HAND_SHAPES: no per-entry ids -> their recorded
  source must be tokens/CTL.json or train frames, frames_scanned <= train frames), STATS_v36_1 (rows not recorded -> reported), the
  semantic holdout manifest.
    python leak_check_ex002.py [--out LEAK_CHECK.json]      exit 1 on any FAIL"""
import argparse, glob, hashlib, json, sys, zlib
from collections import Counter
from pathlib import Path
DS = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'; B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'
VW = DS + '/train-v1-20261101/views'
ap = argparse.ArgumentParser(); ap.add_argument('--out', default=None); a = ap.parse_args()
res, fails = {}, []


def rec(cid):
    """recording key: BEAT2 -> the bare take name (12_zhao_2_100_100); others -> clip id before '#' / '__'."""
    c = str(cid).split('#')[0].split('__')[0]
    for p in ('beat2-zh-', 'beat2-en-'):
        if c.startswith(p): return c[len(p):]
    return c


def check(name, ok, **info):
    res[name] = dict(ok=bool(ok), **info); print(('PASS ' if ok else 'FAIL ') + name, json.dumps(info, ensure_ascii=False)[:600], flush=True)
    if not ok: fails.append(name)


# ---- held-out ids
held, held_split, dev_spk, test_spk, n_rec = set(), Counter(), {'2_scott', '13_lu'}, set(), Counter()
mans = sorted(glob.glob(DS + '/unified-ex002-manifest-v3_5/beat2_*.jsonl')) + [DS + f'/unified-ex002-manifest-v3_6/{n}.jsonl' for n in
                                                                                ('bilibili', 'embody_dyadic', 'embody_speech', 'semsynth_r7p1')]
train_rec_man = set()
for fn in mans:
    for l in open(fn):
        r = json.loads(l); s = r.get('split'); k = rec(r['clip_id']); n_rec[(Path(fn).stem, s)] += 1
        spk = '_'.join(k.split('_')[:2]) if str(r.get('source', '')).startswith('beat2') else None   # BEAT2 take name = <id>_<name>_...
        if s == 'train': train_rec_man.add(k); continue
        held.add(k); held_split[s] += 1
        if s == 'dev' and spk: dev_spk.add(str(spk))
        if s == 'test' and spk: test_spk.add(str(spk))
res['manifest_records'] = {f'{a_}:{b_}': v for (a_, b_), v in sorted(n_rec.items())}
base = json.load(open(VW + '/data-v36_1/SPLIT.json'))
held |= {rec(i) for i in base['val_ids']}
ints = json.load(open(B + '/ex002-interrupt-1107/interrupts_dev_ex002.json'))['interrupts']; held |= {rec(r['clip']) for r in ints}
fresh = {Path(f).stem.split('@')[0] for f in glob.glob(B + '/ex002-interrupt-1107/tails_fresh/seg/*.npz')}; held |= fresh
both = held & train_rec_man
check('manifest_train_vs_heldout_disjoint', not both, n_heldout_recordings=len(held), heldout_records_by_split=dict(held_split), overlap=sorted(both)[:10],
      dev_speakers=sorted(dev_spk), n_test_speakers=len(test_spk), fresh_tail_recordings=len(fresh))

# ---- views
def view_check(v):
    V = Path(VW) / v; sp = json.load(open(V / 'SPLIT.json')); tr = set(sp['train_ids'])
    items = json.load(open(V / 'train.json')); sel = [it for it in items if it['id'] in tr]
    recs = {rec(it['clip']) for it in sel}
    bad = sorted(recs & held); spk = Counter(it['speaker'] for it in sel if str(it['speaker']) in dev_spk)
    hint = Counter(it.get('split_hint') for it in sel)
    tsp = Counter(str(it['speaker']) for it in sel if str(it['speaker']) in test_spk and str(it['speaker']) not in dev_spk)
    check(f'view:{v}', not bad and not spk and set(hint) <= {'train'}, n_train_ids=len(tr), n_train_rows_selected=len(sel), n_recordings=len(recs),
          heldout_recordings_in_train=bad[:10], dev_speaker_items=dict(spk), split_hint=dict(hint),
          info_train_items_of_test_speakers=int(sum(tsp.values())), split_sha256=hashlib.sha256((V / 'SPLIT.json').read_bytes()).hexdigest())
    return tr, sel, hashlib.sha256((V / 'SPLIT.json').read_bytes()).hexdigest()


tr_base, sel_base, sha_base = view_check('data-v36_1'); tr_tail, sel_tail, sha_tail = view_check('data-v36_1-tail')
new = tr_tail - tr_base
check('tail_view_new_ids_not_heldout', not ({rec(i) for i in new} & held), new_ids=len(new), dropped_ids=len(tr_base - tr_tail))

# ---- control tokens
for d, tr, sha in (('tokens', tr_base, sha_base), ('tokens-duplex', tr_tail, sha_tail)):
    D = Path(B) / 'control-token-1105/ex002' / d; prep = json.load(open(D / 'PREP.json')); ctl = json.load(open(D / 'CTL.json'))
    ks = [k for k in ctl if isinstance(ctl[k], list)]; ev = Counter(e.get('split') for k in ks for e in ctl[k])
    bad = sorted({rec(k) for k in ks} & held); notr = [k for k in ks if k not in tr]
    check(f'ctl:{d}', not bad and not notr and set(ev) <= {'train'} and prep.get('split_sha256') == sha, n_items=len(ks), event_split=dict(ev),
          items_not_in_train_ids=notr[:5], heldout_recordings=bad[:5], prep_sha_matches_view=prep.get('split_sha256') == sha)

# ---- intent frames (row-aligned with the view; -1 = no label): no labelled row inside a val item
import numpy as np
for d, v in (('intent-frames-v36_1', 'data-v36_1'), ('intent-frames-v36_1-tail', 'data-v36_1-tail')):
    D = Path(B) / 'models' / d; p = json.load(open(D / 'PREP.json')); it = np.load(D / 'INTENT_FRAMES.npy', mmap_mode='r')
    items = json.load(open(Path(VW) / v / 'train.json')); val = set(json.load(open(Path(VW) / v / 'SPLIT.json'))['val_ids'])
    vr = [(x['offset'], x['frames']) for x in items if x['id'] in val]; lab = int(sum(int((np.asarray(it[o:o + n]) >= 0).sum()) for o, n in vr))
    nq = len(np.load(Path(VW) / v / 'train_q.npy', mmap_mode='r'))
    check(f'intent:{d}', lab == 0 and len(it) == nq, labelled_rows_in_val_items=lab, n_val_items=len(vr), rows=len(it),
          labels_from=p.get('manifest'), note='labels only from the semsynth_r7p1 manifest clips (all split train: manifest check above)')

# ---- BAgger
bj = sorted(glob.glob(B + '/r4-1104/bagger-pai/bagger-data/[0-9]*.json')); bad, n_items, src = set(), 0, Counter()
for fn in bj:
    s = json.load(open(fn)); src[s.get('original_source')] += 1
    for k in [s['clip']] + list(s.get('item_ids', [])):
        n_items += 1
        if rec(k) in held: bad.add(rec(k))
check('bagger-data', not bad, n_samples=len(bj), n_ids=n_items, original_source=dict(src), heldout_recordings=sorted(bad)[:10])

# ---- human tails (half A -> tail view)
idx = [json.loads(l) for fn in sorted(glob.glob(B + '/ex002-interrupt-1107/tails/index.*of28.jsonl')) for l in open(fn)]
idx = [r for r in idx if 'id' in r]; A = [r for r in idx if zlib.crc32(r['recording'].encode()) % 2 == 0]
badA = sorted({r['recording'] for r in A if r['recording'] in held or r['speaker'] in dev_spk})
inview = {rec(it['clip']) for it in sel_tail}
usedA = [r for r in A if r['recording'] in inview]
check('tails_halfA_used', not {r['recording'] for r in usedA} & held, n_index=len(idx), n_halfA=len(A), n_halfA_recording_in_tail_view=len(usedA),
      halfA_heldout_or_dev_speaker=badA[:10], index_split=dict(Counter(r.get('split') for r in idx)),
      note='halfA rows of held-out recordings are dropped by build_tail_view (recording not in the train view / dev speaker); used rows checked here')

# ---- B's libraries
T = Path(B) / 'control-token-1105/ex002/tokens'; libs = {}
hs = json.load(open(T / 'HOLD_STATS_TRAIN.json'))['source']; libs['HOLD_STATS_TRAIN'] = hs.get('ctl') == str(T / 'CTL.json') and hs.get('view', '').endswith('data-v36_1')
libs['HOLD_POOL_DEICTIC_TRAIN'] = 'train' in json.load(open(T / 'HOLD_POOL_DEICTIC_TRAIN.json'))['source']
tr_frames = sum(int(it['frames']) for it in sel_base); hl = json.load(open(T / 'HAND_LIBRARY_TRAIN.json'))
libs['HAND_LIBRARY_TRAIN'] = int(hl['frames_scanned']) <= tr_frames
libs['HAND_SHAPES'] = 'train' in json.load(open(T / 'HAND_SHAPES.json')).get('rule', '')
check('B_libraries', all(libs.values()), checks=libs, hand_library_frames_scanned=int(hl['frames_scanned']), train_frames_base=tr_frames,
      note='HOLD_POOL_TRAIN / TARGET_POOLS carry no ids or source field: built from tokens/CTL.json train events (checked above) by B; not re-derivable here')

# ---- stats / holdout manifest
st = json.load(open(DS + '/scmg/STATS_v36_1.json'))
res['STATS_v36_1'] = dict(keys=list(st), note='no row provenance stored; data.compute_stats default = every row of the view incl. the 380 val rows '
                          '(mean / std normalisation only)')
hm = json.load(open(B + '/semantic-recognizer-v1/manifest-n200/MANIFEST.json'))['entries']; hb = sorted({rec(e['clip_id']) for e in hm} & held)
res['semantic_holdout_manifest'] = dict(n=len(hm), heldout_recordings=hb[:10], note='clips marked by View.holdout (data.py), not a training input')
res['FAILS'] = fails
if a.out: Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
print('LEAK_CHECK', 'OK' if not fails else 'FAIL ' + ','.join(fails)); sys.exit(1 if fails else 0)

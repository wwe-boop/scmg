"""Event-level dataset for the EX002 semantic-gesture recogniser (semantic-recognizer-v1).

Source of truth (verified 2026-11-03):
  * semsynth_r7p1 manifest rows (unified-ex002-manifest-v3_6/semsynth_r7p1.jsonl) carry `events` = the semantic gestures
    that were actually executed in the synthetic motion, with `t_on/t_off` (s, clip time) = the protected gesture CORE
    (MOTION.json specs[].frames, e.g. point:right frames [104,124] -> t_on 2.08 t_off 2.48), key == intended_key.
  * scmg/prep_intent.py maps the same executed intents onto the Qwen2.5 token holding the anchor's first character
    (INTENT_LABEL.npy: -1 unlabelled, 0 none, >=2 VOCAB class).  Every event is cross-linked to its labelled token here.
  * Motion = view train_q.npy (rad, 50 Hz); a manifest clip = the view items with the same clip_id, contiguous (pieces).

Samples (one per window, centre frame c, clip-relative):
  pos        centre = core midpoint of an executed event whose class is kept       label = class
  neg_syn    semsynth clip, no executed core within +-NEG_GAP s of the centre        label = none
  neg_human  random centre in a human-recorded train clip (beat2/streamer/embody...)  label = none
Split by GROUP (manifest split_group U semsynth pair id; human: conv_key / clip): no clip (nor rhythm variant of the same
sentence) in both train and test.  Window = W frames centred on c, stored with +-M frames margin for jitter, edge-held.

Outputs: <out>/INDEX.json + INDEX.npz (small: on CPFS) and <cache>/WINDOWS_f16.npy [N, W+2M, 34] rad (local, large).
    python scripts/build_intent_dataset.py --out <cpfs dir> --cache ~/bulk/semrec-cache
"""
import argparse, hashlib, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scmg.common import P
from scmg.data import VIEW

MAN = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/unified-ex002-manifest-v3_6/semsynth_r7p1.jsonl'
TOK = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/tok-qwen25-v36_1'
INTENT = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/intent-qwen25-v36_1'
LM = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/models/scmg/Qwen2.5-0.5B'
W, M = 160, 25            # 3.2 s window (covers prep bridge ~0.8-1.6 s before a short core and the core itself), +-0.5 s margin
NEG_GAP = 1.0             # s: negative centre at least this far from any executed core
HUMAN_SOURCES = ('beat2_zh', 'beat2_en', 'beat2_en_additional', 'beat2_zh_additional', 'streamer', 'embody_speech',
                 'embody_dyadic', 'bilibili')


def group_test(g, frac, seed):
    h = int(hashlib.sha1(f'{seed}:{g}'.encode()).hexdigest()[:8], 16) / 0xffffffff
    return h < frac


def cut(q, a, L, c, W=W, M=M):
    """Window of W+2M frames centred on clip frame c from clip [a, a+L) of the flat array q; edge-held outside the clip."""
    s = c - W // 2 - M; idx = np.clip(np.arange(s, s + W + 2 * M), 0, L - 1) + a
    return np.asarray(q[idx], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--cache', required=True)
    ap.add_argument('--manifest', default=MAN); ap.add_argument('--view', default=VIEW)
    ap.add_argument('--test-frac', type=float, default=0.2); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-total', type=int, default=30); ap.add_argument('--min-test', type=int, default=5)
    ap.add_argument('--neg-per-clip', type=int, default=2); ap.add_argument('--n-human', type=int, default=15000)
    a = ap.parse_args()
    out, cache = Path(a.out), Path(os.path.expanduser(a.cache)); out.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    v = Path(P(a.view))
    vocab = json.loads((v / 'VOCAB.json').read_text()); inv = {i: k for k, i in vocab.items()}
    items = json.loads((v / 'train.json').read_text())
    sp = json.loads((v / 'SPLIT.json').read_text()); tr_ids, va_ids = set(sp['train_ids']), set(sp['val_ids'])
    q = np.load(v / 'train_q.npy', mmap_mode='r')
    tok_index = json.loads((Path(P(TOK)) / 'TOK_INDEX.json').read_text())
    tok_ids = np.load(Path(P(TOK)) / 'TOK_IDS.npy')
    lab = np.load(Path(P(INTENT)) / 'INTENT_LABEL.npy'); conf = np.load(Path(P(INTENT)) / 'INTENT_CONF.npy')
    st = Counter()
    # ---- view clips (by clip_id for semsynth, by 'clip' for human) ----
    by_cid = defaultdict(list); human = defaultdict(list)
    for it in items:
        if it['id'] in va_ids: st['val_items_' + str(it.get('data_source'))] += 1
        if it['id'] not in tr_ids: continue
        if it.get('data_source') == 'semsynth_r7p1': by_cid[it['clip_id']].append(it)
        elif it.get('data_source') in HUMAN_SOURCES: human[it['clip']].append(it)
    st['semsynth_items_in_val'] = sum(1 for it in items if it['id'] in va_ids and str(it.get('data_source')).startswith('semsynth'))
    for xs in by_cid.values(): xs.sort(key=lambda x: x['offset'])
    # ---- manifest events ----
    rows = []                                     # one per window
    clipinfo = {}
    man_path = P(a.manifest)
    loc = Path(os.path.expanduser('~/bulk/semrec-cache/semsynth_r7p1.jsonl'))
    if loc.exists() and loc.stat().st_size == os.path.getsize(man_path): man_path = str(loc)
    tok_check = Counter()
    for l in open(man_path):
        r = json.loads(l); cid = r['clip_id']; xs = by_cid.get(cid)
        if not xs: st['manifest_clip_not_in_train_view'] += 1; continue
        A = xs[0]['offset']; L = sum(x['frames'] for x in xs)
        if L != r['n_frames'] or any(xs[i + 1]['offset'] != xs[i]['offset'] + xs[i]['frames'] for i in range(len(xs) - 1)):
            st['frames_mismatch'] += 1; continue
        pair = cid.replace('semsynth-r7p1-', '').split('__')[0]
        group = 'semsynth:' + (r.get('split_group') or pair)
        ev = [e for e in r.get('events', []) if e.get('executed')]
        st['events_executed'] += len(ev); st['events_not_executed'] += sum(1 for e in r.get('events', []) if not e.get('executed'))
        st['events_key_ne_intended'] += sum(1 for e in ev if e['key'] != e.get('intended_key'))
        # cross-link each in-vocab event to its labelled LM token (items cover consecutive frame ranges of the clip)
        evs = sorted([e for e in ev if e['key'] in vocab], key=lambda e: e['t_on'])
        st['events_out_of_vocab'] += sum(1 for e in ev if e['key'] not in vocab)
        links = [None] * len(evs)
        toks = []
        for it in xs:
            ix = tok_index.get(it['id'])
            if ix is None: continue
            r0, n = ix
            for j in range(r0, r0 + n):
                if lab[j] > 1: toks.append((it['id'], j - r0, j, int(lab[j]), float(conf[j])))
        if [t[3] for t in toks] == [vocab[e['key']] for e in evs]:
            links = toks; tok_check['class_sequence_match'] += 1
        else:
            tok_check['class_sequence_mismatch'] += 1
            # fall back: greedy match by class in order
            used = set()
            for k, e in enumerate(evs):
                for t_i, t in enumerate(toks):
                    if t_i not in used and t[3] == vocab[e['key']]: links[k] = t; used.add(t_i); break
        clipinfo[cid] = dict(a=int(A), L=int(L), group=group, pair=pair, n_events=len(ev), text=r['text']['plain'],
                             rhythm=r.get('rhythm_source'), items=[x['id'] for x in xs],
                             events=[dict(key=e['key'], t_on=e['t_on'], t_off=e['t_off']) for e in ev])
        for k, e in enumerate(evs):
            c = int(round((e['t_on'] + e['t_off']) / 2 * 50))
            t = links[k]
            rows.append(dict(kind='pos', clip=cid, group=group, c=c, key=e['key'], t_on=e['t_on'], t_off=e['t_off'],
                             tok_item=t[0] if t else None, tok_pos=t[1] if t else None, tok_row=t[2] if t else None,
                             tok_id=int(tok_ids[t[2]]) if t else None, votes_conf=t[4] if t else None))
        # negatives in the same synthetic clip: rhythm-base motion away from every executed core
        cores = np.array([[e['t_on'], e['t_off']] for e in ev]) if ev else np.zeros((0, 2))
        lo, hi = 40, L - 40
        if hi > lo:
            cand = np.arange(lo, hi, 5)
            if len(cores):
                t = cand / 50.
                d = np.maximum(cores[None, :, 0] - t[:, None], t[:, None] - cores[None, :, 1]).min(1)   # >0 outside
                cand = cand[d >= NEG_GAP]
            for c in rng.permutation(cand)[:a.neg_per_clip]:
                rows.append(dict(kind='neg_syn', clip=cid, group=group, c=int(c), key='none'))
    st['tok_link'] = dict(tok_check)
    # ---- human negatives ----
    hc = sorted(human); rng.shuffle(hc)
    nh = 0
    for cl in hc:
        if nh >= a.n_human: break
        xs = sorted(human[cl], key=lambda x: x['offset']); L = sum(x['frames'] for x in xs)
        if L < 200: continue
        A = xs[0]['offset']
        if any(xs[i + 1]['offset'] != xs[i]['offset'] + xs[i]['frames'] for i in range(len(xs) - 1)): st['human_noncontig'] += 1; continue
        group = 'human:' + str(xs[0].get('conv_key') or cl)
        clipinfo['human:' + cl] = dict(a=int(A), L=int(L), group=group, source=xs[0].get('data_source'), items=[x['id'] for x in xs])
        c = int(rng.integers(40, L - 40))
        rows.append(dict(kind='neg_human', clip='human:' + cl, group=group, c=c, key='none', source=xs[0].get('data_source'))); nh += 1
    # ---- split by group, class filter ----
    for r in rows: r['split'] = 'test' if group_test(r['group'], a.test_frac, a.seed) else 'train'
    cnt = Counter(r['key'] for r in rows if r['kind'] == 'pos')
    cnt_te = Counter(r['key'] for r in rows if r['kind'] == 'pos' and r['split'] == 'test')
    keep = sorted([k for k in cnt if cnt[k] >= a.min_total and cnt_te[k] >= a.min_test and cnt[k] - cnt_te[k] >= 20],
                  key=lambda k: -cnt[k])
    classes = ['none'] + keep; cidx = {k: i for i, k in enumerate(classes)}
    dropped = {k: cnt[k] for k in cnt if k not in cidx}
    rows = [r for r in rows if r['key'] in cidx]
    for r in rows: r['y'] = cidx[r['key']]
    # leakage check: no group / clip in both splits
    gs = defaultdict(set)
    for r in rows: gs[r['clip']].add(r['split'])
    assert all(len(s) == 1 for s in gs.values())
    # ---- windows ----
    X = np.lib.format.open_memmap(cache / 'WINDOWS_f16.npy', mode='w+', dtype=np.float16, shape=(len(rows), W + 2 * M, 34))
    for i, r in enumerate(rows):
        ci = clipinfo[r['clip']]; X[i] = cut(q, ci['a'], ci['L'], r['c']).astype(np.float16)
    X.flush()
    # anchor-token text (verification): decode the most common labelled tokens of each kept class
    anchors = {}
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(LM)
        per = defaultdict(Counter)
        for r in rows:
            if r['kind'] == 'pos' and r['tok_id'] is not None:
                per[r['key']][r['tok_id']] += 1
        anchors = {k: [(tk.decode([t]), n) for t, n in per[k].most_common(6)] for k in keep}
    except Exception as e:
        anchors = {'error': repr(e)}
    split_cnt = {s: Counter(r['key'] if r['kind'] == 'pos' else r['kind'] for r in rows if r['split'] == s) for s in ('train', 'test')}
    meta = dict(version='semantic-recognizer-v1', window_frames=W, margin_frames=M, fps=50, neg_gap_s=NEG_GAP,
                test_frac=a.test_frac, seed=a.seed, min_total=a.min_total, min_test=a.min_test, min_train=20,
                classes=classes, class_vocab_id={k: vocab[k] for k in keep}, dropped_classes=dropped,
                stats=st, n_rows=len(rows), split_counts=split_cnt, anchor_tokens=anchors,
                view=str(a.view), manifest=a.manifest, intent=INTENT, tok_cache=TOK, windows_cache=str(cache / 'WINDOWS_f16.npy'))
    (out / 'INDEX.json').write_text(json.dumps(dict(meta=meta, rows=rows, clips=clipinfo), ensure_ascii=False, default=int))
    (out / 'DATASET_STATS.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=int))
    (cache / 'INDEX.json').write_text(json.dumps(dict(meta=meta, rows=rows, clips=clipinfo), ensure_ascii=False, default=int))
    print(json.dumps(dict(stats=st, classes=len(classes), dropped=len(dropped), rows=len(rows),
                          train=sum(split_cnt['train'].values()), test=sum(split_cnt['test'].values())), default=int))


if __name__ == '__main__':
    main()

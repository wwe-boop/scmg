"""Frame-level intent track of the training view (11-03, Claude): one int16 per row of train_q.npy.
  -1  no intent information (human recordings, TTS-only items, frames outside semsynth clips)  -> no intent loss
   0  semsynth frame outside every executed gesture                                            -> class "none"
  >=2 VOCAB class of the executed semantic gesture whose core [t_on - lead, t_off] covers the frame
Source of truth = the semsynth_r7p1 manifest `events` (executed ones; t_on/t_off = protected gesture core, clip time), mapped
onto the view exactly like scripts/build_intent_dataset.py (manifest clip_id -> contiguous view items, first offset = frame 0).
    python -m scmg.prep_intent_frames --out <dir> [--lead 0.3]
Outputs INTENT_FRAMES.npy, PREP.json (counts, class histogram)."""
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .common import P, FPS
from .data import VIEW

MAN = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/unified-ex002-manifest-v3_6/semsynth_r7p1.jsonl'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', default=VIEW); ap.add_argument('--manifest', default=MAN); ap.add_argument('--out', required=True)
    ap.add_argument('--lead', type=float, default=0.3, help='s before the core that already counts as the gesture (preparation)')
    a = ap.parse_args()
    v = Path(P(a.view)); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    vocab = json.loads((v / 'VOCAB.json').read_text())
    items = json.loads((v / 'train.json').read_text())
    nq = np.load(v / 'train_q.npy', mmap_mode='r').shape[0]
    lab = np.full(nq, -1, np.int16)
    by_cid = defaultdict(list)
    for it in items:
        if it.get('data_source') == 'semsynth_r7p1' and it.get('clip_id'): by_cid[it['clip_id']].append(it)
    for xs in by_cid.values(): xs.sort(key=lambda x: x['offset'])
    st = Counter(); hist = Counter()
    for line in open(P(a.manifest)):
        r = json.loads(line); xs = by_cid.get(r['clip_id'])
        if not xs: st['clip_not_in_view'] += 1; continue
        A = xs[0]['offset']; L = sum(x['frames'] for x in xs)
        if L != r['n_frames'] or any(xs[i + 1]['offset'] != xs[i]['offset'] + xs[i]['frames'] for i in range(len(xs) - 1)):
            st['frames_mismatch'] += 1; continue
        lab[A:A + L] = 0; st['clips'] += 1; st['frames_known'] += L
        for e in r.get('events', []):
            if not e.get('executed'): continue
            if e['key'] not in vocab: st['events_out_of_vocab'] += 1; continue
            f0 = max(0, int(round((e['t_on'] - a.lead) * FPS))); f1 = min(L, int(round(e['t_off'] * FPS)) + 1)
            if f1 <= f0: st['events_empty'] += 1; continue
            lab[A + f0:A + f1] = vocab[e['key']]; st['events'] += 1; hist[e['key']] += 1; st['frames_intent'] += f1 - f0
    np.save(out / 'INTENT_FRAMES.npy', lab)
    meta = dict(view=str(v), manifest=a.manifest, lead_s=a.lead, rows=int(nq), stats=dict(st), n_classes_seen=len(hist),
                top_classes=hist.most_common(20),
                train_json_sha256=hashlib.sha256((v / 'train.json').read_bytes()).hexdigest())
    (out / 'PREP.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps(dict(st)), 'classes', len(hist))


if __name__ == '__main__':
    main()

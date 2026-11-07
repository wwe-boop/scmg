"""11-07 EX002 interruption benchmark (3b), step 1: the utterance table of the BEAT2 items of data-v36_1 in BEAT2 take time, in the
format of scripts/interrupt/human_gaps.py --utt (lang chinese / english, recording = BEAT2 take, start / end = speech span of the
item in take seconds = clip_start_s + item speech [s0, s1], speaker, split train / val (SPLIT.json), stem = item id).
    python scripts/ex002/ex002_utterances.py --view V --out utterances_ex002.json"""
import argparse, json
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
v = Path(a.view); items = json.loads((v / 'train.json').read_text()); sp = json.loads((v / 'SPLIT.json').read_text())
tr, va = set(sp['train_ids']), set(sp['val_ids'])
rows = []
for it in items:
    if not str(it.get('source', '')).startswith('beat2'): continue
    split = 'train' if it['id'] in tr else 'val' if it['id'] in va else None
    if split is None or it.get('text_null'): continue
    rec = it['clip'].split('-', 2)[2] if it['clip'].startswith('beat2-') else it['clip']
    s0, s1 = it['speech']
    rows.append(dict(stem=it['id'], lang='chinese' if it['lang'] == 'zh' else 'english', recording=rec, speaker=it['speaker'], split=split,
                     start=float(it['clip_start_s']) + float(s0), end=float(it['clip_start_s']) + float(s1), view_frames=int(it['frames'])))
Path(a.out).write_text(json.dumps(rows))
from collections import Counter
print(len(rows), Counter(r['split'] for r in rows), len({(r['lang'], r['recording']) for r in rows}), 'takes')

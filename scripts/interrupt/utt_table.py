"""Utterance table of the ECHO-G G1 data (11-05, interruption benchmark): one row per packaged clip with its BEAT2 take,
take-time span and split. Input = ECHO-G word annotations (ds/annotations/words/<stem>.json, source.start_sec/end_sec are
take times; the packaged clip ends exactly at its last word, verified: tail 0 s for all 14,987 train clips).
    python scripts/interrupt/utt_table.py --ds <echog ds> --splits <official-meta/splits> --out utterances.json
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--ds', required=True); ap.add_argument('--splits', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
rows = []
for split in ('train_drop', 'val_common'):
    for s in Path(a.splits, f'{split}.txt').read_text().split():
        d = json.loads(Path(a.ds, 'annotations/words', f'{s}.json').read_text()); so = d['source']; w = d['words']
        rows.append(dict(stem=s, split=split, lang=so['language'], recording=so['recording'], start=float(so['start_sec']),
                         end=float(so['end_sec']), last_word_end=float(so['start_sec']) + (max(x['end'] for x in w) if w else 0.),
                         words=[[float(x['start']), float(x['end'])] for x in w],
                         speaker='_'.join(so['recording'].split('_')[:2])))
Path(a.out).write_text(json.dumps(rows))
print(len(rows), 'utterances', len({r['recording'] for r in rows}), 'takes')

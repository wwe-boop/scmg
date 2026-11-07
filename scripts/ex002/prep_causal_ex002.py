"""11-07 EX002 causal-input assets (the G1 precompute_causal.py inputs, built from the EX002 view itself):
  words  <out>/words/<item id>.json  {"words": [{"text", "start", "end"}]} = the view's alignment units (alignment.json:
         zh characters / en words with onset_s / offset_s, the units the view's prosody columns were built from)
  audio  <out>/audio/<item id>.wav -> the item's manifest audio16k.wav (symlink; every item has its own file, offset_s 0)
  pstats <out>/PROSODY_STATS_<split>.json  (causal_feat.prosody_stats over 400 train items, fixed statistics of the prefix prosody)
    python scripts/ex002/prep_causal_ex002.py --view V --out O [--splits train,val] [--pstats]"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--splits', default='train,val'); ap.add_argument('--pstats', action='store_true')
a = ap.parse_args()
v = Path(a.view); o = Path(a.out); (o / 'words').mkdir(parents=True, exist_ok=True); (o / 'audio').mkdir(exist_ok=True)
items = json.loads((v / 'train.json').read_text()); sp = json.loads((v / 'SPLIT.json').read_text())
keep = set()
for s in a.splits.split(','): keep |= set(sp['train_ids' if s == 'train' else 'val_ids'])
n = dict(items=0, words=0, missing_audio=0, empty=0)
for it in items:
    if it['id'] not in keep: continue
    n['items'] += 1
    wf = o / 'words' / f"{it['id']}.json"
    if not wf.exists():
        al = json.loads((v / it['dir'] / 'alignment.json').read_text()) if (v / it['dir'] / 'alignment.json').exists() else []
        ws = [dict(text=str(u['char']), start=float(u['onset_s']), end=float(u['offset_s'])) for u in al]
        n['empty'] += int(not ws); n['words'] += len(ws)
        wf.write_text(json.dumps(dict(words=ws, source='alignment.json units of ' + it['dir'])))
    af = o / 'audio' / f"{it['id']}.wav"
    if not af.exists():
        ma = it['manifest_audio'] if isinstance(it['manifest_audio'], dict) else eval(it['manifest_audio'])
        if float(ma.get('offset_s', 0.)) != 0.: raise SystemExit(f"{it['id']}: audio offset {ma['offset_s']}")
        if not os.path.exists(ma['path']): n['missing_audio'] += 1; continue
        os.symlink(ma['path'], af)
print(json.dumps(n))
if a.pstats:
    from scmg import causal_feat as CF
    st = CF.prosody_stats(str(v), 'train', str(o / 'audio'), str(o / 'words'))
    (o / 'PROSODY_STATS_ex002_v36_1_train.json').write_text(json.dumps(st, indent=1)); print(st)

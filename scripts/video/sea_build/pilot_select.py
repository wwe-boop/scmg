"""Option-3 pilot: pick 100 whole Seamless tracks (one per participant, seed 20261107) from tars still on disk; Seamless train only;
participants of Seamless dev/test and participants already built (build/seamless/GATED.json, view-add-seamless-1103) excluded.
    python pilot_select.py <build dir> <out PILOT_TRACKS.jsonl> [n=100]"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
S = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/seamless-interaction'); B = Path(sys.argv[1]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
pid = lambda key: 'seamless-P' + key.split('_P')[-1]
held, used, cand = set(), set(), {}
for d in sorted((S / 'stream/items').iterdir()):
    tf = d / 'TRACKS.jsonl'
    if not tf.exists(): continue
    label, split, batch, tar = d.name.split('_')
    for l in open(tf):
        t = json.loads(l)
        if split != 'train': held.add(pid(t['key'])); continue
        tp = S / 'raw' / label / split / batch / f'{tar}.tar'
        if tp.exists() and t['valid'] >= 60 * 30: cand.setdefault(pid(t['key']), []).append(dict(t, split=split, batch=batch, tar=tar, tar_path=str(tp)))
for f in (B / 'seamless/GATED.json',):
    if f.exists(): used |= {r['speaker'] for r in json.loads(f.read_text())}
v = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/view-add-seamless-1103/train.json')
if v.exists(): used |= {r['speaker'] for r in json.loads(v.read_text())}
ok = sorted(p for p in cand if p not in held and p not in used)
rng = np.random.default_rng(20261107); pick = [ok[i] for i in sorted(rng.choice(len(ok), min(n, len(ok)), replace=False))]
rows = [cand[p][int(rng.integers(len(cand[p])))] | dict(participant=p) for p in pick]
Path(sys.argv[2]).write_text(''.join(json.dumps(r) + '\n' for r in rows))
print(json.dumps(dict(candidates_participants=len(cand), held=len(held), used=len(used), eligible=len(ok), picked=len(rows),
                      track_min=round(sum(r['valid'] for r in rows) / 30 / 60, 1), tars=len({r['tar_path'] for r in rows}))))

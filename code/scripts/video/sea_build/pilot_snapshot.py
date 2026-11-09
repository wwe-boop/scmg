"""Option-3 pilot: retarget outputs -> snapshot rows in the SEAMLESS_SNAPSHOT.jsonl schema used by regate_seamless / select_near /
extract (id, frames, seconds, strict, result, q, q_sha256, source).   python pilot_snapshot.py <pilot dir>"""
import glob, hashlib, json, sys
from pathlib import Path
P = Path(sys.argv[1]); rows = [json.loads(l) for f in sorted(glob.glob(str(P / 'ITEMS_*.jsonl'))) for l in open(f)]
snap, rej, miss = [], 0, 0
for r in rows:
    d = P / 'retarget' / r['id']
    if (d / 'RESULT.json').exists():
        res = json.loads((d / 'RESULT.json').read_text())
        snap.append(dict(id=r['id'], frames=res['frames'], seconds=res['frames'] / 50, strict=res['strict'], source=r, result=str(d / 'RESULT.json'),
                         q=str(d / 'q_ctrl.npy'), q_sha256=hashlib.sha256((d / 'q_ctrl.npy').read_bytes()).hexdigest()))
    elif d.exists(): rej += 1
    else: miss += 1
(P / 'SNAPSHOT.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in snap))
s = dict(windows=len(rows), results=len(snap), rejected=rej, missing=miss, window_hours=sum(r['frames'] for r in rows) / 30 / 3600,
         result_hours=sum(r['seconds'] for r in snap) / 3600, speakers=len({r['source_id'] for r in rows}))
(P / 'SNAPSHOT_SUMMARY.json').write_text(json.dumps(s, indent=1)); print('SNAPSHOT_DONE', json.dumps(s))

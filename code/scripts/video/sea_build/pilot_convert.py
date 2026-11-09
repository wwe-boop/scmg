"""Option-3 pilot: whole tracks -> release07 video_smplx human.npz windows (convert_seamless.py conv(), 'all' mode: consecutive 20 s
windows inside every smplh:is_valid run >= 5 s, windows >= 5 s), rows in the convert_seamless schema. Tars are untrusted: member bytes
only, allow_pickle=False.   python pilot_convert.py <PILOT_TRACKS.jsonl> <out dir> <shard> <n shards>"""
import hashlib, io, json, sys, tarfile
from pathlib import Path
import numpy as np
sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/seamless-interaction-20261102')
from convert_seamless import conv, runs
tracks = [json.loads(l) for l in open(sys.argv[1])]; out = Path(sys.argv[2]); sh, ns = int(sys.argv[3]), int(sys.argv[4]); W = 20 * 30
rows = []
for t in tracks[sh::ns]:
    key = t['key']
    with tarfile.open(t['tar_path']) as tf:
        with np.load(io.BytesIO(tf.extractfile(key + '.npz').read()), allow_pickle=False) as zz:
            z = {k: zz[k] for k in zz.files if k.startswith('smplh:')}
    valid = z['smplh:is_valid'].astype(bool)
    for s, e in runs(valid):
        if e - s < 5 * 30: continue
        for a in range(s, e, W):
            b = min(e, a + W)
            if b - a < 150: continue
            cid = f"seamless-{t['label'][:3]}-{key}-{a:06d}"; p = out / 'items' / cid / 'human.npz'; p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, **conv(z, a, b))
            rows.append(dict(id=cid, kind='video_smplx', source='video', path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                             license='CC-BY-NC-4.0 internal-research-only', standing_status='UNKNOWN', source_id=t['participant'],
                             subset=t['label'], split=t['split'], batch=t['batch'], tar=t['tar'], seamless_key=key, start_frame=a,
                             training_admitted=False, frames=b - a, fps=30, pilot='option3-1107'))
(out / f'ITEMS_{sh}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows)); print('CONVERT_DONE', sh, len(rows))

"""Audit an isolated collection: hashes, reversal, conditions, seed provenance and train/holdout isolation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scmg.common import CHUNK_FRAMES, STRIDE

p=argparse.ArgumentParser();p.add_argument('collection',type=Path);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();m=json.loads((a.collection/'MANIFEST.json').read_text())
view=Path(m['view']);items=json.loads((view/'train.json').read_text());split=json.loads((view/'SPLIT.json').read_text())
train=set(split['train_ids']);val=set(split['val_ids']);byid={it['id']:it for it in items}
hold={e['clip_id'].split('__')[0] for e in json.loads(Path(m['holdout']).read_text())['entries']}
qraw=np.load(view/'train_q.npy',mmap_mode='r')
fail=[];mxseed=0.;max_jump=0.
for row in m['rows']:
    f=a.collection/row['file'];bad=[]
    if hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']:bad.append('sha256')
    ids=row['item_ids'];its=[byid[x] for x in ids]
    if not set(ids)<=train or set(ids)&val:bad.append('split')
    if any(str(it.get('clip_id','')).split('__')[0] in hold for it in its):bad.append('holdout')
    with np.load(f,allow_pickle=False) as z:
        q,r=z['q_forward'],z['q_reversed'];feat,rf=z['feat_forward'],z['feat_reversed']
        if not np.array_equal(q[::-1],r):bad.append('joint_reverse')
        if not np.array_equal(rf,np.zeros_like(feat) if row['cond']=='null' else feat[::-1]):bad.append('condition')
        if q.shape!=(row['frames'],34) or z['lat_reencoded'].shape[0]!=row['frames']//STRIDE:bad.append('shape')
        if int(z['loss_from_chunk'])!=row['loss_from_chunk'] or row['loss_from_chunk']!=row['frames']//CHUNK_FRAMES*2//3:bad.append('loss_start')
        if not all(np.isfinite(z[k]).all() for k in ('q_forward','q_reversed','lat_reencoded','feat_forward','feat_reversed')):bad.append('finite')
        off=min(it['offset'] for it in its)+row['start'];n=m['seed_chunks']*CHUNK_FRAMES
        err=float(np.max(np.abs(q[:n]-qraw[off:off+n])));mxseed=max(mxseed,err)
        if err>1e-6:bad.append('ground_truth_seed')
        max_jump=max(max_jump,float(np.max(np.abs(np.diff(q,axis=0)))))
    if bad:fail.append(dict(file=row['file'],errors=bad))
out=dict(all_ok=not fail,count=len(m['rows']),failures=fail,max_seed_error=mxseed,max_frame_joint_jump_rad=max_jump,
         note='Structural/provenance audit only; not WBC/physical certification. Raw on-policy drift is intentionally retained as context.')
a.out.write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
raise SystemExit(0 if out['all_ok'] else 1)

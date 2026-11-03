"""Recompute an erroneously raw-spliced seed using the original model/seed.

No smoothing/filtering: replay the first generated chunk, then decode the GT
latent seed with its real look-ahead. The causal VAE needs only one future
latent. Verify the overlapping generated frames, and optionally verify a
complete rollout against the reconstructed trajectory. Preserve input files.
This recovery utility is only for the initial paired, same-model collection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from bagger_collect import atomic_json, COND_KEYS, D
from scmg.common import CHUNK, CHUNK_FRAMES, STRIDE
from scmg.evaluate import load
from scmg.infer import Streamer


@torch.no_grad()
def main(a):
    old=[json.loads((p/'MANIFEST.json').read_text()) for p in (a.input,a.pair_input)]
    for key in ('checkpoint','seed','frames','seed_chunks','n_gen'):
        if old[0][key]!=old[1][key]:raise ValueError(f'paired collection differs: {key}')
    count=min(a.count or old[0]['count'],old[0]['count'])
    stpath=old[0].get('stats',D+'/STATS_v36_1.json');st=json.loads(Path(stpath).read_text())
    qm,qs=(np.array(st[k],np.float32) for k in ('q_mean','q_std'))
    view=Path(old[0]['view']);items={x['id']:x for x in json.loads((view/'train.json').read_text())}
    qraw=np.load(view/'train_q.npy',mmap_mode='r')
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('BAGGER_MEMORY_FRACTION','.48')))
    model,vae=load(old[0]['checkpoint']['path'],old[0]['vae'],'cuda')
    specs=[]
    for source,dest,meta in zip((a.input,a.pair_input),(a.out,a.pair_out),old):
        dest.mkdir(parents=True,exist_ok=True)
        spec={k:v for k,v in meta.items() if k not in ('rows','elapsed_seconds')}
        spec.update(count=count,stats=stpath,seed_representation='VAE reconstruction of ground-truth seed latents; no raw-joint splice',
                    redecoded_from=str(source),redecode_method='causal VAE seed replay, generated suffix unchanged')
        if (dest/'COLLECTION.json').exists() and json.loads((dest/'COLLECTION.json').read_text())!=spec:
            raise ValueError('output spec mismatch')
        atomic_json(dest/'COLLECTION.json',spec);specs.append(spec)
    allrows=[[],[]];n=old[0]['seed_chunks']*CHUNK_FRAMES;t0=time.monotonic();errors=[];seams=[]
    for i in range(count):
        rows=[m['rows'][i] for m in old]
        data=[]
        for source,row in zip((a.input,a.pair_input),rows):
            f=source/row['file']
            if hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('input checksum mismatch')
            with np.load(f,allow_pickle=False) as z:data.append({k:z[k].copy() for k in z.files})
        x=data[0]
        if not np.array_equal(x['q_forward'],data[1]['q_forward']):raise ValueError('paired rollouts are not identical')
        off=min(items[k]['offset'] for k in rows[0]['item_ids'])+rows[0]['start']
        seed=np.asarray(qraw[off:off+n],np.float32).copy()
        torch.manual_seed(rows[0]['seed'])
        mu=vae.encode(torch.from_numpy((seed-qm)/qs).cuda()[None])[0]
        lat=((mu-model.lat_mean)/model.lat_std)[0]
        cond={k:x['original_'+k] for k in COND_KEYS if 'original_'+k in x}
        s=Streamer(model,vae,x['feat_forward'],cond,st,H=10**9,steps=10,seed=rows[0]['seed'],
                   standby=st['q_mean'],ctx_t=model.ctx_t,cfg_w=old[0]['checkpoint']['cfg_w'],
                   trim_start=True,cfg_null_fix=True,window_lm_text=model.cfg.planner=='voxcpm')
        s.k=old[0]['seed_chunks'];s.lat=list(lat.clone().unbind());s.tick()
        # Match the original convolution shape as well as its causal inputs;
        # short vs full decoder calls can choose different TF32 kernels.
        prefix_lat=torch.stack(s.lat)
        padded=torch.cat([prefix_lat,prefix_lat[-1:].expand(old[0]['frames']//STRIDE-len(prefix_lat),-1)])
        prefix=vae.decode((padded*model.lat_std+model.lat_mean)[None].float())[0,:len(prefix_lat)*STRIDE].cpu().numpy()*qs+qm
        # Last STRIDE frames of this prefix still need the next generated chunk.
        err=float(np.max(np.abs(prefix[n:-STRIDE]-x['q_forward'][n:len(prefix)-STRIDE])))
        if err>2e-5:raise ValueError(f'replay mismatch at {i}: {err}')
        forward=x['q_forward'].copy();forward[:n]=prefix[:n]
        fullerr=None
        if i==a.verify_full:
            for _ in range(old[0]['n_gen']-1):s.tick()
            full=vae.decode((torch.stack(s.lat)*model.lat_std+model.lat_mean)[None].float())[0].cpu().numpy()*qs+qm
            fullerr=float(np.max(np.abs(full-forward)))
            if fullerr>2e-5:raise ValueError(f'full rollout mismatch: {fullerr}')
            print('FULL_REPLAY',json.dumps(dict(index=i,max_abs=fullerr)),flush=True)
        reverse=forward[::-1].copy()
        rl=(vae.encode(torch.from_numpy((reverse-qm)/qs).cuda()[None])[0][0]-model.lat_mean)/model.lat_std
        before=float(np.max(np.abs(x['q_forward'][n]-x['q_forward'][n-1])))
        after=float(np.max(np.abs(forward[n]-forward[n-1])))
        errors.append(err);seams.append((before,after))
        for j,(dest,row,payload) in enumerate(zip((a.out,a.pair_out),rows,data)):
            payload.update(q_forward=forward,q_reversed=reverse,lat_reencoded=rl.cpu().numpy(),q_seed_truth=seed)
            f=dest/row['file'];tmp=Path(str(f)+'.tmp')
            with tmp.open('wb') as o:np.savez_compressed(o,**payload)
            tmp.replace(f)
            new=dict(row,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),original_sha256=row['sha256'],
                     replay_max_abs=err,seed_seam_before=before,seed_seam_after=after,full_replay_max_abs=fullerr)
            atomic_json(dest/f'{i:06d}.json',new);allrows[j].append(new)
        if i%25==0:print(json.dumps(dict(index=i,seconds=time.monotonic()-t0,seam_before=before,seam_after=after)),flush=True)
    elapsed=time.monotonic()-t0
    for dest,spec,rows in zip((a.out,a.pair_out),specs,allrows):
        atomic_json(dest/'MANIFEST.json',dict(**spec,rows=rows,elapsed_seconds=elapsed))
        files=[dest/'COLLECTION.json',dest/'MANIFEST.json']+[dest/f'{i:06d}{s}' for i in range(count) for s in ('.npz','.json')]
        (dest/'FILES.sha256').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n' for f in files))
        atomic_json(dest/'FROZEN.json',dict(count=count,source='bagger_reversed',split='train',
            scope='isolated experiment; not canonical data admission',physical_gate='not certified',
            files_sha256=hashlib.sha256((dest/'FILES.sha256').read_bytes()).hexdigest()))
    result=dict(count=count,seconds=elapsed,replay_max_abs=max(errors),
                seam_before=np.quantile(np.array(seams)[:,0],[.5,.9,.99,1]).tolist(),
                seam_after=np.quantile(np.array(seams)[:,1],[.5,.9,.99,1]).tolist())
    atomic_json(a.out/'REDECODE.json',result);print('REDECODE_DONE',json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('input','pair-input','out','pair-out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--count',type=int,default=0);p.add_argument('--verify-full',type=int,default=11)
    main(p.parse_args())

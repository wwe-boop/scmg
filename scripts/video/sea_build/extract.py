"""G6 data build step 2: copy of newdata-1103/code/seamless_extract.py (sha 077ffcc3...) with ONLY these changes: the selection is
SELECT.jsonl ((c)1 newly accepted, Seamless-train, no back-off) instead of grade A; output under <build dir>; sharded by tar group;
grade label regate1107. Utterance / VAD / word-time / held-speaker / eval-text rules unchanged.
    python extract.py <build dir> <shard> <n shards>"""
import io,json,hashlib,tarfile,collections,math,sys
from pathlib import Path
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
G=Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture');O=G/'scmg/newdata-1103';B=Path(sys.argv[1]);D=B/'seamless';D.mkdir(exist_ok=True);SH,NS=int(sys.argv[2]),int(sys.argv[3])
rows=[json.loads(l) for l in (O/'SEAMLESS_SNAPSHOT.jsonl').open()];sel=[json.loads(l) for l in (B/'SELECT.jsonl').open()];assert all(r['source']['split']=='train' for r in sel)
sys.path.insert(0,'/mnt/cpfs/zbl-cpfs-new/USERS/zehan/unified-ex002-data-v3_5-20261025');from direct_boundary_v35 import Boundary
refs=[]
for p in sorted((G/'unified-ex002-manifest-v3_2').glob('*.jsonl')):
 for l in p.open():
  r=json.loads(l)
  if r.get('split') in ('dev','test') and 'clip_id' in r:refs.append(r)
boundary=Boundary(refs)
held={r['source']['source_id'] for r in rows if r['source']['split']!='train'}
groups=collections.defaultdict(list)
for r in sel:
 x=r['source'];groups[(x['subset'],x['split'],x['batch'],x['tar'])].append(r)
accepted=[];audit=[];seen=set()
for gi,(key,rs) in enumerate(sorted(groups.items())[SH::NS]):
 rel=Path(*key[:3])/(key[3]+'.tar');tp=G/'seamless-interaction/raw'/rel
 if not tp.exists():tp=Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture/seamless-interaction/raw')/rel
 try:
  with tarfile.open(tp) as t:
   members={m.name:m for m in t.getmembers()}
   for r in rs:
    x=r['source'];k=x['seamless_key'];detail=dict(id=r['id'],tar=str(tp),selected=0,reasons=[])
    if x['source_id'] in held:detail['reasons'].append('speaker_in_official_dev_test');audit.append(detail);continue
    jb=t.extractfile(members[k+'.json']).read();md=json.loads(jb);assert md['id']==k
    qbytes=Path(r['q']).read_bytes();assert hashlib.sha256(qbytes).hexdigest()==r['q_sha256'];q=np.load(io.BytesIO(qbytes),allow_pickle=False);assert q.shape==(r['frames'],34) and np.isfinite(q).all()
    origin=x['start_frame']/x['fps'];end=origin+len(q)/50
    vad=np.zeros(len(q),np.float32)
    for v in md['metadata:vad']:
     a=max(0,math.ceil((v['start']-origin)*50));b=min(len(q),math.ceil((v['end']-origin)*50));vad[a:max(a,b)]=1
    # Each transcript segment is an entire official utterance; never truncate boundary-crossing utterances.
    candidates=[]
    for ui,u in enumerate(md['metadata:transcript']):
     if not isinstance(u.get('start'),(int,float)) or not isinstance(u.get('end'),(int,float)):
      detail['reasons'].append(dict(reason='missing_utterance_time',utterance=ui));continue
     if not (origin<=u['start']<u['end']<=end):continue
     a=max(0,math.floor((u['start']-origin)*50));b=min(len(q),math.ceil((u['end']-origin)*50))
     if b-a<50 or vad[a:b].sum()<25 or vad[a:b].mean()<.5:continue
     # Do not retain long listening gaps inside a transcript container.
     off=np.flatnonzero(np.diff(np.r_[0,(vad[a:b]==0).astype(int),0]));longest=max((off[1::2]-off[::2]),default=0)
     if longest>50:continue
     words=[w for w in u.get('words',[]) if w.get('word','').strip()]
     if not words:continue
     if any(not isinstance(w.get('start'),(int,float)) or not isinstance(w.get('end'),(int,float)) for w in words):
      detail['reasons'].append(dict(reason='missing_word_time',utterance=ui));continue
     if any(not (origin+a/50-1e-6<=w['start']<w['end']<=origin+b/50+1e-6) for w in words):continue
     text=' '.join(w['word'].strip() for w in words);hits=boundary.text_hits(text)
     if hits:detail['reasons'].append(dict(reason='eval_text_overlap',utterance=ui,hits=hits));continue
     identity=(k,round(origin+a/50,3),round(origin+b/50,3))
     if identity in seen:detail['reasons'].append('duplicate_same_participant_interval');continue
     seen.add(identity);candidates.append((ui,a,b,words,text))
    if not candidates:detail['reasons'].append('no_complete_own_speech_utterance_after_vad_and_isolation')
    else:
     wb=t.extractfile(members[k+'.wav']).read();wave,sr=sf.read(io.BytesIO(wb),dtype='float32');assert wave.ndim==1
     for ui,a,b,words,text in candidates:
      n=b-a;start=origin+a/50;i0=round(start*sr);i1=round((start+n/50)*sr);audio=wave[i0:i1];assert len(audio)==i1-i0
      g=math.gcd(sr,16000);audio=resample_poly(audio,16000//g,sr//g);assert len(audio)>=n*320;audio=audio[:n*320]
      cid=r['id']+f'-u{ui:04d}';d=D/'clips'/cid;d.mkdir(parents=True,exist_ok=True)
      np.save(d/'q34.npy',q[a:b]);np.save(d/'vad.npy',vad[a:b]);sf.write(d/'audio.wav',audio,16000,subtype='PCM_16')
      ww=[dict(text=w['word'].strip(),start_time=w['start']-start,end_time=w['end']-start,score=w.get('score')) for w in words]
      rec=dict(id=cid,parent_id=r['id'],frames=n,speaker=x['source_id'],conversation=k.rsplit('_P',1)[0],participant_key=k,absolute_start_s=start,absolute_end_s=start+n/50,parent_frame_range=[a,b],data_source='seamless_stream',grade='regate1107',text=text,words=ww,path=str(d),q=str(d/'q34.npy'),audio=str(d/'audio.wav'),vad=str(d/'vad.npy'),audio_sha256=hashlib.sha256((d/'audio.wav').read_bytes()).hexdigest(),parent_q_sha256=r['q_sha256'],metadata_sha256=hashlib.sha256(jb).hexdigest(),source_tar=str(tp),source_audio_member=k+'.wav',source_metadata_member=k+'.json',own_vad_fraction=float(vad[a:b].mean()),source_user_review='PASS_USER_20261103',text_isolation='PASS_FIXED_V32',training_admitted=False,pending=['video_calibration_visibility_review','final_manifest_review'],license=x['license'],transition_required=True)
      (d/'META.json').write_text(json.dumps(rec));accepted.append(rec);detail['selected']+=1
    audit.append(detail)
 except Exception as e:audit.append(dict(tar=str(tp),ids=[r['id'] for r in rs],error=repr(e)))
 if gi%20==0:print('tar',gi,len(groups),'segments',len(accepted),flush=True)
(D/f'SEGMENTS_{SH}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in accepted));(D/f'EXTRACTION_AUDIT_{SH}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in audit))
summary=dict(shard=SH,selected=len(sel),segments=len(accepted),frames=sum(r['frames'] for r in accepted),hours=sum(r['frames'] for r in accepted)/180000,speakers=len({r['speaker'] for r in accepted}),eval_reference_rows=len(refs),errors=[r for r in audit if 'error' in r],policy='Only complete official utterances inside A/train, own participant mono denoised WAV, own VAD >=50% and >=0.5s, no >1s nonvoice gap, minimum 1s; official heldout participant exclusion; exact and near text isolation.')
(D/f'EXTRACTION_SUMMARY_{SH}.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

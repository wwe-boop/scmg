"""G6 data build step 3: copy of newdata-1103/code/seamless_features.py (sha 1fa17f76...) lines 1-51 (tp-isolate-r3 text isolation,
feat89 features, CAM++ speaker vectors) unchanged except the input (all SEGMENTS_*.jsonl of <build dir>) and the end: the ready items are
written to ITEMS_READY.json (the view is built by gate_merge.py after the subwindow re-gate).
    python features.py <build dir>"""
import sys,json,hashlib,gzip,collections,os,time
from pathlib import Path
import numpy as np,soundfile as sf,torch,onnxruntime as ort
import torchaudio.compliance.kaldi as kaldi
G=Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture');B=Path(sys.argv[1]);D=B/'seamless'
sys.path.insert(0,'/mnt/cpfs/zbl-cpfs-new/USERS/zehan/unified-train-A-20261025');import feat89 as F
sys.path.insert(0,str(G/'d1k-textpool-r1'));from tp.isolate import Isolator
sys.path.insert(0,'/mnt/cpfs/zbl-cpfs-new/USERS/zehan/speaker-style-emb-20261030/code');from audio_window import speech_window
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[json.loads(l) for f in sorted(D.glob('SEGMENTS_*.jsonl')) for l in f.open()]
# Original tp-isolate-r3 against dev614 and EVERY user/assistant turn in text-pool dev/test.
texts=[];pins={}
for split in ['dev','test']:
 for p in sorted((G/'d1k-textpool-r1/derived/pool-r6-20260930/pool'/split).glob('*.jsonl.gz')):
  pins[str(p)]=sha(p)
  with gzip.open(p,'rt') as f:
   for line in f:
    r=json.loads(line)
    for ti,t in enumerate(r['turns']):
     text=t.get('raw_text') or t.get('text') or t.get('normalized') or ''
     if text:texts.append(dict(id=str(r.get('id',r.get('conv_id')))+f':{ti}',text=text))
p=D/'POOL_EVAL_TEXTS.json';p.write_text(json.dumps(texts));dev=G/'rsi-glmotion-v1/fm-full2/test.json'
assert 'sealed' not in str(dev.resolve()) and 'testset-primary' not in str(dev.resolve())
iso=Isolator({'dev614':dev,'pool_dev_test':p});isolation=[];badconv=set()
for r in rows:
 hit=iso.check(r['text']);isolation.append(dict(id=r['id'],conversation=r['conversation'],hit=hit))
 if hit:badconv.add(r['conversation'])
(D/'ISOLATION.json').write_text(json.dumps(dict(audit=iso.audit,pool_pins=pins,pool_turns=len(texts),rows=isolation,excluded_conversations=sorted(badconv)),indent=1))
rows=[r for r in rows if r['conversation'] not in badconv]
sel=json.loads(Path('/mnt/cpfs/zbl-cpfs-new/USERS/zehan/speaker-style-emb-20261030/MODEL_SELECTION.json').read_text());assert sha(sel['weight'])==sel['weight_sha256']
so=ort.SessionOptions();so.intra_op_num_threads=1;so.inter_op_num_threads=1
model=ort.InferenceSession(sel['weight'],sess_options=so,providers=['CPUExecutionProvider']);assert model.get_providers()==['CPUExecutionProvider']
items=[];errors=[];vectors=[];N=sum(r['frames'] for r in rows)
# Stage per clip first; aggregate only verified clips, no holes after errors.
for i,r in enumerate(rows):
 d=Path(r['path'])
 try:
  T=r['frames'];q=np.load(d/'q34.npy');w,sr=sf.read(d/'audio.wav',dtype='int16');vad=np.load(d/'vad.npy');assert sr==16000 and len(w)==T*320 and q.shape==(T,34)
  words=[{k:x[k] for k in ['text','start_time','end_time']} for x in r['words']]
  rec=F.Recording(None,None,'en',T,words=words,pcm_fs=(w,sr));seg=rec.segment(0,T);feat=seg['feat'];feat[:,-1]=vad
  assert feat.shape==(T,89) and np.isfinite(feat).all() and seg['alignment']
  chunk,crop=speech_window(w.astype(np.float32)/32768,10.)
  assert len(chunk)>=8000
  fb=kaldi.fbank(torch.from_numpy(chunk)[None],num_mel_bins=80,dither=0,sample_frequency=16000);fb=fb-fb.mean(0,keepdim=True)
  emb=model.run(None,{model.get_inputs()[0].name:fb[None].numpy()})[0].reshape(-1);assert emb.shape==(192,) and np.isfinite(emb).all() and np.linalg.norm(emb)>1e-6;emb=(emb/np.linalg.norm(emb)).astype(np.float32)
  np.save(d/'feat.npy',feat);np.save(d/'spk.npy',emb);(d/'alignment.json').write_text(json.dumps(seg['alignment']));(d/'spk.json').write_text(json.dumps(dict(model='campplus',model_sha256=sel['weight_sha256'],audio_sha256=r['audio_sha256'],crop=crop,scope='own_participant_utterance')))
  k=len(items);items.append(dict(id=r['id'],clip_id=r['parent_id'],dir=str(d),offset=0,frames=T,speech=seg['speech'],events=[],text=rec.text,text_null=False,conv_key=r['conversation'],speaker=r['speaker'],lang='en',clip=r['id'],clip_start_s=r['absolute_start_s'],source='seamless_stream',data_source='seamless_stream',label_source=None,split_hint='train',prev_clip_id=None,prosody_scope='clip',frame_valid_frac=1.,source_token='seamless_stream',speed_id=0,spk=dict(path=str(d/'spk.npy'),sha256=sha(d/'spk.npy'),valid=True,policy='verified_own_participant_vad'),training_admitted=False,gaps=r['pending'],source_user_review='PASS_USER_20261103',text_isolation='PASS_V32_DEV614_POOL',license=r['license'],absolute_end_s=r['absolute_end_s'],own_vad_fraction=r['own_vad_fraction'],q_sha256=sha(d/'q34.npy'),audio_sha256=r['audio_sha256'],metadata_sha256=r['metadata_sha256'],participant_key=r['participant_key']))
  vectors.append(emb)
 except Exception as e:errors.append(dict(id=r['id'],error=repr(e)))
 if i%25==0:print(i,len(rows),'ready',len(items),'errors',len(errors),flush=True)
(D/'ITEMS_READY.json').write_text(json.dumps(items,ensure_ascii=False))
(D/'FEATURES_SUMMARY.json').write_text(json.dumps(dict(items=len(items),frames=sum(r['frames'] for r in items),hours=sum(r['frames'] for r in items)/180000,
  speakers=len({r['speaker'] for r in items}),errors=errors,excluded_conversations=len(badconv),feature_code=F.code_digests()),indent=1))
print('FEATURES_DONE',len(items),len(errors),flush=True)

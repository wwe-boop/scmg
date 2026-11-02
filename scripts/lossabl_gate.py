"""Read-only preflight / completion audit. Never starts training or submits DLC."""
import argparse,json,importlib.util,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('phase',choices=['dry-run','finalize']);p.add_argument('--out');a=p.parse_args()
C=Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg')
D=Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg')
if a.phase=='dry-run':
 paths=[C/'dlc-scmg-train-r6-1102/main-peft/best.pt',C/'trained/vae-r3/final.pt',D/'STATS_v36_1.json',D/'text-e5-v36_1/TEXT_FEAT.npy',Path('/mnt/cpfs/zbl-cpfs-new/USERS/zehan/eval-v2-20261102/pack.pkl')]
 deps=['torch','numpy','scipy','eval2.gen_proxy']
 result={'files':{str(x):x.is_file() for x in paths},'dependencies':{x:importlib.util.find_spec(x) is not None for x in deps},'pod_count':1,'gpu_per_pod':8,'cpu':80,'memory':'1280Gi','SchedulingStrategy':'Auto','live_schedulability':'requires pai-train list_nodes + submit_job dry_run before submission'}
 assert all(result['files'].values()) and all(result['dependencies'].values()),result
else:
 if not a.out:raise SystemExit('--out required')
 root=Path(a.out);lanes=['ctl','s1','s1-s2','s1-nostream','s1-nocont','s1-sf10','s1-dfm','s1-mg'];result={}
 for lane in lanes:
  d=root/lane
  assert (d/'LOSSABL_DONE').is_file(),lane
  done=json.loads((d/'DONE.json').read_text());assert done['step']==4000,(lane,done)
  best=json.loads((d/'BEST.json').read_text());assert best['key']=='gen_fkd',(lane,best)
  ev=json.loads((d/'eval-best/RESULTS.json').read_text());assert 'inf' in ev and '0' in ev
  if lane=='s1-mg':assert (d/'eval-best-cfg1/RESULTS.json').is_file()
  result[lane]={'steps':done['step'],'best':best,'offline':ev['inf'],'H0':ev['0']}
print(json.dumps({'phase':a.phase,'status':'PASS','result':result},indent=2))

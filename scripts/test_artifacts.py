"""Failure injection against the actual lane/final/SEMAUX shell control flow; never launches a model."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

FAKE = r'''import json, os, sys
from pathlib import Path
import numpy as np
args=sys.argv[1:];fail=os.environ.get('FAIL','');bad=os.environ.get('BAD','')
with open(os.environ['CALLS'],'a') as f:f.write(json.dumps(args)+'\n')
if args[:2]==['-m','scmg.token_cache']:sys.exit(0)
if args[:2]==['-m','scmg.train']:
 out=Path(args[args.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
 (out/'DONE.json').write_text('{"step":3}');sys.exit(0)
if args[:2]==['-m','scmg.evaluate']:
 tag='eval';out=Path(args[args.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
 (out/'RESULTS.json').write_text(json.dumps({'inf':{'clips':1},'0':{'clips':1}}));sys.exit(1 if fail==tag else 0)
name=Path(args[0]).name
if name=='gen_semantic.py':
 tag='gen';out=Path(args[2]);label=args[3].split('=')[0]
 for vv in args[args.index('--variants')+1].split(','):
  p=out/label/vv;p.mkdir(parents=True,exist_ok=True);np.save(p/'gen_0.npy',np.zeros((32,34),np.float32))
 sys.exit(1 if fail==tag else 0)
if name=='semantic_eval.py':
 vv=Path(args[args.index('--gen-dir')+1]).name;tag='score-'+vv
 out=Path(args[args.index('--out')+1]);out.write_text('' if bad==tag else json.dumps(dict(centre=dict(top1=.5,top5=1.,balanced_top1=.5),n_neg_windows=1)))
 sys.exit(1 if fail==tag else 0)
tag='planner' if name=='planner_probe.py' else 'rollout'
if fail==tag:sys.exit(42)
if bad==tag:print('{}');sys.exit(0)
print(json.dumps(dict(label='test',plan_shrink=1.,plan_r2=0.,h_change_no_audio=.1,h_change_no_text=.1) if tag=='planner' else dict(label='test',free=dict(speed=1.,still=.1,copy=.5),oracle=dict(speed=1.,still=.1,copy=.5))))
'''


def main(root):
    src=Path(__file__).resolve().parent;code=root/'code';(code/'scripts').mkdir(parents=True)
    man=root/'manifest';man.mkdir();(man/'MANIFEST.json').write_text(json.dumps(dict(entries=[dict(ci=0,frames=32)])))
    for name in ('artifacts.py','artifacts.sh','ft_lane.sh','ft_final.sh'):
        s=(src/name).read_text().replace('$B/semantic-recognizer-v1/manifest-n200',str(man))
        (code/'scripts'/name).write_text(s)
    fake=root/'fake.py';fake.write_text(FAKE);wrapper=root/'fake-python';wrapper.write_text(f'#!/bin/bash\nexec {sys.executable} {fake} "$@"\n');wrapper.chmod(0o755)
    env=dict(os.environ,PY=str(wrapper),ARTIFACT_PY=sys.executable,CALLS=str(root/'calls.jsonl'))
    def invoke(script,case,fail='',bad=''):
        folder=root/case;folder.mkdir(exist_ok=True)
        args=['bash',str(code/'scripts'/script),'ignored-gpu',str(folder),'lane']
        if script=='ft_lane.sh':args+=['fake.pt','{}','{}']
        return subprocess.run(args,env=dict(env,FAIL=fail,BAD=bad),capture_output=True,text=True),folder
    for script in ('ft_final.sh','ft_lane.sh'):
        for failed in ('planner','gen','score-own','score-notext','score-swap','rollout')+ (('eval',) if script=='ft_lane.sh' else ()):
            r,folder=invoke(script,script+'-'+failed,fail=failed)
            assert r.returncode!=0 and '_DONE lane' not in r.stdout,(script,failed,r.stdout,r.stderr)
            semantic=folder/'lane'/('final-eval/semantic' if script=='ft_final.sh' else 'semantic')
            if failed in ('gen','score-own','score-notext','score-swap'):assert not (semantic/'DONE').exists()
            rr,_=invoke(script,folder.name);assert rr.returncode==0,(script,failed,rr.stdout,rr.stderr)
            # Corrupt valid artifacts even with existing semantic DONE: restart must repair and validate.
            (semantic/'score-own.json').write_text('')
            probe=semantic.parent/'planner_probe.json';probe.write_text('{broken')
            rr,_=invoke(script,folder.name);assert rr.returncode==0,(rr.stdout,rr.stderr)
            assert json.loads(probe.read_text())['label']=='test' and json.loads((semantic/'score-own.json').read_text())['centre']
        for bad in ('planner','score-own'):
            r,_=invoke(script,script+'-bad-'+bad,bad=bad);assert r.returncode!=0,(script,bad)
        print('F2',script,'all step failures propagate, corrupt outputs rejected, retry repairs DONE artifacts',flush=True)
    # Execute the exact SEMAUX block with lightweight train/evaluate stubs and the real semantic artifact wrapper.
    pipeline=(src/'pipeline_dlc.sh').read_text();block=pipeline.split('if [ "${SEMAUX:-0}" = 1 ]; then',1)[1].split('if [ "${VOXQ:-0}" = 1 ]; then',1)[0]
    block='if [ "${SEMAUX:-0}" = 1 ]; then'+block
    block=block.replace('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/semantic-recognizer-v1/manifest-n200',str(man))
    driver=root/'semaux.sh';driver.write_text('''#!/bin/bash
set -uo pipefail
CODE=$1;R=$2;mkdir -p "$R";SEMAUX=1;IFD=unused;SERIAL=0
source "$CODE/scripts/artifacts.sh"
steps() { echo 4; }
g() { echo ignored-gpu; }
train() { mkdir -p "$R/$2"; }
evaluate() { [ "${FAIL:-}" != eval ]; }
'''+block)
    for fail in ('eval','score-own','score-notext','score-swap',''):
        out=root/('semaux-'+(fail or 'success'))
        r=subprocess.run(['bash',str(driver),str(code),str(out)],env=dict(env,FAIL=fail),capture_output=True,text=True)
        assert (r.returncode==0)==(not fail),(fail,r.stdout,r.stderr)
        assert (out/'SEMAUX_DONE').exists()==(not fail)
    print('F2 SEMAUX eval/score failures and every background PID are checked; success publishes DONE',flush=True)
    print('ALL_OK shell failure injection',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);main(a.out)

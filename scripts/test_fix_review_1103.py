"""F1-F9 regression, synthetic data + actual model/loss/shell paths. Run on CUDA alongside existing acceptance."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import torch
from scmg.data import View, WindowDataset
from scmg.common import J, FEAT, SPK, CHUNK_FRAMES
from scmg.train import ADDABLE, validate_additions, build_model
from scmg.losses import StageCfg, step_losses, chunk_intent_labels
from scmg.token_cache import resolve_token_cache, reject_ttsh, validate_manifest_tokens, validate_manifest_cache, cache_metadata
from test_stream_text import tiny_lm, model, fixture


def expect_error(fn, message=''):
    try: fn()
    except (ValueError, FileNotFoundError) as e:
        assert message in str(e), str(e)
    else: raise AssertionError('expected rejection')


def data_test(root):
    root.mkdir(); (root/'cond').mkdir()
    items=[dict(id=f'i{i}', clip=f'c{i}', clip_id=f'p{i if i < 2 else 1}__rhythm', offset=i*64, frames=64,
                speaker='shared', data_source='semsynth_r7p1', text_null=True) for i in range(3)]
    (root/'train.json').write_text(json.dumps(items)); (root/'SPLIT.json').write_text(json.dumps(dict(train_ids=[x['id'] for x in items],val_ids=[])))
    (root/'cond'/'COND.json').write_text(json.dumps(dict(items={})))
    np.save(root/'cond'/'SPK_EMB.npy',np.zeros((0,SPK),np.float32))
    np.save(root/'train_q.npy',np.repeat(np.arange(3)*7.,64)[:,None]*np.ones((1,J)))
    np.save(root/'train_feat.npy',np.zeros((192,FEAT),np.float32))
    (root/'holdout.json').write_text(json.dumps(dict(entries=[dict(clip_id='p1')])))
    stats=dict(q_mean=[0.]*J,q_std=[1.]*J,f_mean=[0.]*FEAT,f_std=[1.]*FEAT)
    os.environ['SCMG_NO_GLITCH']='1'
    v=View(root,stats=stats,holdout=root/'holdout.json'); ds=WindowDataset(v,64,ref_frames=32)
    assert v.by_spk['shared']==[0] and ds.w.tolist()==[1.,0.,0.]
    for i in range(20): assert ds.sample(0,0,np.random.default_rng(i))['ref_valid']==0
    text=root/'text';causal=root/'causal';text.mkdir();causal.mkdir()
    for d in (text,causal):
        np.save(d/'TEXT_FEAT.npy',np.zeros((1,768),np.float32));np.save(d/'TEXT_TIME.npy',np.zeros(1,np.float32))
        (d/'TEXT_INDEX.json').write_text(json.dumps(dict(i0=[0,1])))
    (causal/'PREP.json').write_text('{"causal":false}');os.environ['SCMG_TEXT_CAUSAL']=str(causal)
    expect_error(lambda:View(root,stats=stats,text_cache=text),'prefix-causal')
    (causal/'PREP.json').write_text('{"causal":true}');View(root,stats=stats,text_cache=text)
    np.save(causal/'TEXT_TIME.npy',np.ones(1,np.float32))
    expect_error(lambda:View(root,stats=stats,text_cache=text),'do not match')
    os.environ.pop('SCMG_TEXT_CAUSAL')
    print('F1 actual View excludes holdout + rhythm variants from primary/reference sampling',flush=True)


def additions_test():
    for k in ADDABLE:
        start,end=(0,3) if k=='vox_slots' else (False,True)
        validate_additions({k:start},{k:end}); validate_additions({k:end},{k:end})
        expect_error(lambda:validate_additions({k:end},{k:start}),'unsupported module transition')
    for value in (-1,2,True,0): expect_error(lambda:validate_additions({'vox_slots':3},{'vox_slots':value}))
    # Validate effective extra config too (before any model construction / weight loading).
    expect_error(lambda:build_model({}, {'vox_slots':3}, {'vox_slots':2}, {}))
    print('F3 monotone config transitions, resizing/type errors, effective extra config',flush=True)


def audio_test():
    for kind in ('chunk','vox'):
        for tts in (False,True):
            m,v=model(kind,mtc=True,tts_h=tts,tts_h_dim=8)
            if tts:
                with torch.no_grad():m.tts_in[1].weight.normal_(0,.1)
            m.cfg.drop_audio=m.cfg.drop_text=m.cfg.drop_spk=m.cfg.drop_time=0.
            feat,c,st=fixture(); T=64
            batch={k:x[None] for k,x in c.items()}
            batch.update(q=torch.randn(1,T,J,device='cuda'), mask=torch.ones(1,T,device='cuda'),feat=feat[:T][None],
                         gest=torch.zeros(1,T,dtype=torch.long,device='cuda'))
            batch['gest'][:,32:48]=2
            if tts:batch.update(tts_h=torch.randn(1,T//4,8,device='cuda'),tts_hm=torch.ones(1,T//4,device='cuda'))
            stats=dict(qm=torch.zeros(J,device='cuda'),qs=torch.ones(J,device='cuda'))
            def run(b,enabled):
                rec={}; funcs={name:getattr(m,name) for name in ('plan','mt_memory','flow_velocity')}
                def plan(*a,**kw):
                    h=funcs['plan'](*a,**kw);rec['h']=h.detach().clone();return h
                def mt(*a,**kw):
                    x,mask=funcs['mt_memory'](*a,**kw);rec['memory']=x.detach().clone();rec['memory_mask']=mask.clone();return x,mask
                def flow(*a,**kw):
                    rec['local']=a[5].detach().clone();rec['local_mask']=a[6].clone();return funcs['flow_velocity'](*a,**kw)
                m.plan,m.mt_memory,m.flow_velocity=plan,mt,flow
                try:
                    torch.manual_seed(19)
                    with torch.no_grad():step_losses(m,v,b,StageCfg(p_sel_audio=1.,sel_audio_premask=enabled,self_force_p=0.),stats)
                finally:
                    for name,fn in funcs.items():setattr(m,name,fn)
                return rec
            changed={k:x.clone() for k,x in batch.items()};changed['feat'][:,32:48]+=20
            if tts:changed['tts_h'][:,8:12]+=30
            a,b=run(batch,True),run(changed,True)
            assert all(torch.equal(a[k],b[k]) for k in a),(kind,tts,[(k,(a[k]!=b[k]).sum().item()) for k in a])
            a0,b0=run(batch,False),run(changed,False)
            assert not torch.equal(a0['memory'],b0['memory']), 'negative control should expose the old leak'
            print('F4',kind,'TTS=',tts,'h, MTC, local audio AND masks bit-identical; old mode leaks',flush=True)


def intent_test():
    rows=torch.tensor([[2]*15+[99], [2]*3+[3]*3+[0]*10, [2]*8+[3]*8, [-1]*16, [0]*16],device='cuda')
    labels=chunk_intent_labels(rows,100)
    assert labels.tolist()==[2,0,2,-1,0],labels
    print('F5 dominant counts, >=4 frames, deterministic tie, none and unknown',flush=True)


def token_test(root):
    root.mkdir()
    cases=[('', 'Qwen2.5-0.5B',False),('Qwen2.5-0.5B','Qwen2.5-0.5B',False),('Qwen3-0.6B-Base','Qwen3-0.6B-Base',False),
           ('MiniCPM4-0.5B','MiniCPM4-0.5B',False),('VoxCPM1.5','VoxCPM1.5',True)]
    for i,(lm,meta,chars) in enumerate(cases):
        d=root/str(i);d.mkdir();np.save(d/'TOK_IDS.npy',np.array([1,2],np.int32))
        (d/'TOK_INDEX.json').write_text(json.dumps(dict(i=[0,2])))
        prep=dict(model=meta,items=1,tokens=2,voxcpm_chars=chars);(d/'PREP.json').write_text(json.dumps(prep))
        cfg=SimpleNamespace(planner='voxcpm',vox_lm=lm)
        assert resolve_token_cache(cfg,str(d))==str(d)
        validate_manifest_cache(dict(token_cache_metadata=cache_metadata(d)),d)
        if i >= 2:
            expect_error(lambda:validate_manifest_cache(dict(version='semantic-eval-manifest-v1'),d),'mismatch')
        expect_error(lambda:resolve_token_cache(SimpleNamespace(planner='voxcpm',vox_lm='MiniCPM4-0.5B' if 'Mini' not in lm else ''),str(d)),'mismatch')
        (d/'TOK_INDEX.json').write_text(json.dumps(dict(i=[1,2])))
        expect_error(lambda:resolve_token_cache(cfg,str(d)),'TOK_INDEX')
    for lm, _, _ in cases:
        selected=resolve_token_cache(SimpleNamespace(planner='voxcpm',vox_lm=lm))
        assert Path(selected).is_dir()
    expect_error(lambda:validate_manifest_tokens(dict(tids=np.array([1]),tidm=np.array([True])),dict(tids=np.array([2]),tidm=np.array([True])),0),'mismatch')
    for entry in ('planner_probe.py','rollout_probe.py','gen_semantic.py','ft_lane.sh'):
        expect_error(lambda:reject_ttsh(SimpleNamespace(tts_h=True),entry),'TTSH')
    checkpoint=root/'tts.pt';torch.save(dict(base_cfg=dict(tts_h=True),extra_cfg={},ema={}),checkpoint)
    manifest=root/'manifest';manifest.mkdir();(manifest/'MANIFEST.json').write_text('{"entries":[]}')
    scripts=Path(__file__).resolve().parent
    for script in ('planner_probe.py','rollout_probe.py','gen_semantic.py'):
        args=[sys.executable,str(scripts/script)]
        args+=([str(manifest),str(root/'gen')] if script=='gen_semantic.py' else ['unused-vae.pt'])
        args+=[f'tts={checkpoint}']
        result=subprocess.run(args,capture_output=True,text=True)
        assert result.returncode!=0 and 'TTSH' in result.stderr,(script,result.stderr)
    result=subprocess.run(['bash',str(scripts/'ft_lane.sh'),'unused-gpu',str(root/'lanes'),'tts',str(checkpoint),'{}','{}'],
                          env=dict(os.environ,PY=sys.executable),capture_output=True,text=True)
    assert result.returncode!=0 and 'TTSH' in result.stderr,result.stderr
    # The command-line parser must expose all streaming knobs; x2 with no causal cache fails before loading data/models.
    result=subprocess.run([sys.executable,str(scripts/'gen_semantic.py'),'--help'],capture_output=True,text=True)
    assert result.returncode==0 and all(flag in result.stdout for flag in ('--tok-cache','--text-mode','--x2-lag','--x2-rate',
        '--deploy-timing','--window-lm-text','--trim-start','--cfg-null-fix'))
    result=subprocess.run([sys.executable,str(scripts/'gen_semantic.py'),'unused-man','unused-out','m=unused.pt','--text-mode','x2'],
                          env={k:v for k,v in os.environ.items() if k!='SCMG_TEXT_CAUSAL'},capture_output=True,text=True)
    assert result.returncode!=0 and 'x2 requires causal E5' in result.stderr,result.stderr
    print('F6/F7 tokenizer families, metadata, corrupt indices, manifest mismatch; four real CLI TTSH guards + x2 prerequisite',flush=True)


def timing_test():
    sys.path.insert(0,str(Path(__file__).parent/'g1cross'))
    from timing import check_duration,replace_upper
    check_duration(51,50,31,30)
    for n in (2,49,53,100):expect_error(lambda:check_duration(n,50,31,30),'duration mismatch')
    ref=np.ones((30,39));pred=np.zeros((30,39));u=replace_upper(ref,pred,[10,11]);assert not u[:,10:12].any()
    expect_error(lambda:replace_upper(ref,pred[:2],[10,11]),'duration mismatch')
    print('F8 correct fps rounding accepted, short/long inputs refused, no GT upper tail',flush=True)


def fsq_test():
    m,v=model('vox',vox_fsq='tanh');m.cfg.drop_audio=m.cfg.drop_text=0.
    feat,c,_=fixture();b={k:x[None] for k,x in c.items()};b.update(q=torch.randn(1,64,J,device='cuda'),mask=torch.ones(1,64,device='cuda'),feat=feat[:64][None])
    with torch.no_grad():m.vox.fsq_in.bias.fill_(3.)
    o=step_losses(m,v,b,StageCfg(fsq_sat_w=1.,self_force_p=0.),dict(qm=torch.zeros(J,device='cuda'),qs=torch.ones(J,device='cuda')))
    assert o['fsq_sat']>0; o['fsq_sat'].backward();assert m.vox.fsq_in.weight.grad.abs().sum()>0
    print('F9 tanh FSQ saturation loss is positive with nonzero gradients',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    tiny_lm(a.out/'tiny-qwen');data_test(a.out/'data');additions_test();audio_test();intent_test();token_test(a.out/'tokens');timing_test();fsq_test()
    print('ALL_OK F1 F3 F4 F5 F6 F7 F8 F9',flush=True)

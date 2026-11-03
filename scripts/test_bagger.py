"""CUDA acceptance: subprocess bitwise legacy comparison and corrective-mask gradients."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_selfforce_v2 import setup, snapshot, loss


def main(a):
    from scmg.bagger import expand_source, BaggerPool
    from scmg.common import CHUNK, CHUNK_FRAMES
    from scmg.losses import step_losses, StageCfg
    checks = []
    def check(name, ok, **details):
        checks.append(dict(name=name, ok=bool(ok), **details))
        print(json.dumps(checks[-1]), flush=True)
    child = a.out / 'legacy.pt'
    env = dict(os.environ, BAGGER_TEST_CHILD=str(child), PYTHONPATH=str(a.orig)+':/workspace/zehan/env/scmg-deps')
    subprocess.run([sys.executable, str(Path(__file__).resolve())],
                   env=env, cwd=a.orig, check=True)
    old, new = torch.load(child, weights_only=False), snapshot()
    for case in old:
        errors = []
        for part in ('losses', 'grads'):
            if old[case][part].keys() != new[case][part].keys(): errors.append(part+'/keys')
            for key, value in old[case][part].items():
                if key not in new[case][part] or not torch.equal(value, new[case][part][key]): errors.append(part+'/'+key)
        for key in ('cpu_rng','cuda_rng'):
            if not torch.equal(old[case][key],new[case][key]): errors.append(key)
        check('disabled bitwise '+case, not errors, errors=errors)
    del old,new
    for kind in ('chunk','vox','mtc'):
        m,v,b,st = setup(kind, B=2, T=96)
        state = torch.get_rng_state().clone(); cstate = torch.cuda.get_rng_state().clone()
        original = m.source.weight.detach().clone()
        expand_source(m)
        check('source zero extension '+kind, torch.equal(state,torch.get_rng_state()) and
              torch.equal(cstate,torch.cuda.get_rng_state()) and torch.equal(m.source.weight[:-1],original)
              and bool((m.source.weight[-1]==0).all()))
        b['source'][1] = 7
        b['loss_from_chunk'] = torch.tensor([-1,4],device='cuda')
        b['bagger_null'] = torch.tensor([False,True],device='cuda')
        b['spk_valid'][1] = 0; b['ref_valid'][1] = 0; b['tidm'][1] = False; b['tmask'][1] = False
        flow_records, aux_records, histories = [],[],[]
        orig_flow, orig_plan = m.flow_velocity, m.plan
        def flow(*args,**kwargs):
            o = orig_flow(*args,**kwargs)
            if o.requires_grad:
                o.retain_grad(); flow_records.append((args,o))
            return o
        def plan(*args,**kwargs):
            histories.append(args[0].detach().clone())
            return orig_plan(*args,**kwargs)
        def capture_aux(_module,_args,o):
            o.retain_grad(); aux_records.append(o)
        m.flow_velocity,m.plan=flow,plan
        hook=m.p_aux.register_forward_hook(capture_aux)
        o=loss(m,v,b,st,self_force_p=1.,self_force_rounds=2,p_noctx=1.,ctx_noise_p=1.,outpaint=True,
               w_cont=.1,w_rhy=.1)
        o['loss'].backward(); hook.remove()
        args, vel = flow_records[-1]
        grad=vel.grad.view(2,6,*vel.shape[1:])
        check('masked flow gradients '+kind, bool((grad[1,:4]==0).all()) and bool(grad[1,4:].abs().sum()>0))
        check('masked auxiliary gradients '+kind, bool((aux_records[-1].grad[1,:4]==0).all()) and
              bool(aux_records[-1].grad[1,4:].abs().sum()>0))
        with torch.no_grad():
            gt=((v.encode(b['q'])[0]-m.lat_mean)/m.lat_std).view(2,6,CHUNK,-1)
        known=args[2].view(2,6,-1); xt=args[0].view(2,6,*args[0].shape[1:]); ts=args[1].view(2,6,-1)
        check('clean BAgger history '+kind, torch.equal(histories[-1][1],gt[1]) and
              torch.equal(xt[1,1:,:m.cfg.ctx],gt[1,:-1,-m.cfg.ctx:]) and bool((ts[1,1:,:m.cfg.ctx]==1).all())
              and bool(known[1,:4,m.cfg.ctx:].all()))
        check('null audio '+kind, torch.equal(args[5].view(2,6,*args[5].shape[1:])[1],
              m.audio_null.expand(6,*args[5].shape[1:])))
        check('finite backward '+kind, all(bool(torch.isfinite(x).all()) for x in o.values()) and
              all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None))
    # Data-level smoke: full sequence reversal, tail selection, nulls and matching schemas.
    from types import SimpleNamespace
    from scmg.data import WindowDataset, collate
    root=a.out/'fixture'; root.mkdir(exist_ok=True)
    q=np.arange(384*34,dtype=np.float32).reshape(384,34)/10000
    f=np.ones((384,89),np.float32)
    np.savez(root/'0.npz',q_reversed=q[::-1].copy(),feat_reversed=f,
             original_spk=np.ones(192,np.float32),original_spk_valid=np.float32(1),
             original_ref=np.ones((128,34),np.float32),original_ref_valid=np.float32(1))
    (root/'MANIFEST.json').write_text(json.dumps(dict(source='bagger_reversed',split='train',
        seed_representation='VAE reconstruction of ground-truth seed latents; no raw-joint splice',rows=[
        dict(file='0.npz',frames=384,loss_from_chunk=16,cond='reverse')])))
    view=SimpleNamespace(qm=np.zeros(34),qs=np.ones(34),text_c=None,tok=None,intent=None,tts=None,bagger_dir=str(root),
                         weights=lambda share:np.ones(1),clips=[dict(frames=384)])
    ds=WindowDataset(view,bagger_share=.33)
    sample=ds.bagger.sample(np.random.default_rng(0))
    check('tail/window mask offset', sample['loss_from_chunk']==8 and np.array_equal(sample['q'],q[::-1][128:]))
    check('reverse retains speaker/style, blanks text',sample['source']==7 and not sample['tmask'].any() and
          sample['spk_valid']==1 and sample['ref_valid']==1 and bool((sample['spk']==1).all()))
    ds.bagger.rows[0][1]['cond']='null'
    null=ds.bagger.sample(np.random.default_rng(0))
    check('null blanks speaker/style',null['spk_valid']==0 and null['ref_valid']==0 and bool((null['spk']==0).all()))
    ds.bagger.rows[0][1]['cond']='reverse'
    ds.sample=lambda ci,s,rng:{**sample,'source':np.int64(0)}
    it=iter(ds); rows=[next(it) for _ in range(1000)]
    share=np.mean([x['source']==7 for x in rows]); batch=collate(rows[:32])
    check('sampling share/schema',.28<share<.38 and batch['loss_from_chunk'].shape==(32,), measured_share=float(share))
    result=dict(all_ok=all(x['ok'] for x in checks),checks=checks)
    (a.out/'RESULTS.json').write_text(json.dumps(result,indent=2))
    return 0 if result['all_ok'] else 1


if __name__=='__main__':
    torch.set_num_threads(2)
    if os.environ.get('BAGGER_TEST_CHILD'):
        torch.save(snapshot(),os.environ['BAGGER_TEST_CHILD'])
    else:
        p=argparse.ArgumentParser(); p.add_argument('--orig',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
        a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
        tiny=a.out/'tiny-qwen';tiny.mkdir(exist_ok=True)
        (tiny/'config.json').write_text(json.dumps(dict(model_type='qwen2',vocab_size=128,hidden_size=64,
            intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=2048,use_cache=False)))
        os.environ['SCMG_LM_CONFIG']=str(tiny)
        sys.exit(main(a))

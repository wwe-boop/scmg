"""Regression tests for the 2026-11-02 second review; one GPU, synthetic inputs only.

PYTHONPATH=. python scripts/test_fix1102.py --original /path/to/pristine/tree \
    --checkpoints /path/to/dlc-scmg-train-r4-1102 --vae /path/to/vae.pt --artifacts /path/to/results
The original tree is a read-only snapshot. Artifacts are retained, never removed.
"""
import argparse
import ast
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scmg.common import CHUNK, CHUNK_FRAMES, J, FEAT, SPK, TEXT_DIM
from scmg.data import require_causal_text
from scmg.infer import Streamer, WIN, LOOK
from scmg.losses import StageCfg, step_losses
from scmg.model import Config, SCMG, INF
from scmg.vae import MotionVAE
from scripts.p3_finetune import physics_conditions, physics_flow_conditions, physics_batch

ROOT = Path(__file__).resolve().parents[1]
DEV = 'cuda'
ARGS = None


def original_module(name):
    if 'fix1102_original' not in sys.modules:
        mod = types.ModuleType('fix1102_original')
        mod.__path__ = [str(Path(ARGS.original) / 'scmg')]
        sys.modules[mod.__name__] = mod
    return importlib.import_module('fix1102_original.' + name)


def sample(B=1, K=WIN):
    torch.manual_seed(12)
    rand = lambda *s: torch.randn(*s, device=DEV)
    full = lambda shape, x, dtype: torch.full(shape, x, dtype=dtype, device=DEV)
    T = K * CHUNK_FRAMES; nt = 6
    return dict(q=rand(B, T, J), mask=full((B, T), 1., torch.float32), feat=rand(B, T, FEAT),
                text=rand(B, nt, TEXT_DIM), text_c=rand(B, nt, TEXT_DIM),
                ttime=torch.arange(nt, device=DEV).float()[None].expand(B, -1) / 3,
                tseg=full((B, nt), 0., torch.float32), tmask=full((B, nt), True, torch.bool),
                tids=full((B, nt), 3, torch.long), tidm=full((B, nt), True, torch.bool),
                titime=torch.arange(nt, device=DEV).float()[None].expand(B, -1) / 3,
                tiseg=full((B, nt), 0., torch.float32), spk=rand(B, SPK),
                spk_valid=full((B,), 1., torch.float32), source=full((B,), 1, torch.long),
                ref=rand(B, 32, J), ref_valid=full((B,), 1., torch.float32))


def small(**kw):
    m = SCMG(Config(z=4, d=32, heads=4, audio_layers=1, planner_layers=1, flow_layers=1, **kw)).to(DEV)
    # Nonzero learned gates/styles prevent missing-condition regressions from passing at zero initialization.
    with torch.no_grad():
        m.f_ada.weight.normal_(0, .03)
        if m.cfg.style: m.style_head.out.weight.normal_(0, .1)
        if m.cfg.mtc:
            for block in list(m.planner) + list(m.flow): block.g.fill_(.3)
        if hasattr(m, 'stream_emb'): m.stream_emb.weight.normal_(0, .1)
    return m


def stats():
    return dict(q_mean=[0.] * J, q_std=[1.] * J)


def cond(b):
    return {k: v[0] for k, v in b.items() if k not in ('q', 'mask', 'feat')}


def streamer(m, vae, b, cls=Streamer, **kw):
    return cls(m, vae, b['feat'][0], cond(b), stats(), device=DEV, steps=1, seed=73, **kw)


class Fix1102(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert torch.cuda.is_available(), 'Run GPU tests on the authorized single GPU.'
        assert torch.cuda.device_count() == 1, 'Expose exactly one GPU.'
        torch.set_num_threads(2)
        print(json.dumps(dict(pid=os.getpid(), cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                              torch=torch.__version__, gpu=torch.cuda.get_device_name(0))), flush=True)
        # Record the actual test process on its physical UUID once the CUDA context exists.
        torch.zeros(1, device=DEV)
        query = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv'], text=True)
        mine = [line for line in query.splitlines() if line.split(',')[0].strip() == str(os.getpid())]
        print('PID_GPU_VERIFICATION', mine, flush=True)
        assert len(mine) == 1, query
        (Path(ARGS.artifacts) / 'pid_gpu.txt').write_text('\n'.join(mine) + '\n')

    def assertExact(self, a, b, label):
        self.assertTrue(torch.equal(a, b), f'{label}: max diff {(a.float()-b.float()).abs().max().item()}')

    @torch.no_grad()
    def compare_p3_streamer(self, m, vae, label, valid):
        b = sample(); b['ref_valid'].fill_(valid)
        cur = WIN - 1 - LOOK
        hist = torch.randn(1, WIN, CHUNK, m.cfg.z, device=DEV); hist[:, cur:] = 0
        s = streamer(m, vae, b); s.k = cur
        s.lat = list(hist[0, :cur].flatten(0, 1).unbind())
        s.n_emitted = cur * CHUNK_FRAMES - 4
        captured = []
        actual_flow = m.flow_velocity
        def capture(*a, **kw):
            v = actual_flow(*a, **kw)
            captured.append((a, kw, v.clone()))
            return v
        with patch.object(m, 'flow_velocity', side_effect=capture): s.tick()
        self.assertEqual(len(captured), 1)
        a, kw, expected = captured[0]
        fc = physics_flow_conditions(m, physics_conditions(m, b, hist), [cur],
                                     generator=torch.Generator(device=DEV).manual_seed(73))
        got = actual_flow(*a[:3], **fc)
        diff = float((got - expected).abs().max())
        self.assertLessEqual(diff, 1e-5)
        # Compare every condition as well as the final velocity.
        for idx, key in [(3,'h'), (4,'g'), (5,'aw'), (6,'aw_mask'), (7,'style_tok'), (9,'mt'), (10,'mt_mask')]:
            if a[idx] is None: self.assertIsNone(fc[key])
            elif a[idx].dtype == torch.bool: self.assertExact(a[idx], fc[key], key)
            else: torch.testing.assert_close(a[idx], fc[key], rtol=0, atol=1e-5)
        print(f'P3_STREAMER {label} ref_valid={valid} max_abs={diff:.9g}', flush=True)

    def test_p3_conditions_nonzero_gates(self):
        vae = MotionVAE(z=4, c=16).to(DEV).eval()
        for extra in ({}, dict(mtc=True), dict(style=True), dict(mtc=True, style=True, lora_r=2),
                      dict(mtc=True, style=True, adsc=True)):
            m = small(**extra).eval()
            for valid in (0., 1.): self.compare_p3_streamer(m, vae, str(extra), valid)

    def test_p3_real_checkpoints_and_default_infer(self):
        if not ARGS.checkpoints or not ARGS.vae: self.skipTest('Supply --checkpoints and --vae for real checkpoint regression')
        from scmg.evaluate import load
        old_streamer = original_module('infer').Streamer
        for lane in ('base', 'main-peft', 'main-causal', 'nomtc-causal'):
            path = Path(ARGS.checkpoints) / lane / 'best.pt'
            m, vae = load(str(path), ARGS.vae, DEV)
            for valid in (0., 1.): self.compare_p3_streamer(m, vae, lane, valid)
            b = sample()
            for mode in ('final', 'x2', 'segment'):
                for horizon in (INF, 2):
                    new = streamer(m, vae, b, text_mode=mode, H=horizon)
                    old = streamer(m, vae, b, cls=old_streamer, text_mode=mode, H=horizon)
                    for _ in range(2):
                        self.assertExact(new.tick(), old.tick(), f'{lane}/{mode}/{horizon}')
                    self.assertExact(torch.stack(new.lat), torch.stack(old.lat), 'latents')
                    print(f'DEFAULT_BITWISE {lane} mode={mode} H={horizon} ticks=2 PASS', flush=True)
            del m, vae
            torch.cuda.empty_cache()

    def test_p3_candidate_batch_and_base_bitwise(self):
        b = sample(B=2); m = small().train(); ref = copy.deepcopy(m).eval()
        hist = torch.randn(2, WIN, CHUNK, m.cfg.z, device=DEV)
        cand = torch.randn(2, WIN, 3, CHUNK, m.cfg.z, device=DEV)
        tot = torch.randn(2, WIN, 3, device=DEV); el = torch.ones_like(tot, dtype=torch.bool)
        el[:, :, -1] = False
        # Execute the original physics_batch body, with only its I/O supplied by fixtures.
        tree = ast.parse((Path(ARGS.original)/'scripts/p3_finetune.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'physics_batch')
        class DS:
            def sample(self, ci, start, rng): return {k:v[ci].cpu().numpy() for k,v in b.items()}
        W = [dict(ci=i, start=0, committed=hist[i].flatten(0,1).cpu().numpy(), cand_lat=cand[i].cpu().numpy(),
                  total=tot[i].cpu().numpy(), eligible=el[i].cpu().numpy()) for i in range(2)]
        from scmg.data import collate
        ns = dict(torch=torch, np=np, ds=DS(), W=W, dev=DEV, m=m, ref=ref, cfg=m.cfg, L=m.cfg.ctx+CHUNK,
                  CHUNK=CHUNK, INF=INF, TAU=.5, MODE='awr', MARGIN=.5, collate=collate, tscore=lambda w:w['total'])
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<original physics_batch>', 'exec'), ns)
        torch.manual_seed(105); old = ns['physics_batch']([0,1]); old[0].backward()
        grads = {n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}; m.zero_grad()
        torch.manual_seed(105); new = physics_batch(m, ref, b, hist, cand, tot, el); new[0].backward()
        for a,v in zip(old[:2],new[:2]): self.assertExact(a,v,'base P3 loss')
        for n,p in m.named_parameters():
            if n in grads: self.assertExact(grads[n], p.grad, 'base P3 gradient '+n)
        print(f'P3_BASE_BITWISE losses+{len(grads)} gradients PASS', flush=True)
        m = small(mtc=True, style=True, lora_r=2).train(); ref=copy.deepcopy(m).eval()
        for mode in ('awr', 'rft'):
            seen=[]; actual=m.flow_velocity; actual_ref=ref.flow_velocity
            def capture(*a, **kw): seen.append(kw); return actual(*a, **kw)
            def capture_ref(*a, **kw): seen.append(kw); return actual_ref(*a, **kw)
            with patch.object(m,'flow_velocity',side_effect=capture), patch.object(ref,'flow_velocity',side_effect=capture_ref):
                losses=physics_batch(m,ref,b,hist,cand,tot,el,mode=mode,margin=-10)
            self.assertEqual(len(seen),2)
            for k in ('style_tok','mt','mt_mask'):
                self.assertIsNotNone(seen[0][k]); self.assertIs(seen[0][k],seen[1][k])
                val=seen[0][k].reshape(2,15,3,*seen[0][k].shape[1:])
                self.assertExact(val[:,:,0],val[:,:,2],k+' shared among candidates')
            m.zero_grad(); (losses[0]+losses[1]).backward()
            self.assertTrue(torch.isfinite(losses[0]))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.flow.parameters() if p.grad is not None),0)
            print(f'P3_BATCH {mode} candidate ordering/shared reference/finite gradient PASS',flush=True)

    def test_causal_text_guards_and_regression(self):
        m=small(adsc=True); vae=MotionVAE(z=4,c=16).to(DEV).eval(); b=sample(B=2,K=4)
        st=dict(qm=torch.zeros(J,device=DEV),qs=torch.ones(J,device=DEV))
        missing={k:v for k,v in b.items() if k!='text_c'}
        for seg in (0.,.3,.999):
            # p_adsc_stream=0 still must reject a configuration capable of token streaming.
            with self.assertRaisesRegex(ValueError,'SCMG_TEXT_CAUSAL'):
                step_losses(m,vae,missing,StageCfg(p_async=.1,p_async_seg=seg,p_adsc_stream=0),st)
        with self.assertRaisesRegex(ValueError,'SCMG_TEXT_CAUSAL'):
            streamer(m,vae,missing,text_mode='x2')
        with self.assertRaisesRegex(ValueError,'SCMG_TEXT_CAUSAL'):
            require_causal_text(None,True,'evaluate --text-mode x2')
        # Real evaluate.main must reject BEFORE loading any checkpoint (including empty dev views).
        import scmg.evaluate as ev
        fake=types.SimpleNamespace(text_c=None)
        stats_path=Path(ARGS.artifacts)/'stats.json'; stats_path.write_text(json.dumps(stats()))
        argv=['evaluate','--ckpt','unused','--vae','unused','--out',str(Path(ARGS.artifacts)/'eval'),
              '--stats',str(stats_path),'--text-cache','unused','--text-mode','x2']
        with patch.object(sys,'argv',argv), patch.object(ev,'View',return_value=fake), patch.object(ev,'load') as load:
            with self.assertRaisesRegex(ValueError,'SCMG_TEXT_CAUSAL'): ev.main()
            load.assert_not_called()
        old_loss=original_module('losses').step_losses
        for async_p,seg,has_c in [(0,.3,False),(.8,1.,False),(.8,.3,True),(.8,0.,True)]:
            bb=b if has_c else missing
            sc=StageCfg(p_async=async_p,p_async_seg=seg,self_force_p=0,ctx_noise_p=0)
            torch.manual_seed(23); expected=old_loss(m,vae,bb,sc,st)
            torch.manual_seed(23); actual=step_losses(m,vae,bb,sc,st)
            self.assertEqual(set(expected),set(actual))
            for k in actual: self.assertExact(expected[k],actual[k],'loss '+k)
            print(f'ADSC_BITWISE p_async={async_p} p_async_seg={seg} text_c={has_c} PASS',flush=True)

    def test_vox_fsq_modes(self):
        from transformers import Qwen2Config
        from scmg.voxplanner import FSQ
        config_dir=Path(ARGS.artifacts)/'tiny-qwen'; config_dir.mkdir(exist_ok=True)
        Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                    num_attention_heads=4,num_key_value_heads=2).save_pretrained(config_dir)
        b=sample(K=4); vae=MotionVAE(z=4,c=16).to(DEV).eval()
        st=dict(qm=torch.zeros(J,device=DEV),qs=torch.ones(J,device=DEV))
        sc=StageCfg(self_force_p=0,ctx_noise_p=0)
        with patch.dict(os.environ,{'SCMG_LM_CONFIG':str(config_dir)}):
            for mode in (True,False,'tanh'):
                torch.manual_seed(53); m=small(planner='voxcpm',vox_fsq=mode)
                optimizer=torch.optim.AdamW(m.parameters(),lr=1e-4)
                out=step_losses(m,vae,b,sc,st); self.assertTrue(torch.isfinite(out['loss']))
                out['loss'].backward()
                self.assertGreater(float(m.vox.fsq_in.weight.grad.abs().sum()),0)
                optimizer.step(); m.eval()
                s=streamer(m,vae,b); self.assertTrue(torch.isfinite(s.tick()).all()); self.assertEqual(len(s.lat),CHUNK)
                # Inspect the actual bottleneck, using the identical LM input for every mode.
                lat=torch.randn(1,4,CHUNK,m.cfg.z,device=DEV); au=torch.randn(1,4,m.cfg.d,device=DEV)
                g=torch.randn(1,m.cfg.d,device=DEV); captured=[]
                handle=m.vox.fsq_in.register_forward_hook(lambda mod,inp,res:captured.append(res.detach()))
                with torch.no_grad(): plan,q=m.vox(lat,au,b['tids'],b['tidm'],g)
                handle.remove(); raw=captured[0]
                expected=FSQ()(raw) if mode is True else torch.tanh(raw) if mode=='tanh' else raw
                self.assertExact(q,expected,'bottleneck '+str(mode))
                if mode is True:
                    old=original_module('voxplanner').VoxPlanner(m.cfg).to(DEV).eval(); old.load_state_dict(m.vox.state_dict(),strict=True)
                    with torch.no_grad(): old_plan,old_q=old(lat,au,b['tids'],b['tidm'],g)
                    self.assertExact(plan,old_plan,'FSQ True plan'); self.assertExact(q,old_q,'FSQ True codes')
                    # Whole model output/gradient, not just FSQ arithmetic.
                    prior=copy.deepcopy(m); prior.vox=old
                    torch.manual_seed(93); now=step_losses(m,vae,b,sc,st)
                    torch.manual_seed(93); before=step_losses(prior,vae,b,sc,st)
                    for k in now: self.assertExact(now[k],before[k],'FSQ True train '+k)
                    self.assertExact(streamer(m,vae,b).tick(),streamer(prior,vae,b).tick(),'FSQ True tick')
                    print('FSQ_TRUE_BITWISE plan/codes/train losses/generated chunk PASS',flush=True)
                print(f'VOX_FSQ {mode!r} train_step_loss={float(out["loss"]):.6g} generated_latents={len(s.lat)} PASS',flush=True)
            with self.assertRaisesRegex(ValueError,'vox_fsq'): small(planner='voxcpm',vox_fsq='typo')

    def test_pipeline_device_dry_run(self):
        script=ROOT/'scripts/pipeline_dlc.sh'
        subprocess.run(['bash','-n',str(script)],check=True)
        dry=Path(ARGS.artifacts)/'pipeline'; dry.mkdir(exist_ok=True)
        # Stub the Python entrypoint: no import torch, model load, CUDA allocation, or actual train/eval.
        stub=dry/'python_stub.py'
        stub.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
a=sys.argv[1:]
if a==['-']: sys.stdin.read(); sys.exit(0)
if '-m' in a:
 module=a[a.index('-m')+1]
 rec=dict(module=module,cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),args=a)
 with open(os.environ['DRY_LOG'],'a') as f: f.write(json.dumps(rec)+'\\n')
 if module=='scmg.train':
  out=Path(a[a.index('--out')+1]); out.mkdir(parents=True,exist_ok=True); (out/'DONE.json').write_text('{}')
 elif module=='scmg.evaluate':
  out=Path(a[a.index('--out')+1]); out.mkdir(parents=True,exist_ok=True); (out/'RESULTS.json').write_text('{}')
''')
        stub.chmod(0o755)
        # Do not execute the legacy pipeline's rm; keep every artifact. Shorten polling only.
        bashenv=dry/'bashenv.sh'; bashenv.write_text('rm() { return 0; }\nsleep() { command sleep 0.03; }\n')
        for kind,extra in [('full',{}),('vox',{'VOXABL':'1'}),('adsc',{'ONLY_ADSC':'1'})]:
            output=dry/kind; output.mkdir(exist_ok=True); log=dry/(kind+'.jsonl')
            if kind=='adsc':
                for lane in ('base','main-peft','main-causal'):
                    (output/lane).mkdir(exist_ok=True); (output/lane/'DONE.json').write_text('{}')
            # Missing VAE exercises stage-0's check_vae invocation as well.
            vae=output/'vae'/'final.pt'
            if kind!='full': vae.parent.mkdir(exist_ok=True); vae.touch()
            env=os.environ.copy(); env.update(NGPU='1',CUDA_VISIBLE_DEVICES='1',PY=str(stub),DRY_LOG=str(log),
                                               BASH_ENV=str(bashenv),SCALE='.001',**extra)
            env.pop('ONLY_ADSC',None) if kind!='adsc' else None
            env.pop('VOXABL',None) if kind!='vox' else None
            result=subprocess.run(['bash',str(script),str(ROOT),str(output),str(vae)],env=env,text=True,
                                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=40)
            (dry/(kind+'.log')).write_text(result.stdout)
            self.assertEqual(result.returncode,0,result.stdout)
            records=[json.loads(line) for line in log.read_text().splitlines()]
            calls=[r for r in records if r['module'] in ('scmg.train','scmg.evaluate')]
            self.assertGreater(len(calls),0)
            self.assertTrue(all(r['cuda']=='1' for r in calls),calls)
            if kind=='vox':
                trained=[Path(r['args'][r['args'].index('--out')+1]).name for r in calls if r['module']=='scmg.train']
                self.assertEqual(len(trained),9); self.assertLess(trained.index('vox-full-seed1'),trained.index('vox-fsqtanh'))
                tanh=next(r for r in calls if 'vox-fsqtanh' in ' '.join(r['args']) and r['module']=='scmg.train')
                self.assertEqual(json.loads(tanh['args'][tanh['args'].index('--cfg')+1])['vox_fsq'],'tanh')
            print(f'PIPELINE_DRY_RUN {kind} train/evaluate_calls={len(calls)} all_CUDA_VISIBLE_DEVICES=1 PASS',flush=True)
        # Static expansion of g for both absent CVD and UUID/MIG-style identifiers.
        source=script.read_text(); a=source.index('VISIBLE_DEVS='); z=source.index('\n\ntrain()',a)
        for visible,expected in [(None,'0'),('1','1'),('GPU-abc','GPU-abc')]:
            env=os.environ.copy(); env['NGPU']='1'
            if visible is None: env.pop('CUDA_VISIBLE_DEVICES',None)
            else: env['CUDA_VISIBLE_DEVICES']=visible
            run=subprocess.run(['bash','-uc',source[a:z]+'\nfor i in {0..7}; do g $i; done'],env=env,
                               text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertEqual(run.returncode,0,run.stderr); self.assertEqual(run.stdout.splitlines(),[expected]*8)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--original',required=True); ap.add_argument('--checkpoints'); ap.add_argument('--vae')
    ap.add_argument('--artifacts',required=True)
    ARGS, rest=ap.parse_known_args()
    Path(ARGS.artifacts).mkdir(parents=True,exist_ok=True)
    unittest.main(argv=[sys.argv[0],*rest],verbosity=2)

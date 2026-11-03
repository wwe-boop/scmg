"""Self-force acceptance on CUDA (no dataset / sealed-set access).

python scripts/test_selfforce_v2.py --orig <pre-change code> --out <artifacts>
Optional --benchmark-ckpt / --vae measure the lane architecture, B=32/T=256.
A compares every loss, parameter gradient and final RNG state in a separate process.
B drives the real Streamer with identical GT context / initial target noise at chunk 1.
   Training's dropout, teacher noise and start-chunk zero context are intentionally
   disabled/excluded for this controlled comparison; batched matmul roundoff is allowed.
C inspects planner AND local-flow history; D checks finite, nonzero gradients.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


def seed(n):
    torch.manual_seed(n)
    np.random.seed(n)


def setup(kind='chunk', B=2, T=64):
    from scmg.common import J, FEAT, SPK, TEXT_DIM
    from scmg.model import SCMG, Config
    from scmg.vae import MotionVAE
    seed(71)
    extra = dict(planner='voxcpm', vox_lm_init='random') if kind == 'vox' else {}
    if kind == 'mtc': extra.update(mtc=True, style=True, sem_head=True, sem_cond=True, adsc=True)
    m = SCMG(Config(d=64, heads=4, audio_layers=1, planner_layers=2, flow_layers=2, **extra)).cuda()
    # Nonzero gates/output: a zero-init flow would conceal broken CFG construction.
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith(('ada.weight', 'f_out.weight', 'sem_emb.weight', '.g')):
                p.normal_(0, .03)
    vae = MotionVAE(z=m.cfg.z, c=32).cuda().eval().requires_grad_(False)
    b = dict(q=torch.randn(B, T, J, device='cuda'), mask=torch.ones(B, T, device='cuda'),
             feat=torch.randn(B, T, FEAT, device='cuda'), text=torch.randn(B, 6, TEXT_DIM, device='cuda'),
             ttime=torch.arange(6, device='cuda').float()[None].expand(B, -1) * .1,
             tmask=torch.ones(B, 6, dtype=torch.bool, device='cuda'),
             spk=torch.randn(B, SPK, device='cuda'), spk_valid=torch.ones(B, device='cuda'),
             source=torch.zeros(B, dtype=torch.long, device='cuda'),
             ref=torch.randn(B, T, J, device='cuda'), ref_valid=torch.ones(B, device='cuda'),
             tids=torch.arange(6, device='cuda')[None].expand(B, -1) + 10,
             tidm=torch.ones(B, 6, dtype=torch.bool, device='cuda'))
    b['tseg'] = b['ttime'].clone(); b['titime'] = b['ttime'].clone(); b['tiseg'] = b['ttime'].clone()
    b['text_c'] = b['text'].clone()
    stats = dict(qm=torch.zeros(J, device='cuda'), qs=torch.ones(J, device='cuda'))
    return m, vae, b, stats


def loss(m, v, b, stats, n=123, **kw):
    from scmg.losses import StageCfg, step_losses
    seed(n)
    return step_losses(m, v, b, StageCfg(**kw), stats)


def snapshot():
    out = {}
    for kind in ('chunk', 'vox', 'mtc'):
        m, v, b, st = setup(kind)
        for label, kw in [('defaults', {}), ('forced2', dict(self_force_p=1., self_force_rounds=2)),
                          ('off', dict(self_force_p=0.))]:
            m.train().zero_grad(set_to_none=True)
            o = loss(m, v, b, st, **kw)
            o['loss'].backward()
            out[kind + '/' + label] = dict(losses={k: x.detach().cpu() for k, x in o.items()},
                grads={k: p.grad.cpu() for k, p in m.named_parameters() if p.grad is not None},
                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        del m, v
    return out


def capture(m, v, b, st, **kw):
    plans, flows = [], []
    orig_plan, orig_flow = m.plan, m.flow_velocity
    def plan(*a, **k):
        plans.append(a[0].detach().clone())
        return orig_plan(*a, **k)
    def flow(*a, **k):
        flows.append(tuple(x.detach().clone() if torch.is_tensor(x) else x for x in a))
        return orig_flow(*a, **k)
    m.plan, m.flow_velocity = plan, flow
    try:
        o = loss(m, v, b, st, **kw)
    finally:
        m.plan, m.flow_velocity = orig_plan, orig_flow
    return o, plans, flows


def run(a):
    from scmg.common import CHUNK, CHUNK_FRAMES
    from scmg.infer import Streamer
    from scmg.losses import StageCfg
    checks = []
    def check(name, passed, **details):
        passed = bool(passed)
        checks.append(dict(name=name, ok=passed, **details))
        print(name, 'ok' if passed else 'FAIL', json.dumps(details), flush=True)
    # A: PYTHONPATH and cwd point only to immutable pre-change source in the child.
    ref = a.out / 'baseline.pt'
    env = dict(os.environ, SF_CHILD='1', SF_REF=str(ref), PYTHONPATH=str(a.orig))
    cmd = [sys.executable, '-c', f'exec(compile(open({str(Path(__file__).resolve())!r}).read(), {str(Path(__file__).resolve())!r}, "exec"))']
    subprocess.run(cmd, cwd=a.orig, env=env, check=True)
    old, new = torch.load(ref, weights_only=False), snapshot()
    for case in old:
        mismatches = []
        for section in ('losses', 'grads'):
            if old[case][section].keys() != new[case][section].keys(): mismatches.append(section + '/keys')
            for k, x in old[case][section].items():
                if k not in new[case][section] or not torch.equal(x, new[case][section][k]): mismatches.append(section + '/' + k)
        for k in ('cpu_rng', 'cuda_rng'):
            if not torch.equal(old[case][k], new[case][k]): mismatches.append(k)
        check('A bitwise ' + case, not mismatches, mismatches=mismatches)
    del old, new
    # B: same real window at chunk 1, source=0, no condition/teacher noise, known previous chunk.
    for kind in ('chunk', 'vox', 'mtc'):
        m, v, b, st = setup(kind, B=1)
        m.cfg.style = False  # style RNG draw shapes differ in parallel training / single-chunk inference
        m.cfg.drop_text = m.cfg.drop_audio = m.cfg.drop_spk = m.cfg.drop_time = 0.
        m.cfg.sem_head = m.cfg.sem_cond = False  # semantic route covered in D
        if hasattr(m, 'stream_emb'): del m.stream_emb  # ADSC bookkeeping covered in D
        m.eval()
        kw = dict(self_force_p=1., self_force_cfg=2., self_force_steps=10, sigma_tf=0., ctx_noise_p=0., p_noctx=0.)
        with torch.no_grad():
            _, plans, flows = capture(m, v, b, st, **kw)
            gt = (v.encode(b['q'])[0] - m.lat_mean) / m.lat_std
        target = plans[-1][0, 1]
        cond = {k: x[0] for k, x in b.items() if k not in ('q', 'mask', 'feat')}
        s = Streamer(m, v, b['feat'][0], cond, dict(q_mean=st['qm'].cpu().tolist(), q_std=st['qs'].cpu().tolist()),
                     steps=10, cfg_w=2., cfg_null_fix=True, trim_start=True, seed=123)
        s.k = 1; s.lat = list(gt[0, :CHUNK].clone().unbind())
        # The same seed alone is insufficient: training consumes condition/dropout RNG draws and samples B*K chunks.
        # Replay the actual initial noise to isolate CFG/ODE parity, while keeping Streamer's seed fixed.
        s.noise_override = lambda *_: flows[0][0][1:2, m.cfg.ctx:].clone()
        rec = []
        orig = m.flow_velocity
        def record(*args, **kwargs):
            rec.append(tuple(x.detach().clone() if torch.is_tensor(x) else x for x in args))
            return orig(*args, **kwargs)
        m.flow_velocity = record
        try: s.tick()
        finally: m.flow_velocity = orig
        actual = torch.stack(s.lat[-CHUNK:])
        err = float((actual - target).abs().max())
        check('B Streamer CFG2 ' + kind, torch.allclose(actual, target, atol=3e-5, rtol=3e-5), max_abs=err)
        # Inspect all ten null-branch masks, planner h and optional MTC memories (rather than only output).
        aligned = True
        max_cond_error = 0.
        for i in range(20):
            for j in (2, 3, 4, 5, 6, 9, 10):
                x, y = flows[i][j], rec[i][j]
                if x is None or y is None:
                    aligned &= x is None and y is None
                elif x.dtype == torch.bool:
                    aligned &= torch.equal(x[1:2], y)
                else:
                    max_cond_error = max(max_cond_error, float((x[1:2] - y).abs().max()))
                    aligned &= torch.allclose(x[1:2], y, atol=3e-5, rtol=3e-5)
        check('B branch inputs/masks ' + kind, aligned and bool(rec[1][6].any()), max_abs=max_cond_error)
        del m, v
    # C: every round must receive GT, and the final training context must be unchanged (noise cannot touch GT).
    m, v, b, st = setup()
    with torch.no_grad():
        gt = ((v.encode(b['q'])[0] - m.lat_mean) / m.lat_std).view(2, -1, CHUNK, m.cfg.z)
        _, plans, flows = capture(m, v, b, st, self_force_p=1., self_force_rounds=2, self_force_mix=1.,
                                  self_force_noise=.1, sigma_tf=0., ctx_noise_p=0., p_noctx=0.)
        _, no_plans, no_flows = capture(m, v, b, st, self_force_p=0., sigma_tf=0., ctx_noise_p=0., p_noctx=0.)
    check('C mix=1 all planner histories are GT', all(torch.equal(x, gt) for x in plans) and torch.equal(plans[-1], no_plans[-1]))
    known_ctx = flows[-1][2][:, :m.cfg.ctx]
    check('C mix=1 local context equals no self-force', torch.equal(known_ctx, no_flows[-1][2][:, :m.cfg.ctx]) and
          torch.equal(flows[-1][0][:, :m.cfg.ctx][known_ctx], no_flows[-1][0][:, :m.cfg.ctx][known_ctx]))
    # Partial mix is a chunk decision, not a token decision. Invalid positions remain GT after noise.
    b['mask'][:, -4:] = 0
    with torch.no_grad():
        _, plans, _ = capture(m, v, b, st, n=79, self_force_p=1., self_force_mix=.5, self_force_noise=.1, sigma_tf=0.)
    equal = (plans[-1] == gt).all(-1)
    check('C partial mix whole chunks + padding preserved', bool(equal[:, -1, -1].all()) and
          bool(equal[:, :-1].all(-1).any()) and bool((~equal[:, :-1].any(-1)).any()))
    del m, v
    # D: all switches individually/together, plus style/MTC/semantic/ADSC paths.
    for kind in ('chunk', 'vox', 'mtc'):
        m, v, b, st = setup(kind)
        for label, kw in [('cfg', dict(self_force_cfg=2., self_force_steps=10)), ('mix', dict(self_force_mix=.3)),
                          ('noise', dict(self_force_noise=.1)), ('all', dict(self_force_cfg=2., self_force_mix=.3, self_force_noise=.1))]:
            m.train().zero_grad(set_to_none=True)
            if kind == 'mtc': kw = dict(kw, p_async=1., p_async_seg=0.)
            if label == 'cfg':
                o, plans, _ = capture(m, v, b, st, self_force_p=1., self_force_rounds=2, **kw)
                check('D CFG shares teacher-noised history ' + kind,
                      torch.equal(plans[0], plans[1]) and torch.equal(plans[2], plans[3]))
            else:
                o = loss(m, v, b, st, self_force_p=1., self_force_rounds=2, **kw)
            o['loss'].backward()
            grads = [p.grad for p in m.parameters() if p.grad is not None]
            check('D backward ' + kind + '/' + label,
                  all(bool(torch.isfinite(x).all()) for x in [*o.values(), *grads]) and
                  any(bool(x.abs().sum() > 0) for x in grads), loss=float(o['loss']))
        del m, v
    # CLI resolution / ARGS.json payload: explicit flags win, omitted flags do not erase --sc values.
    from scmg.train import add_self_force_args, resolve_self_force_args
    p = argparse.ArgumentParser(); p.add_argument('--sc', default='{}'); add_self_force_args(p)
    ns = p.parse_args(['--sc', '{"self_force_cfg": 3, "self_force_mix": 0.3}', '--self-force-cfg', '2'])
    resolve_self_force_args(ns)
    payload = json.loads(json.dumps(vars(ns)))
    check('CLI precedence + ARGS serialization', payload['self_force_cfg'] == 2 and payload['self_force_mix'] == .3
          and json.loads(payload['sc'])['self_force_noise'] == 0.)
    timings = benchmark(a) if a.benchmark_ckpt else []
    result = dict(checks=checks, timings=timings, all_ok=all(x['ok'] for x in checks))
    (a.out / 'RESULTS.json').write_text(json.dumps(result, indent=2))
    print('ALL_OK' if result['all_ok'] else 'SOME_FAIL', flush=True)
    return 0 if result['all_ok'] else 1


def benchmark(a):
    from scmg.train import build_model
    from scmg.vae import MotionVAE
    from scmg.losses import StageCfg, step_losses
    ck = torch.load(a.benchmark_ckpt, map_location='cpu', weights_only=False)
    m, _ = build_model(ck['base_cfg'], {}, ck['extra_cfg'], ck['ema'])
    m.cfg.vox_prefsq_norm = True; m.add_modules(); m.cuda().train()
    del ck
    vc = torch.load(a.vae, map_location='cpu', weights_only=False)
    v = MotionVAE(**vc['cfg']).cuda().eval().requires_grad_(False); v.load_state_dict(vc['ema']); del vc
    # Synthetic, resident batch isolates step cost (no data I/O); actual lane model, fp32 like scmg.train.
    small, unused, b, st = setup('chunk', B=32, T=256); del small, unused
    seed(123)
    b['tids'] = torch.randint(10, 1000, (32, 128), device='cuda'); b['tidm'] = torch.ones_like(b['tids'], dtype=torch.bool)
    rows = []
    for label, kw in [('sf4', dict(self_force_rounds=4, self_force_steps=4)),
                      ('sf-cfg', dict(self_force_rounds=2, self_force_steps=10, self_force_cfg=2.)),
                      ('sf-mix', dict(self_force_rounds=2, self_force_steps=10, self_force_cfg=2., self_force_mix=.3)),
                      ('sf-lite', dict(self_force_rounds=1, self_force_steps=10)),
                      ('sf-noise', dict(self_force_rounds=1, self_force_steps=4, self_force_noise=.1)), ('no-sf', {})]:
        sc = StageCfg(self_force_p=0. if label == 'no-sf' else 1., **kw)
        elapsed = []
        torch.cuda.reset_peak_memory_stats()
        for i in range(4):
            m.zero_grad(set_to_none=True); torch.cuda.synchronize(); start = time.perf_counter()
            o = step_losses(m, v, b, sc, st); o['loss'].backward(); torch.cuda.synchronize()
            if i: elapsed.append(time.perf_counter() - start)
        seconds = float(np.median(elapsed))
        row = dict(lane=label, forced_step_seconds=seconds, forced_it_s=1/seconds,
                   peak_gib=torch.cuda.max_memory_allocated()/2**30, batch=32, frames=256, text_ids=128, samples=3)
        rows.append(row); print('BENCH', json.dumps(row), flush=True)
    base = rows[-1]['forced_step_seconds']
    for row in rows:
        p = 1. if row['lane'] == 'sf4' else (0. if row['lane'] == 'no-sf' else .5)
        row['lane_expected_seconds'] = p * row['forced_step_seconds'] + (1-p)*base
        row['lane_expected_it_s'] = 1 / row['lane_expected_seconds']
    return rows


if __name__ == '__main__':
    torch.set_num_threads(4)
    if os.environ.get('SF_CHILD'):
        torch.save(snapshot(), os.environ['SF_REF'])
    else:
        p = argparse.ArgumentParser()
        p.add_argument('--orig', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
        p.add_argument('--benchmark-ckpt'); p.add_argument('--vae')
        args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
        # Small real Qwen2 architecture for acceptance; full checkpoint architecture for benchmark.
        config = args.out / 'tiny-qwen'; config.mkdir(exist_ok=True)
        (config / 'config.json').write_text(json.dumps(dict(model_type='qwen2', vocab_size=128, hidden_size=64,
            intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=2048, use_cache=False)))
        original_lm_config = os.environ.get('SCMG_LM_CONFIG')
        os.environ['SCMG_LM_CONFIG'] = str(config)
        # benchmark loads the full config, not the tiny test fixture
        orig_benchmark = benchmark
        def benchmark(a):
            if original_lm_config: os.environ['SCMG_LM_CONFIG'] = original_lm_config
            else: os.environ.pop('SCMG_LM_CONFIG', None)
            return orig_benchmark(a)
        sys.exit(run(args))

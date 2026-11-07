"""11-08 BAgger x causal inputs (data.causal_placeholder; BaggerPool.sample adds the 'not covered' cz rows; the WindowDataset guard removed).
 a without BAgger: the EX002 WindowDataset sample stream (12 samples, every key) == BASE_TREE bitwise (causal_rows refactor)
 b BAgger share 0.5 + causal: 24 samples collate; BAgger rows cz_ok False, cz_feat == feat, cz_tend all NaN; view rows unchanged keys
 c loss probe on a mixed batch with the czm-sib recipe (sibag sc + causal flags): finite losses
    SCMG_J=34 python scripts/tests/test_bagger_causal.py      env BASE_TREE SP"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve(); CODE = HERE.parents[2]
B_ = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; D_ = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'
os.environ.setdefault('SCMG_TEXT_CAUSAL', B_ + '/models/text-e5-causal-v36_1')
SP = Path(os.environ['SP']); SP.mkdir(parents=True, exist_ok=True); FAIL = []


def ok(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not cond: FAIL.append(name)


GEN = r'''
import sys, json, numpy as np, torch
sys.path.insert(0, sys.argv[1])
from scmg.data import View, WindowDataset, collate
from scmg.losses import StageCfg
B_, D_ = sys.argv[2], sys.argv[3]; share = float(sys.argv[4])
st = json.load(open(D_ + '/scmg/STATS_v36_1.json'))
v = View(D_ + '/train-v1-20261101/views/data-v36_1', 'train', D_ + '/scmg/text-e5-v36_1', st, mmap=True, tok_cache=D_ + '/scmg/tok-qwen25-v36_1',
         bagger_dir=B_ + '/r4-1104/bagger-pai/bagger-data' if share > 0 else None)
v.load_causal(B_ + '/ex002-causal-1107/pre-train-slim2m', StageCfg().horizons, spk_atol=1e-3, check_spk=False)
it = iter(WindowDataset(v, 256, max_tokens=256, seed=11, bagger_share=share)); smp = [next(it) for _ in range(int(sys.argv[6]))]
torch.save(smp, sys.argv[5])
'''
(SP / 'gen.py').write_text(GEN)
def gen(code, share, out, n):
    r = subprocess.run([sys.executable, str(SP / 'gen.py'), str(code), B_, D_, str(share), str(out), str(n)], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(code)))
    if r.returncode: print(r.stderr[-2500:])
    return torch.load(out, weights_only=False) if r.returncode == 0 else None
a0 = gen(os.environ['BASE_TREE'], 0.0, SP / 'a0.pt', 12); a1 = gen(CODE, 0.0, SP / 'a1.pt', 12)
same = a0 is not None and a1 is not None and all(set(x) == set(y) and all(np.array_equal(np.asarray(x[k]), np.asarray(y[k]), equal_nan=True) for k in x) for x, y in zip(a0, a1))
ok('a no BAgger: sample stream == BASE_TREE bitwise (every key)', same)
b = gen(CODE, 0.5, SP / 'b.pt', 24)
sys.path.insert(0, str(CODE))
from scmg.data import collate
good = b is not None
if good:
    bag = [x for x in b if int(x['clip']) < 0]; vw = [x for x in b if int(x['clip']) >= 0]
    good = len(bag) > 0 and len(vw) > 0 and all(not bool(x['cz_ok']) and np.array_equal(x['cz_feat'], x['feat']) and np.isnan(x['cz_tend']).all() for x in bag)
    try: bt = collate(b); torch.save(bt, SP / 'bmix.pt'); good &= bool((~bt['cz_ok']).sum() >= len(bag))
    except Exception as e: print(repr(e)); good = False
ok('b BAgger + causal: samples collate; BAgger rows not covered', good, '' if b is None else f'{len(bag)} BAgger / {len(vw)} view')
SC = dict(p_stream=0.67, p_async=1.0, p_async_seg=0.0, w_int=0.3, sem_w=2.0, self_force_cfg=0.0, self_force_mix=0.0, self_force_noise=0.0, self_force_p=0.0,
          causal_front=True, causal_spk=True, causal_text=True)
PRB = SP / 'probe_bagger.py'                 # the shared probe + expand_source as scmg.train does when bagger_share > 0
PRB.write_text((HERE.parent / '_causal_loss_probe.py').read_text().replace("m.add_modules(); m.to(dev).train()",
               "m.add_modules()\nfrom scmg.bagger import expand_source\nexpand_source(m)\nm.to(dev).train()"))
r = subprocess.run([sys.executable, str(PRB), str(SP / 'bmix.pt'), str(SP / 'pmix.pt'), B_ + '/dlc-scmg-train-r4-1102/base/best.pt',
                    B_ + '/trained/vae-r3/final.pt', D_ + '/scmg/STATS_v36_1.json', json.dumps({'adsc': True, 'intent_aux': True}), json.dumps(SC)],
                   capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(CODE)), cwd=str(CODE))
if r.returncode: print(r.stderr[-2500:])
p = torch.load(SP / 'pmix.pt', weights_only=False) if r.returncode == 0 else None
ok('c mixed BAgger / view batch, czm-sib losses finite', p is not None and all(bool(torch.isfinite(x).all()) for x in p['loss'].values()))
print('ALL_OK' if not FAIL else f'SOME_FAIL {FAIL}'); sys.exit(1 if FAIL else 0)

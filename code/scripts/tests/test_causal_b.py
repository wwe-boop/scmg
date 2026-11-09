"""11-06 B lanes: WindowDataset p_start0 (B1) and StageCfg spk_empty_mean / CausalInputs spk_empty (B2).
    SCMG_J=39 CUDA_VISIBLE_DEVICES=1 python scripts/tests/test_causal_b.py      env CK VAE STATS HOSV TEXT TEXTC DS PSTATS BASE_TREE SP
 a B1 default: the WindowDataset sample stream (40 samples, every key) is bitwise equal with p_start0 left out and with
   p_start0=0 (no extra random draw); p_start0=0.5 starts ~half the windows at frame 0 (clips longer than the window)
 b default losses + 406 gradients bitwise equal to BASE_TREE (the code before B) on a real batch; the batch with the B2 key
   cz_spkmean added and spk_empty_mean off is bitwise the same
 c B2 training == inference: for every psync tick of dev clips, the training speaker of chunk k (causal_spk_select + the
   empty -> mean rule) == CausalInputs(spk_refresh='grades', spk_empty=mean).spk(t_k); before 1 s it is the mean vector, valid 1
 d B2 is live: with spk_empty_mean the losses change on a batch whose windows start at frame 0"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve(); CODE = HERE.parents[2]
sys.path.insert(0, str(CODE))
import torch
E = os.environ; SP = Path(E['SP']); SP.mkdir(parents=True, exist_ok=True)
os.environ['SCMG_TEXT_CAUSAL'] = E['TEXTC']
from scmg.common import CHUNK, STRIDE, U80
from scmg import causal_feat as CF
from scmg.data import View, WindowDataset, collate
from scmg.losses import StageCfg, causal_spk_select
FAIL = []


def ok(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not cond: FAIL.append(name)


stats = json.loads(Path(E['STATS']).read_text())
# a
vt = View(E['HOSV'], 'train', E['TEXT'], stats)
def stream(**kw):
    it = iter(WindowDataset(vt, 256, max_tokens=256, seed=5, **kw)); return [next(it) for _ in range(40)]
s0, s1 = stream(), stream(p_start0=0.)
same = all(set(x) == set(y) and all(np.array_equal(np.asarray(x[k]), np.asarray(y[k]), equal_nan=True) for k in x) for x, y in zip(s0, s1))
ok('a B1 default == p_start0 0 (40 samples, every key bitwise)', same)
s5 = stream(p_start0=.5); long_ = [x for x in s5 if vt.clips[int(x['clip'])]['frames'] > 256]
f0 = np.mean([int(x['start']) == 0 for x in long_]) if long_ else 0.
ok('a B1 p_start0 0.5: share of windows at the clip start', 0.3 < f0 < 0.8, f'{f0:.2f} of {len(long_)} long-clip windows')
# b
SC = dict(p_stream=0.67, p_async=1.0, p_async_seg=0.0, self_force_p=0, w_safe=300.)
def probe(code, bp, sc, tag):
    out = SP / f'probe_{tag}.pt'
    r = subprocess.run([sys.executable, str(HERE.parent / '_causal_loss_probe.py'), str(bp), str(out), E['CK'], E['VAE'], E['STATS'],
                        '{"adsc": true}', json.dumps(sc)], env=dict(os.environ, PYTHONPATH=str(code)), cwd=str(code), capture_output=True, text=True)
    if r.returncode: print(r.stderr[-2500:])
    return torch.load(out, weights_only=False) if r.returncode == 0 else None
def same_probe(p, q):
    if p is None or q is None: return False
    return all(torch.equal(p['loss'][k], q['loss'][k]) for k in p['loss']) and all(torch.equal(p['grad'][k], q['grad'][k]) for k in p['grad'])
bA = collate(s0[:8]); torch.save(bA, SP / 'bA.pt')
ok('b default losses + grads == BASE_TREE (sf0 + w_safe 300 StageCfg)', same_probe(probe(E['BASE_TREE'], SP / 'bA.pt', SC, 'base'), probe(CODE, SP / 'bA.pt', SC, 'new')))
vd = View(E['HOSV'], 'val', E['TEXT'], stats); cz = Path(E['CZDEV'])
vd.load_causal(cz, StageCfg().horizons)
idx = {c['items'][0]['id']: i for i, c in enumerate(vd.clips)}
stems = sorted(idx)[:6]; dsd = WindowDataset(vd, 256, max_tokens=256)
smp = [dsd.sample(idx[s], 0, np.random.default_rng(idx[s])) for s in stems]
bC = collate(smp); torch.save(bC, SP / 'bC.pt'); bC0 = {k: v for k, v in bC.items() if k != 'cz_spkmean'}; torch.save(bC0, SP / 'bC0.pt')
SCc = dict(SC, causal_front=True, causal_spk=True, causal_text=True)
ok('b cz_spkmean key with spk_empty_mean off: bitwise no-op (causal recipe)', same_probe(probe(CODE, SP / 'bC0.pt', SCc, 'c0'), probe(CODE, SP / 'bC.pt', SCc, 'c1')))
pe = probe(CODE, SP / 'bC.pt', dict(SCc, spk_empty_mean=True), 'c2'); p1 = torch.load(SP / 'probe_c1.pt', weights_only=False)
ok('d B2 live: spk_empty_mean changes the losses on clip-start windows', pe is not None and not same_probe(pe, p1))
# c
mean = vd.causal['spk_mean']; pst = CF.load_pstats(E['PSTATS']); enc = CF.speaker_encoder()
items = {it['id']: it for it in json.loads((Path(E['HOSV']) / 'train.json').read_text())}
n = bad = nmean = 0; H2 = 2 * U80
for j, st in enumerate(stems):
    b = {k: v[j:j + 1] for k, v in bC.items()}; L = vd.clips[idx[st]]['frames']; K = 256 // 16
    spk_c, val_c, _ = causal_spk_select(b, torch.tensor([H2]), K)
    emp = val_c < .5; spk_c = torch.where(emp[..., None], torch.as_tensor(mean)[None, None], spk_c); val_c = torch.where(emp, torch.ones_like(val_c), val_c)
    pcm, wave = CF.clip_audio(Path(E['DS']) / 'audio' / f'{st}.wav')
    ci = CF.CausalInputs(pcm, wave, [], items[st]['lang'], L, pst, vd.fm, vd.fs, enc, spk_refresh='grades', spk_empty=mean)
    for k in range(K):
        tk = min(L, (k * CHUNK + H2) * STRIDE)
        v, ok_ = ci.spk(tk); n += 1; nmean += int(tk < 50)
        bad += int(not (np.array_equal(v.astype(np.float32), spk_c[0, k].numpy().astype(np.float32)) and float(ok_) == float(val_c[0, k])))
ok('c B2 training speaker == CausalInputs(grades, spk_empty=mean) at every tick', bad == 0, f'{n} ticks ({nmean} before 1 s -> mean vector)')
print('ALL_OK' if not FAIL else f'SOME_FAIL {FAIL}'); sys.exit(1 if FAIL else 0)

"""11-07 interruption continuation under the causal psync protocol (scripts/interrupt/infer_retract.py --protocol psync).
    SCMG_J=39 CUDA_VISIBLE_DEVICES=k CK=.. VAE=.. STATS=.. VIEW=.. TEXT=.. TEXTC=.. WAV=.. WORDS=.. PSTATS=.. CAMPPLUS=..
    STEMS=<3+ dev stems> REF_TREE=<tree whose infer_test.py made the main-table psync runs> python scripts/tests/test_retract_psync.py
All runs: dev split (val), --spk-refresh grades --causal-feat inc --seed-mode per_clip --seed 1, CFG 2.5, 10 steps (= the
main-table psync setting).
A  --no-interrupt output == REF_TREE infer_test.py --protocol psync, bitwise (robot_repr), and == this tree's infer_test.py
B  interrupted runs (restart chunk and latent, audio end, fractions 0.3 / 0.6): the frames emitted before the interruption is
   applied == the uninterrupted psync run's frames, bitwise (50 Hz command frames)
C  causality after the interruption: replacing the PCM after max(t_i, the audio the last pre-interruption tick had seen) by
   noise leaves the interrupted output bitwise unchanged; the same noise changes the uninterrupted run (the region is read
   when it is not interrupted); every post-interruption feat89 call reads at most the frames before t_i
D  text: the kept tokens are exactly the tokens whose words all ended by t_i; tokens are withdrawn at the interruption"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve(); CODE = HERE.parents[2]
sys.path.insert(0, str(CODE))
FAIL = []


def ok(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not cond: FAIL.append(name)


import torch
E = os.environ; SP = Path(E.get('TEST_DIR') or tempfile.mkdtemp(prefix='retract_psync_')); SP.mkdir(parents=True, exist_ok=True)
stems = [s.strip() for s in open(E['STEMS']) if s.strip()][:3]
(SP / 'stems.txt').write_text('\n'.join(stems) + '\n')
PS = ['--protocol', 'psync', '--text-causal', E['TEXTC'], '--words-dir', E['WORDS'], '--pstats', E['PSTATS'], '--campplus', E['CAMPPLUS'],
      '--spk-refresh', 'grades', '--causal-feat', 'inc', '--seed-mode', 'per_clip', '--seed', '1', '--cfg-w', '2.5', '--steps', '10', '--split', 'val']
BASE = ['--ckpt', E['CK'], '--vae', E['VAE'], '--stats', E['STATS'], '--view', E['VIEW'], '--text-cache', E['TEXT']]
env = dict(os.environ)


def run(code, script, out, extra, wav=None):
    cmd = [sys.executable, script, *BASE, '--out', str(SP / out), *PS, '--wav-dir', wav or E['WAV'], *extra]
    r = subprocess.run(cmd, cwd=str(code), capture_output=True, text=True,
                       env=dict(env, PYTHONPATH=':'.join([str(code)] + [p for p in env.get('PYTHONPATH_EXTRA', '').split(':') if p])))
    if r.returncode: print(r.stdout[-2000:], r.stderr[-3000:]); raise SystemExit(f'{out} failed')
    return SP / out


def pts(d, names):
    return {n: torch.load(d / f'{n}.pt', weights_only=False) for n in names}


# ---- A
ref = run(E['REF_TREE'], 'scripts/echog_g1/infer_test.py', 'ref_infer_test', ['--stems', str(SP / 'stems.txt'), '--save50'])
own = run(CODE, 'scripts/echog_g1/infer_test.py', 'own_infer_test', ['--stems', str(SP / 'stems.txt'), '--save50'])
noi = run(CODE, 'scripts/interrupt/infer_retract.py', 'noint', ['--no-interrupt', '--stems', str(SP / 'stems.txt')])
R, O, N = pts(ref, stems), pts(own, stems), pts(noi, stems)
ok('A no-interrupt == REF_TREE infer_test psync (bitwise)', all(torch.equal(R[s]['robot_repr'], N[s]['robot_repr']) for s in stems))
ok('A this tree infer_test == REF_TREE infer_test (bitwise)', all(torch.equal(R[s]['robot_repr'], O[s]['robot_repr']) for s in stems))
q_ref = {s: np.load(ref / f'{s}.q50.npy') for s in stems}

# ---- B
rows = []
for s in stems:
    n30 = int(R[s]['real_num_frames'])
    for fr in (0.3, 0.6):
        f_i = int(round(fr * n30)); rows.append(dict(stem=s, frac=fr, n30=n30, f_i=f_i, t_i=f_i / 30.))
(SP / 'ints.json').write_text(json.dumps(dict(interrupts=rows)))
names = [f"{r['stem']}@{r['frac']:.1f}" for r in rows]
IA = ['--interrupts', str(SP / 'ints.json'), '--audio', 'end']
res = {}
for rs in ('chunk', 'latent'):
    d = run(CODE, 'scripts/interrupt/infer_retract.py', f'int_{rs}', [*IA, '--restart', rs])
    P = pts(d, names); res[rs] = (d, P)
    good, info = True, []
    for r, n in zip(rows, names):
        g = P[n]['generator']['interrupt']; q = np.load(d / f'{n}.q50.npy')
        keep = (g['k_int'] * 4 - 1) * 4 if rs == 'chunk' else 4 * g['l_r']        # frames emitted before the apply
        same = np.array_equal(q[:keep], q_ref[r['stem']][:keep]); diff_after = not np.array_equal(q[keep:keep + 50], q_ref[r['stem']][keep:keep + 50])
        good &= same and diff_after and keep > 0; info.append(f'{n}:{keep}{"" if same else "!"}{"" if diff_after else "?"}')
        if rs == 'chunk':
            calls = P[n]['generator']['post_feat_calls']; n50 = int(round(r['t_i'] * 50))
            ok(f'C post-interruption feat89 reads <= t_i ({n})', bool(calls) and max(calls) <= n50, f'calls {calls} n50 {n50}')
    ok(f'B restart {rs}: pre-interruption frames == uninterrupted psync (bitwise), later frames differ', good, ' '.join(info))

# ---- D
from scmg import causal_feat as CF
from scmg.data import View, WindowDataset
from scmg.common import CHUNK_FRAMES
stats = json.loads(Path(E['STATS']).read_text()); view = View(E['VIEW'], 'val', E['TEXT'], stats)
idx = {c['items'][0]['id']: i for i, c in enumerate(view.clips)}
tokz = CF.e5_tokenizer(); dgood = True; dinfo = []
for r, n in zip(rows, names):
    ci = idx[r['stem']]; c = view.clips[ci]; it = c['items'][0]; L = c['frames']
    s_ = WindowDataset(view, -(-L // CHUNK_FRAMES) * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
    w = CF.clip_words(Path(E['WORDS']) / f"{r['stem']}.json", L); kn = int(np.asarray(s_['tmask']).sum())
    ex = CF.text_extras(it, w, tokz, kn, len(s_['tmask']))
    ends = np.asarray(ex['wtime']).reshape(-1, 2)[:, 1]; tw = np.asarray(ex['tword'])
    exp = sum(1 for k in range(kn) if tw[k, 0] >= 0 and all(ends[j] <= r['t_i'] + 1e-9 for j in range(tw[k, 0], tw[k, 1] + 1)))
    g = res['chunk'][1][n]['generator']['interrupt']
    dgood &= g['tokens_kept'] == exp and g['tokens_kept'] < g['tokens_total']; dinfo.append(f"{n}:{g['tokens_kept']}/{exp}/{g['tokens_total']}")
ok('D kept tokens = tokens whose words ended by t_i (and some withdrawn)', dgood, ' '.join(dinfo))

# ---- C (noise after the cut, frac 0.6, chunk restart)
import soundfile as sf
rng = np.random.default_rng(0); nw = SP / 'wav_noise'; nw.mkdir(exist_ok=True)
rows6 = [r for r in rows if r['frac'] == 0.6]; (SP / 'ints6.json').write_text(json.dumps(dict(interrupts=rows6)))
cuts = {}
for r in rows6:
    n = f"{r['stem']}@0.6"; g = res['chunk'][1][n]['generator']['interrupt']
    seen = ((g['k_int'] - 1) * 4 + 2) * 4; cut = max(int(round(r['t_i'] * 50)), seen); cuts[r['stem']] = cut
    x, sr = sf.read(str(Path(E['WAV']) / f"{r['stem']}.wav"), dtype='int16', always_2d=True)
    if sr != 16000: raise SystemExit('16 kHz expected')
    x = x.copy(); x[cut * 320:] = rng.integers(-6000, 6000, x[cut * 320:].shape, dtype=np.int16)
    sf.write(str(nw / f"{r['stem']}.wav"), x, sr, subtype='PCM_16')
dn = run(CODE, 'scripts/interrupt/infer_retract.py', 'int_chunk_noise', ['--interrupts', str(SP / 'ints6.json'), '--audio', 'end', '--restart', 'chunk'], wav=str(nw))
PN = pts(dn, [f"{r['stem']}@0.6" for r in rows6]); PC = res['chunk'][1]
ok('C noise after the cut: interrupted output unchanged (bitwise)', all(torch.equal(PN[n]['robot_repr'], PC[n]['robot_repr']) for n in PN),
   ' '.join(f'{s}:cut{c}' for s, c in cuts.items()))
nn = run(CODE, 'scripts/interrupt/infer_retract.py', 'noint_noise', ['--no-interrupt', '--stems', str(SP / 'stems.txt')], wav=str(nw))
NN = pts(nn, stems)
ok('C the same noise changes the uninterrupted run (positive control)', all(not torch.equal(NN[s]['robot_repr'], N[s]['robot_repr']) for s in stems))
print('ALL_OK' if not FAIL else f'FAILED {FAIL}'); sys.exit(1 if FAIL else 0)

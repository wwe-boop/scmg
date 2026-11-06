"""11-06 inference seed protocol (scripts/echog_g1/infer_test.py --seed-mode).
    python scripts/tests/test_seed_mode.py                       part A (CPU, no model)
    SCMG_J=39 CUDA_VISIBLE_DEVICES=k REAL=1 BASE_TREE=<code before the change> CK=.. VAE=.. STATS=.. VIEW=.. TEXT=.. STEMS=<3+ dev stems>
        python scripts/tests/test_seed_mode.py                   + part B (real inference, offline protocol)
A  clip_seed: 'run' returns the run seed for every clip; 'per_clip' = seed + crc32(stem): deterministic, different for
   different stems and for different run seeds; unknown modes refused; default of --seed-mode is 'run'.
B  default (no --seed-mode) output bitwise equal to BASE_TREE; --seed-mode run == default; per_clip: clips differ from
   the run-mode output, and each clip's output does not depend on which other clips / which order / which shard ran."""
import importlib.util, os, subprocess, sys, tempfile, zlib
from pathlib import Path
HERE = Path(__file__).resolve(); CODE = HERE.parents[2]
FAIL = []


def ok(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not cond: FAIL.append(name)


src = (CODE / 'scripts/echog_g1/infer_test.py').read_text()
ns = {}; exec(compile(src[src.index('def clip_seed'):src.index('def main')], 'clip_seed', 'exec'), {'zlib': zlib}, ns); cs = ns['clip_seed']
st = ['english_2_scott_0_1_1_utt_0005', 'english_6_carla_0_74_74_utt_0003', 'english_2_scott_0_2_2_utt_0002']
ok('A run mode = run seed for every clip', all(cs(3, 'run', s) == 3 for s in st))
ok('A per_clip = seed + crc32(stem)', all(cs(1, 'per_clip', s) == 1 + zlib.crc32(s.encode()) for s in st))
ok('A per_clip distinct across stems', len({cs(0, 'per_clip', s) for s in st}) == len(st))
ok('A per_clip differs between run seeds', all(cs(0, 'per_clip', s) != cs(1, 'per_clip', s) for s in st))
try: cs(0, 'bogus', st[0]); ok('A unknown mode refused', False)
except ValueError: ok('A unknown mode refused', True)
ok("A --seed-mode default is 'run'", "ap.add_argument('--seed-mode', default='run'" in src)

if os.environ.get('REAL') == '1':
    import torch
    E = os.environ; SP = Path(tempfile.mkdtemp(prefix='seedmode_'))
    stems = [s.strip() for s in open(E['STEMS']) if s.strip()][:3]
    (SP / 'all.txt').write_text('\n'.join(stems) + '\n'); (SP / 'rev.txt').write_text('\n'.join(stems[::-1]) + '\n'); (SP / 'one.txt').write_text(stems[-1] + '\n')

    def run(code, out, stl, extra=(), shard='0/1'):
        cmd = [sys.executable, 'scripts/echog_g1/infer_test.py', '--ckpt', E['CK'], '--vae', E['VAE'], '--stats', E['STATS'], '--view', E['VIEW'],
               '--text-cache', E['TEXT'], '--out', str(SP / out), '--stems', str(SP / stl), '--split', 'val', '--cfg-w', '2.5', '--seed', '1', '--shard', shard, *extra]
        r = subprocess.run(cmd, cwd=str(code), capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(code)))
        if r.returncode: print(r.stdout[-1500:], r.stderr[-2500:])
        return {s: torch.load(SP / out / f'{s}.pt', weights_only=False)['robot_repr'] for s in stems if (SP / out / f'{s}.pt').exists()}

    base = run(E['BASE_TREE'], 'base', 'all.txt'); dflt = run(CODE, 'dflt', 'all.txt'); rm = run(CODE, 'runm', 'all.txt', ('--seed-mode', 'run'))
    ok('B default == BASE_TREE bitwise', len(base) == 3 and all(torch.equal(base[s], dflt[s]) for s in stems))
    ok('B --seed-mode run == default', len(rm) == 3 and all(torch.equal(rm[s], dflt[s]) for s in stems))
    pc = run(CODE, 'pc', 'all.txt', ('--seed-mode', 'per_clip')); pr = run(CODE, 'pcrev', 'rev.txt', ('--seed-mode', 'per_clip'))
    po = run(CODE, 'pcone', 'one.txt', ('--seed-mode', 'per_clip')); ps = run(CODE, 'pcsh', 'all.txt', ('--seed-mode', 'per_clip'), '1/2')
    ok('B per_clip differs from run mode', len(pc) == 3 and all(not torch.equal(pc[s], dflt[s]) for s in stems))
    ok('B per_clip independent of clip order', len(pr) == 3 and all(torch.equal(pc[s], pr[s]) for s in stems))
    ok('B per_clip independent of the other clips', torch.equal(po[stems[-1]], pc[stems[-1]]))
    ok('B per_clip independent of the shard', len(ps) >= 1 and all(torch.equal(ps[s], pc[s]) for s in ps))
print('ALL_OK' if not FAIL else f'FAILED {FAIL}'); sys.exit(1 if FAIL else 0)

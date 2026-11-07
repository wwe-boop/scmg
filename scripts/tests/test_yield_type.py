"""11-07 exp/duplex: the YIELD control-token type appended to scmg/ctl.py (coordinator-approved, append-only).
    SCMG_J=39 SCMG_NO_GLITCH=1 SCMG_LOCAL_MIRROR=~/bulk/cpfs-mirror CUDA_VISIBLE_DEVICES=k ORIG=<export of the tree before the change>
    CTL=<ctl-tokens-hos> CK=<a ctl checkpoint with NTYPES_V1 rows, e.g. local/ivl/train/final.pt> python scripts/tests/test_yield_type.py
A  every existing type keeps its name and id; NTYPES_V1 = the old NTYPES; YIELD = NTYPES_V1
B  token_value / window_arrays of every token of CTL.json: bitwise equal to the old code
C  a fresh ctl model (seeded): every parameter equal to the old code's, ctl_type rows < NTYPES_V1 equal, the YIELD row zero
D  an old checkpoint (NTYPES_V1 rows) loads; offline inference with lexicon tokens on 2 dev clips: bitwise equal to the old tree"""
import importlib.util, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, torch
CODE = Path(__file__).resolve().parents[2]; ORIG = Path(os.environ['ORIG']); E = os.environ; FAIL = []


def ok(n, c, i=''):
    print(('PASS ' if c else 'FAIL ') + n + (f'  {i}' if i else ''), flush=True)
    if not c: FAIL.append(n)


def load(tree, name):
    spec = importlib.util.spec_from_file_location(f'{name}_{abs(hash(str(tree)))}', tree / 'scmg' / f'{name}.py', submodule_search_locations=[])
    return spec


sys.path.insert(0, str(CODE))
from scmg import ctl as new
code_old = subprocess.run([sys.executable, '-c', f"import sys, json; sys.path.insert(0, '{ORIG}'); from scmg import ctl; "
                           "print(json.dumps(dict(names=ctl.TYPE_NAMES, ntypes=ctl.NTYPES)))"], capture_output=True, text=True)
old = json.loads(code_old.stdout.strip().splitlines()[-1])
ok('A existing names / ids unchanged', new.TYPE_NAMES[:old['ntypes']] == old['names'] and new.NTYPES_V1 == old['ntypes'])
ok('A YIELD appended', new.YIELD == old['ntypes'] and new.NTYPES == old['ntypes'] + 1 and new.group_of(new.YIELD) == 0)
raw = json.loads((Path(E['CTL']) / 'CTL.json').read_text())
toks = [t for ts in raw.values() for t in ts]
SP = Path(tempfile.mkdtemp(prefix='yieldtype_')); (SP / 'toks.json').write_text(json.dumps(toks[:4000]))
dump = ("import sys, json, numpy as np; sys.path.insert(0, sys.argv[1]); from scmg import ctl; T = json.load(open(sys.argv[2])); "
        "v = [ctl.token_value(t) for t in T]; np.savez(sys.argv[3], ty=np.array([x[0] for x in v]), val=np.stack([x[1] for x in v]), vm=np.stack([x[2] for x in v]), "
        "**{k: x for k, x in ctl.window_arrays([(t['t'], t['a'], t['b'], t.get('apex', float('nan')), *ctl.token_value(t)) for t in T[:16]]).items()})")
for tree, o in ((ORIG, 'old.npz'), (CODE, 'new.npz')): subprocess.run([sys.executable, '-c', dump, str(tree), str(SP / 'toks.json'), str(SP / o)], check=True)
a, b = np.load(SP / 'old.npz'), np.load(SP / 'new.npz')
ok('B token_value / window_arrays bitwise', set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k], equal_nan=True) for k in a.files), f'{len(toks[:4000])} tokens')
mk = ("import sys, torch; sys.path.insert(0, sys.argv[1]); from scmg.model import SCMG, Config; torch.manual_seed(0); "
      "m = SCMG(Config(z=32, ctl=True, ctl_flow=True, adsc=True)); torch.save({k: v.clone() for k, v in m.state_dict().items()}, sys.argv[2])")
try:
    for tree, o in ((ORIG, 'old.pt'), (CODE, 'new.pt')): subprocess.run([sys.executable, '-c', mk, str(tree), str(SP / o)], check=True, capture_output=True)
    so, sn = torch.load(SP / 'old.pt'), torch.load(SP / 'new.pt')
    same = set(so) == set(sn) and all(torch.equal(so[k], sn[k]) for k in so if k != 'ctl_type.weight')
    ct = sn['ctl_type.weight']
    ok('C fresh ctl model: all parameters as before, YIELD row zero', same and torch.equal(ct[:old['ntypes']], so['ctl_type.weight']) and float(ct[old['ntypes']:].abs().sum()) == 0)
except subprocess.CalledProcessError as e:
    ok('C fresh ctl model', False, e.stderr[-800:])
B_ = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; C_ = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'
stems = [l.strip() for l in open(Path(E['CTL']) / 'dev_ann_stems.txt')][:2]; (SP / 'st.txt').write_text('\n'.join(stems) + '\n')
def run(tree, out):
    cmd = [sys.executable, 'scripts/echog_g1/infer_test.py', '--ckpt', E['CK'], '--vae', B_ + '/echog-g1-decft-1105/decft-w3d0m05/final.pt',
           '--stats', C_ + '/scmg/STATS_echog_g1_v1.json', '--view', B_ + '/views/echog-g1-v1-hos', '--text-cache', C_ + '/scmg/text-e5-echog-g1-v1',
           '--out', str(SP / out), '--stems', str(SP / 'st.txt'), '--split', 'val', '--cfg-w', '2.5', '--seed', '0', '--seed-mode', 'per_clip',
           '--ctl-tokens', E['CTL'], '--ctl-mode', 'auto']
    r = subprocess.run(cmd, cwd=str(tree), capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(tree)))
    if r.returncode: print(r.stderr[-1500:])
    return {s: torch.load(SP / out / f'{s}.pt', weights_only=False)['robot_repr'] for s in stems if (SP / out / f'{s}.pt').exists()}
o_, n_ = run(ORIG, 'inf_old'), run(CODE, 'inf_new')
ok('D old checkpoint loads; offline + lexicon tokens bitwise equal to the old tree', len(o_) == 2 and len(n_) == 2 and all(torch.equal(o_[s], n_[s]) for s in stems))
print('ALL_OK' if not FAIL else f'FAILED {FAIL}'); sys.exit(1 if FAIL else 0)

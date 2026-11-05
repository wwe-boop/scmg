"""Tests for the revised scripts/echog_g1/executability.py (11-05).
    PYTHONPATH=<mujoco>:$OFF/src python scripts/tests/test_executability.py <gt_dir> <stems> <pred_dir> <urdf> <mjcf> <exclusion.json> [N]
1 default arguments: every number of the original checker (git 4bbc910) is reproduced bit for bit (all old json keys).
2 --exclusion-file: two disjoint evaluation subsets use the identical excluded-pair set (= the file), also for a synthetic
  one-pair table (the real training-set table is empty), and that pair no longer appears among the violations.
3 --collision-stride 1 runs, reports GT violations per link pair, and per-pair counts are frame counts (sum >= coll frames).
"""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
NEW = os.path.join(ROOT, 'scripts/echog_g1/executability.py')
gt, stems_f, pred, urdf, mjcf, excl = sys.argv[1:7]; N = int(sys.argv[7]) if len(sys.argv) > 7 else 120
ok = True
def check(name, cond, info=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True); ok &= bool(cond)

tmp = tempfile.mkdtemp(prefix='exectest-', dir=os.environ.get('TMPDIR', '/tmp/claude-1000'))
old = os.path.join(tmp, 'executability_4bbc910.py')
open(old, 'w').write(subprocess.check_output(['git', '-C', ROOT, 'show', '4bbc910:scripts/echog_g1/executability.py'], text=True))
stems = [s.strip() for s in open(stems_f) if s.strip()]
sub = {}
for k, sl in dict(all=stems[:N], a=stems[:N // 2], b=stems[N // 2:N]).items():
    sub[k] = os.path.join(tmp, f'stems_{k}.txt'); open(sub[k], 'w').write('\n'.join(sl) + '\n')


def run(script, stems, out, *extra):
    cmd = [sys.executable, script, '--ref-dir', gt, '--stems', stems, '--urdf', urdf, '--mjcf', mjcf, '--out', out, *extra, f'pred={pred}']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: print(r.stdout, r.stderr); raise SystemExit(f'failed: {cmd}')
    return r.stdout, json.load(open(out))


# 1 default behaviour unchanged
so, jo = run(old, sub['all'], os.path.join(tmp, 'old.json'))
sn, jn = run(NEW, sub['all'], os.path.join(tmp, 'new.json'))
same = all(jo[l][k] == jn[l][k] for l in jo for k in jo[l]) and set(jo) == set(jn)
check('default args: all original json fields identical', same, f'labels {sorted(jo)}; GT any {jo["GT"]["any"]!r} pred any {jo["pred"]["any"]!r}')
check('default args: original printed lines unchanged', all(l in sn.splitlines() for l in so.splitlines()))
print(sn.strip())

# 2 fixed exclusion table: identical across disjoint subsets
syn = os.path.join(tmp, 'syn_excl.json'); rec = json.load(open(excl))
check('real table loads', 'excluded' in rec, f'{len(rec["excluded"])} pairs from {rec["n_clips"]} clips / {rec["n_frames"]} frames')
pair = ['torso_link', 'right_shoulder_yaw_link']
json.dump(dict(rec, excluded=[pair + [1.0]]), open(syn, 'w'))
for name, tab, want in (('training-set table', excl, sorted([x, y] for x, y, _ in rec['excluded'])), ('synthetic one-pair table', syn, [pair])):
    ex = []
    for k in ('a', 'b'):
        _, j = run(NEW, sub[k], os.path.join(tmp, f'ex_{k}.json'), '--exclusion-file', tab)
        ex.append(j['GT']['checker']['excluded_pairs'])
        if tab == syn:
            check(f'synthetic: excluded pair absent from violations (subset {k})',
                  all([x, y] != pair for lab in ('GT', 'pred') for x, y, _ in j[lab]['pairs_all']))
    check(f'{name}: excluded set identical on disjoint subsets and equal to the file', ex[0] == ex[1] == want, str(ex[0]))

# 3 per-frame collisions + per-pair GT listing
s1, j1 = run(NEW, sub['all'], os.path.join(tmp, 'stride1.json'), '--collision-stride', '1', '--exclusion-file', excl)
g = j1['GT']
check('stride 1: per-pair counts are frame counts (sum >= collision frames)',
      sum(c for _, _, c in g['pairs_all']) >= round(g['coll'] * g['frames']) and g['pairs_unit'].endswith('stride 1)'))
check('stride 1: GT per-pair line printed', any(l.startswith('GT violations by link pair') for l in s1.splitlines()))
print(s1.strip())
print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)

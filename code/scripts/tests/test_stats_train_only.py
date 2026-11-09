"""11-08 audit decision: train-only normalisation stats behind the default-off flag scmg.train --stats-train-only.
    BASE=<commit before the change, e.g. 23b9c87> python scripts/tests/test_stats_train_only.py      (CPU, seconds)
A  default path bitwise: scmg/train.py of BASE and of this tree have the identical AST once the new `if a.stats_train_only:` branch
   is replaced by its `else:` body and the new argparse line is dropped; every top-level def / class of BASE's scmg/data.py has an
   identical AST here (the change only ADDS train_rows / stats_train_only / load_train_only_stats). With the flag off the code that
   runs is therefore the old code, token for token.
B  flag on, synthetic view (val rows set to 1e3): stats_train_only == mean / std of the non-val rows exactly (n >= pool -> every row),
   the val rows have no influence, provenance written; load_train_only_stats computes when missing, re-reads its own file, and
   refuses a file without rows = train_only or with another SPLIT.json sha256."""
import ast, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
CODE = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(CODE)); FAIL = []


def ok(n, c, i=''):
    print(('PASS ' if c else 'FAIL ') + n + (f'  {i}' if i else ''), flush=True)
    if not c: FAIL.append(n)


def old_src(path): return subprocess.run(['git', '-C', str(CODE), 'show', f"{os.environ.get('BASE', '23b9c87')}:{path}"], capture_output=True, text=True, check=True).stdout


class Strip(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        t = node.test
        if isinstance(t, ast.Attribute) and t.attr == 'stats_train_only': return node.orelse
        return node

    def visit_Expr(self, node):
        c = node.value
        if isinstance(c, ast.Call) and getattr(c.func, 'attr', '') == 'add_argument' and c.args and getattr(c.args[0], 'value', '') == '--stats-train-only': return None
        return node


new_t = Strip().visit(ast.parse((CODE / 'scmg/train.py').read_text())); old_t = ast.parse(old_src('scmg/train.py'))
ok('A train.py default path == BASE (AST)', ast.dump(new_t) == ast.dump(old_t))
nd = {n.name: ast.dump(n) for n in ast.parse((CODE / 'scmg/data.py').read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
od = {n.name: ast.dump(n) for n in ast.parse(old_src('scmg/data.py')).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
ok('A data.py: every BASE def / class unchanged', all(nd.get(k) == v for k, v in od.items()), f'added {sorted(set(nd) - set(od))}')

from scmg.data import stats_train_only, load_train_only_stats, train_rows
with tempfile.TemporaryDirectory() as td:
    v = Path(td) / 'view'; v.mkdir(); rng = np.random.default_rng(0)
    q = rng.normal(size=(300, 34)).astype(np.float32); f = rng.normal(size=(300, 89)).astype(np.float32)
    items = [dict(id=f'r{k}#0', offset=k * 50, frames=50) for k in range(6)]; val = ['r2#0', 'r5#0']
    for it in items:
        if it['id'] in val: q[it['offset']:it['offset'] + 50] = 1e3; f[it['offset']:it['offset'] + 50] = 1e3
    np.save(v / 'train_q.npy', q); np.save(v / 'train_feat.npy', f)
    (v / 'train.json').write_text(json.dumps(items)); (v / 'SPLIT.json').write_text(json.dumps(dict(train_ids=[i['id'] for i in items if i['id'] not in val], val_ids=val)))
    r = train_rows(v); keep = np.ones(300, bool); keep[100:150] = False; keep[250:300] = False
    ok('B train_rows = rows outside val items', np.array_equal(r, np.nonzero(keep)[0]), f'{len(r)} rows')
    st = stats_train_only(v, n=10 ** 6)
    qe, fe = q[keep], f[keep]; fs = np.maximum(fe.std(0), 1e-3); fs[-1] = 1.; fm = fe.mean(0); fm[-1] = 0.
    ok('B stats == exact train-row stats', np.allclose(st['q_mean'], qe.mean(0), atol=0, rtol=0) and np.allclose(st['q_std'], np.maximum(qe.std(0), 0.03), atol=0, rtol=0)
       and np.allclose(st['f_mean'], fm, atol=0, rtol=0) and np.allclose(st['f_std'], fs, atol=0, rtol=0))
    ok('B val rows (1e3) have no influence', max(np.abs(st['q_mean']).max(), np.abs(st['f_mean']).max()) < 1.)
    ok('B provenance', st['rows'] == 'train_only' and st['n_pool_rows'] == 200 and len(st['split_sha256']) == 64)
    p = Path(td) / 'S.json'; s1, made = load_train_only_stats(p, v); p.write_text(json.dumps(s1)); s2, made2 = load_train_only_stats(p, v)
    ok('B compute when missing, re-read own file', made and not made2 and s1['q_mean'] == s2['q_mean'])
    bad = dict(s1); bad.pop('rows'); p.write_text(json.dumps(bad))
    try: load_train_only_stats(p, v); ok('B refuses a file without rows=train_only', False)
    except ValueError: ok('B refuses a file without rows=train_only', True)
    p.write_text(json.dumps(dict(s1, split_sha256='0' * 64)))
    try: load_train_only_stats(p, v); ok('B refuses another SPLIT sha', False)
    except ValueError: ok('B refuses another SPLIT sha', True)
print('ALL_OK' if not FAIL else 'FAILED ' + ', '.join(FAIL)); sys.exit(1 if FAIL else 0)

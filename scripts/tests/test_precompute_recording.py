"""11-07 regression test of precompute_causal.write --recording entries (the item-id == recording-key collision of a1bfb78):
synthetic results where one recording's only item has the recording's own id; every recording must keep its clip entry, the
colliding item's token ends must live in it, other items get token-end entries; write() + the run guard count them."""
import json, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'echog_g1'))
import precompute_causal as PC
from scmg.causal_feat import UNIT_COLS
FAIL = []
def ok(n, c, i=''):
    print(('PASS ' if c else 'FAIL ') + n + (f'  {i}' if i else '')); FAIL.append(n) if not c else None
z = lambda: (np.zeros((5, 192), np.float32), np.zeros(5, np.uint8), [np.zeros((3, 89), np.float16)] * 5, np.zeros((2, len(UNIT_COLS))))
res = [('recA', 300, *z(), [('recA#0', np.array([1., 2.], np.float32)), ('recA#1', np.array([3.], np.float32))]),
       ('recB', 120, *z(), [('recB', np.array([0.5, 0.7, 0.9], np.float32))]),           # item id == recording key
       ('recC', 80, *z(), [('recC#0', np.array([], np.float32))])]
d = Path(tempfile.mkdtemp()); PC.write(d, res); idx = json.loads((d / 'INDEX.json').read_text()); te = np.load(d / 'TOKEN_END.npy')
ok('every recording keeps its clip entry', all('feat' in idx[k] for k in ('recA', 'recB', 'recC')), str({k: sorted(v) for k, v in idx.items()}))
ok('colliding item token ends live in the recording entry', idx['recB']['tend'][1] == 3 and np.allclose(te[idx['recB']['tend'][0]:idx['recB']['tend'][0] + 3], [0.5, 0.7, 0.9]))
ok('other items have token-end entries', idx['recA#1']['tend'][1] == 1 and 'feat' not in idx['recA#1'] and idx['recC#0']['tend'][1] == 0)
ok('clip entry count == results', sum('feat' in e for e in idx.values()) == len(res))
print('ALL_OK' if not FAIL else f'FAILED {FAIL}'); sys.exit(1 if FAIL else 0)

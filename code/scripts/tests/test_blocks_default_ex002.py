"""3c-7 (D1) default-path test: ex002_cbf_dev's layer construction with --blocks none is the 3c-4b ProjectionLayer; its output on
the first 640 frames of enc-s0 psync s0 ci 7 and ci 37 equals the stored 3c-4b run (runs/ex002-proj4b-enc) bitwise (the layer is
causal, so a stream prefix reproduces the stored prefix). Also: the block layer's support rule puts same-hand finger rows in F_L / F_R.
    PYTHONPATH=.:~/bulk/pydeps/safeproj python scripts/tests/test_blocks_default_ex002.py"""
import os, sys, numpy as np
from pathlib import Path
os.environ.setdefault('SCMG_SAFETY_KKT', 'qdldl'); os.environ.setdefault('MUJOCO_GL', 'disable'); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import safety_proj as S
from scmg.safety_ex002_3f import EX002RMLimits, K_3F
from scmg.safety_proj_ex002 import EX002Collision
from scmg.safety_blocks_ex002 import block_of
E = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/ex002-sfscope-1108/enc-s0/psync/s0'; R = os.path.expanduser('~/bulk/causal-train-1105/runs/ex002-proj4b-enc/psync/s0')
lim = EX002RMLimits(); coll = EX002Collision(lim.E); fail = []
for ci in (7, 37):
    L = S.ProjectionLayer(lim, coll, 'l2', K=K_3F, B=13, d_safe=1e-3, tail_smooth=1e-3, recursive=True, max_iter=8); L.start = 'rest'; L.reset()
    q = np.load(f'{E}/{ci}.npy').astype(np.float64)[:640]; y, _ = S.run_stream(L, q, K_3F)
    ok = np.array_equal(y.astype(np.float32), np.load(f'{R}/{ci}.npy')[:640]); print(('PASS' if ok else 'FAIL'), 'default bitwise ci', ci)
    if not ok: fail.append(ci)
g = np.zeros(34); g[20] = 0.05; g[3] = 7e-17; gr = np.zeros(34); gr[27] = 0.03; gm = np.zeros(34); gm[5] = 0.02; gm[21] = 0.01
ok = (block_of(g), block_of(gr), block_of(gm)) == ('L', 'R', 'A'); print(('PASS' if ok else 'FAIL'), 'support rule')
if not ok: fail.append('support')
print('ALL_OK' if not fail else f'FAILED {fail}'); sys.exit(1 if fail else 0)

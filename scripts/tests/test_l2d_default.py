"""L2d (design D3) default-path test: rest_ramp_ex002.py without --version (= l2) reproduces the stored L2 diagnostic outputs
(eval-enc-s0-halfB/L2diag, written before the L2d change) bitwise on every interruption.
    python scripts/tests/test_l2d_default.py"""
import subprocess, sys, tempfile, os
from pathlib import Path
import numpy as np
B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; X = B + '/ex002-interrupt-1107'; E = B + '/ex002-sfscope-1108/enc-s0'
CODE = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as td:
    subprocess.run([sys.executable, str(CODE / 'scripts/ex002/rest_ramp_ex002.py'), '--in-dir', E + '/int_psync_s0', '--prefix-from', E + '/psync/s0',
                    '--out-dir', td, '--interrupts', X + '/interrupts_dev_ex002.json', '--human', X + '/ref-halfB/human_ref_ex002'], check=True, capture_output=True)
    fs = sorted(Path(td).glob('*.npy')); bad = [f.name for f in fs if not np.array_equal(np.load(f), np.load(Path(X + '/eval-enc-s0-halfB/L2diag') / f.name))]
    print('PASS' if fs and not bad else 'FAIL', 'default l2 bitwise vs stored L2diag', len(fs), 'files', len(bad), 'differ')
    sys.exit(1 if bad or not fs else 0)

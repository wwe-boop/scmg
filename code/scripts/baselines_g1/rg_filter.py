"""Official RoboGesture default (filtered) branch on an existing raw generation: offline.infer_wav(raw=False) =
CollisionFilter.filter(smooth_model_blocks(model_to_robot(raw)), zero legs) -> qpos[:,7:60] = expand @ control.
Identical to the filtered branch of baseline-repro-1104/scripts/rg_generate.py (which only filtered seed 0).
Run with the RoboGesture env (robogesture-py310) on zehan-server, 1 core.
  rg_filter.py --raw <gen/raw_seedK> --out <filtered_seedK> --stems <txt>
"""
import argparse, sys, time
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/robogesture')
sys.path.insert(0, str(ROOT / 'official'))
from robogesture.motion import model_to_robot, smooth_model_blocks  # noqa: E402
from robogesture.collision import CollisionFilter  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--raw', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--stems', required=True); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
filt = CollisionFilter()
for i, s in enumerate(l.strip() for l in open(a.stems) if l.strip()):
    fp = out / f'{s}.npz'
    if fp.exists():
        continue
    raw = np.load(Path(a.raw) / f'{s}.npz')['raw_motion41']
    m30 = smooth_model_blocks(model_to_robot(raw)); t = time.perf_counter()
    res = filt.filter(m30, np.zeros(12), np.zeros(30, dtype=np.float32))
    q = np.zeros((len(raw), 60), np.float32)
    for k, fr in enumerate(res.frames):
        c, _ = filt._to_control(fr); q[k, 7:60] = filt.expand @ c
    np.savez_compressed(fp, qpos=q, fps=np.float32(30), safe=res.safe, corrected=res.corrected, held=res.held,
                        minimum_clearance=res.minimum_clearance, filter_seconds=np.float64(time.perf_counter() - t))
    print('FILT', i, s, res.safe, res.corrected, res.held, flush=True)
print('ALL_DONE', flush=True)

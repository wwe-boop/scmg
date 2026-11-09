"""RoboGesture official weights on one wav (fig-1109), seed 0, raw + official filtered branch; same calls as baseline-repro-1104/scripts/rg_generate.py."""
import os, sys, inspect, textwrap
from pathlib import Path
import numpy as np
ROOT = Path("/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/robogesture"); OUT = Path(sys.argv[2]); WAV = Path(sys.argv[1])
os.environ.setdefault("ROBOGESTURE_WEIGHTS_ROOT", str(ROOT / "weights")); sys.path.insert(0, str(ROOT / "official"))
import torch
import robogesture.config as config
config.MOTION_MODEL_PATH = ROOT / "weights" / "model.pt"
from robogesture import model as official
from robogesture.offline import read_wav
from robogesture.motion import model_to_robot, smooth_model_blocks
from robogesture.collision import CollisionFilter
torch.set_num_threads(2); gen = official.MotionGenerator("cuda")
ref = np.load(ROOT / "smoke/official_raw.npz")["raw_motion41"]
assert np.array_equal(gen.generate(read_wav(ROOT / "official/examples/sample_06.wav")), ref); print("PARITY_OK", flush=True)
raw = gen.generate(read_wav(WAV)); OUT.mkdir(parents=True, exist_ok=True)
q = np.zeros((len(raw), 60), np.float32); q[:, 19:] = raw
np.savez_compressed(OUT / "raw" / f"{WAV.stem}.npz" if False else OUT / f"raw_{WAV.stem}.npz", qpos=q, raw_motion41=raw, fps=np.float32(30))
filt = CollisionFilter(); m30 = smooth_model_blocks(model_to_robot(raw)); res = filt.filter(m30, np.zeros(12), np.zeros(30, dtype=np.float32))
q = np.zeros((len(raw), 60), np.float32)
for k, fr in enumerate(res.frames):
    c, _ = filt._to_control(fr); q[k, 7:60] = filt.expand @ c
np.savez_compressed(OUT / f"filtered_{WAV.stem}.npz", qpos=q, fps=np.float32(30), safe=res.safe, corrected=res.corrected, held=res.held)
print("RG_DONE", len(raw), res.safe, res.corrected, res.held, flush=True)

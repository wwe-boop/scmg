"""Run the unmodified EMAGE/GestureLSM SMPL-X -> G1 converter (baseline-repro-1104/gmr-repro/scripts/smplx_npz_to_g1.py:
GMR unitree_g1 smplx_to_g1.json, height 1.66+0.1*beta0, Savitzky-Golay(7,2) on qpos, scripts/g1_repr.py encoding) from any
host.  The original tree links weights/smplx_models and third_party/GMR/assets to zehan-server-local paths
(/workspace/...) that the 5090 host and DLC pods cannot see, so (1) the module constant R (used only for
R/'weights/smplx_models') is pointed at <assets> holding weights/smplx_models/smplx/SMPLX_NEUTRAL.npz (byte-identical
SMPLX_NEUTRAL_2020.npz, sha256 bdf06146...), and (2) general_motion_retargeting is imported from <assets>/GMR, an rsync of
the same GMR checkout (commit 25f56db, code unchanged) plus its assets/unitree_g1.  The converter, calibration helpers
(gmr_calib / smooth_qpos / g1_repr) and every option come from the original CPFS tree.

  gmr_convert.py [--assets DIR] -- <smplx_npz_to_g1.py args: --src --out --stems --shard i/n --generator ...>
"""
import argparse, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--assets', default='/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baselines-g1-1105/assets')
ap.add_argument('--gmr-root', default='/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/gmr-repro')
a, rest = ap.parse_known_args()
rest = rest[1:] if rest[:1] == ['--'] else rest
root = Path(a.assets)
assert (root / 'weights/smplx_models/smplx/SMPLX_NEUTRAL.npz').exists(), root
assert (root / 'GMR/assets/unitree_g1/g1_mocap_29dof.xml').exists(), root
sys.path.insert(0, str(root / 'GMR'))
import general_motion_retargeting  # noqa: E402,F401  (cached before the converter prepends the original GMR path)
assert Path(general_motion_retargeting.__file__).resolve().is_relative_to((root / 'GMR').resolve())
sys.path.insert(0, str(Path(a.gmr_root) / 'scripts'))
import smplx_npz_to_g1 as M  # noqa: E402  (inserts <gmr-root>/third_party/GMR itself)
M.R = root
sys.argv = ['smplx_npz_to_g1.py'] + rest
M.main()

"""Inference-time joint-limit layer (11-05, G1 safety): clamp the 29 joints of each robot_repr .pt into the Unitree URDF
[lower + margin, upper - margin] (scmg/g1_safety.URDF_LIMITS, canonical G1_JOINT_NAMES order). Root channels 0:10 are
copied. Everything else in the .pt dict is kept; 'generator' gets a suffix.
    python scripts/echog_g1/clamp_limits.py --in-dir D --out-dir D_clamp [--margin 0.0]
Needs PYTHONPATH=<ECHO-G official>/src (joint names).
"""
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import g1_safety as S

ap = argparse.ArgumentParser(); ap.add_argument('--in-dir', required=True); ap.add_argument('--out-dir', required=True)
ap.add_argument('--margin', type=float, default=0.0); a = ap.parse_args()
c = S._consts('cpu', torch.float64); lo, hi = c['lo'] + a.margin, c['hi'] - a.margin
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
n = ch = fr = 0
for p in sorted(Path(a.in_dir).glob('*.pt')):
    x = torch.load(p, map_location='cpu', weights_only=False)
    rr = torch.as_tensor(x['robot_repr']).double(); q = rr[:, 10:]
    qc = torch.minimum(torch.maximum(q, lo), hi)
    ch += int((qc != q).any(1).sum()); fr += len(q); n += 1
    rr2 = rr.clone(); rr2[:, 10:] = qc
    x['robot_repr'] = rr2.to(torch.as_tensor(x['robot_repr']).dtype)
    x['generator'] = str(x.get('generator', '')) + f' | clamp_limits margin {a.margin}'
    torch.save(x, out / p.name)
print(f'{n} clips, {fr} frames, clamped frames {ch} ({100 * ch / max(fr, 1):.2f}%)')

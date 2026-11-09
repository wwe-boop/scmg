"""Resampling floor of the official metrics: GT robot_repr[:aligned_frames] -> 50 FPS -> 30 FPS (scmg/g1.py), written as
an evaluator-ready prediction dir. Evaluating it against the untouched GT shows what the 30 <-> 50 conversion alone does
to FGD / Div / BA / jerk / foot metrics (a perfect 50 Hz model would score this).
    gt_roundtrip.py <motion_39d_30fps dir> <audit csv> <stem list> <out dir> [cubic|linear]"""
import csv, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import g1

mdir, audit, lst, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]); kind = sys.argv[5] if len(sys.argv) > 5 else 'cubic'
al = {r['stem']: int(r['aligned_frames']) for r in csv.DictReader(open(audit))}
out.mkdir(parents=True, exist_ok=True)
stems = [s for s in Path(lst).read_text().split() if s]
for s in stems:
    rr = torch.load(mdir / f'{s}.pt', map_location='cpu', weights_only=True)['robot_repr'].double().numpy()[:al[s]]
    rb = g1.to30(g1.to50(rr, kind), len(rr), kind)
    torch.save(dict(robot_repr=torch.from_numpy(rb).float(), real_num_frames=len(rb), fps=30.0, representation_units='physical',
                    representation_schema='projecthermes-g1-39d-standard-v2', source=s, roundtrip=kind), out / f'{s}.pt')
print('ROUNDTRIP_DONE', len(stems), kind)

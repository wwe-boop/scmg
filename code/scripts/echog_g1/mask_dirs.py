"""Report-only 'leg masking' for the official evaluator (default OFF; the official ECHO-G benchmark always uses all 39
channels). Writes masked copies of a prediction dir AND the GT dir with the same rule (scmg/g1.py mask_channels):
  legs       12 leg joints (channels 10..21) -> 0 rad
  legs_root  + root (0..9): identity orientation, yaw change 0, root velocity 0
Then run the official eval_g1_motion_cls.py on <out>/pred and <out>/gt (same flags as the full benchmark) and report the
numbers as an extra 'upper body' column next to the official 39-channel result.
    mask_dirs.py <legs|legs_root> <pred dir> <gt motion dir> <stem list> <out dir>"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import g1

mode, pred, gt, lst, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
stems = [s for s in Path(lst).read_text().split() if s]
for sub, src in (('pred', pred), ('gt', gt)):
    (out / sub).mkdir(parents=True, exist_ok=True)
    for s in stems:
        d = torch.load(src / f'{s}.pt', map_location='cpu', weights_only=False)
        d = dict(d) if isinstance(d, dict) else dict(robot_repr=d)
        d['robot_repr'] = torch.from_numpy(g1.mask_channels(d['robot_repr'].double().numpy(), mode)).float()
        d['leg_mask'] = mode
        torch.save(d, out / sub / f'{s}.pt')
print('MASKED', mode, len(stems), out)

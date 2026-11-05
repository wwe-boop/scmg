"""Negative-control motion sources for the G1 semantic metric (semantic-baseline-1105), written as <stem>.pt like generator outputs.
  gt-mismatch   real human motion of ANOTHER val_common clip of similar length (sorted by length, neighbour swap; cropped /
                edge-held to this clip's length): human-like motion with no relation to this clip's speech -> floor ~0.5
  gt-shift2s    this clip's own ground truth circularly shifted by 2 s: same speaker / content, timing broken
    python scripts/semantic/g1_make_controls.py --data ~/bulk/semantic-baseline-1105/g1/data --out ~/bulk/semantic-baseline-1105/g1/outputs
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); d = Path(a.data).expanduser(); out = Path(a.out).expanduser()
    z = np.load(d / 'MOTION.npz'); sp = json.loads((d / 'EVENTS.json').read_text())['split']
    stems = [str(s) for s in z['stems']]; te = [i for i, s in enumerate(stems) if sp[s] == 'test']
    Q, OFF, LEN = z['q'], z['off'], z['len']
    get = lambda i: Q[OFF[i]:OFF[i] + LEN[i]]
    order = sorted(te, key=lambda i: z['len'][i])
    partner = {order[k]: order[k + 1 if k % 2 == 0 and k + 1 < len(order) else k - 1] for k in range(len(order))}
    for name in ('gt-mismatch', 'gt-shift2s'): (out / name).mkdir(parents=True, exist_ok=True)
    for i in te:
        L = int(z['len'][i]); x = get(partner[i])
        x = x[:L] if len(x) >= L else np.concatenate([x, np.repeat(x[-1:], L - len(x), 0)])
        torch.save(dict(robot_repr=torch.tensor(x), real_num_frames=L, fps=30.0, control='gt-mismatch', partner=stems[partner[i]]),
                   out / 'gt-mismatch' / f'{stems[i]}.pt')
        torch.save(dict(robot_repr=torch.tensor(np.roll(get(i), 60, 0)), real_num_frames=L, fps=30.0, control='gt-shift2s'),
                   out / 'gt-shift2s' / f'{stems[i]}.pt')
    print('wrote', len(te), 'clips x 2 controls')


if __name__ == '__main__':
    main()

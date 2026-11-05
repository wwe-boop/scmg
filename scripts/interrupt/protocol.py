"""Interruption protocol list (11-05): which clips are interrupted where.

Clips: utterances of >= --min-dur s (clip duration = ECHO-G 30 fps frame count / 30) of a stem list; deterministic subset =
the first --n stems in sha1(stem) order. Each clip is interrupted at 30 % / 50 % / 70 % of its length: interrupt frame
f_i = round(frac * n30) (30 fps; t_i = f_i / 30 s; the robot's motion up to frame f_i - 1 is the normal output).
    python scripts/interrupt/protocol.py --stems val_common.txt --motion-dir <GT 30 fps> --n 600 --out interrupts_test.json
"""
import argparse, hashlib, json
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--stems', required=True); ap.add_argument('--motion-dir', required=True); ap.add_argument('--n', type=int, default=600)
ap.add_argument('--min-dur', type=float, default=3.0); ap.add_argument('--fracs', default='0.3,0.5,0.7'); ap.add_argument('--out', required=True)
a = ap.parse_args()
stems = sorted(open(a.stems).read().split(), key=lambda s: hashlib.sha1(s.encode()).hexdigest())
rows, n_ok, n_seen = [], 0, 0
for s in stems:
    if n_ok >= a.n: break
    n30 = int(torch.load(f'{a.motion_dir}/{s}.pt', map_location='cpu', weights_only=False)['real_num_frames']); n_seen += 1
    if n30 / 30. < a.min_dur: continue
    n_ok += 1
    for fr in (float(x) for x in a.fracs.split(',')):
        f = int(round(fr * n30)); rows.append(dict(stem=s, frac=fr, n30=n30, f_i=f, t_i=f / 30.))
json.dump(dict(stems_file=a.stems, n_clips=n_ok, scanned=n_seen, min_dur=a.min_dur, rule='sha1(stem) order, first n with duration >= min_dur',
               interrupts=rows), open(a.out, 'w'), indent=0)
print(a.out, n_ok, 'clips', len(rows), 'interrupts (scanned', n_seen, ')')

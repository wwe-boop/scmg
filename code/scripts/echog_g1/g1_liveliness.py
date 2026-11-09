"""11-08 G1 liveliness (coordinator: main-table claims must not hide under-motion; read-only on existing outputs): the EX002
static_share.py definitions carried to G1 -- per frame the speeds (central difference, 30 fps) of the 6 upper-body points shoulder
(roll link) / elbow / wrist (yaw link) x left / right from the official pose-only FK (identity base: joint motion only), each divided by
that point's mean speed over the dev GT (D-spk dev 400, fixed normaliser); still = mean normalised speed < 0.15; wrist speed = mean of
both wrists (m/s); wrist amplitude = mean per-axis std of the wrist positions per clip (m).
    PYTHONPATH=<ECHO-G official src> python g1_liveliness.py --norm-gt GTDIR --norm-stems S --stems S2 --out O.json label=<dir of <stem>.pt> ..."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from echo_g.evaluation import g1_motion_cls as E
from echo_g.evaluation import g1_kinematics as K

BODIES = ['left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_elbow_link', 'right_elbow_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link']
IDX = [K.G1_BFS_BODY_NAMES.index(b) for b in BODIES]


def pts(pt):
    x = torch.load(pt, map_location='cpu', weights_only=False); r = x['robot_repr'] if isinstance(x, dict) else x
    n = int(x.get('real_num_frames', len(r))) if isinstance(x, dict) else len(r)
    P = np.asarray(E.fk_positions_ba(torch.as_tensor(np.asarray(r[:n]), dtype=torch.float64)))
    return P[:, IDX]


def speed(P):
    v = np.zeros(P.shape[:2]); v[1:-1] = np.linalg.norm(P[2:] - P[:-2], axis=-1) * 15.; v[0], v[-1] = v[1], v[-2]; return v


ap = argparse.ArgumentParser(); ap.add_argument('--norm-gt', required=True); ap.add_argument('--norm-stems', required=True)
ap.add_argument('--stems', required=True); ap.add_argument('--out', required=True); ap.add_argument('--thr', type=float, default=.15)
ap.add_argument('specs', nargs='+')
a = ap.parse_args()
ns = [l.strip() for l in open(a.norm_stems) if l.strip()]
norm = np.concatenate([speed(pts(Path(a.norm_gt) / f'{s}.pt')) for s in ns]).mean(0)
stems = [l.strip() for l in open(a.stems) if l.strip()]; res = dict(_meta=dict(norm=norm.tolist(), norm_stems=len(ns), bodies=BODIES, thr=a.thr))
for spec in a.specs:
    lab, d = spec.split('=', 1); st = fr = 0; ws, wa = [], []
    for s in stems:
        p = Path(d) / f'{s}.pt'
        if not p.exists(): continue
        P = pts(p); sp = speed(P); vv = (sp / norm).mean(1); st += int((vv < a.thr).sum()); fr += len(P)
        ws.append(float(sp[:, 4:6].mean())); wa.append(float(P[:, 4:6].std(0).mean()))
    res[lab] = dict(clips=len(ws), frames=fr, still_share=st / max(fr, 1), wrist_speed=float(np.mean(ws)) if ws else None, wrist_amp=float(np.mean(wa)) if wa else None)
    print(lab, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[lab].items()}), flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))

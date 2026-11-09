"""Unified metrics for SAVED trajectories of any model (same dev clips / GT / metric code as eval_dots.py and scmg.evaluate).
    python scripts/eval_traj.py <gen dir> <out.json> [--seed1 <dir>] [--period 50] [--first 0]
gen dir holds gen_Hinf_<ci>.npy and gt_<ci>.npy ([T, 34] rad, 50 Hz) for the eval_dots.py clip order.
--period/--first: the generator's own chunk grid in 50 Hz frames (RoboGesture: 1 s chunks -> 50; SCMG/dots: 16)."""
import argparse, json
from pathlib import Path
import numpy as np
from scmg.common import PROSODY_COLS
from scmg.data import View, WindowDataset, VIEW
from scmg.evaluate import kin_feats, frechet, beat_align

ap = argparse.ArgumentParser()
ap.add_argument('gen'); ap.add_argument('out'); ap.add_argument('--seed1', default=None)
ap.add_argument('--period', type=int, default=16); ap.add_argument('--first', type=int, default=0)
a = ap.parse_args()
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st = json.load(open(D + '/STATS_v36_1.json'))
v = View(VIEW, 'val', D + '/text-e5-v36_1', st, mmap=True); v.glitch = None
meta = json.load(open(f'{VIEW}/META.json')); lo, hi = np.array(meta['lower']), np.array(meta['upper']); A = slice(4, 18)


def boundary(q):
    j = np.linalg.norm(np.diff(q[:, A], 3, axis=0), axis=-1); f = np.arange(len(j)) + 1
    r = (f - a.first) % a.period; at = (r <= 1) | (r >= a.period - 1)
    return float(j[at].mean() / max(j[~at].mean(), 1e-9))


clips = [ci for ci in range(len(v.clips)) if v.clips[ci]['frames'] // 16 >= 3][:43]
G = Path(a.gen); Fg, Ft, ba, bat, br, brt, sp, dv, lim, mx, per = [], [], [], [], [], [], [], [], [], [], []
for ci in clips:
    f = G / f'gen_Hinf_{ci}.npy'
    if not f.exists(): continue
    y = np.load(f).astype(np.float64); gt = np.load(G / f'gt_{ci}.npy').astype(np.float64); T = min(len(y), len(gt))
    y, gt = y[:T], gt[:T]
    s = WindowDataset(v, T // 16 * 16, ref_frames=16, max_tokens=64).sample(ci, 0, np.random.default_rng(ci))
    beat = np.clip(s['feat'][:, PROSODY_COLS['onset']] + s['feat'][:, PROSODY_COLS['beat']], 0, None)
    n = min(T, len(beat)); y, gt, beat = y[:n], gt[:n], beat[:n]
    Fg += kin_feats(y); Ft += kin_feats(gt)
    ba.append(beat_align(y, beat)); bat.append(beat_align(gt, beat)); br.append(boundary(y)); brt.append(boundary(gt))
    sp.append(np.abs(np.diff(y[:, A], axis=0)).mean() / max(np.abs(np.diff(gt[:, A], axis=0)).mean(), 1e-6))
    lim.append(float(((y < lo) | (y > hi)).mean())); mx.append(float(np.abs(y).max()))
    if a.seed1 and (Path(a.seed1) / f'gen_Hinf_{ci}.npy').exists():
        y1 = np.load(Path(a.seed1) / f'gen_Hinf_{ci}.npy')[:n]; dv.append(float(np.abs(y1[:, A] - y[:, A]).mean()))
    per.append(dict(ci=ci, speed=float(sp[-1]), beat=float(ba[-1]), lang=v.clips[ci]['items'][0].get('lang')))
res = dict(clips=len(sp), FKD=frechet(Fg, Ft), beat_align=float(np.mean(ba)), beat_align_gt=float(np.mean(bat)),
           boundary=float(np.mean(br)), boundary_gt=float(np.mean(brt)), speed_ratio=float(np.mean(sp)),
           diversity=float(np.mean(dv)) if dv else None, joint_limit_fraction=float(np.mean(lim)), max_abs_rad=float(np.max(mx)),
           speed_ratio_en=float(np.mean([p['speed'] for p in per if p['lang'] == 'en'] or [np.nan])),
           speed_ratio_zh=float(np.mean([p['speed'] for p in per if p['lang'] == 'zh'] or [np.nan])))
Path(a.out).write_text(json.dumps(dict(inf=res, per_clip=per), indent=1)); print(json.dumps(res))

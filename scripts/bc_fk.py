"""BC (beat consistency) with the OFFICIAL position-based definition, on EX002 via exact forward kinematics.
    python scripts/bc_fk.py <proxy.npz> <onsets.json> <out.json> <label=dir> [<label=dir> ...]
EMAGE / BEAT (references/metric.py `alignment`): per upper-body joint, speed = |d position / dt| (3-D, metres) normalised by
the joint's mean speed over the training data; motion beats = local minima (argrelextrema order 7 @ 30 fps -> 12 @ 50 fps)
with normalised speed > 0.3; score per joint = mean over AUDIO onsets of exp(-d^2 / (2 sigma^2)), d = distance to the
nearest motion beat, sigma 0.3 s; BC = mean over joints.  The previous EX002 port used JOINT-ANGLE velocities: angle
minima are dense and noisy, so BC did not drop when audio and motion were misaligned and random motion scored 0.78.
Upper-body joints here = EX002 shoulder / elbow / wrist / hand bodies of both arms (positions from scmg.kinematics FK).
Validation (`--validate`): ground truth vs ground truth with audio shifted by +-0.5 / +-1 s, another clip's motion, and
i.i.d. random motion; BC must drop under misalignment."""
import argparse, glob, json
from pathlib import Path
import numpy as np, torch
from scipy.signal import argrelextrema
from scmg.kinematics import Kinematics

ap = argparse.ArgumentParser()
ap.add_argument('proxy'); ap.add_argument('onsets'); ap.add_argument('out'); ap.add_argument('dirs', nargs='*')
ap.add_argument('--bodies', default=''); ap.add_argument('--validate', default='')
ap.add_argument('--sigma', type=float, default=0.3); ap.add_argument('--fps', type=int, default=50)
a = ap.parse_args()
kin = Kinematics(a.proxy, 'cpu')
z = np.load(a.proxy)
# body indices of the arm chains: every body driven (directly) by an arm joint, plus the hand base bodies
names = [str(x) for x in z['joints']]
arm_j = [i for i, n in enumerate(names) if 'arm_joint' in n]
bodies = [b for b in range(kin.nb) if kin.jidx[b] in arm_j]
if a.bodies: bodies = [int(x) for x in a.bodies.split(',')]
order = max(1, round(7 * a.fps / 30))


def positions(q):
    with torch.no_grad():
        _, p = kin.fk(torch.as_tensor(np.asarray(q, np.float32)))
    return p[:, bodies].numpy()                                    # [T, nb_sel, 3]


def speeds(q):
    p = positions(q)
    v = np.gradient(p, 1 / a.fps, axis=0)
    return np.linalg.norm(v, axis=-1)                               # [T, nb_sel] m/s


def bc(sp, onsets, scale):
    on = np.asarray(onsets, float); on = on[(on >= 0) & (on < len(sp) / a.fps)]
    if not len(on): return None
    s = sp / np.maximum(scale, 1e-8); out = []
    for j in range(s.shape[1]):
        ids = argrelextrema(s[:, j], np.less, order=order)[0]; ids = ids[s[ids, j] > 0.3]
        if not len(ids): out.append(0.); continue
        d = np.abs(on[:, None] - ids[None] / a.fps).min(1); out.append(float(np.exp(-d * d / (2 * a.sigma ** 2)).mean()))
    return float(np.mean(out))


ons = json.load(open(a.onsets)); ons = ons.get('onsets', ons) if isinstance(ons, dict) else ons
def onset_of(ci):
    v = ons[str(ci)] if str(ci) in ons else ons[int(ci)]
    return v['times_s'] if isinstance(v, dict) else v
pairs = {}
for spec in a.dirs:
    lab, d = spec.split('=', 1)
    pairs[lab] = {int(Path(f).stem.split('_')[-1]): f for f in glob.glob(f'{d}/gen_Hinf_*.npy')}
gt_dir = a.validate or (a.dirs[0].split('=', 1)[1] if a.dirs else '')
gts = {int(Path(f).stem.split('_')[-1]): np.load(f) for f in glob.glob(f'{gt_dir}/gt_*.npy')}
cis = sorted(gts)
# per-body speed normalisation from the ground truth of the dev clips (training-data scale in the official code;
# dev GT is used here because it is what every row is compared against -- same scale for all rows)
scale = np.mean(np.concatenate([speeds(gts[ci]) for ci in cis]), 0)
res = {}
if a.validate:
    rng = np.random.default_rng(0); rows = {}
    for name, fn in [('gt', lambda ci, sp: bc(sp, onset_of(ci), scale))] + \
            [(f'shift{s:+.1f}s', (lambda s: lambda ci, sp: bc(sp, np.asarray(onset_of(ci)) + s, scale))(s)) for s in (0.5, 1.0, -0.5, -1.0)]:
        rows[name] = float(np.mean([v for v in (fn(ci, speeds(gts[ci])) for ci in cis) if v is not None]))
    other = []
    for i, ci in enumerate(cis):
        cj = cis[(i + 7) % len(cis)]; n = min(len(gts[ci]), len(gts[cj]))
        other.append(bc(speeds(gts[cj][:n]), onset_of(ci), scale))
    rows['other_clip_motion'] = float(np.mean([v for v in other if v is not None]))
    rnd = [bc(speeds(gts[ci][0] + np.cumsum(rng.normal(0, 0.01, gts[ci].shape), 0)), onset_of(ci), scale) for ci in cis]
    rows['random_walk_motion'] = float(np.mean([v for v in rnd if v is not None]))
    res['validation'] = rows; print(json.dumps(rows, indent=1))
for lab, files in pairs.items():
    vals = [bc(speeds(np.load(f)), onset_of(ci), scale) for ci, f in sorted(files.items()) if ci in gts]
    res[lab] = dict(BC=float(np.mean([v for v in vals if v is not None])), clips=len(vals)); print(lab, res[lab])
res['meta'] = dict(bodies=bodies, order=order, sigma=a.sigma, fps=a.fps, clips=len(cis))
Path(a.out).write_text(json.dumps(res, indent=1))

"""Unified evaluation of a dots-motion checkpoint (direct joint-space AR flow) on the SAME dev clips and metrics as
SCMG (scmg/evaluate.py): FKD, speed ratio, beat alignment, chunk-boundary jerk ratio, diversity, joint-limit fraction.
Codex's own evaluator/status are NOT used (user 11-02: only its code counts).
    python scripts/eval_dots.py <dots code dir> <ckpt> <out dir> [n_clips] [ode_steps]
Generation: start from the clip's first two GT frames (legal q0/dq0), then free-running 16-frame chunks with the
model's own history (no clamp / filter), 320 ms audio look-ahead as the model was trained. Trajectories of the first
8 clips are saved as gen_Hinf_<ci>.npy / gt_<ci>.npy (rad) -> scripts/physics_eval.py and render_compare.py."""
import json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, sys.argv[1])
from dots_motion.model import MotionFlow, build_chunk              # noqa: E402
from scmg.data import View, WindowDataset, collate, VIEW            # noqa: E402
from scmg.evaluate import kin_feats, frechet, beat_align, boundary_ratio   # noqa: E402
from scmg.common import PROSODY_COLS                                 # noqa: E402

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
ck = torch.load(sys.argv[2], map_location='cpu', weights_only=False); out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
NC = int(sys.argv[4]) if len(sys.argv) > 4 else 43; STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 8
st = ck['stats']; dev = 'cuda'
meta = json.load(open(f'{VIEW}/META.json')); lo, hi = np.array(meta['lower']), np.array(meta['upper'])
v = View(VIEW, 'val', D + '/text-e5-v36_1', st, mmap=True)
v.glitch = None                                                        # evaluate against the raw ground truth
model = MotionFlow(**ck['model_config']).to(dev); model.load_state_dict(ck['model']); model.eval()
qm, qs = np.asarray(st['q_mean'], np.float32), np.asarray(st['q_std'], np.float32)
A = slice(4, 18)


def boundary_dots(q, first=2):
    """Jerk at dots chunk boundaries (chunks start at frame 2, 18, 34, ...) / jerk elsewhere."""
    j = np.linalg.norm(np.diff(q[:, A], 3, axis=0), axis=-1); f = np.arange(len(j)) + 1
    r = (f - first) % 16; at = (r <= 1) | (r >= 15)
    return float(j[at].mean() / max(j[~at].mean(), 1e-9))


@torch.no_grad()
def rollout(s, T, seed):
    b = {k: x.to(dev) for k, x in collate([s]).items()}
    for k in list(b):
        if b[k].is_floating_point(): b[k] = b[k].float()
    work = {k: x.clone() for k, x in b.items()}
    g = torch.Generator(device=dev).manual_seed(seed)
    with torch.autocast('cuda', torch.bfloat16):
        for index in range(1, T - 1, 16):
            c, _, _ = build_chunk(work, index, model.history)
            pred = model.sample(c, steps=STEPS, noise=torch.randn(1, 16, 34, device=dev, generator=g)).float()
            n = min(16, T - 1 - index); work['q'][:, index + 1:index + 1 + n] = pred[:, :n]
    return work['q'][0].float().cpu().numpy() * qs + qm


Fg, Ft, ba, bat, br, brt, sp, dv, lim_frac, maxabs = [], [], [], [], [], [], [], [], [], []
clips = [ci for ci in range(len(v.clips)) if v.clips[ci]['frames'] // 16 >= 3][:NC]
for ci in clips:
    T = v.clips[ci]['frames'] // 16 * 16
    s = WindowDataset(v, T, ref_frames=16, max_tokens=64).sample(ci, 0, np.random.default_rng(ci))
    gt = s['q'] * qs + qm
    beat = np.clip(s['feat'][:, PROSODY_COLS['onset']] + s['feat'][:, PROSODY_COLS['beat']], 0, None)
    y = rollout(s, T, 0)
    Fg += kin_feats(y); Ft += kin_feats(gt)
    ba.append(beat_align(y, beat)); bat.append(beat_align(gt, beat)); br.append(boundary_dots(y)); brt.append(boundary_dots(gt))
    sp.append(np.abs(np.diff(y[:, A], axis=0)).mean() / max(np.abs(np.diff(gt[:, A], axis=0)).mean(), 1e-6))
    lim_frac.append(float(((y < lo) | (y > hi)).mean())); maxabs.append(float(np.abs(y).max()))
    if len(dv) < 20: dv.append(float(np.abs(rollout(s, T, 1)[:, A] - y[:, A]).mean()))
    if ci < 8: np.save(out / f'gen_Hinf_{ci}.npy', y); np.save(out / f'gt_{ci}.npy', gt)
res = {'inf': dict(clips=len(clips), FKD=frechet(Fg, Ft), beat_align=float(np.mean(ba)), beat_align_gt=float(np.mean(bat)),
                   boundary=float(np.mean(br)), boundary_gt=float(np.mean(brt)), speed_ratio=float(np.mean(sp)),
                   diversity=float(np.mean(dv)), joint_limit_fraction=float(np.mean(lim_frac)), max_abs_rad=float(np.max(maxabs)),
                   tick_ms_p50=float('nan'), tick_ms_p95=float('nan'))}
(out / 'RESULTS.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res))

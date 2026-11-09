"""Probe real checkpoints for startup continuity from multiple joint poses.

Run from a frozen copy of the SCMG package; writes raw trials and JSON metrics.
No smoothing, safety projection, or changes to the generator are applied.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from scmg.common import CHUNK_FRAMES, J, U80
from scmg.data import View, WindowDataset, VIEW
from scmg.infer import Streamer
from scmg.model import INF
from scmg.safety import limits34
from scmg.train import build_model
from scmg.vae import MotionVAE

ARMS = slice(4, 18)
COND = ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm')


def digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def metrics(cmd, q0, lim):
    # Three stationary history frames make the initial velocity/acceleration zero.
    startup = np.concatenate([np.repeat(q0[None], 3, axis=0), cmd[:CHUNK_FRAMES]])
    vel = np.diff(startup, axis=0) * 50
    acc = np.diff(vel, axis=0) * 50
    jerk = np.diff(acc, axis=0) * 50
    first = np.abs(cmd[0] - q0)
    return dict(
        finite=bool(np.isfinite(cmd).all()),
        first_arm_max_deg=float(np.rad2deg(first[ARMS].max())),
        first_all_max_deg=float(np.rad2deg(first.max())),
        first_arm_mean_deg=float(np.rad2deg(first[ARMS].mean())),
        first_arm_speed_ratio=float((first[ARMS] * 50 / lim['v'][ARMS]).max()),
        startup_arm_speed_ratio=float((np.abs(vel[:, ARMS]) / lim['v'][ARMS]).max()),
        startup_arm_acc_ratio=float((np.abs(acc[:, ARMS]) / lim['a'][ARMS]).max()),
        startup_arm_jerk_ratio=float((np.abs(jerk[:, ARMS]) / lim['j'][ARMS]).max()),
        position_violation_frames=int(((cmd < lim['lo']) | (cmd > lim['hi'])).any(-1).sum()),
    )


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--vae', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--chunks', type=int, default=6); ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--view', default=VIEW)
    ap.add_argument('--data', default='/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg')
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    stats = json.loads((Path(args.data) / 'STATS_v36_1.json').read_text())
    meta = json.loads((Path(args.view) / 'META.json').read_text())
    qm, qs = np.asarray(stats['q_mean'], np.float32), np.asarray(stats['q_std'], np.float32)
    lo, hi = np.array(meta['lower']), np.array(meta['upper'])
    lim = limits34(lo, hi, margin=0)
    clip_pose = lambda q: np.clip(q, lo + .005, hi - .005).astype(np.float32)
    view = View(args.view, 'val', str(Path(args.data) / 'text-e5-v36_1'), stats, mmap=True)
    available = [i for i, c in enumerate(view.clips) if c['frames'] >= args.chunks * CHUNK_FRAMES]
    # Spread the fixed sample across the dev list instead of taking adjacent clips.
    selected = [available[i] for i in np.linspace(0, len(available)-1, args.n, dtype=int)]
    ds = WindowDataset(view, args.chunks * CHUNK_FRAMES, max_tokens=256)
    samples = [ds.sample(i, 0, np.random.default_rng(i)) for i in selected]
    print('loading', args.ckpt, flush=True)
    vc = torch.load(args.vae, map_location='cpu', weights_only=False)
    vae = MotionVAE(**vc['cfg']); vae.load_state_dict(vc['ema']); vae.to('cuda').eval()
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model, _ = build_model(ck['base_cfg'], {}, ck['extra_cfg'])
    model.load_state_dict(ck['ema'])
    ident = dict(path=args.ckpt, step=ck['step'], stage=ck['stage'], ema_sha256=digest(ck['ema']),
                 vae_path=args.vae, vae_ema_sha256=digest(vc['ema']), config=model.cfg.dict(), sc=ck.get('sc'))
    sc = ck.get('sc') or {}
    ctx_t = float(sc.get('ctx_infer_t', .9)) if sc.get('ctx_noise_p', 0) > 0 else 1.
    model.to('cuda').eval(); del ck, vc
    (out / 'MODEL.json').write_text(json.dumps(ident, indent=2))
    records = []; trajectories = {}; t0 = time.time()
    for si, (ci, sample) in enumerate(zip(selected, samples)):
        gt = sample['q'] * qs + qm
        cond = {k: sample[k] for k in COND}
        alternate = samples[(si+1) % len(samples)]['q'][-1] * qs + qm
        plus, minus = qm.copy(), qm.copy(); plus[ARMS] += .3; minus[ARMS] -= .3
        rng = np.random.default_rng(9000 + ci)
        random_pose = qm.copy(); random_pose[ARMS] = rng.uniform(lo[ARMS]+.05, hi[ARMS]-.05)
        poses = dict(mean=qm, gt_start=gt[0], other_real=alternate,
                     arms_plus_03=plus, arms_minus_03=minus, random_in_limits=random_pose)
        for pose_name, pose in poses.items():
            q0 = clip_pose(pose)
            for horizon, seed in ((4, 0), (4, 1), ('inf', 0)):
                # Also reset global RNG: existing StyleHead bypasses Streamer's generator.
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                stream = Streamer(model, vae, sample['feat'], cond, stats,
                                  H=INF if horizon == 'inf' else horizon * U80,
                                  steps=args.steps, seed=seed, standby=q0, ctx_t=ctx_t)
                for _ in range(args.chunks): stream.tick()
                stream.flush()
                cmd = torch.stack(stream.frames).cpu().numpy()
                rec = dict(clip=view.clips[ci]['clip'], clip_index=ci, pose=pose_name, H=horizon,
                           seed=seed, **metrics(cmd, q0, lim))
                # Diagnostic only: decode exactly the same generated latents with standby history.
                # No new samples are generated and this is NOT counted as the production result.
                prefix = stream.hist0.repeat(4, 1)
                lat = torch.cat([prefix, torch.stack(stream.lat)], 0)
                yp = vae.decode((lat * model.lat_std + model.lat_mean)[None])[0]
                yp = yp[len(prefix) * vae.stride:] * stream.qs + stream.qm
                rec['diagnostic_decode_history_first_arm_max_deg'] = float(
                    torch.rad2deg((yp[0] - torch.as_tensor(q0, device='cuda')).abs()[ARMS]).max())
                key = f'{si}_{pose_name}_{horizon}_{seed}'
                trajectories[key + '_cmd'] = cmd; trajectories[key + '_q0'] = q0
                records.append(rec)
                with (out / 'trials.jsonl').open('a') as f: f.write(json.dumps(rec) + '\n')
        print('clip', si+1, '/', args.n, 'trials', len(records), 'seconds', round(time.time()-t0,1), flush=True)
    summary = []
    for horizon in (4, 'inf'):
        for pose_name in poses:
            rows = [r for r in records if r['H'] == horizon and r['pose'] == pose_name]
            maxima = [r['first_arm_max_deg'] for r in rows]
            summary.append(dict(H=horizon, pose=pose_name, n=len(rows),
                first_arm_max_deg_median=float(np.median(maxima)), first_arm_max_deg_max=max(maxima),
                first_speed_pass=sum(r['first_arm_speed_ratio'] <= 1+1e-6 for r in rows),
                startup_va_pass=sum(r['startup_arm_speed_ratio'] <= 1+1e-6 and r['startup_arm_acc_ratio'] <= 1+1e-6 for r in rows),
                startup_jerk_pass=sum(r['startup_arm_jerk_ratio'] <= 1+1e-6 for r in rows),
                diagnostic_decode_history_first_arm_max_deg_median=float(np.median([r['diagnostic_decode_history_first_arm_max_deg'] for r in rows]))))
    np.savez_compressed(out / 'trajectories.npz', **trajectories)
    result = dict(model=ident, args=vars(args), trials=len(records), seconds=time.time()-t0,
                  all_finite=all(r['finite'] for r in records), summary=summary,
                  caveats=['Joint-range-valid random poses are not collision-validated.',
                           'Startup is measured from a stationary pose, 50 Hz, raw unfiltered commands.',
                           'Finite tests cannot establish support for every possible initial pose.'])
    (out / 'RESULTS.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['summary'], indent=1), flush=True)


if __name__ == '__main__':
    main()

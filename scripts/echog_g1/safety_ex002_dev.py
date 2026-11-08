"""EX002 development check of the C3 safety layers (scmg/safety_proj_ex002.py) on EX002 joint trajectories.

    python scripts/echog_g1/safety_ex002_dev.py --gen NAME=DIR ... --out J [--t0 250 --T 1500 --workers 12]

DIR holds <recording>.npy q37 trajectories at 50 Hz (layout ex002_3f.LAYOUT37: lift, waist_yaw, waist_pitch, head_yaw,
head_pitch, left_arm_joint1-7, right_arm_joint1-7, fingers). The gripper embodiment controls the 14 arm joints + 2 grippers:
arms = q37[:, 5:19], grippers held at the anchor, waist / head not modelled (fixed at the collision model's reset pose) --
an approximation of the real waist motion, stated in the output. Window [t0, t0 + T) of every recording present in all
DIRs; each window is an independent stream starting at rest at the nearest safe pose of its first frame.
Per layer and generator: independent validator ex002_safety.validate_sequence (frames + midpoints, D2.2 table) counts as %
of frames, fallback %, RMSE vs raw (rad), |dJerk| = |mean |j| (arm joints, rad/s^3) - same for the GT window|, per-chunk
solve time. No FGD (the EX002 FGD v2 scorer is not wired locally).
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from scmg import safety_proj as S
from scmg import safety_proj_ex002 as X

_W = {}
KEYS = ('position', 'velocity', 'acceleration', 'jerk', 'collision', 'collision_midpoint')


def _init(layers):
    lim = X.EX002Limits(); coll = X.EX002Collision(lim.E)
    _W.update(lim=lim, coll=coll, layers={n: X.make_layer_ex002(n, lim, coll) for n in layers})


def _one(job):
    gen, path, t0, T = job
    lim = _W['lim']
    q37 = np.load(path).astype(np.float64)[t0:t0 + T]
    q = np.repeat(lim.E.anchor[None], len(q37), 0); q[:, :14] = q37[:, 5:19]
    out = {}
    for n, L in _W['layers'].items():
        y, st = S.run_stream(L, q, X.K_EX)
        v = X.validate(lim, y, L.q_start if L.q_start is not None else y[0])
        j = np.abs(np.diff(y[:, :14], 3, axis=0)) / lim.dt ** 3
        out[n] = dict(viol={k: int(v[k]) for k in KEYS}, frames=len(y), fb=int(st['fallback_frames']), times=st['times'],
                      se=float(((y - q) ** 2).sum()), jerk=float(j.mean()))
    return gen, os.path.basename(path), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--t0', type=int, default=250); ap.add_argument('--T', type=int, default=1500)
    ap.add_argument('--workers', type=int, default=12); ap.add_argument('--layers', default=','.join(X.LAYERS_EX))
    ap.add_argument('--ref', default='gt', help='generator name whose window is the jerk reference')
    a = ap.parse_args()
    gens = dict(g.split('=', 1) for g in a.gen); layers = a.layers.split(',')
    common = sorted(set.intersection(*[{f for f in os.listdir(d) if f.endswith('.npy')} for d in gens.values()]))
    jobs = [(g, os.path.join(d, f), a.t0, a.T) for g, d in gens.items() for f in common]
    import multiprocessing as mp
    with mp.get_context('fork').Pool(a.workers, initializer=_init, initargs=(layers,)) as pool:
        res = pool.map(_one, jobs, chunksize=1)
    agg = {}
    ref_jerk = {f: o for g, f, o in res if g == a.ref}
    for g in gens:
        for n in layers:
            rows = [(f, o[n]) for gg, f, o in res if gg == g]
            F = sum(r['frames'] for _, r in rows)
            t = np.concatenate([np.array(r['times']) for _, r in rows]) * 1e3
            jr = np.mean([ref_jerk[f]['none']['jerk'] for f, _ in rows]) if a.ref in gens else float('nan')
            agg[f'{g}/{n}'] = dict(
                **{f'{k}_pct': 100 * sum(r['viol'][k] for _, r in rows) / F for k in KEYS},
                fallback_pct=100 * sum(r['fb'] for _, r in rows) / F, rmse=float(np.sqrt(sum(r['se'] for _, r in rows) / (F * 16))),
                jerk=float(np.mean([r['jerk'] for _, r in rows])), dJerk=float(abs(np.mean([r['jerk'] for _, r in rows]) - jr)),
                t_p50=float(np.percentile(t, 50)), t_p95=float(np.percentile(t, 95)), t_p99=float(np.percentile(t, 99)),
                t_max=float(t.max()), clips=len(rows), frames=F)
            r = agg[f'{g}/{n}']
            print(f"{g:6s} {n:10s} pos {r['position_pct']:6.2f} vel {r['velocity_pct']:6.2f} acc {r['acceleration_pct']:6.2f} "
                  f"jerk {r['jerk_pct']:6.2f} coll {r['collision_pct']:6.2f} mid {r['collision_midpoint_pct']:6.2f} | fb {r['fallback_pct']:5.2f} "
                  f"rmse {r['rmse']:.4f} |dJerk| {r['dJerk']:8.2f} | t p50 {r['t_p50']:6.1f} p99 {r['t_p99']:6.1f} max {r['t_max']:6.1f} ms", flush=True)
    json.dump(dict(gens=gens, recordings=common, t0=a.t0, T=a.T, fps=50, layers=layers, rows=agg,
                   note='arms only (q37[:, 5:19]); grippers at anchor; waist / head fixed at the collision model reset pose'),
              open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

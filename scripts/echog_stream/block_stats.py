"""Block-boundary artefacts of streamed motion vs ground truth (11-05).

For every label (prediction dir) and the GT, on the common prefix of each clip:
  accel ratio = mean |2nd difference| at frames straddling a block boundary (centres f_k - 1, f_k) / mean elsewhere
  jerk  ratio = mean |3rd difference| at windows touching a boundary (f_k - 2 .. f_k) / mean elsewhere
  (|.| = per-body Euclidean norm of official G1 BA FK positions (identity root, zero translation), averaged over the 17
   upper bodies; --space joint uses the raw representation channels instead, e.g. EX002 without a G1 FK)
  block grid: f_k = ceil(k * block_s * fps) (the streaming grid, anchored at the clip start), k >= 1;
  beats: official BeatAlign definition (speed minima of moving upper bodies), phase in the block period;
         beat_phase_maxmin = max / min share over 10 phase bins (uniform = 1).
GT ratios are the reference (GT has no blocks: ~1.0); *_vs_gt = label ratio / GT ratio.
    PYTHONPATH=<official>/src python scripts/echog_stream/block_stats.py --ref-dir R --stems S --mmae-file M --out stats.json \
        label=pred_dir [label=pred_dir ...]
--ex002-dev (EX002 joint space, 50 FPS): panels are eval-dev export dirs (gen_Hinf_<ci>.npy, gt_<ci>.npy, 34 joints in rad);
  clips = the ci present in every panel; |.| = mean over the first --joints (18 = waist 2, head 2, arms 14; fingers excluded,
  the upper-body set of the EX002 FGD v2 BC) of |difference| in rad; block grid = the streaming grid of every data-dev chunk
  (chunk start + ceil(k * block_s * fps), k >= 1, inside the chunk: each chunk is streamed from a cold start); frames within
  3 of a chunk seam (an offline cut shared by every row, also the official one) are left out of both the boundary and the
  elsewhere sets; beats = FGD v2 metrics_std.motion_beats (angular speed minima of the 18 upper joints, BC calibration),
  phase in the block period of the chunk-local grid.
    python scripts/echog_stream/block_stats.py --ex002-dev --out stats.json label=eval-dev/row [...]
"""
import argparse, json, math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.signal

A = None


def grid(n, block_s, fps):
    out, k = [], 1
    while True:
        f = int(math.ceil(k * block_s * fps - 1e-6))
        if f >= n: return np.array(out, int)
        out.append(f); k += 1


def per_clip(stem):
    from echo_g.evaluation import g1_motion_cls as E
    E._G1_MEAN_VEL_PATH = A.mmae_file
    reps = [E.load_repr(f'{A.ref_dir}/{stem}.pt', A.fps)] + [E.load_repr(f'{d}/{stem}.pt', A.fps) for _, d in A.cols]
    if any(r is None for r in reps): return None
    n = min(r.shape[0] for r in reps)
    if n < 8: return None
    g = grid(n, A.block_ms / 1000, A.fps)
    ba = E.BeatAlign(A.fps); mmae = E._g1_mean_vel() if A.space == 'fk' else None
    acc_b = np.zeros(n, bool); jerk_b = np.zeros(n, bool)
    for f in g:
        acc_b[[x for x in (f - 1, f) if 1 <= x < n - 1]] = True
        jerk_b[[x for x in (f - 2, f - 1, f) if 1 <= x < n - 2]] = True
    out = []
    for r in reps:
        if A.space == 'fk':
            J = E.fk_positions_ba(r[:n])[:, ba.upper_bodies].astype(np.float64)              # T, 17, 3
            nrm = lambda d: np.linalg.norm(d, axis=-1).mean(-1)
        else:
            J = r[:n].double().numpy()[:, :, None]
            nrm = lambda d: np.abs(d[..., 0]).mean(-1)
        acc = np.zeros(n); acc[1:-1] = nrm(J[2:] - 2 * J[1:-1] + J[:-2])
        jk = np.zeros(n); jk[1:-2] = nrm(J[3:] - 3 * J[2:-1] + 3 * J[1:-2] - J[:-3])
        ia = np.zeros(n, bool); ia[1:-1] = True; ij = np.zeros(n, bool); ij[1:-2] = True
        res = [acc[acc_b].sum(), acc_b.sum(), acc[ia & ~acc_b].sum(), (ia & ~acc_b).sum(),
               jk[jerk_b].sum(), jerk_b.sum(), jk[ij & ~jerk_b].sum(), (ij & ~jerk_b).sum()]
        if A.space == 'fk':
            spd = E.body_speed_central(E.fk_positions_ba(r[:n]), A.fps); bt = []
            for b in ba.upper_bodies:
                v = spd[:, b]
                m = np.intersect1d(scipy.signal.argrelextrema(v, np.less, order=ba.order)[0], np.where(v / (mmae[b] + 1e-8) > ba.bc_threshold)[0])
                bt.append(m / A.fps)
            bt = np.concatenate(bt); ph = np.histogram((bt % (A.block_ms / 1000)) / (A.block_ms / 1000), bins=10, range=(0, 1))[0]
        else:
            ph = np.zeros(10)
        out.append((np.array(res, np.float64), ph, n))
    return out


def ex002_dev_clip(ci):
    """--ex002-dev: one dev clip -> [(sums, phase hist, frames)] for GT + every panel"""
    import sys
    sys.path.insert(0, A.fgd_code)
    from metrics_std import motion_beats
    chunks = [c for c in A.chunks if c['ci'] == ci]
    reps = [np.load(f'{A.cols[0][1]}/gt_{ci}.npy')] + [np.load(f'{d}/gen_Hinf_{ci}.npy') for _, d in A.cols]
    n = min(len(r) for r in reps)
    acc_b = np.zeros(n, bool); jerk_b = np.zeros(n, bool); seam = np.zeros(n, bool); bs = A.block_ms / 1000
    for c in chunks:
        s0, e0 = c['start_frame'], min(c['end_frame'], n)
        if s0 > 0: seam[max(0, s0 - 3):s0 + 3] = True
        for f in grid(e0 - s0, bs, A.fps) + s0:
            acc_b[[x for x in (f - 1, f) if 1 <= x < n - 1]] = True
            jerk_b[[x for x in (f - 2, f - 1, f) if 1 <= x < n - 2]] = True
    starts = np.array([c['start_frame'] for c in chunks]) / A.fps
    out = []
    for r in reps:
        J = np.asarray(r[:n], np.float64)[:, :A.joints]
        nrm = lambda d: np.abs(d).mean(-1)
        acc = np.zeros(n); acc[1:-1] = nrm(J[2:] - 2 * J[1:-1] + J[:-2])
        jk = np.zeros(n); jk[1:-2] = nrm(J[3:] - 3 * J[2:-1] + 3 * J[1:-2] - J[:-3])
        ia = np.zeros(n, bool); ia[1:-1] = True; ij = np.zeros(n, bool); ij[1:-2] = True
        ab, jb = acc_b & ~seam, jerk_b & ~seam; ae, je = ia & ~acc_b & ~seam, ij & ~jerk_b & ~seam
        res = [acc[ab].sum(), ab.sum(), acc[ae].sum(), ae.sum(), jk[jb].sum(), jb.sum(), jk[je].sum(), je.sum()]
        bt = np.concatenate(motion_beats(np.asarray(r[:n], np.float64), A.fps))
        loc = bt - starts[np.searchsorted(starts, bt + 1e-9, side='right') - 1]          # time since the chunk start
        ph = np.histogram((loc % bs) / bs, bins=10, range=(0, 1))[0]
        out.append((np.array(res, np.float64), ph, n))
    return out


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-dir'); ap.add_argument('--stems'); ap.add_argument('--out', required=True)
    ap.add_argument('--ex002-dev', action='store_true'); ap.add_argument('--joints', type=int, default=18)
    ap.add_argument('--dev-manifest', default='/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/data-dev/MANIFEST.json')
    ap.add_argument('--fgd-code', default='/mnt/cpfs/zbl-cpfs-new/USERS/zehan/fgd-ex002-v2-20261102/code')
    ap.add_argument('--mmae-file'); ap.add_argument('--fps', type=float, default=30.0); ap.add_argument('--block-ms', type=float, default=320.)
    ap.add_argument('--space', choices=['fk', 'joint'], default='fk'); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('panels', nargs='+')
    A = ap.parse_args(); A.cols = [tuple(p.split('=', 1)) for p in A.panels]
    if A.ex002_dev:
        import glob, os
        if A.fps == 30.0: A.fps = 50.0
        A.space = 'joint-ex002'; A.chunks = json.load(open(A.dev_manifest))
        sets = [{int(os.path.basename(p)[9:-4]) for p in glob.glob(f'{d}/gen_Hinf_*.npy')} for _, d in A.cols]
        stems = sorted(set.intersection(*sets))[: A.limit or None]; fn = ex002_dev_clip
    else:
        if not (A.ref_dir and A.stems): raise SystemExit('--ref-dir and --stems required')
        stems = [s.strip() for s in open(A.stems) if s.strip()][: A.limit or None]; fn = per_clip
    labels = ['GT'] + [l for l, _ in A.cols]
    tot = {l: np.zeros(8) for l in labels}; ph = {l: np.zeros(10) for l in labels}; dur = 0.0; nclip = 0
    with ProcessPoolExecutor(A.workers) as ex:
        for res in ex.map(fn, stems, chunksize=1 if A.ex002_dev else 8):
            if res is None: continue
            nclip += 1; dur += res[0][2] / A.fps
            for l, (r, h, _) in zip(labels, res): tot[l] += r; ph[l] += h
    out = dict(clips=nclip, minutes=dur / 60, block_ms=A.block_ms, fps=A.fps, space=A.space, labels={})
    nb = 17 if A.space == 'fk' else (18 if A.ex002_dev else 0)           # beat channels (G1 upper bodies / EX002 upper joints)
    for l in labels:
        t = tot[l]; ar = (t[0] / t[1]) / (t[2] / t[3]); jr = (t[4] / t[5]) / (t[6] / t[7]); h = (ph[l] + 1) / (ph[l].sum() + 10)   # +1 per bin: finite on small sets
        out['labels'][l] = dict(accel_ratio=ar, jerk_ratio=jr, accel_mean=(t[0] + t[2]) / (t[1] + t[3]), jerk_mean=(t[4] + t[6]) / (t[5] + t[7]),
                                beats_per_s=float(ph[l].sum() / dur / nb) if nb else None,
                                beat_phase_maxmin=float(h.max() / max(h.min(), 1e-9)) if nb else None, beat_phase_hist=h.tolist())
    g = out['labels']['GT']
    for l in labels:
        out['labels'][l]['accel_ratio_vs_gt'] = out['labels'][l]['accel_ratio'] / g['accel_ratio']
        out['labels'][l]['jerk_ratio_vs_gt'] = out['labels'][l]['jerk_ratio'] / g['jerk_ratio']
    json.dump(out, open(A.out, 'w'), indent=1)
    print(f"{nclip} clips {dur / 60:.1f} min, {A.block_ms:.0f} ms grid")
    print(f"{'label':28s} accelR  vsGT  jerkR  vsGT  beats/s  phase max/min")
    for l in labels:
        r = out['labels'][l]
        print(f"{l:28s} {r['accel_ratio']:.3f} {r['accel_ratio_vs_gt']:.3f} {r['jerk_ratio']:.3f} {r['jerk_ratio_vs_gt']:.3f} "
              f"{(r['beats_per_s'] or 0):.3f}  {(r['beat_phase_maxmin'] or 0):.2f}")


if __name__ == '__main__':
    main()

"""11-08 report-only arm-acceleration statistics of EX002 streams (module 5c report; no gate).
Per frame a = max over arms of || 2nd difference of the arm's 7 joint angles || * 50^2 (rad/s^2). Per spec (label=dir[,dir...]):
share of frames > 100, p99, max, and the peak over 0.2 s windows (stride 0.1 s): median / p95 / max. Seeds of one label are pooled.
    python accel_stats.py --out O.json GT=<dir with gt_*.npy> none=<d0>,<d1> tok5b=... tok5c=..."""
import argparse, glob, json, os
import numpy as np
ARM = (list(range(4, 11)), list(range(11, 18))); FPS = 50


def files(d):
    g = sorted(glob.glob(d + '/gt_*.npy'), key=lambda p: int(os.path.basename(p)[3:-4]))
    return g if g else sorted((f for f in glob.glob(d + '/*.npy') if os.path.basename(f)[:-4].isdigit()), key=lambda p: int(os.path.basename(p)[:-4]))


def stat(dirs):
    a_all, pk = [], []
    for d in dirs:
        for f in files(d):
            y = np.load(f); a = np.max([np.linalg.norm(np.diff(y[:, jj], n=2, axis=0), axis=1) * FPS ** 2 for jj in ARM], 0); a_all.append(a)
            pk += [a[i:i + 11].max() for i in range(0, len(a) - 10, 5)]
    a = np.concatenate(a_all); pk = np.array(pk)
    return dict(frames=int(len(a)), share_gt100=float((a > 100).mean()), p99=float(np.percentile(a, 99)), max=float(a.max()),
                win_median=float(np.median(pk)), win_p95=float(np.percentile(pk, 95)), win_max=float(pk.max()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('specs', nargs='+'); a = ap.parse_args()
    R = {}
    for s in a.specs:
        lab, ds = s.split('=', 1); R[lab] = stat(ds.split(',')); print(lab, json.dumps({k: round(v, 4) for k, v in R[lab].items()}))
    open(a.out, 'w').write(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()

"""EX002 aiming error (ours, 11-08; coordinator after the qualitative review): per aim command, at the apex frame of the evaluate rule
(APEX.json detector on +-1 s around the command), the angle between the ray (forearm: wrist origin, elbow->wrist direction = the
official ray; index: index base->tip) and the direction wrist -> target, split into azimuth (angle between the horizontal
projections, absolute) and elevation (ray elevation - target elevation, signed: negative = the ray points below the target).
Reports median / mean / p90 over events and seeds per run; same plan semantics as evaluate (shuffle runs scored against the
shuffled command with --shuffled-run).
    python ex002_aim_error.py --plan plan.json --ctl tokens --out res.json [--shuffled-run] label=<dir> ..."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
FPS = 50


def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12), -1, 1))))


def parts(ray, tgt):
    el = lambda x: np.degrees(np.arctan2(x[2], np.linalg.norm(x[:2])))
    return ang(ray, tgt), ang(np.r_[ray[:2], 0.], np.r_[tgt[:2], 0.]), el(ray) - el(tgt)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', required=True); ap.add_argument('--ctl', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--shuffled-run', action='store_true'); ap.add_argument('specs', nargs='+')
    a = ap.parse_args()
    from scmg import ex002_ctl as E
    from ex002_apex import Detector
    det = Detector(json.loads((Path(a.ctl) / 'APEX.json').read_text()))
    P = json.loads(Path(a.plan).read_text()); by = {}
    for e in P['events']: by.setdefault(e['ci'], []).append(e)
    groups = {}
    for spec in a.specs:
        lab, d = spec.split('=', 1); grp = lab.rsplit('_s', 1)[0]; rows = []
        for f in sorted(Path(d).glob('*.npy'), key=lambda p: int(p.stem)):
            ci = int(f.stem); y = np.load(f); n = len(y); geo = E.points(y)
            for e in by.get(ci, []):
                if e['kind'] != 'aim': continue
                fe = int(round(e['t'] * FPS))
                if fe >= n: continue
                v = dict(e, **e['shuffle']) if a.shuffled_run else e; h = v['hand']
                ah = det.apex(geo, h, max(0, fe - FPS), min(n, fe + FPS)); w = geo['wrist'][ah, h]; tg = np.asarray(v['target']) - w
                fa = parts(w - geo['elbow'][ah, h], tg); ix = parts(geo['index_tip'][ah, h] - geo['index_base'][ah, h], tg)
                sh = geo['shoulder'][ah, h]; above = bool(v['target'][2] > sh[2])
                rows.append(dict(ci=ci, t=e['t'], fa=fa, ix=ix, target_above_shoulder=above, wrist_minus_shoulder_z=float(w[2] - sh[2])))
        groups.setdefault(grp, []).extend(rows)
    res = {}
    for grp, rows in groups.items():
        r = {}
        for k in ('fa', 'ix'):
            A = np.array([x[k] for x in rows])
            r[k] = dict(total_median=float(np.median(A[:, 0])), total_mean=float(A[:, 0].mean()), total_p90=float(np.percentile(A[:, 0], 90)),
                        azimuth_median=float(np.median(A[:, 1])), elevation_median=float(np.median(A[:, 2])), elevation_mean=float(A[:, 2].mean()))
            up = np.array([x['target_above_shoulder'] for x in rows])
            if up.any(): r[k]['elevation_median_target_above_shoulder'] = float(np.median(A[up, 2])); r[k]['total_median_target_above_shoulder'] = float(np.median(A[up, 0]))
            if (~up).any(): r[k]['elevation_median_target_below_shoulder'] = float(np.median(A[~up, 2]))
        r['events'] = len(rows); r['share_target_above_shoulder'] = float(np.mean([x['target_above_shoulder'] for x in rows]))
        r['wrist_minus_shoulder_z_median'] = float(np.median([x['wrist_minus_shoulder_z'] for x in rows]))
        res[grp] = r; print(grp, json.dumps({k: ({kk: round(vv, 1) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 3)) for k, v in r.items()}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

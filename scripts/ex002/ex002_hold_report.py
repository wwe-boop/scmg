"""9c report-only extras (11-08; criteria "Module 9c"): per aim event with an insertion (KEYFRAMES t_hold), the forearm angular error
over the whole hold window (median over the hold frames, then over events), the share of events whose apex frame (APEX.json detector,
+-1 s as ex002_aim_error.py) lies inside the hold, and the mean wrist speed over the hold (m/s; median / IQR) next to the train pool stats.
    SCMG_J=34 python ex002_hold_report.py --plan P --ctl TOKENS --out O.json [--pool-json HOLD_POOL_TRAIN.json] label=<dir> ..."""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from ex002_aim_error import parts
FPS = 50


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', required=True); ap.add_argument('--ctl', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--pool-json', default=None); ap.add_argument('specs', nargs='+'); a = ap.parse_args()
    from scmg import ex002_ctl as E
    from ex002_apex import Detector
    det = Detector(json.loads((Path(a.ctl) / 'APEX.json').read_text()))
    P = json.loads(Path(a.plan).read_text()); ev = {(e['ci'], round(e['t'], 3)): e for e in P['events'] if e['kind'] == 'aim'}
    G = {}
    for spec in a.specs:
        lab, d = spec.split('=', 1); grp = lab.rsplit('_s', 1)[0]
        KF = [x for f in glob.glob(d + '/KEYFRAMES.*.json') for x in json.loads(Path(f).read_text())['events'] if x['kind'] == 'aim' and 't_hold' in x]
        by = {}
        for x in KF: by.setdefault(x['ci'], []).append(x)
        for ci, xs in by.items():
            y = np.load(f'{d}/{ci}.npy'); n = len(y); geo = E.points(y)
            for x in xs:
                e = ev.get((ci, round(x['t'], 3)))
                if e is None: continue
                h = e['hand']; tg = np.asarray(e['target']); h0, h1 = (int(round(v * FPS)) for v in x['t_hold']); h1 = min(h1, n - 1)
                errs = [parts(geo['wrist'][f, h] - geo['elbow'][f, h], tg - geo['wrist'][f, h])[0] for f in range(h0, h1 + 1)]
                fe = int(round(e['t'] * FPS)); ah = det.apex(geo, h, max(0, fe - FPS), min(n, fe + FPS))
                sp = float(np.linalg.norm(np.diff(geo['wrist'][h0:h1 + 1, h], axis=0), axis=-1).mean() * FPS) if h1 > h0 else 0.
                G.setdefault(grp, []).append((float(np.median(errs)), h0 <= ah <= h1, sp))
    R = {}
    for grp, rows in G.items():
        r = np.array(rows, dtype=float)
        R[grp] = dict(events=len(rows), hold_angle_median=float(np.median(r[:, 0])), apex_in_hold_share=float(r[:, 1].mean()),
                      hold_speed_median=float(np.median(r[:, 2])), hold_speed_q25=float(np.percentile(r[:, 2], 25)), hold_speed_q75=float(np.percentile(r[:, 2], 75)))
        print(grp, json.dumps({k: round(v, 4) for k, v in R[grp].items()}), flush=True)
    if a.pool_json: R['_train_pool'] = json.loads(Path(a.pool_json).read_text())['stats']
    Path(a.out).write_text(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()

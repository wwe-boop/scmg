"""9c'' report (11-08; coordinator): STG of aim events split by whether a PRESENT hold of the reference insertion run overlaps the event's
+-1 s apex window (flags from the KEYFRAMES of --ref; the same flags apply to every row). Score_a x Score_b as ex002_ctl_run.evaluate
(forearm ray, APEX.json detector, given commands); the none row gives the gain per split.
    SCMG_J=34 python stg_split.py --plan P --ctl TOKENS --ref <ref run dir with KEYFRAMES> --out O.json label=<dir> ...  (labels *_s<k> pooled)"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3])); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
FPS = 50


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', required=True); ap.add_argument('--ctl', required=True); ap.add_argument('--ref', required=True, nargs='+')
    ap.add_argument('--out', required=True); ap.add_argument('specs', nargs='+'); a = ap.parse_args()
    from scmg import ex002_ctl as E, ctl
    from ex002_apex import Detector
    det = Detector(json.loads((Path(a.ctl) / 'APEX.json').read_text()))
    P = json.loads(Path(a.plan).read_text()); aims = [e for e in P['events'] if e['kind'] == 'aim']
    holds = {}
    for r in a.ref:
        for f in glob.glob(r + '/KEYFRAMES.*.json'):
            for e in json.loads(Path(f).read_text())['events']:
                if e['kind'] == 'present' and 't_hold' in e: holds.setdefault(e['ci'], []).append(tuple(e['t_hold']))
    flag = {(e['ci'], round(e['t'], 3)): any(h0 <= e['t'] + 1 and h1 >= e['t'] - 1 for h0, h1 in holds.get(e['ci'], [])) for e in aims}
    R = {}
    for spec in a.specs:
        lab, d = spec.split('=', 1); grp = lab.rsplit('_s', 1)[0]; acc = R.setdefault(grp, {True: [], False: []})
        for f in sorted(Path(d).glob('[0-9]*.npy'), key=lambda p: int(p.stem)):
            ci = int(f.stem); y = np.load(f); n = len(y); geo = E.points(y)
            for e in aims:
                if e['ci'] != ci: continue
                fe = int(round(e['t'] * FPS))
                if fe >= n: continue
                h = e['hand']; ah = det.apex(geo, h, max(0, fe - FPS), min(n, fe + FPS)); sa = ctl.score_a(ah / FPS, e['t']); rr = E.rays(geo, ah, h)
                sb, _ = ctl.score_b(rr['forearm'][0], rr['forearm'][1], e['target']); acc[flag[(ci, round(e['t'], 3))]].append(sa * sb)
    out = {g: {('with_present_hold' if k else 'without'): dict(n=len(v), stg=float(np.mean(v)) if v else None) for k, v in d.items()} for g, d in R.items()}
    out['_share_with_present_hold'] = float(np.mean(list(flag.values())))
    print(json.dumps(out)); Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

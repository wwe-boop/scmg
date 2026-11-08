"""Module 10 (a) independent shape check over the SAMPLED events (criteria "Module 10"): for every finger command of a plan (given
values) and hand, the library entry the override uses (same sha1 rule as finger_override) is classified by joint angle
(hand_library.angle_count) and compared with the commanded count. Prints / writes agreement per (h, n) and overall.
    python shape_check.py --plan P --library HAND_LIBRARY_TRAIN.npz --shapes HAND_SHAPES.json --out O.json"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand_library import angle_count


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', required=True); ap.add_argument('--library', required=True); ap.add_argument('--shapes', required=True)
    ap.add_argument('--out', required=True); a = ap.parse_args()
    lib = dict(np.load(a.library)); tmpl = json.loads(Path(a.shapes).read_text())['shapes']; R = {}
    for e in json.loads(Path(a.plan).read_text())['events']:
        if e['kind'] != 'finger': continue
        for h in (0, 1):
            n = int(e['fingers'][h]); key = f'{h}_{n}'
            i_ = int(hashlib.sha1(f"{e['clip']}:{e['t']:.3f}:10a:{h}".encode()).hexdigest()[:12], 16) % len(lib[key])
            R.setdefault(key, []).append(int(angle_count(lib[key][i_:i_ + 1], h, tmpl)[0] == n))
    out = {k: dict(events=len(v), angle_agreement=float(np.mean(v))) for k, v in sorted(R.items())}
    out['all'] = dict(events=sum(len(v) for v in R.values()), angle_agreement=float(np.mean([x for v in R.values() for x in v])))
    print(json.dumps(out)); Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

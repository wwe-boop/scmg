"""Freeze the noise units of the joint Div/BA/clarity criteria (CRITERIA.md) from a CONTROL's own runs only.

Inputs: diag_beats.py JSONs of the control (postproc.sh: one prediction label + GT per file), psync and offline, inference seeds 0 and 1.
u_i = 2 * sqrt(SE_boot^2 + (x_s0 - x_s1)^2 / 4), SE_boot = mean of the two seeds' clip-bootstrap SEs / sqrt(2) (2-seed mean);
T3 gap uses SE_boot = sqrt(SE_psync^2 + SE_offline^2) of Div. Refuses to overwrite an existing file (frozen once written).
    python noise_units.py --label L --psync s0.json s1.json --offline s0.json s1.json --out NOISE_UNITS_<control>.json
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--label', required=True, help='name recorded as the control'); ap.add_argument('--psync', nargs=2, required=True); ap.add_argument('--offline', nargs=2, required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
if os.path.exists(a.out): raise SystemExit(f'{a.out} exists: noise units are frozen once written')
ld = lambda f: next(v for k, v in json.load(open(f))['labels'].items() if k != 'GT')   # the one prediction label of a postproc diag.json
P = [ld(f) for f in a.psync]; O = [ld(f) for f in a.offline]


def unit(xs, ses):
    se = float(np.mean(ses)) / np.sqrt(2)
    return 2 * float(np.sqrt(se ** 2 + (xs[0] - xs[1]) ** 2 / 4)), float(np.mean(xs))


u1, m1 = unit([p['ba_lift'] for p in P], [p['ba_lift_se'] for p in P])
u2, m2 = unit([p['shape_l1'] for p in P], [p['shape_l1_se'] for p in P])
gaps = [abs(p['div'] - o['div']) for p, o in zip(P, O)]
se_gap = [float(np.hypot(p['div_se'], o['div_se'])) for p, o in zip(P, O)]
u3, m3 = unit(gaps, se_gap)
res = dict(control=a.label, inputs=dict(psync=a.psync, offline=a.offline),
           T1_ba_lift=dict(control=m1, unit=u1), T2_shape_l1=dict(control=m2, unit=u2), T3_div_gap=dict(control=m3, unit=u3),
           div_psync=float(np.mean([p['div'] for p in P])), div_offline=float(np.mean([o['div'] for o in O])),
           ba_null_psync=float(np.mean([p['ba_null'] for p in P])), total_power_rel_gt_psync=float(np.mean([p['total_power_rel_gt'] for p in P])))
fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL); os.write(fd, json.dumps(res, indent=1).encode()); os.close(fd)
print(json.dumps(res, indent=1))

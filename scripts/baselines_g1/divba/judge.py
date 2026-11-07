"""Joint Div/BA/clarity judgement (CRITERIA.md, joint version) of dev candidates against one control.

Each run = (official benchmark JSON from the DLC job, postproc.sh dir with diag.json / EXEC.json / safe/cbf_qp/stats.jsonl),
for psync and offline x inference seeds 0 and 1; every number is the 2-seed mean. Noise units come from the frozen
NOISE_UNITS file of the control (noise_units.py); this script never writes them.
    python judge.py --units NOISE_UNITS_x.json --ctrl name=BENCH_FMT@POST_FMT --cand name=BENCH_FMT@POST_FMT ... [--out J]
BENCH_FMT / POST_FMT: paths with {p} (psync|offline) and {s} (0|1), e.g.
  /cpfs/.../zt/ctrl/{p}/benchmark_seed_00{s}.json@~/bulk/divba-1106/post/zt/ctrl_{p}_seed_00{s}
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--units', required=True); ap.add_argument('--ctrl', required=True); ap.add_argument('--cand', nargs='*', default=[])
ap.add_argument('--out', default=None)
a = ap.parse_args()
U = json.load(open(a.units))


def load(spec):
    name, fmts = spec.split('=', 1); bf, pf = fmts.split('@')
    r = dict(name=name, missing=[])
    for p in ('psync', 'offline'):
        acc = {}
        for s in (0, 1):
            bj, pd = os.path.expanduser(bf.format(p=p, s=s)), os.path.expanduser(pf.format(p=p, s=s))
            if not (os.path.exists(bj) and os.path.exists(f'{pd}/DONE')): r['missing'].append(f'{p} s{s}'); continue
            m = json.load(open(bj))['metrics']; d = next(v for k, v in json.load(open(f'{pd}/diag.json'))['labels'].items() if k != 'GT')
            ex = json.load(open(f'{pd}/EXEC.json')); st = [json.loads(l) for l in open(f'{pd}/safe/cbf_qp/stats.jsonl')]
            T = np.array([x['T'] for x in st], float); rm = np.array([x['rmse'] for x in st]); mx = np.array([x['max_abs'] for x in st])
            fb = sum(x.get('fallback_frames', 0) for x in st) / T.sum()
            row = dict(FGD=m['FGD'], Div=m['Div'], BA=m['BA']['mean'], JerkLW=m['Jerk_length_weighted']['mean'],
                       JerkLW_GT=m['Jerk_GT_length_weighted']['mean'], lift=d['ba_lift'], ba_null=d['ba_null'], shape=d['shape_l1'],
                       power=d['total_power_rel_gt'], exec_raw=ex['raw']['clean_clips'], exec_cbf=ex['cbf_qp']['clean_clips'],
                       cbf_rmse=float((rm * T).sum() / T.sum()), cbf_rmse_p99=float(np.percentile(rm, 99)), cbf_max=float(mx.max()), cbf_fallback=fb)
            for k, v in row.items(): acc.setdefault(k, []).append(v)
        r[p] = {k: float(np.mean(v)) for k, v in acc.items()}; r[p]['n_seeds'] = len(acc.get('FGD', []))
    return r


C = load(a.ctrl); res = dict(control=C, candidates=[], units=U)
u1, u2, u3 = U['T1_ba_lift']['unit'], U['T2_shape_l1']['unit'], U['T3_div_gap']['unit']
gap = lambda r: abs(r['psync']['Div'] - r['offline']['Div'])
for spec in a.cand:
    X = load(spec); cp, xp = C['psync'], X['psync']
    if X['missing'] or C['missing']: X['verdict'] = f"incomplete {X['missing'] or C['missing']}"; res['candidates'].append(X); continue
    imp = dict(T1=(xp['lift'] - cp['lift']) / u1, T2=(cp['shape'] - xp['shape']) / u2, T3=(gap(C) - gap(X)) / u3)
    guards = dict(
        G1_fgd=xp['FGD'] <= cp['FGD'] + 0.03,
        G3_jerk=abs(xp['JerkLW'] - xp['JerkLW_GT']) <= abs(cp['JerkLW'] - cp['JerkLW_GT']),
        G4_exec=xp['exec_cbf'] >= cp['exec_cbf'], G4_rmse=xp['cbf_rmse'] <= 1.2 * cp['cbf_rmse'],
        G4_fallback=xp['cbf_fallback'] <= cp['cbf_fallback'] + 0.005,
        amp_psync=abs(xp['Div'] - cp['Div']) <= 0.05, amp_offline=abs(X['offline']['Div'] - C['offline']['Div']) <= 0.05)
    no_worse = all(v >= -1.0 for v in imp.values())
    planned = (xp['lift'] >= cp['lift'] + 0.0015) or (xp['shape'] <= 0.7 * cp['shape'])
    X.update(improvement_in_units=imp, score=float(sum(imp.values())), guards=guards, no_objective_worse=no_worse, planned_gain=planned,
             admitted=bool(all(guards.values()) and no_worse and planned), dJerk_raw=dict(psync=xp['JerkLW'] - xp['JerkLW_GT'],
             offline=X['offline']['JerkLW'] - X['offline']['JerkLW_GT']), div_gap=gap(X))
    X['verdict'] = 'ADMITTED' if X['admitted'] else 'rejected: ' + ', '.join([k for k, v in guards.items() if not v] +
                   ([] if no_worse else ['objective worse > 1 unit']) + ([] if planned else ['no planned gain']))
    res['candidates'].append(X)
adm = sorted([x for x in res['candidates'] if x.get('admitted')], key=lambda x: (-x['score'], x['psync']['FGD']))
if len(adm) > 1 and adm[0]['score'] - adm[1]['score'] < 1:      # tie -> lower psync FGD
    adm[:2] = sorted(adm[:2], key=lambda x: x['psync']['FGD'])
res['ranking'] = [x['name'] for x in adm]
f = lambda v: f'{v:.4f}'
print(f"units: T1 {u1:.4f} T2 {u2:.4f} T3 {u3:.4f} | control {C['name']} psync lift {f(C['psync']['lift'])} shape {f(C['psync']['shape'])} "
      f"FGD {C['psync']['FGD']:.3f} Div {C['psync']['Div']:.3f}/{C['offline']['Div']:.3f} exec_cbf {C['psync']['exec_cbf']:.3f} cbf_rmse {C['psync']['cbf_rmse']:.5f}")
for x in res['candidates']:
    if 'score' not in x: print(f"{x['name']:12s} {x['verdict']}"); continue
    xp = x['psync']
    print(f"{x['name']:12s} lift {f(xp['lift'])} shape {f(xp['shape'])} gap {x['div_gap']:.3f} | units T1 {x['improvement_in_units']['T1']:+.2f} "
          f"T2 {x['improvement_in_units']['T2']:+.2f} T3 {x['improvement_in_units']['T3']:+.2f} score {x['score']:+.2f} | FGD {xp['FGD']:.3f} "
          f"Div {xp['Div']:.3f}/{x['offline']['Div']:.3f} dJerk {x['dJerk_raw']['psync']:+.2f}/{x['dJerk_raw']['offline']:+.2f} "
          f"exec_cbf {xp['exec_cbf']:.3f} cbf_rmse {xp['cbf_rmse']:.5f} fb {xp['cbf_fallback']:.4f} | {x['verdict']}")
print('ranking:', res['ranking'])
if a.out: json.dump(res, open(a.out, 'w'), indent=1)

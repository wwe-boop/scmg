"""Markdown table of eval_interrupt.py results (11-05).  python scripts/interrupt/make_table.py res.json [label ...]"""
import json, sys

r = json.load(open(sys.argv[1])); labs = sys.argv[2:] or [k for k in r if k != '_meta']
f = lambda x, d=2, p=1.: '-' if x is None else f'{x * p:.{d}f}'
print('| method | n | yield time s (R_sil) | yield rate % | abrupt % | RKD | unsafe % (pre / new) | react s | peak jerk med | energy | RKD all | FGD | yield s / rate (R_gt) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for k in labs:
    o = r[k]
    print(f"| {k} | {o['n']} | {f(o['yield_s_R_sil'])} | {f(o['yield_rate_R_sil'], 1, 100)} | {f(o['abrupt_rate'], 1, 100)} | {f(o['rkd'])} | "
          f"{f(o['unsafe_any'], 1, 100)} ({f(o.get('unsafe_pre'), 1, 100)} / {f(o.get('unsafe_new'), 1, 100)}) | {f(o['react_s'])} | {f(o['jerk_median'], 0)} | {f(o['energy_median'])} | {f(o['rkd_all'])} | {f(o['fgd'])} | "
          f"{f(o['yield_s_R_gt'])} / {f(o['yield_rate_R_gt'], 1, 100)} |")
m = r['_meta']
print(f"\nT_r {m['T_r']:.2f} s | human jerk P95 {m['human_jerk_p95']:.0f} m/s^3 | human RKD half-split floor {m['rkd_human_half_split']:.2f} "
      f"(all {m['rkd_all_human_half_split']:.2f}; all-vs-retract {m['rkd_all_vs_retract']:.2f}) | FGD human half-split {m['fgd_human_half_split']:.2f} | "
      f"human retraction windows {m['n_human_retract']} / {m['human_windows']}")

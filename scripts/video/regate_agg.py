"""(c)1 aggregation of regate_seamless.py outputs: hours per rule (single and cumulative) among R0 rejects, the accept rule, and
the G2 / G3 / G4 comparisons between newly accepted and R0-passing items.   python regate_agg.py <dir with regate_<i>.jsonl>"""
import json, glob, sys
import numpy as np
rows = [json.loads(l) for f in sorted(glob.glob(sys.argv[1] + '/regate_[0-9].jsonl')) for l in open(f) if l.strip()]
err = [r for r in rows if 'error' in r]; rows = [r for r in rows if 'error' not in r]
H = lambda rs: sum(r['seconds'] for r in rs) / 3600
ok = lambda r: r['R1'] and r['R2'] and r['R3'] and r['R4']
print('items', len(rows), 'errors', len(err), 'hours %.2f' % H(rows))
p0 = [r for r in rows if r['R0']]; rej = [r for r in rows if not r['R0']]
print('R0 pass %d / %.2f h; R0 reject %d / %.2f h' % (len(p0), H(p0), len(rej), H(rej)))
for k in ('R1', 'R2', 'R3', 'R4'):
    a = [r for r in rej if r[k]]; print(f'  among R0 rejects, {k} pass: {len(a)} / {H(a):.2f} h')
c = rej
for k in ('R1', 'R2', 'R3', 'R4'):
    c = [r for r in c if r[k]]; print(f'  cumulative R1..{k}: {len(c)} / {H(c):.2f} h')
acc = [r for r in rows if ok(r)]; new = [r for r in acc if not r['R0']]; lost = [r for r in p0 if not ok(r)]
print('accept rule: %d / %.2f h total; new %d / %.2f h; R0-passing items failing the new rule %d' % (len(acc), H(acc), len(new), H(new), len(lost)))
m = lambda rs, k: float(np.mean([r[k] for r in rs if r.get(k) is not None])) if rs else float('nan')
newids = {r['id'] for r in new}
for nm, rs in (('R0-pass', p0), ('new', new), ('R0-reject-not-new', [r for r in rej if r['id'] not in newids])):
    print(nm, 'n', len(rs), 'h %.2f' % H(rs), 'wrist_out %.4f' % m(rs, 'wrist_out_frac'), 'far_repair %.3f' % m(rs, 'far_repair_frac'),
          'near_frac %.2f' % m(rs, 'near_frac'), 'far_wrist_cm %.2f' % m(rs, 'far_wrist_cm'), 'near_wrist_cm %.2f' % m(rs, 'near_wrist_cm'),
          'self_contact %.3f' % m(rs, 'self_contact_frac'), 'post_jerk %.3f' % m(rs, 'post_jerk'), 'R1 %.3f' % m(rs, 'R1'), 'R2 %.3f' % m(rs, 'R2'),
          'backoff<1: %d' % sum(1 for r in rs if (r.get('backoff') or 1) < 1), 'speakers', len({r['speaker'] for r in rs}))
print('far_repair quantiles of R0 rejects', np.round(np.quantile([r['far_repair_frac'] for r in rej], [.1, .25, .5, .75, .9]), 3).tolist())
json.dump(dict(new=sorted(newids)), open(sys.argv[1] + '/NEW_ACCEPT.json', 'w'))

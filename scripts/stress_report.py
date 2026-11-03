"""Aggregate stress_eval rows: python stress_report.py <rows.jsonl|dir> ...  -> table per model x setting x condition.
Setting = tm/H/text/seed.  Columns: still, speed, hf (jitter share), jerk, lim_pos (waist+arms, GT ref), lim_vel, beat_gap,
post-silence stillness (silence3), drift = stillness of the last 10 s bin minus the first (long-form), tick p95."""
import json, sys, glob, collections
import numpy as np
rows = []
for a in sys.argv[1:]:
    for f in (sorted(glob.glob(a + '/*.jsonl')) if not a.endswith('.jsonl') else [a]):
        rows += [json.loads(l) for l in open(f) if l.startswith('{')]
G = collections.defaultdict(list)
for r in rows: G[(r['model'], f"{r['tm']}/{r['H']}/{r['text']}/s{r['seed']}", r['cond'])].append(r)
def m(v, k):
    x = [r[k] for r in v if r.get(k) is not None]; return np.mean(x) if x else float('nan')
print('%-12s %-20s %-9s %3s %6s %5s %6s %5s %6s %6s %6s %7s %6s %6s' % ('model', 'setting', 'cond', 'n', 'still', 'speed', 'hf', 'jerk', 'limP', 'limV', 'beat', 'postSil', 'drift', 'tick95'))
for k in sorted(G):
    v = G[k]; drift = np.mean([r['still_bins'][-1] - r['still_bins'][0] for r in v if len(r['still_bins']) > 1])
    print('%-12s %-20s %-9s %3d %5.1f%% %5.2f %6.4f %5.2f %5.1f%% %5.1f%% %+6.3f %6.1f%% %+6.2f %6.0f' % (k[0], k[1], k[2], len(v), 100 * m(v, 'still'), m(v, 'speed'), m(v, 'hf'), m(v, 'jerk'),
          100 * m(v, 'lim_pos'), 100 * m(v, 'lim_vel'), m(v, 'beat_gap'), 100 * m(v, 'post_sil'), drift, m(v, 'tick_p95')))
gt = [r['lim_pos_gt'] for r in rows if r.get('lim_pos_gt') is not None]
if gt: print('GT waist+arm position-limit share (same clips): %.1f%%' % (100 * np.mean(gt)))

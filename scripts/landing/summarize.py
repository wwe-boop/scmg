"""One table for the landing runs (Claude 11-05).

    python scripts/landing/summarize.py <out.json> --fgd FGD.json[+FGD2.json] --metrics metrics.jsonl[+...]
        --row label:fgd_label:metrics_label:sem_own_s0,sem_swap_s0;sem_own_s1,sem_swap_s1[:rollout_s0,rollout_s1] ...
follow = top-1(own text) - top-1(swapped text) at the event centre, mean over the listed seeds (semantic_eval.py score).
FGD = score_dirs.py FGD_mean (two evaluator seeds) of the seed-0 export; FGD_s1 = the same label in the --fgd1 file(s)
(score_dirs.py run on the seed-1 exports).
"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--fgd', default=''); ap.add_argument('--fgd1', default=''); ap.add_argument('--metrics', default='')
ap.add_argument('--row', action='append', default=[])
a = ap.parse_args()
fgd, fgd1 = {}, {}
for src, dst in ((a.fgd, fgd), (a.fgd1, fgd1)):
    for f in filter(None, src.split('+')):
        for k, v in json.load(open(f))['methods'].items(): dst[k] = v['all']
met = {}
for f in filter(None, a.metrics.split('+')):
    for l in open(f): r = json.loads(l); met[r['label']] = r
rows = []
for spec in a.row:
    parts = spec.split(':')
    lab, fl, ml, sem = parts[:4]; ro = parts[4] if len(parts) > 4 else ''
    fo = []
    for pair in filter(None, sem.split(';')):
        own, swap = pair.split(',')
        fo.append(json.load(open(own))['centre']['top1'] - json.load(open(swap))['centre']['top1'])
    st = [json.load(open(p))['free'] for p in filter(None, ro.split(','))]
    m = met.get(ml)
    r = dict(label=lab, FGD=fgd.get(fl, {}).get('FGD_mean'), FGD_s1=fgd1.get(fl, {}).get('FGD_mean'),
             BC=fgd.get(fl, {}).get('BC'), follow=float(np.mean(fo)) if fo else None, follow_seeds=fo,
             rollout_still=float(np.mean([x['still'] for x in st])) if st else None,
             rollout_speed=float(np.mean([x['speed'] for x in st])) if st else None)
    if m:
        r.update(wrist_out=m['gen']['wrist_out'], upper_out=m['gen']['upper_out'], still=m['gen']['still'], still_gt=m['gt']['still'],
                 still_gap=m['still_gap'], freeze=m['gen']['freeze'], freeze_gt=m['gt']['freeze'], overspeed=m['gen']['overspeed'],
                 speed=m['speed'], amp_upper=m['amp_upper'], amp_wrist=m['amp_wrist'], style=m['gen'].get('style'))
    rows.append(r)
json.dump(rows, open(a.out, 'w'), indent=1)
f = lambda x, fmt: fmt % x if x is not None else '-'
pct = lambda r, k, fmt: f(100 * r[k] if r.get(k) is not None else None, fmt)
print('%-14s %6s %6s %7s %6s %6s %6s %6s %6s %6s %5s %5s' % ('label', 'FGD', 'FGDs1', 'follow', 'wrist', 'upper', 'still', 'freeze', 'over', 'speed', 'ampU', 'ampW'))
for r in rows:
    print('%-14s %6s %6s %7s %6s %6s %6s %6s %6s %6s %5s %5s' % (r['label'], f(r['FGD'], '%.1f'), f(r['FGD_s1'], '%.1f'), f(r['follow'], '%+.3f'),
          pct(r, 'wrist_out', '%.2f%%'), pct(r, 'upper_out', '%.2f%%'), pct(r, 'still', '%.1f%%'), pct(r, 'freeze', '%.2f%%'),
          pct(r, 'overspeed', '%.1f%%'), f(r.get('speed'), '%.3f'), f(r.get('amp_upper'), '%.2f'), f(r.get('amp_wrist'), '%.2f')))

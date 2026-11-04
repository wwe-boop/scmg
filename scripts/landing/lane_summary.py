"""Per-lane summary of the full-tier EX002 eval (11-04): stillness / speed (rollout-s*.json, free run), semantic follow
(own - swap top-1 at the event centre, semantic-s*/score-*.json), FGD (score_dirs.py JSON, optional), arm metrics (optional).
Means over inference seeds.     python scripts/landing/lane_summary.py [--fgd off.json,eng.json] [--arms arms.jsonl] label=<lane dir> ...
"""
import sys, json, glob, os, argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('lanes', nargs='+'); ap.add_argument('--fgd', default=''); ap.add_argument('--arms', default='')
a = ap.parse_args()
fgd = {}
for f, p in zip(a.fgd.split(',') if a.fgd else [], ('offline', 'engine')):
    for f1 in f.split('+'):
        if os.path.exists(f1):
            for k, v in json.load(open(f1))['methods'].items(): fgd[(k, p)] = v['all']['FGD_mean']
arms = {}
if a.arms:
    for f in a.arms.split('+'):
        for l in open(f): r = json.loads(l); arms[r['label']] = r
print('%-16s %-7s %7s %7s %6s %7s | %6s %6s %6s' % ('lane', 'proto', 'FGD', 'still', 'speed', 'follow', 'contact', 'upAmp', 'wrAmp'))
for spec in a.lanes:
    lab, d = spec.split('=', 1)
    for p in ('offline', 'engine'):
        e = f'{d}/eval/final/{p}'
        st = [json.load(open(f))['free'] for f in sorted(glob.glob(e + '/rollout-s*.json'))]
        fo = []
        for s in sorted(glob.glob(e + '/semantic-s*')):
            try: fo.append(json.load(open(s + '/score-own.json'))['centre']['top1'] - json.load(open(s + '/score-swap.json'))['centre']['top1'])
            except FileNotFoundError: pass
        ar = arms.get(f'{lab}-{p}')                       # arm_audit labels = <lane>-<protocol>
        con = ar and sum(ar['rate']['gen'][k] for k in ('hand_torso', 'hand_hand', 'forearm_torso'))
        f_ = fgd.get((lab, p))
        print('%-16s %-7s %7s %6.1f%% %6.2f %+7.3f | %6s %6s %6s' % (lab, p, '%.1f' % f_ if f_ else '-', 100 * np.mean([x['still'] for x in st]) if st else float('nan'),
              np.mean([x['speed'] for x in st]) if st else float('nan'), np.mean(fo) if fo else float('nan'),
              '%.1f%%' % (100 * con) if ar else '-', '%.2f' % ar['amp_ratio']['upper_std'] if ar else '-', '%.2f' % ar['amp_ratio']['wrist_std'] if ar else '-'))

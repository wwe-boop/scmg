"""Summary table of ctl_g1.sh (11-06): per output = official FGD / |dDiv| / |dBA| (benchmark json), pair_win (mean over the 3
recognisers; paired difference vs ctrl-offline with the per-recogniser 95 % CI), control metrics (CTL_EVAL.json) and text
sensitivity (TEXT_SENS.json). Writes <root>/SUMMARY.json and prints a markdown table.
    python scripts/semantic/ctl_summary.py <ctl_g1 out root>"""
import json, sys
from pathlib import Path

O = Path(sys.argv[1]); R = O / 'dev'
rows = {}


def bench_json(p):
    f = p.parent / f'benchmark_{p.name}.json'
    if not f.exists(): return None
    m = json.loads(f.read_text())['metrics']
    return dict(FGD=m['FGD'], dDiv=m['Div_gap'], dBA=m['BA_gap'])


for lane in sorted(x for x in R.iterdir() if x.is_dir()) if R.exists() else []:
    lab = lane.name
    for p in ('offline', 'engine'):
        b = bench_json(lane / 'bench' / p / 'seed_000')
        if b: rows[f'{lab}-{p}'] = b
    for d in sorted((lane / 'ctl').glob('*-*/seed_000')) if (lane / 'ctl').exists() else []:
        b = bench_json(d)
        if b: rows[f"{lab}-{d.parent.name.replace('offline', 'off').replace('engine', 'eng')}"] = b
    for d in sorted((lane / 'cfgt').glob('ct*/seed_000')) if (lane / 'cfgt').exists() else []:
        b = bench_json(d)
        if b: rows[f'{lab}-{d.parent.name}'] = b
    for nm in ('notext', 'swap'):
        b = bench_json(lane / 'ts' / nm / 'seed_000')
        if b: rows[f'{lab}-{nm}'] = b
    ts = lane / 'ts' / 'TEXT_SENS.json'
    if ts.exists():
        t = json.loads(ts.read_text())
        for k in ('swap', 'notext'):
            if k in t: rows.setdefault(f'{lab}-offline', {})[f'text_{k}_ratio'] = t[k]['upper']['ratio']
    ce = lane / 'ctl' / 'CTL_EVAL.json'
    if ce.exists():
        for k, v in json.loads(ce.read_text()).items():
            g = v.get('given') or v.get('gt') or {}
            key = f"{lab}-{k.replace('offline', 'off').replace('engine', 'eng')}" if not k.startswith('none') else f"{lab}-{k.split('-')[1]}"
            r = rows.setdefault(key, {})
            for m in ('point_stg', 'point_score_a', 'point_score_b', 'point_ray_deg', 'amp_hit', 'amp_mean', 'act_03'):
                if g.get(m, {}).get('mean') is not None: r[m] = g[m]['mean']
            if 'dq_span_rad' in v: r['dq_span_rad'] = v['dq_span_rad']['mean']
rep = O / 'sem' / 'REPORT.json'
if rep.exists():
    rp = json.loads(rep.read_text())
    for src, t in rp['table'].items():
        rows.setdefault(src, {})['pair_win'] = t['pair_win']['mean']
    for pr, per in rp['pairs'].items():
        x = pr.split(':')[0]
        if per: rows.setdefault(x, {})['pw_diff_vs_ctrl'] = [round(v['pair_win']['diff'], 4) for v in per.values()]; \
            rows[x]['pw_ci_vs_ctrl'] = [[round(c, 4) for c in v['pair_win']['ci95']] for v in per.values()]
(O / 'SUMMARY.json').write_text(json.dumps(rows, indent=1))
cols = ['FGD', 'dDiv', 'dBA', 'pair_win', 'pw_diff_vs_ctrl', 'text_swap_ratio', 'text_notext_ratio', 'point_stg', 'point_score_a',
        'point_score_b', 'amp_hit', 'act_03', 'dq_span_rad']
print('| output | ' + ' | '.join(cols) + ' |'); print('|' + '---|' * (len(cols) + 1))
fmt = lambda v: '' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v))
for k in sorted(rows): print(f'| {k} | ' + ' | '.join(fmt(rows[k].get(c)) for c in cols) + ' |')

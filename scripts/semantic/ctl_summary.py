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
            if k in t:
                rr = rows.setdefault(f'{lab}-offline', {})
                rr[f'text_{k}_d'] = t[k]['upper']['d_swap']; rr[f'text_{k}_ratio'] = t[k]['upper']['ratio_of_means']
    wsf = lane / 'ts' / 'WINDOW_SENS.json'
    if wsf.exists():
        for k, v in json.loads(wsf.read_text()).items():
            r = rows.setdefault(f'{lab}-offline' if k == 'base' else f'{lab}-{k}', {})
            r.update(ws_r_text=v['r_text'], ws_r_audio=v['r_audio'], ws_diff=v['diff'], ws_ci=[round(c, 3) for c in v['ci95']])
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
        x, y = pr.split(':')
        if not per: continue
        key = 'pw_gain_over_notext' if y.endswith('-notext') else 'pw_diff_vs_ctrl'
        rows.setdefault(x, {})[key] = round(float(sum(v['pair_win']['diff'] for v in per.values()) / len(per)), 4)
        rows[x][key + '_ci'] = [[round(c, 3) for c in v['pair_win']['ci95']] for v in per.values()]
        rows[x]['det_auc_' + key] = round(float(sum(v['det_auc']['diff'] for v in per.values()) / len(per)), 4)
    for src, t in rp['table'].items():
        if 'det_auc' in t: rows.setdefault(src, {})['det_auc'] = t['det_auc']['mean']
(O / 'SUMMARY.json').write_text(json.dumps(rows, indent=1))
cols = ['FGD', 'dDiv', 'dBA', 'pair_win', 'det_auc', 'pw_diff_vs_ctrl', 'pw_gain_over_notext', 'pw_gain_over_notext_ci', 'ws_r_text', 'ws_r_audio',
        'ws_diff', 'ws_ci', 'text_swap_d', 'text_notext_d', 'text_swap_ratio', 'text_notext_ratio', 'point_stg', 'point_score_a',
        'point_score_b', 'amp_hit', 'act_03', 'dq_span_rad']
print('| output | ' + ' | '.join(cols) + ' |'); print('|' + '---|' * (len(cols) + 1))
fmt = lambda v: '' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v))
for k in sorted(rows): print(f'| {k} | ' + ' | '.join(fmt(rows[k].get(c)) for c in cols) + ' |')

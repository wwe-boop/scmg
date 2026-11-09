"""11-06 G1 main table on val_common (3242): every row = runs (training seed x inference seed) of one method / protocol;
official metrics: mean, 95 % t-interval over the runs, min-max; executability on the 406-clip every-8th subset (independent
checker, --collision-stride 1): raw output and after the CBF-QP safety layer (exp/safe-proj safety_run.py cbf_qp), mean over runs.
    python scripts/echog_g1/final_table_g1.py --out T.json --md T.md  'label=bench1.json:exec1.json,bench2.json:exec2.json,...'
(exec*.json = the EXEC.json of post_final3.sh with panels raw / cbf_qp; may be '-')"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats

KEYS = ('FGD', 'dDiv', 'dBA', 'dJerkW', 'foot', 'slide')


def met(f):
    d = json.loads(Path(f).read_text()); m = d['metrics']
    return dict(FGD=m['FGD'], dDiv=m['Div_gap'], dBA=m['BA_gap'], dJerkW=m['Jerk_length_weighted_gap'],
                foot=m['foot_ground_error']['mean'], slide=m['contact_sliding_speed']['mean'], n=d['n_clips'])


def ci(x):
    x = np.asarray(x, float); m = float(x.mean())
    if len(x) < 2: return m, None, None
    h = float(stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return m, m - h, m + h


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--md'); ap.add_argument('rows', nargs='+')
    a = ap.parse_args(); rows = []
    for spec in a.rows:
        lab, rest = spec.split('=', 1); runs = []
        for part in rest.split(','):
            b, e = part.split(':') if ':' in part else (part, '-')
            r = met(b)
            if e != '-' and Path(e).exists():
                ex = json.loads(Path(e).read_text())
                for k, pan in (('raw', 'raw'), ('cbf', 'cbf_qp')):
                    s = ex[pan]; s = s.get('summary', s)
                    r[f'exec_{k}_clean'] = s['clean_clips']; r[f'exec_{k}_any'] = s['any']
            r['bench'] = b; runs.append(r)
        row = dict(label=lab, runs=runs, n_runs=len(runs))
        for k in KEYS + ('exec_raw_clean', 'exec_raw_any', 'exec_cbf_clean', 'exec_cbf_any'):
            v = [r[k] for r in runs if k in r]
            if v: m, lo, hi = ci(v); row[k] = dict(mean=m, ci95=[lo, hi], min=float(min(v)), max=float(max(v)), n=len(v))
        rows.append(row)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    if a.md:
        f = lambda v: f'{v:.4f}' if abs(v) < .1 else f'{v:.3f}'
        cell = lambda d: '' if d is None else (f(d['mean']) + (f" [{f(d['ci95'][0])}, {f(d['ci95'][1])}]" if d['ci95'][0] is not None else ''))
        pct = lambda d: '' if d is None else f"{100 * d['mean']:.1f}%"
        L = ['| row | runs | FGD (95% CI) | FGD min-max | dDiv | dBA | dJerkW | foot | slide | clean raw / CBF-QP | viol. frames raw / CBF-QP |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
        for r in rows:
            g = lambda k: r.get(k)
            L.append(f"| {r['label']} | {r['n_runs']} | {cell(g('FGD'))} | {f(g('FGD')['min'])}-{f(g('FGD')['max'])} | {cell(g('dDiv'))} | "
                     f"{cell(g('dBA'))} | {cell(g('dJerkW'))} | {cell(g('foot'))} | {cell(g('slide'))} | "
                     f"{pct(g('exec_raw_clean'))} / {pct(g('exec_cbf_clean'))} | {pct(g('exec_raw_any'))} / {pct(g('exec_cbf_any'))} |")
        Path(a.md).write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()

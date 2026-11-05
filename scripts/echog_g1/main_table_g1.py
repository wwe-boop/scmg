"""11-06 G1 main-table rows on val_common (3242): official ECHO-G metrics of each row's benchmark JSONs (inference seeds
listed), plus the executability of seed 0 on the 406-clip every-8th subset (executability.py --collision-stride 1).
    python scripts/echog_g1/main_table_g1.py --out T.json --md T.md  label=bench0.json,bench1.json[@exec.json:panel] ...
"""
import argparse, json
from pathlib import Path
import numpy as np


def met(f):
    m = json.loads(Path(f).read_text())['metrics']
    return dict(FGD=m['FGD'], Div=m['Div'], dDiv=m['Div_gap'], dDiv_s=m['Div_signed_gap'], BA=m['BA']['mean'], dBA=m['BA_gap'],
                dJerkW=m['Jerk_length_weighted_gap'], dJerkW_s=m['Jerk_length_weighted_signed_gap'],
                foot=m['foot_ground_error']['mean'], slide=m['contact_sliding_speed']['mean'], n=json.loads(Path(f).read_text())['n_clips'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--md'); ap.add_argument('rows', nargs='+')
    a = ap.parse_args(); rows = []
    for spec in a.rows:
        lab, rest = spec.split('=', 1); ex = None
        if '@' in rest: rest, ex = rest.split('@', 1)
        ms = [met(f) for f in rest.split(',')]
        r = dict(label=lab, seeds=len(ms), per_seed=ms, **{k: float(np.mean([m[k] for m in ms])) for k in ms[0] if k != 'n'}, n=ms[0]['n'])
        if ex:
            f, panel = ex.split(':'); e = json.loads(Path(f).read_text())[panel]['summary']
            r.update(exec_pass=e['pass_rate'] if 'pass_rate' in e else e.get('clips_pass'), exec_frames_any=e.get('frames_any', e.get('any')), exec_raw=e)
        rows.append(r)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    if a.md:
        cols = ('FGD', 'dDiv', 'dDiv_s', 'BA', 'dBA', 'dJerkW', 'foot', 'slide', 'exec_pass', 'exec_frames_any')
        L = ['| row | seeds | ' + ' | '.join(cols) + ' |', '|' + '---|' * (2 + len(cols))]
        f = lambda v: '' if v is None else (f'{v:.4f}' if abs(v) < .1 else f'{v:.3f}')
        for r in rows:
            fg = ' / '.join(f"{m['FGD']:.3f}" for m in r['per_seed'])
            L.append(f"| {r['label']} | {r['seeds']} (FGD {fg}) | " + ' | '.join(f(r.get(c)) for c in cols) + ' |')
        Path(a.md).write_text('\n'.join(L) + '\n')
    print(Path(a.md).read_text() if a.md else json.dumps(rows))


if __name__ == '__main__':
    main()

"""Print the official ECHO-G G1 metrics of benchmark JSONs as one table.
  table.py name=path.json [name=path.json ...]
Columns: FGD, BA (|BA-BA_GT|), Div (|Div-Div_GT|), dJerk = Jerk_length_weighted_gap (m/s^3, T-3 weighted), foot ground
error (m), contact sliding speed (m/s), MM20 (null unless 20 runs); n clips / frames."""
import json, sys

rows = []
for arg in sys.argv[1:]:
    name, path = arg.split('=', 1)
    d = json.load(open(path)); m = d['metrics']
    rows.append((name, d['n_clips'], m['FGD'], m['BA']['mean'], m['BA_gap'], m['Div'], m['Div_gap'],
                 m['Jerk_length_weighted']['mean'], m['Jerk_length_weighted_gap'], m['foot_ground_error']['mean'],
                 m['contact_sliding_speed']['mean'], m['Multimodality']))
hdr = ('method', 'n', 'FGD', 'BA', '|dBA|', 'Div', '|dDiv|', 'Jerk', '|dJerk|', 'foot_err', 'slide', 'MM20')
print('| ' + ' | '.join(hdr) + ' |'); print('|' + '---|' * len(hdr))
for r in rows:
    f = [r[0], str(r[1])] + [('%.4f' % x) if isinstance(x, float) else str(x) for x in r[2:]]
    print('| ' + ' | '.join(f) + ' |')

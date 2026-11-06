"""Appendix ablation table of the interruption study (11-06): every method on the dev subset (every 3rd sorted dev stem,
300 interruptions, interruption-causal prefix conditions), all three abruptness definitions (v3 main, v2, v1), from the
eval_interrupt.py result files. Prints a markdown table and writes <out>.tsv.
    python scripts/interrupt/make_ablation_table.py --metrics <dir> --out <dir>/appendix_ablation
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument('--metrics', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
M = Path(a.metrics)
R = {}
for f in ('dev_v3.json', 'dev_v3_blends.json', 'dev_v3_r0causal.json', 'dev_v3_extra.json'):
    p = M / f
    if p.exists(): R.update({k: v for k, v in json.loads(p.read_text()).items() if k != '_meta'})
GROUPS = [
 ('baselines', [('B1 freeze', 'B1'), ('B2_p damped retraction (T_r 1.8) + projection', 'B2_p'), ('B2fast_p (T_r 1.0)', 'B2fast_p'),
                ('B3c continuation (causal)', 'B3c'), ('GT + B2', 'GT_B2')]),
 ('ours: zero-training (selected)', [('R0causal_p blend + texture + projection', 'R0causal_p'), ('R0_p (non-causal continuation)', 'R0_p')]),
 ('training: floor token / tail data', [('ctrl (2000 more steps, no tails)', 'ctrl'), ('yld tails, floor 2', 'yld_f2'), ('yld tails, floor 0', 'yld'),
                ('lab tails, floor 2', 'lab_f2'), ('lab tails, floor 1', 'lab_f1'), ('tail05w2 (tail share 0.5, w_tail 2)', 'tail05w2'),
                ('tail05w2td05 (+ w_tdelta 0.5)', 'tail05w2td05')]),
 ('yield guidance (audio-state CFG), yld model', [('s = 1', 'yld_g1'), ('s = 2', 'yld_g2'), ('s = 4', 'yld_g4')]),
 ('model-difference guidance v_yld + s (v_yld - v_sf0), _p', [('s = 1', 'd1_p'), ('s = 2', 'd2_p'), ('s = 4', 'd4_p')]),
 ('rest-pose inpainting on yld, chunk restart, _p', [('tr2 T_r 1.0 (pins at T_r only)', 'tr2_10_p'), ('tr2 T_r 1.4', 'tr2_14_p'),
                ('tr2h T_r 1.0 (+ pin later chunk ends)', 'tr2h10_p'), ('tr2h T_r 1.4', 'tr2h14_p'), ('tr2h T_r 1.4 soft w 0.5', 'tr2h14w05_p'),
                ('tr2h T_r 1.4 soft w 0.3', 'tr2h14w03_p'), ('tr2hd T_r 1.4 damped target', 'tr2hd14_p')]),
 ('yld + joint-space blend, _p', [('blend + texture 0.5', 'yld_tex05_p'), ('blend, no texture', 'yld_bl_p')]),
]
pc = lambda x: '-' if x is None else f'{100 * x:.1f}'
hdr = ['group', 'method', 'yield s', 'yield %', 'v3 %', 'v3 ratio med / p90', 'v2 %', 'v1 %', 'v1 pre %', 'RKD', 'unsafe %', 'react s']
rows = []
for g, items in GROUPS:
    for name, key in items:
        o = R.get(key)
        if o is None: rows.append([g, name] + ['n/a'] * (len(hdr) - 2)); continue
        rows.append([g, name, f"{o['yield_s_R_sil']:.2f}", pc(o['yield_rate_R_sil']), pc(o.get('abrupt_v3')),
                     '-' if o.get('v3_ratio_med') is None else f"{o['v3_ratio_med']:.2f} / {o['v3_ratio_p90']:.2f}",
                     pc(o.get('abrupt_induced')), pc(o['abrupt_rate']), pc(o.get('abrupt_pre')), f"{o['rkd']:.3f}", pc(o['unsafe_any']),
                     '-' if o.get('react_s') is None else f"{o['react_s']:.3f}"])
md = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)] + ['| ' + ' | '.join(r) + ' |' for r in rows]
print('\n'.join(md))
Path(a.out + '.tsv').write_text('\n'.join('\t'.join(r) for r in [hdr] + rows) + '\n')
DEF = """
Abruptness definitions (all on peak hand jerk, official FK, 30 fps third differences):
  v1 (11-05) peak jerk on [t_i - 0.1 s, t_i + 2.13 s] above the human post-speech P95 (262 m/s^3). GT normal speech: 81 %.
  v2 (11-06, after round 1106b) peak jerk on [t_i, t_i + 0.5 s] above the P95 of the same motion's 2.13 s before t_i.
     Structural base rate without interruption: ours 36.4 %, GT 45.6 %, human retraction windows 36.7 %.
  v3 (11-06, final) peak jerk over [t_i, t_i + 1.8 s] / the same peak of the method's own uninterrupted output > 1.5.
     Baseline 0 by construction; threshold 1.5 a priori. criteria/metric_change_1106.json, criteria/metric_change_v3_1106.json."""
print(DEF)

"""11-08 audit: does the EX002 interruption evaluator's splice (motion = U[:f] + y[f:], reference = U) depend on the prefix mismatch
between a continuation file y and the uninterrupted reference U?  Per interruption: max |y[:f] - U[:f]| (rad), the earliest differing
frame counted back from f (1 = frame f-1), and v3_ratio two ways:
  cur : motion U[:f] + y[f:]  vs reference U                         (eval_interrupt_ex002 'executed' rule)
  alt : motion y[:f] + y[f:]  vs reference y[:f] + U[f:]              (both share the continuation's own prefix)
    python prefix_check_ex002.py --out O.json label=<ydir>:<Udir> ..."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import icommon_ex002 as I
X = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/ex002-interrupt-1107'
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('rows', nargs='+'); a = ap.parse_args()
ints = json.loads(Path(X + '/interrupts_dev_ex002.json').read_text())['interrupts']; res = {}
for spec in a.rows:
    lab, rest = spec.split('=', 1); yd, ud = rest.split(':'); per = []
    for r in ints:
        fn = Path(yd) / f"{r['ci']}@{r['item']}@{r['frac']:.1f}.npy"
        if not fn.exists(): continue
        f = r['f_i']; y = np.load(fn).astype(np.float64); U = np.load(f"{ud}/{r['ci']}.npy").astype(np.float64)
        n = min(len(y), f + I.WIN)
        if n < f + I.WIN or len(U) < f: continue
        d = np.abs(y[:f] - U[:f]).max(1); nz = np.nonzero(d > 0)[0]
        cur_m = np.concatenate([U[:f], y[f:n]]); alt_r = np.concatenate([y[:f], U[f:n]]) if len(U) >= n else None
        Pc, PU = I.fk(cur_m), I.fk(U[:n]); Pa = I.fk(y[:n])
        vc = I.v3_ratio(Pc, PU, f); va = I.v3_ratio(Pa, I.fk(alt_r), f) if alt_r is not None else None
        per.append(dict(ci=r['ci'], item=r['item'], frac=r['frac'], maxdiff=float(d.max()), first_back=int(f - nz[0]) if len(nz) else 0, v3_cur=vc, v3_alt=va))
    md = np.array([p['maxdiff'] for p in per]); fb = np.array([p['first_back'] for p in per])
    vc = np.array([p['v3_cur'] for p in per if p['v3_cur'] is not None and p['v3_alt'] is not None]); va = np.array([p['v3_alt'] for p in per if p['v3_cur'] is not None and p['v3_alt'] is not None])
    res[lab] = dict(n=len(per), share_bitwise_prefix=float((md == 0).mean()), maxdiff_p50=float(np.median(md)), maxdiff_p95=float(np.percentile(md, 95)), maxdiff_max=float(md.max()),
                    share_diff_only_last3=float(((fb >= 1) & (fb <= 3)).mean()), share_diff_earlier_than_3=float((fb > 3).mean()), first_back_max=int(fb.max()),
                    v3_cur=float((vc > I.V3_RATIO).mean()), v3_alt=float((va > I.V3_RATIO).mean()), v3_pp_change=float(100 * ((va > I.V3_RATIO).mean() - (vc > I.V3_RATIO).mean())),
                    v3_flips=int(((vc > I.V3_RATIO) != (va > I.V3_RATIO)).sum()), items=per)
    print(lab, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res[lab].items() if k != 'items'}), flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))

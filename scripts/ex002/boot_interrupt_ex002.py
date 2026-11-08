"""11-08 audit H1: recording-clustered bootstrap CIs of the EX002 interruption rows (eval_interrupt_ex002 reports point estimates only).
Unit = dev recording (ci, 43): resample recordings with replacement, keep every interruption of a drawn recording; the same draw for
every row (paired differences). Metrics as eval_interrupt_ex002 / yield_v4_ex002: yield3d (need3d rows), yield-height (need rows),
v3 = share v3_ratio > V3_RATIO, unsafe_any (WBC), RKD-post (kin features vs the human windows, the human set held fixed),
and (if --v4 per-item given) yield v4 activity / withdrawal / v4.
    python boot_interrupt_ex002.py --per-item A.jsonl [B.jsonl ...] [--v4 V.jsonl] --human H --rows R1 R2 ... --pairs R1-R2 ... --out O.json"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import icommon_ex002 as I
ap = argparse.ArgumentParser(); ap.add_argument('--per-item', nargs='+', required=True); ap.add_argument('--v4', default=None)
ap.add_argument('--human', required=True); ap.add_argument('--rows', nargs='+', required=True); ap.add_argument('--pairs', nargs='*', default=[])
ap.add_argument('--n', type=int, default=2000); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
a = ap.parse_args()
hw = np.load(a.human + '_windows.npz'); HK = np.stack([I.kin_features(I.fk(w), I.SEC) for w in hw['q_ctx'].astype(np.float64)])
R = {}
for fn in a.per_item:
    for l in open(fn):
        r = json.loads(l)
        if r['method'] in a.rows: R.setdefault(r['method'], {})[(r['ci'], r['item'], round(r['frac'], 1))] = r
if a.v4:
    for l in open(a.v4):
        r = json.loads(l); k = (r['ci'], r['item'], round(r['frac'], 1))
        if r['method'] in R and k in R[r['method']]: R[r['method']][k].update(v4_activity=r['activity'], v4_withdrawal=r['withdrawal'], v4=r['v4'])
recs = sorted({k[0] for m in R.values() for k in m})


def metrics(rows):
    s3 = [x for x in rows if x['need3d']]; sh = [x for x in rows if x['need']]; v3 = [x['v3_ratio'] for x in rows if x.get('v3_ratio') is not None]
    out = dict(yield3d=np.mean([x['yield3d_s'] is not None for x in s3]) if s3 else np.nan,
               yield_height=np.mean([x['yield_s'] is not None for x in sh]) if sh else np.nan,
               v3=np.mean(np.array(v3) > I.V3_RATIO) if v3 else np.nan, unsafe=np.mean([x['unsafe_any'] for x in rows]),
               rkd_post=I.kin_distance(np.stack([x['kin'] for x in rows]), HK)[0])
    if rows and 'v4' in rows[0]:
        for k in ('v4_activity', 'v4_withdrawal', 'v4'): out[k] = np.mean([x[k] for x in rows])
    return out


by = {m: {c: [v for k, v in R[m].items() if k[0] == c] for c in recs} for m in R}
point = {m: metrics([x for c in recs for x in by[m][c]]) for m in R}
rng = np.random.default_rng(a.seed); BS = {m: [] for m in R}
for _ in range(a.n):
    draw = rng.choice(recs, len(recs), replace=True)
    for m in R: BS[m].append(metrics([x for c in draw for x in by[m][c]]))
res = dict(_meta=dict(unit='dev recording (ci)', n_recordings=len(recs), n_boot=a.n, seed=a.seed, human_windows=int(len(HK)),
                      note='RKD-post: human windows held fixed (not resampled); pairs use the same recording draw'))
for m in R:
    res[m] = {k: dict(point=float(v), ci95=[float(np.nanpercentile([b[k] for b in BS[m]], 2.5)), float(np.nanpercentile([b[k] for b in BS[m]], 97.5))])
              for k, v in point[m].items()}
    res[m]['n'] = len(R[m]); print(m, json.dumps({k: [round(v['point'], 4)] + [round(x, 4) for x in v['ci95']] for k, v in res[m].items() if k != 'n'}), flush=True)
for p in a.pairs:
    x, y = p.split('-', 1); d = {}
    for k in point[x]:
        diff = np.array([bx[k] - by_[k] for bx, by_ in zip(BS[x], BS[y])])
        d[k] = dict(point=float(point[x][k] - point[y][k]), ci95=[float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5))])
    res[f'{x}-{y}'] = d; print(p, json.dumps({k: [round(v['point'], 4)] + [round(z, 4) for z in v['ci95']] for k, v in d.items()}), flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))

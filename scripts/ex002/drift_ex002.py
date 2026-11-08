"""11-08 design D3 side metric: outward drift after the interruption. Per event and wrist: max over frames [f, f + 15] (0.3 s) of the
wrist displacement from frame f - 1 along the unit wrist velocity at f - 1 (central difference of frames f - 3 .. f - 1, exact FK of
icommon_ex002); the larger of the two wrists; reported as median / p95 / max (m) over events. Executed stream = U[:f] + y[f:].
    python drift_ex002.py --out O.json label=<dir>:<U dir> ..."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import icommon_ex002 as I
X = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/ex002-interrupt-1107'
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('rows', nargs='+'); a = ap.parse_args()
ints = json.loads(Path(X + '/interrupts_dev_ex002.json').read_text())['interrupts']; res = {}
for spec in a.rows:
    lab, rest = spec.split('=', 1); d, ud = rest.split(':'); dr = []
    for r in ints:
        fn = Path(d) / f"{r['ci']}@{r['item']}@{r['frac']:.1f}.npy"
        if not fn.exists(): continue
        f = r['f_i']; y = np.load(fn).astype(np.float64); U = np.load(f"{ud}/{r['ci']}.npy").astype(np.float64)
        z = np.concatenate([U[:f], y[f:]])[:f + 16]; P = I.fk(z[f - 3:])          # frames f-3 .. f+15
        v = P[2] - P[0]; u = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
        disp = ((P[3:] - P[2][None]) * u[None]).sum(-1)                              # [16, wrists]
        dr.append(float(disp.max(0).max()))
    dr = np.array(dr); res[lab] = dict(n=len(dr), median=float(np.median(dr)), p95=float(np.percentile(dr, 95)), max=float(dr.max()))
    print(lab, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res[lab].items()}), flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))

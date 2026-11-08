"""11-08 audit F2: write the interruption baselines B1 (freeze at f_i), B2 (critically damped to the reference rest pose R_sil, T_r 1.8 s)
and FRONT (damped to the fixed FRONT pose) as stream files <out>/<B1|B2|FRONT>/<ci>@<item>@<frac>.npy (frames up to f_i + 110), built
exactly as eval_interrupt_ex002.py builds them from the base's own uninterrupted psync output U, so they can go through the same
post chain (safety_int_ex002.py --isc) as the method rows.
    python baseline_streams_ex002.py --base E --human H --out O"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import icommon_ex002 as I
X = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/ex002-interrupt-1107'
ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True); ap.add_argument('--human', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
H = json.load(open(a.human + '.json')); rest = np.array(H['rest']['R_sil']['joints']); rest_z = np.array(H['rest']['R_sil']['hand_z'])
ints = json.loads(Path(X + '/interrupts_dev_ex002.json').read_text())['interrupts']
gts = {ci: np.load(f'{a.base}/psync/gt_{ci}.npy').astype(np.float64) for ci in sorted({r['ci'] for r in ints})}
cand = []
rest_P = I.fk(rest[None])[0]
for ci_, q_ in sorted(gts.items()):
    P_ = I.fk(q_[::5]); dz = np.abs(P_[:, :, 2] - rest_z[None]).max(1); dd = np.linalg.norm(P_ - rest_P[None], axis=-1).min(1)
    for t_ in np.nonzero(dz <= 0.05)[0]: cand.append((float(dd[t_]), ci_, int(t_ * 5)))
cand.sort(reverse=True); front = gts[cand[0][1]][cand[0][2]].copy()
for lab in ('B1', 'B2', 'FRONT'): (Path(a.out) / lab).mkdir(parents=True, exist_ok=True)
for r in ints:
    f = r['f_i']; U = np.load(f"{a.base}/psync/s0/{r['ci']}.npy").astype(np.float64); n = f + 110; name = f"{r['ci']}@{r['item']}@{r['frac']:.1f}.npy"
    for lab, y in (('B1', I.freeze_after(U, f)), ('B2', I.retract_after(U, f, rest, 1.8)), ('FRONT', I.retract_after(U, f, front, 1.8))):
        np.save(Path(a.out) / lab / name, y[:n].astype(np.float32))
print('BASELINE_STREAMS_DONE', len(ints), 'front', cand[0][1:])

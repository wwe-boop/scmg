"""11-08 EX002 interruption benchmark, deployed chain: generator -> (R0causal / Funnel post) -> streaming safety projection.
For every interruption r the EXECUTED stream z = U[:f_i] (the method's own uninterrupted psync output) + y[f_i:] (its continuation),
truncated to f_i + WIN, is streamed from the recording start through the EX002 safety layer (proj_l2, the 3c-4 settings: B 13,
d_safe 1 mm, qdldl, tail_smooth 1e-3, recursive feasibility) in 16-frame chunks; the output keeps the input file name, so
eval_interrupt_ex002.py scores it as B3:<out>:<projected U> (the projected prefix is identical: the layer is causal and deterministic).
Needs the exp/safe-proj code on PYTHONPATH (scmg.safety_proj, scmg.safety_ex002_3f, scmg.safety_proj_ex002).
    python scripts/ex002/safety_int_ex002.py --in-dir Y --prefix-from U --out-dir O --interrupts I.json [--shard i/n]"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
os.environ.setdefault('SCMG_SAFETY_KKT', 'qdldl')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import icommon_ex002 as I


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--in-dir', required=True); ap.add_argument('--prefix-from', required=True)
    ap.add_argument('--out-dir', required=True); ap.add_argument('--interrupts', required=True); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--B', type=int, default=13); ap.add_argument('--tail-smooth', type=float, default=1e-3)
    a = ap.parse_args()
    from scmg import safety_proj as S
    from scmg.safety_ex002_3f import EX002RMLimits, K_3F
    from scmg.safety_proj_ex002 import EX002Collision
    lim = EX002RMLimits(); coll = EX002Collision(lim.E)
    L = S.ProjectionLayer(lim, coll, 'l2', K=K_3F, B=a.B, d_safe=1e-3, tail_smooth=a.tail_smooth, recursive=True); L.start = 'rest'
    ints = json.loads(Path(a.interrupts).read_text())['interrupts']; si, sn = (int(x) for x in a.shard.split('/'))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); n = 0; t0 = time.time(); fb = 0
    for r in ints[si::sn]:
        name = f"{r['ci']}@{r['item']}@{r['frac']:.1f}.npy"; f = r['f_i']
        if (out / name).exists() or not (Path(a.in_dir) / name).exists(): continue
        y = np.load(Path(a.in_dir) / name).astype(np.float64); U = np.load(Path(a.prefix_from) / f"{r['ci']}.npy").astype(np.float64)
        z = np.concatenate([U[:f], y[f:]])[:f + I.WIN]
        ys, st = S.run_stream(L, z, K_3F); fb += st['fallback_frames']
        np.save(out / name, ys.astype(np.float32)); n += 1
    print('SAFETY_INT_DONE', n, 'fallback_frames', fb, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

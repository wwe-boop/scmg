"""11-08 EX002 3c dev runner (exp/duplex docs/criteria/EX002_CBF_1108.md): the three-finger CBF-QP on bench_ex002 dev outputs.
    python scripts/ex002_cbf_dev.py run --src <bench dir with <P>/s<k>/<ci>.npy + <P>/gt_*.npy> --out O --proto psync --seed 0
    python scripts/ex002_cbf_dev.py check --src SRC --out O            (gates 1 / 3 + raw comparison over all runs present)
Every recording is one stream (layer.reset), fed 16-frame chunks; outputs O/<P>/s<k>/<ci>.npy (float32, 50 Hz q34) + stats.jsonl."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(a):
    os.environ.setdefault('SCMG_SAFETY_KKT', 'qdldl')     # 11-08: pinned KKT backend (EX002_CBF_1108.md 3c-3b amendment)
    from scmg import safety_proj as S
    from scmg.safety_ex002_3f import make_cbf_3f, make_slack_3f, EX002RMLimits, K_3F
    if a.layer == 'cbf_qp': L, lim, coll = make_cbf_3f()
    elif a.layer == 'cbf_slack': L, lim, coll = make_slack_3f()
    else:                                    # proj_l2: the C3 chunk projection (module 3c-3)
        from scmg.safety_proj_ex002 import EX002Collision, B_EX
        lim = EX002RMLimits(); coll = EX002Collision(lim.E); L = S.ProjectionLayer(lim, coll, 'l2', K=K_3F, B=a.B or B_EX, d_safe=1e-3, tail_smooth=a.tail_smooth, recursive=a.recursive, max_iter=a.max_iter, pred_d=a.pred_d, carry=a.carry); L.start = 'rest'; L.reset()
    src = Path(a.src) / a.proto / f's{a.seed}'; out = Path(a.out) / a.proto / f's{a.seed}'; out.mkdir(parents=True, exist_ok=True)
    for g in (Path(a.src) / a.proto).glob('gt_*.npy'):
        d = Path(a.out) / a.proto / g.name
        try: os.symlink(g, d)
        except FileExistsError: pass                       # parallel seed runs of one protocol share the GT links
    with open(out / 'stats.jsonl', 'w') as fo:
        for f in sorted(src.glob('*.npy'), key=lambda p: int(p.stem)):
            q = np.load(f).astype(np.float64); t = time.perf_counter(); y, st = S.run_stream(L, q, K_3F)
            np.save(out / f.name, y.astype(np.float32))
            fo.write(json.dumps(dict(ci=int(f.stem), T=len(q), wall=time.perf_counter() - t, times=st['times'], fallback_frames=st['fallback_frames'],
                                     fallback_chunks=st['fallback_chunks'], rmse=float(np.sqrt(((y - q) ** 2).mean())),
                                     qp=st.get('qp', 0), qp_fail=st.get('qp_fail', 0), solves=st.get('solves_per_chunk', [])[-len(st['times']):] if st.get('solves_per_chunk') else [])) + '\n')
    print('CBF_RUN_DONE', a.proto, a.seed, os.getloadavg())


def check(a):
    from scmg.safety_ex002_3f import EX002RMLimits
    lim = EX002RMLimits(); X = lim.X; res = {}
    for P in ('offline', 'psync'):
        for k in range(3):
            o = Path(a.out) / P / f's{k}'
            if not (o / 'stats.jsonl').exists(): continue
            n = wb = pos = vel = wraw = 0; times = []; fb = 0; rm = []
            for line in open(o / 'stats.jsonl'):
                r = json.loads(line); times += r['times']; fb += r['fallback_frames']; rm.append(r['rmse'])
                y = np.load(o / f"{r['ci']}.npy").astype(np.float64); x = np.load(Path(a.src) / P / f's{k}' / f"{r['ci']}.npy").astype(np.float64)
                n += len(y); wb += sum(not X.clearance(z)[0] for z in y); wraw += sum(not X.clearance(z)[0] for z in x)
                pos += int(((y < lim.lo - 1e-6) | (y > lim.hi + 1e-6)).any(1).sum())
                vel += int((np.abs(np.diff(y, axis=0)) / lim.dt > lim.vmax + 1e-6).any(1).sum())
            t = np.array(times) * 1e3
            res[f'{P}-s{k}'] = dict(frames=n, wbc_violating_frames=wb, raw_wbc_violating_frames=wraw, pos_frames=pos, vel_frames=vel,
                                    all_clear=bool(wb == 0 and pos == 0 and vel == 0), fallback_frames=fb, rmse_mean=float(np.mean(rm)),
                                    t_p50=float(np.percentile(t, 50)), t_p95=float(np.percentile(t, 95)), t_p99=float(np.percentile(t, 99)), t_max=float(t.max()))
            print(P, k, json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in res[f'{P}-s{k}'].items()}), flush=True)
    Path(a.out, 'CHECK.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run'); r.add_argument('--src', required=True); r.add_argument('--out', required=True); r.add_argument('--proto', required=True)
    r.add_argument('--seed', type=int, required=True); r.add_argument('--layer', default='cbf_qp', choices=['cbf_qp', 'cbf_slack', 'proj_l2']); r.add_argument('--B', type=int, default=0)
    r.add_argument('--pred-d', type=float, default=0.0, help='3c-5: predictive initial rows within act + d (m)')
    r.add_argument('--carry', action='store_true', help='3c-5: carry the active / cut pairs of chunk k into chunk k+1')
    r.add_argument('--max-iter', type=int, default=4, help='QP solves (cutting-plane rounds) per chunk (3c-4b: 8)')
    r.add_argument('--recursive', action='store_true', help='3c-4: collision rows admit the shifted verified plan (recursive feasibility)')
    r.add_argument('--tail-smooth', type=float, default=1e-3, help='proj_l2 braking-tail velocity weight (11-08: 1e-3, chosen by conditioning)')
    c = sp.add_parser('check'); c.add_argument('--src', required=True); c.add_argument('--out', required=True)
    a = ap.parse_args(); run(a) if a.cmd == 'run' else check(a)

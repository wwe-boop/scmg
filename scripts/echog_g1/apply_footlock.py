"""Apply the G1 two-foot lock (scmg/g1_footlock.py) to every <stem>.pt of a prediction dir -> new dir (11-05).
    python scripts/echog_g1/apply_footlock.py <in dir> <out dir> [--workers 8] [--fps 30]"""
import argparse, sys, os
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--fps', type=float, default=30.)
a = ap.parse_args()


def one(f):
    import torch
    torch.set_num_threads(1)
    from scmg.g1_footlock import foot_lock
    o = Path(a.dst) / f.name
    if o.exists(): return 0
    x = torch.load(f, weights_only=False)
    rr = torch.as_tensor(x['robot_repr']).double()
    y = foot_lock(rr, a.fps).to(x['robot_repr'].dtype)
    if not torch.isfinite(y).all(): raise ValueError(f'{f.name}: non-finite after foot lock')
    x = dict(x); x['robot_repr'] = y
    g = x.get('generator'); x['generator'] = dict(g, foot_lock='scmg/g1_footlock.py') if isinstance(g, dict) else dict(source=g, foot_lock='scmg/g1_footlock.py')
    tmp = o.with_suffix('.pt.tmp'); torch.save(x, tmp); os.replace(tmp, o); return 1


if __name__ == '__main__':
    Path(a.dst).mkdir(parents=True, exist_ok=True)
    fs = sorted(Path(a.src).glob('*.pt'))
    with Pool(a.workers) as p: n = sum(p.map(one, fs, chunksize=4))
    print('FOOTLOCK_DONE', len(fs), 'files,', n, 'new')

"""GMR SMPL-X -> G1 retargeting cost (the EMAGE / GestureLSM / SemTalk benchmark converter, smplx_npz_to_g1.retarget_npz:
Y-up -> Z-up, SMPL-X FK, GMR IK per frame, Savitzky-Golay(7,2), 39-d encoding), CPU, 1 thread, on the same 50 clips.
  python lat_gmr.py --src <smplx npz dir> --tag <name> --out DIR [--assets DIR]"""
import argparse, json, os, resource, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lat_common as LC  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--src', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--assets', default='/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baselines-g1-1105/assets')
    ap.add_argument('--gmr-root', default='/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/gmr-repro')
    a = ap.parse_args()
    root = Path(a.assets); sys.path.insert(0, str(root / 'GMR'))
    import general_motion_retargeting  # noqa: F401
    sys.path.insert(0, str(Path(a.gmr_root) / 'scripts'))
    import smplx_npz_to_g1 as M
    M.R = root
    host0 = LC.host_snapshot(); stems = LC.stems50()[:a.n or None]; rows = []
    for k, stem in enumerate(stems):
        f = Path(a.src) / f'{stem}.npz'
        d = np.load(f, allow_pickle=True); n = len(d['poses'])
        t0 = time.perf_counter(); rr, _, _, _ = M.retarget_npz(f); tt = time.perf_counter() - t0
        rows.append(dict(stem=stem, dur=n / 30., frames30=n, stages=dict(gmr=tt), compute=tt, schedule=[(n / 30., tt, 0.0, n, 30.)]))
        print(f'{k + 1}/{len(stems)} {stem} n={n} gmr={tt:.3f}s', flush=True)
    extra = dict(src=a.src, cpu_threads=int(os.environ.get('OMP_NUM_THREADS', '1')), host_start=host0, host_end=LC.host_snapshot(),
                 note='dur = output frames / 30 (= the clip length); latency figures here are GMR alone after the whole SMPL-X sequence')
    LC.write(a.out, f'gmr_{a.tag}', rows, extra)


if __name__ == '__main__':
    main()

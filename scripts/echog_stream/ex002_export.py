"""EX002 dev export of native 50 FPS / 34D chunk predictions into the eval-dev format (11-05).

Same computation as the frozen dlc-package-r2/code/export_dev.py --retrained branch (that file is read-only and always exports
all 43 dev clips; this one also takes a clip subset for smoke runs and writes the audit_exports.py checks for the subset):
  per dev clip ci (robogesture/audio/MANIFEST.json), the data-dev chunks (data-dev/MANIFEST.json, <= 12 s, explicit offline
  cuts) are rejoined on the 50 FPS timeline (PCHIP at the chunk-local frame times, exact at the knots), clipped to the EX002
  position limits (views/data-v36_1/META.json -- the limits export_dev reads through Retargeter.lo / .hi), and the first 10
  frames are blended from the GT first frame (smoothstep), so frame 0 == GT exactly.
  Outputs <out>/<prefix>{,_seed1}/gen_Hinf_<ci>.npy, gt_<ci>.npy, meta_<ci>.json.
test_ex002.py checks this file against export_dev.py's released smoke export bit for bit.
    python scripts/echog_stream/ex002_export.py --native preds/stream_prefix --prefix stream_prefix --output eval-dev [--ci 0 1 2]
"""
import argparse, hashlib, json, time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

R = Path('/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official')
ROWS = R.parent / 'robogesture/audio/MANIFEST.json'
CHUNKS = R / 'data-dev/MANIFEST.json'
GT_DIR = R.parent / 'robogesture/audio'
META = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1/META.json')


def limits():
    m = json.loads(META.read_text())
    return m['joints'], np.array(m['ex002']['lower']), np.array(m['ex002']['upper'])


def export_clip(row, chunks, native_seed_dir, lo, hi):
    """export_dev.py retrained branch for one clip. returns y [T, 34] float32, gt, quality, partmeta"""
    ci = row['ci']; T = row['T']; fps = 50; N = T
    timeline = np.arange(N) / fps; native = np.zeros((N, 34), np.float32); cover = np.zeros(N, np.int32); partmeta = []
    for c in [c for c in chunks if c['ci'] == ci]:
        path = native_seed_dir / (c['stem'] + '.pt'); payload = torch.load(path, map_location='cpu', weights_only=False)
        z = torch.as_tensor(payload['robot_repr']).numpy()
        assert z.shape[1] == native.shape[1] and float(payload['fps']) == fps, (path, z.shape, payload['fps'])
        start = c['start_frame'] / 50; end = c['end_frame'] / 50; mask = (timeline >= start - 1e-9) & (timeline < end - 1e-9)
        times = np.minimum(timeline[mask] - start, (len(z) - 1) / fps)
        native[mask] = PchipInterpolator(np.arange(len(z)) / fps, z, axis=0)(times); cover[mask] += 1
        partmeta.append(dict(stem=c['stem'], start_frame=c['start_frame'], end_frame=c['end_frame'], native=str(path), native_frames=len(z),
                             source_word_spans_over_12s=c.get('source_word_spans_over_12s', []), word_interval_policy=c.get('word_interval_policy'),
                             streaming=payload.get('streaming')))
    assert np.all(cover == 1), (ci, np.unique(cover, return_counts=True))
    gt = np.load(GT_DIR / f'gt_{ci}.npy'); assert gt.shape == (T, 34)
    raw = native.copy(); y = np.clip(raw, lo, hi); alpha = np.minimum(np.arange(T) / 10, 1.); alpha = alpha * alpha * (3 - 2 * alpha)
    y = gt[0][None] * (1 - alpha[:, None]) + y * alpha[:, None]
    y = y.astype(np.float32)
    quality = dict(raw_limit_frame_fraction=float(((raw < lo - 1e-6) | (raw > hi + 1e-6)).any(1).mean()),
                   joint_limit_frame_fraction=float(((y < lo - 1e-6) | (y > hi + 1e-6)).any(1).mean()))
    assert np.isfinite(y).all() and np.array_equal(y[0], gt[0]); assert quality['joint_limit_frame_fraction'] == 0
    return y, gt, quality, partmeta


def audit(d, cis, rows, names, lo, hi):
    """audit_exports.py checks on the exported subset"""
    out = []
    for ci in cis:
        c = rows[ci]; y = np.load(d / f'gen_Hinf_{ci}.npy'); gt = np.load(d / f'gt_{ci}.npy'); ref = np.load(GT_DIR / f'gt_{ci}.npy')
        assert y.shape == gt.shape == ref.shape == (c['T'], 34); assert y.dtype == gt.dtype == np.float32; assert np.isfinite(y).all()
        assert np.array_equal(gt, ref); assert np.array_equal(y[0], gt[0]); assert np.all((y >= lo - 1e-6) & (y <= hi + 1e-6))
        m = json.loads((d / f'meta_{ci}.json').read_text()); assert m['ci'] == ci and m['joints'] == names
        out.append(dict(ci=ci, frames=len(y), sha256=hashlib.sha256((d / f'gen_Hinf_{ci}.npy').read_bytes()).hexdigest()))
    return dict(clips=len(out), frames=sum(r['frames'] for r in out), zero_limit_violations=True, exact_first_gt=True, rows=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--native', type=Path, required=True, help='dir with seed_000/ seed_001/ <stem>.pt (official prediction format)')
    ap.add_argument('--prefix', required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ci', type=int, nargs='*', help='clip subset (default: all 43)'); ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    a = ap.parse_args()
    rows = json.loads(ROWS.read_text()); chunks = json.loads(CHUNKS.read_text()); names, lo, hi = limits()
    byci = {r['ci']: r for r in rows}; cis = sorted(byci) if not a.ci else sorted(a.ci)
    report = {}
    for seed in a.seeds:
        sd = a.native / f'seed_{seed:03d}'
        if not sd.is_dir(): continue
        out = a.output / (a.prefix if seed == 0 else f'{a.prefix}_seed{seed}'); out.mkdir(parents=True, exist_ok=True)
        for ci in cis:
            row = byci[ci]; mp = out / f'meta_{ci}.json'
            if mp.exists(): continue
            t0 = time.perf_counter(); y, gt, quality, partmeta = export_clip(row, chunks, sd, lo, hi)
            np.save(out / f'gen_Hinf_{ci}.npy', y); np.save(out / f'gt_{ci}.npy', gt)
            meta = dict(ci=ci, clip=row['clip'], seed=seed, frames=row['T'], fps=50, units='rad', joints=names, source_wav=row['wav'], segments=partmeta,
                        GT_first_frame_exact=True, retarget_quality=quality, seconds=time.perf_counter() - t0,
                        exporter='scripts/echog_stream/ex002_export.py (= dlc-package-r2 export_dev.py --retrained, clip subset allowed)',
                        segmentation='data-dev chunks <=12s (the official EX002 offline cuts); each chunk is generated independently '
                                     '(streaming rows: a cold-start stream per chunk); no seam smoothing',
                        hands='EX002 model directly generates fingers', safety='position limits only')
            mp.write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False))
        report[out.name] = audit(out, cis, byci, names, lo, hi)
        print('EXPORTED', out, report[out.name]['clips'], 'clips', report[out.name]['frames'], 'frames', flush=True)
    (a.output / f'EXPORT_AUDIT_{a.prefix}.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()

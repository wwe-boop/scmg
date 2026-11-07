"""Still-frame share on EX002 FK (ours; the pause_overlap.py definition carried to EX002): per frame the speeds (central difference,
50 Hz) of the 6 upper-body points shoulder / elbow / wrist x left / right (scmg/ex002_ctl.points, exact FK), each divided by that
point's mean speed over the data-v36_1 dev GT (43 clips, fixed normaliser NORM_DEV.json); still = mean normalised speed < 0.15.
Also the mean elbow flare ((|elbow lateral| - |shoulder lateral|) / shoulder width, lateral = shoulder-line axis).
    python static_share.py norm --out NORM_DEV.json                      (from the dev GT of data-v36_1)
    python static_share.py score --norm NORM_DEV.json label=<dir of *.npy [T,34]> ...   (or label=@<file listing npy paths>)"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import ex002_ctl as E
PTS = ('shoulder', 'elbow', 'wrist')


def geo(q):
    g = E.points(np.asarray(q, np.float32)); return np.concatenate([g[k] for k in PTS], 1)      # [T, 6, 3]


def speed(P):
    v = np.zeros(P.shape[:2]); v[1:-1] = np.linalg.norm(P[2:] - P[:-2], axis=-1) * 25.; v[0], v[-1] = v[1], v[-2]; return v


def flare(q):
    g = E.points(np.asarray(q, np.float32)); sh, el = g['shoulder'], g['elbow']
    w = np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1); lat = (sh[:, 0] - sh[:, 1]) / w[:, None]; mid = sh.mean(1)
    L = lambda p: np.abs(np.einsum('thd,td->th', p - mid[:, None], lat))
    return ((L(el) - L(sh)) / w[:, None]).mean(1)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    n = sub.add_parser('norm'); n.add_argument('--out', required=True)
    n.add_argument('--view', default='/home/xr-25b7/bulk/cpfs-mirror/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1')
    s = sub.add_parser('score'); s.add_argument('--norm', required=True); s.add_argument('--out'); s.add_argument('--thr', type=float, default=.15)
    s.add_argument('specs', nargs='+')
    a = ap.parse_args()
    if a.cmd == 'norm':
        v = Path(a.view); items = json.loads((v / 'train.json').read_text()); val = set(json.loads((v / 'SPLIT.json').read_text())['val_ids'])
        Q = np.load(v / 'train_q.npy', mmap_mode='r'); clips = {}
        for it in items:
            if it['id'] in val: clips.setdefault(it['clip'], []).append(it)
        sp = []
        for c, xs in clips.items():
            xs.sort(key=lambda x: x['offset']); o0 = xs[0]['offset']; o1 = xs[-1]['offset'] + xs[-1]['frames']
            sp.append(speed(geo(Q[o0:o1])))
        sp = np.concatenate(sp); Path(a.out).write_text(json.dumps(dict(points=[f'{k}_{h}' for k in PTS for h in 'LR'], mean_speed=sp.mean(0).tolist(),
                                                                       clips=len(clips), frames=len(sp), view=str(v))))
        print('NORM', sp.mean(0).round(4).tolist(), len(clips)); return
    norm = np.asarray(json.loads(Path(a.norm).read_text())['mean_speed']); res = {}
    for spec in a.specs:
        lab, d = spec.split('=', 1)
        fs = [l.strip() for l in open(d[1:])] if d.startswith('@') else sorted(glob.glob(d + '/*.npy'))
        st, fl, fr, per = 0, [], 0, []
        for f in fs:
            q = np.load(f)
            if q.ndim != 2 or q.shape[1] != 34 or len(q) < 3: continue
            G = geo(q); sp = speed(G); vv = (sp / norm).mean(1); st += int((vv < a.thr).sum()); fr += len(q); fl.append(flare(q))
            # 11-08: per clip wrist speed (m/s, mean of both wrists) and amplitude (m, mean per-axis std of the wrist positions)
            per.append(dict(file=Path(f).name, still_share=float((vv < a.thr).mean()), wrist_speed=float(sp[:, 4:6].mean()),
                            wrist_amp=float(G[:, 4:6].std(0).mean())))
        res[lab] = dict(files=len(fs), frames=fr, still_share=st / max(fr, 1), elbow_flare_mean=float(np.concatenate(fl).mean()) if fl else None,
                        wrist_speed_mean=float(np.mean([x['wrist_speed'] for x in per])) if per else None,
                        wrist_amp_mean=float(np.mean([x['wrist_amp'] for x in per])) if per else None, per_clip=per)
        print(lab, json.dumps({k: v for k, v in res[lab].items() if k != 'per_clip'}), flush=True)
    if a.out: Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

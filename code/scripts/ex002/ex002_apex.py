"""Frozen kinematic apex detector for EX002 pointing (2610.03105 protocol: logistic regression, +-3 frames at 30 fps = +-0.1 s
positives, trained on recordings disjoint from the evaluation). Train = BEAT2 deictic events of the EX002 TRAIN split
(semantic-baseline-1105/ex002-sem/data EVENTS.json, split 'train'; their clips are view train items, never dev): window
[t - 1 s, t + 1 s] around the annotated segment midpoint; hand = the hand with the larger wrist-path diameter; a_ref = the frame
of that hand's largest wrist distance from its rest position (the scmg/ctl.py G1 apex rule, EX002 FK). Features as
scripts/semantic/g1_ctl_apex.py (rest distance, speed, acceleration, height, forward reach, shoulder-wrist extension, ray elevation,
offsets, window-relative values), EX002 rest = META rest34.
    SCMG_J=34 python scripts/ex002/ex002_apex.py --events <EVENTS.json> --view <data-v36_1> --out APEX.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import ex002_ctl as E

FPS = 50; POS = 5; OFFS = (-10, -5, 5, 10)
_REST = {}


def speed(x):
    v = np.zeros(len(x)); v[1:] = np.linalg.norm(np.diff(x, axis=0), axis=-1) * FPS; v[0] = v[1] if len(v) > 1 else 0; return v


def features(geo, h, f0, f1, rest_w):
    W, El, S = geo['wrist'][:, h], geo['elbow'][:, h], geo['shoulder'][:, h]
    dr = np.linalg.norm(W - rest_w[h], axis=-1); sp = speed(W); ac = np.abs(np.gradient(sp)) * FPS
    ray = W - El; el = np.arctan2(ray[:, 2], np.hypot(ray[:, 0], ray[:, 1])); ext = np.linalg.norm(W - S, axis=-1)
    T = len(W); idx = np.arange(f0, f1); sh = lambda x, o: x[np.clip(idx + o, 0, T - 1)]
    cols = [dr[idx], sp[idx], ac[idx], W[idx, 2], W[idx, 0], ext[idx], el[idx]]
    cols += [sh(sp, o) for o in OFFS] + [sh(dr, o) for o in OFFS]
    cols += [dr[idx] - dr[f0:f1].max(), sp[idx] / max(sp[f0:f1].max(), 1e-6)]
    return np.stack(cols, -1)


class Detector:
    def __init__(self, d): self.w, self.b, self.mu, self.sd = (np.asarray(d[k]) for k in ('coef', 'intercept', 'mean', 'std')); self.rest_w = np.asarray(d['rest_wrist'])

    def apex(self, geo, h, f0, f1):
        X = (features(geo, h, f0, f1, self.rest_w) - self.mu) / self.sd
        return f0 + int(np.argmax(X @ self.w + self.b))


def gt_apex(geo, f0, f1, rest_w):
    seg = geo['wrist'][f0:f1]
    diam = [float(np.linalg.norm(seg[:, h].max(0) - seg[:, h].min(0))) for h in (0, 1)]
    h = int(np.argmax(diam)); return h, f0 + int(np.argmax(np.linalg.norm(seg[:, h] - rest_w[h], axis=-1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--events', required=True); ap.add_argument('--view', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--max', type=int, default=1200); a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    V = Path(a.view); meta = json.loads((V / 'META.json').read_text()); rest = np.asarray(meta['ex002']['rest34'], np.float32)
    rest_w = E.points(rest[None])['wrist'][0]
    D = json.loads(Path(a.events).read_text()); items = {x['id']: x for x in json.loads((V / 'train.json').read_text())}
    split = json.loads((V / 'SPLIT.json').read_text()); train_ids = set(split['train_ids'])
    clips = {}
    for x in items.values(): clips.setdefault(x['clip'], []).append(x)
    q = np.load(V / 'train_q.npy', mmap_mode='r')
    ev = [e for e in D['events'] if e['split'] == 'train' and e['cls'] == 'deictic'][:a.max]
    X, y = [], []; n = 0
    for e in ev:
        its = sorted(clips.get(e['stem'], []), key=lambda x: x['offset'])
        if not its or any(x['id'] not in train_ids for x in its): continue
        Qc = np.concatenate([np.asarray(q[x['offset']:x['offset'] + x['frames']], np.float32) for x in its])
        c = int(round(e['t'] * FPS)); f0, f1 = max(0, c - FPS), min(len(Qc), c + FPS)
        if f1 - f0 < 20: continue
        lo, hi = max(0, f0 - 20), min(len(Qc), f1 + 20); geo = E.points(Qc[lo:hi])
        h, apx = gt_apex(geo, f0 - lo, f1 - lo, rest_w)
        F = features(geo, h, f0 - lo, f1 - lo, rest_w); lab = (np.abs(np.arange(f0 - lo, f1 - lo) - apx) <= POS).astype(int)
        X.append(F); y.append(lab); n += 1
    X = np.concatenate(X); y = np.concatenate(y); mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit((X - mu) / sd, y)
    out = dict(coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]), mean=mu.tolist(), std=sd.tolist(), rest_wrist=rest_w.tolist(),
               pos_frames=POS, offsets=list(OFFS), fps=FPS, n_events=n, n_frames=int(len(y)), n_pos=int(y.sum()),
               rule='EX002 train-split BEAT2 deictic events, apex = max wrist distance from rest (scmg ctl rule)')
    Path(a.out).write_text(json.dumps(out)); print(json.dumps({k: v for k, v in out.items() if k not in ('coef', 'mean', 'std', 'rest_wrist')}))


if __name__ == '__main__':
    main()

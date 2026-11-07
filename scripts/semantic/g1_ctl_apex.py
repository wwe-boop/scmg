"""Frozen kinematic apex detector for the pointing score (11-05; arXiv 2610.03105 protocol: logistic regression, +-3 frames at
30 fps = +-0.1 s positives, trained on data disjoint from the evaluation). Train = the TRAIN-split POINT tokens of
g1_ctl_tokens.py on the ground-truth G1 motion (50 Hz, official FK); a_ref = the token apex (scmg/ctl.py rule).
Per frame f of the search window W = [a - 1 s, b + 1 s] (clip-clipped), hand = the token hand, upper-body kinematic features:
  wrist distance from its rest position, wrist speed, wrist acceleration, wrist height (z), forward reach (x), shoulder-wrist
  extension, elbow -> wrist ray elevation; speed and rest distance at offsets -10 / -5 / +5 / +10 frames; rest distance and
  speed relative to their window maxima. a_hat = argmax over W of the detector probability.
Also stores the 'action' speed threshold for the +-0.3 s timing metric: the 25th percentile of the peak wrist speed (max over
hands) inside the annotated span of TRAIN annotation tokens (non-lexical).
    python scripts/semantic/g1_ctl_apex.py --ctl <dir> --view <hos view> --out <dir>/APEX.json
Prints the detector's Score_a on the dev GT (diagnostic of the detector itself)."""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import ctl

FPS = 50; POS = 5; OFFS = (-10, -5, 5, 10)


def speed(x):
    v = np.zeros(len(x)); v[1:] = np.linalg.norm(np.diff(x, axis=0), axis=-1) * FPS; v[0] = v[1] if len(v) > 1 else 0; return v


def features(geo, h, f0, f1):
    """[f1 - f0, n_feat] for frames f0..f1-1 (indices into geo arrays)."""
    W, E, S = geo['wrist'][:, h], geo['elbow'][:, h], geo['shoulder'][:, h]
    rest = ctl.rest_wrist()[h]
    dr = np.linalg.norm(W - rest, axis=-1); sp = speed(W); ac = np.abs(np.gradient(sp)) * FPS
    ray = W - E; el = np.arctan2(ray[:, 2], np.hypot(ray[:, 0], ray[:, 1]))
    ext = np.linalg.norm(W - S, axis=-1)
    T = len(W); idx = np.arange(f0, f1)
    sh = lambda x, o: x[np.clip(idx + o, 0, T - 1)]
    cols = [dr[idx], sp[idx], ac[idx], W[idx, 2], W[idx, 0], ext[idx], el[idx]]
    cols += [sh(sp, o) for o in OFFS] + [sh(dr, o) for o in OFFS]
    cols += [dr[idx] - dr[f0:f1].max(), sp[idx] / max(sp[f0:f1].max(), 1e-6)]
    return np.stack(cols, -1)


def window(t, T):
    return max(0, int(round((t['a'] - 1.0) * FPS))), min(T, int(round((t['b'] + 1.0) * FPS)))


class Detector:
    def __init__(self, d): self.w, self.b, self.mu, self.sd = (np.asarray(d[k]) for k in ('coef', 'intercept', 'mean', 'std'))

    def apex(self, geo, h, f0, f1):
        X = (features(geo, h, f0, f1) - self.mu) / self.sd
        return f0 + int(np.argmax(X @ self.w + self.b))


def gt_geo(view, stems):
    q = np.load(Path(view) / 'train_q.npy', mmap_mode='r')
    items = {it['id']: it for it in json.loads((Path(view) / 'train.json').read_text())}
    return {s: ctl.geometry_np(q[items[s]['offset']:items[s]['offset'] + items[s]['frames'], 10:39]) for s in stems}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ctl', required=True); ap.add_argument('--view', required=True)
    ap.add_argument('--out', required=True); a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    import torch; torch.set_num_threads(4)
    raw = json.loads((Path(a.ctl) / 'CTL.json').read_text())
    use = {s: ts for s, ts in raw.items() if any(ctl.group_of(t['type']) == 0 for t in ts)}
    geo = gt_geo(a.view, list(use))
    X, Y, peaks = [], [], []
    for s, ts in use.items():
        if ts[0]['split'] != 'train': continue
        g = geo[s]; T = len(g['wrist']); spm = np.maximum(speed(g['wrist'][:, 0]), speed(g['wrist'][:, 1]))
        for t in ts:
            if ctl.group_of(t['type']) != 0: continue
            fa, fb = int(round(t['a'] * FPS)), int(round(t['b'] * FPS))
            if fb > fa: peaks.append(float(spm[fa:fb].max()))
            if t['type'] != ctl.POINT: continue
            f0, f1 = window(t, T); fx = features(g, t['hand'], f0, f1); ap_ = int(round(t['apex'] * FPS))
            X.append(fx); Y.append(np.abs(np.arange(f0, f1) - ap_) <= POS)
    X, Y = np.concatenate(X), np.concatenate(Y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    lr = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0).fit((X - mu) / sd, Y)
    det = dict(coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]), mean=mu.tolist(), std=sd.tolist(), pos_frames=POS,
               offsets=OFFS, fps=FPS, n_train_frames=int(len(Y)), n_train_pos=int(Y.sum()),
               action_speed=float(np.percentile(peaks, 25)), action_speed_rule='25th pct of train peak wrist speed in annotated spans')
    D = Detector(det); sa, err = [], []
    for s, ts in use.items():
        if ts[0]['split'] != 'dev': continue
        g = geo[s]; T = len(g['wrist'])
        for t in ts:
            if t['type'] != ctl.POINT: continue
            f0, f1 = window(t, T); ah = D.apex(g, t['hand'], f0, f1) / FPS
            sa.append(ctl.score_a(ah, t['apex'])); err.append(abs(ah - t['apex']))
    det['dev_gt_score_a'] = float(np.mean(sa)); det['dev_gt_abs_err_s_median'] = float(np.median(err)); det['dev_n'] = len(sa)
    Path(a.out).write_text(json.dumps(det, indent=1)); print(json.dumps({k: v for k, v in det.items() if k not in ('coef', 'mean', 'std')}))


if __name__ == '__main__':
    main()

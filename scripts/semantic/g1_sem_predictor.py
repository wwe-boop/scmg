"""Text-only semantic-intent predictor (11-06, BJ50 intent head analogue, deployable: inference sees text only).
Per e5 token of the PREFIX-CAUSAL cache (what a streaming system has when the word arrives): features = token feature +
mean of the previous 3 tokens; label = BEAT2 annotated semantic class (deictic / iconic / metaphoric) whose token time
(g1_ctl_tokens.py: keyword-anchored) is nearest to this token within 0.25 s, else none. Trained on TRAIN-split items only
(MLP, class-balanced CE, CPU). Dev is used only to pick the insertion threshold (selection), never for fitting.
Output PRED.json {dev stem: [token]} in the CTL.json token format (type = class, t = token time, a = t, b = t + 1, split dev,
p = probability), non-maximum-suppressed (>= MIN_GAP s apart); consumed by infer_test.py --ctl-mode pred / predlex.
    python scripts/semantic/g1_sem_predictor.py --ctl <CTL dir> --text <causal e5 cache> --view <hos view> --out <dir>
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

CLS = ['none', 'deictic', 'iconic', 'metaphoric']; TID = {1: 1, 2: 2, 3: 3}; NEAR = 0.25; MIN_GAP = 1.0


def feats(X, k0, kn):
    x = np.asarray(X[k0:k0 + kn], np.float32)
    ctx = np.zeros_like(x)
    for i in range(kn):
        j = max(0, i - 3)
        if i > j: ctx[i] = x[j:i].mean(0)
    return np.concatenate([x, ctx], 1)


def build(raw, X, T, ix, split):
    xs, ys, meta = [], [], []
    for s, ts in raw.items():
        if not ts or ts[0]['split'] != split or s not in ix: continue
        ann = [t for t in ts if t['type'] in TID]
        if split == 'train' and not any(t['type'] <= 4 for t in ts): continue          # items with BEAT2 annotation only
        k0, kn = ix[s]; tt = np.asarray(T[k0:k0 + kn], np.float64); y = np.zeros(kn, np.int64)
        for t in ann:
            ok = np.isfinite(tt)
            if not ok.any(): continue
            d = np.where(ok, np.abs(tt - t['t']), 1e9); i = int(d.argmin())
            if d[i] <= NEAR: y[i] = TID[t['type']]
        xs.append(feats(X, k0, kn)); ys.append(y); meta.append((s, tt))
    return xs, ys, meta


def main():
    ap = argparse.ArgumentParser()
    for k in ('ctl', 'text', 'view', 'out'): ap.add_argument('--' + k, required=True)
    ap.add_argument('--epochs', type=int, default=6); a = ap.parse_args(); torch.manual_seed(0); torch.set_num_threads(4)
    raw = json.loads((Path(a.ctl) / 'CTL.json').read_text())
    X = np.load(Path(a.text) / 'TEXT_FEAT.npy', mmap_mode='r'); T = np.load(Path(a.text) / 'TEXT_TIME.npy')
    ix = json.loads((Path(a.text) / 'TEXT_INDEX.json').read_text())
    xtr, ytr, _ = build(raw, X, T, ix, 'train'); xdv, ydv, mdv = build(raw, X, T, ix, 'dev')
    Xtr, Ytr = torch.tensor(np.concatenate(xtr)), torch.tensor(np.concatenate(ytr))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    cnt = torch.bincount(Ytr, minlength=4).float(); w = (cnt.sum() / (4 * cnt.clamp(min=1))).clamp(max=50)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 4))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xtr)); tot = 0.
        for i in range(0, len(perm), 1024):
            b = perm[i:i + 1024]; loss = F.cross_entropy(net((Xtr[b] - mu) / sd), Ytr[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
        print('epoch', ep, round(tot / len(perm), 4), flush=True)
    net.eval()
    with torch.no_grad():
        P = [torch.softmax(net((torch.tensor(x) - mu) / sd), -1).numpy() for x in xdv]
    n_ann = int(sum((y > 0).sum() for y in ydv))
    # threshold on dev (selection): max-class probability of non-none; choose tau so that the NMS'd count ~= annotation count
    def predict(tau):
        out = {}
        for (s, tt), p in zip(mdv, P):
            pn = 1 - p[:, 0]; cand = [i for i in np.argsort(-pn) if pn[i] >= tau and np.isfinite(tt[i])]; keep = []
            for i in cand:
                if all(abs(tt[i] - tt[j]) >= MIN_GAP for j in keep): keep.append(i)
            out[s] = [dict(type=int(np.argmax(p[i, 1:]) + 1), cls=CLS[int(np.argmax(p[i, 1:]) + 1)], t=round(float(tt[i]), 3),
                           a=round(float(tt[i]), 3), b=round(float(tt[i]) + 1.0, 3), split='dev', p=round(float(pn[i]), 4))
                      for i in sorted(keep)]
        return out
    def score(out):
        tp = 0; tpc = 0; n = sum(len(v) for v in out.values())
        for (s, tt), y in zip(mdv, ydv):
            gt = [(tt[i], y[i]) for i in np.nonzero(y)[0]]
            for t in out.get(s, []):
                m = [c for (g, c) in gt if abs(g - t['t']) <= 0.5]
                tp += bool(m); tpc += t['type'] in m
        return dict(n_pred=n, precision=tp / max(n, 1), recall_any=tp / max(n_ann, 1), class_precision=tpc / max(n, 1))
    rep = {}
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        rep[tau] = score(predict(tau)); print(tau, rep[tau], flush=True)
    tau = min(rep, key=lambda t: abs(rep[t]['n_pred'] - n_ann))
    out = predict(tau); o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / 'PRED.json').write_text(json.dumps(out))
    # base rate: lexicon rule (every LEX2SEM word) scored the same way
    (o / 'PRED_INFO.json').write_text(json.dumps(dict(tau=tau, n_dev_annotations=n_ann, sweep={str(k): v for k, v in rep.items()},
                                                      train_tokens=int(len(Ytr)), train_pos=cnt.tolist(), near_s=NEAR, min_gap_s=MIN_GAP), indent=1))
    torch.save(dict(net=net.state_dict(), mu=mu, sd=sd), o / 'predictor.pt')
    print('TAU', tau, rep[tau])


if __name__ == '__main__':
    main()

"""Anticipatory semantic gesture planning: premise analysis (11-06; train split for fitting / statistics, dev only for scores).
 1. Lag of the BEAT2 semantic gesture relative to its lexical affiliate (the annotated keyword's aligned onset t_kw):
    stroke start a - t_kw, apex - t_kw, end b - t_kw, per class (deictic / iconic / metaphoric), keyword-anchored tokens only.
 2. Predictability from text of an UPCOMING semantic gesture: at every e5 token i (time t_i), label = class of an annotated
    segment whose start a lies in [t_i, t_i + 1 s) (else none). Linear probe (torch logistic regression, class-balanced, train
    items only), dev AUC of 'any gesture starts within 1 s' and macro one-vs-rest class AUC, for feature sets:
      causal      prefix-causal e5 of token i + mean of the previous 3 tokens            (P-sync: words only once voiced)
      anticip     causal + mean of the tokens arriving in (t_i, t_i + 1 s] (bidirectional e5 of those words, TTS look-ahead)
      shuffled    anticip with labels permuted within item (chance floor)
    python scripts/semantic/g1_anticip_analysis.py --ctl <CTL dir> --text <e5 cache> --text-causal <causal e5 cache> --out res.json
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

LOOK = 1.0


def auc(pos, neg):
    s = np.concatenate([pos, neg]); o = np.argsort(s, kind='mergesort'); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / max(1, len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    for k in ('ctl', 'text', 'text_causal', 'out'): ap.add_argument('--' + k.replace('_', '-'), required=True)
    a = ap.parse_args(); torch.manual_seed(0); torch.set_num_threads(4); rng = np.random.default_rng(0)
    raw = json.loads((Path(a.ctl) / 'CTL.json').read_text())
    lag = {c: dict(a=[], apex=[], b=[]) for c in (1, 2, 3)}
    for s, ts in raw.items():
        if not ts or ts[0]['split'] != 'train': continue
        for t in ts:
            if t['type'] in lag and t.get('t_src') == 'keyword':
                lag[t['type']]['a'].append(t['a'] - t['t']); lag[t['type']]['apex'].append(t['apex'] - t['t']); lag[t['type']]['b'].append(t['b'] - t['t'])
    names = {1: 'deictic', 2: 'iconic', 3: 'metaphoric'}
    res = dict(lag={names[c]: {k: dict(n=len(v), q10=float(np.quantile(v, .1)), q25=float(np.quantile(v, .25)), median=float(np.median(v)),
                                        q75=float(np.quantile(v, .75)), q90=float(np.quantile(v, .9)), frac_before_word=float(np.mean(np.asarray(v) < 0)))
                               for k, v in d.items()} for c, d in lag.items()})
    Xb = np.load(Path(a.text) / 'TEXT_FEAT.npy', mmap_mode='r'); Xc = np.load(Path(a.text_causal) / 'TEXT_FEAT.npy', mmap_mode='r')
    T = np.load(Path(a.text_causal) / 'TEXT_TIME.npy'); ix = json.loads((Path(a.text_causal) / 'TEXT_INDEX.json').read_text())
    ixb = json.loads((Path(a.text) / 'TEXT_INDEX.json').read_text())
    data = {'train': [], 'dev': []}
    for s, ts in raw.items():
        if not ts or s not in ix or not any(t['type'] <= 4 for t in ts): continue           # items with BEAT2 annotation
        sp = ts[0]['split']; k0, kn = ix[s]; assert ixb[s] == ix[s]
        tt = np.asarray(T[k0:k0 + kn], np.float64); ok = np.isfinite(tt)
        xc = np.asarray(Xc[k0:k0 + kn], np.float32); xb = np.asarray(Xb[k0:k0 + kn], np.float32)
        y = np.zeros(kn, np.int64)
        for t in ts:
            if t['type'] in (1, 2, 3):
                y[(tt <= t['a']) & (t['a'] < tt + LOOK)] = t['type']
        past = np.stack([xc[max(0, i - 3):i].mean(0) if i > 0 else np.zeros(768, np.float32) for i in range(kn)])
        fut = np.stack([xb[(tt > tt[i]) & (tt <= tt[i] + LOOK)].mean(0) if ((tt > tt[i]) & (tt <= tt[i] + LOOK)).any()
                        else np.zeros(768, np.float32) for i in range(kn)])
        data[sp].append(dict(causal=np.concatenate([xc, past], 1)[ok], anticip=np.concatenate([xc, past, fut], 1)[ok], y=y[ok]))
    for feat in ('causal', 'anticip', 'shuffled'):
        f = 'anticip' if feat == 'shuffled' else feat
        Xtr = torch.tensor(np.concatenate([d[f] for d in data['train']])); Ytr = np.concatenate([d['y'] for d in data['train']])
        if feat == 'shuffled': Ytr = np.concatenate([rng.permutation(d['y']) for d in data['train']])
        Ytr = torch.tensor(Ytr); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
        W = torch.zeros(Xtr.shape[1], 4, requires_grad=True); bb = torch.zeros(4, requires_grad=True)
        cnt = torch.bincount(Ytr, minlength=4).float(); w = cnt.sum() / (4 * cnt.clamp(min=1))
        opt = torch.optim.AdamW([W, bb], 1e-3, weight_decay=1e-2)
        for ep in range(4):
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(perm), 2048):
                b = perm[i:i + 2048]; loss = F.cross_entropy(((Xtr[b] - mu) / sd) @ W + bb, Ytr[b], weight=w)
                opt.zero_grad(); loss.backward(); opt.step()
        Xdv = torch.tensor(np.concatenate([d[f] for d in data['dev']])); Ydv = np.concatenate([d['y'] for d in data['dev']])
        with torch.no_grad(): P = torch.softmax(((Xdv - mu) / sd) @ W + bb, -1).numpy()
        r = dict(any_auc=auc(1 - P[Ydv > 0, 0], 1 - P[Ydv == 0, 0]),
                 class_auc={names[c]: auc(P[Ydv == c, c], P[Ydv != c, c]) for c in (1, 2, 3)},
                 dev_tokens=int(len(Ydv)), dev_pos=int((Ydv > 0).sum()), train_pos=int((Ytr > 0).sum()))
        r['class_auc_macro'] = float(np.mean(list(r['class_auc'].values())))
        res[feat] = r; print(feat, json.dumps(r), flush=True)
    print(json.dumps(res['lag'], indent=1))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

"""Aggregate g1_recognizer.py score files: mean / sd over recogniser seeds per source, and clip-bootstrap paired differences.
    python scripts/semantic/sem_report.py --scores <dir> --rec-prefix rec-sem-v1 --pairs echog-s0:ours-offline-s0,... --out <json>
Score files are <dir>/<rec>--<source>--<split>.json (+ .probs.npz from --save-probs).  Paired differences use the events
both sources share (same stem@time key), resampling clips (stems) with replacement, 2000 draws, per recogniser seed.
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

CLUSTER = lambda stem: stem       # bootstrap cluster of an event: its clip (default) or, with --cluster recording, its recording
KEYS = ('det_auc', 'pair_win', 'bal_top1', 'srr', 'time_bal_top1', 'shuf_spk_bal_top1', 'top1')


def auc(pos, neg):
    s = np.concatenate([pos, neg]); o = np.argsort(s, kind='mergesort'); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True); r = (np.bincount(inv, weights=r) / cnt)[inv]
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def stat(z, idx_lex, idx_bg, which):
    P, Pt, y = z['P'], z['Pt'], z['y']
    lexpos = np.where(z['lex'])[0]                                   # row of P for each lexicon event; Pt rows follow lexpos order
    if which == 'det_auc':
        return auc(1 - P[idx_lex, 0], 1 - P[idx_bg, 0])
    if which == 'pair_win':
        j = np.searchsorted(lexpos, idx_lex); a_, b_ = P[idx_lex, y[idx_lex]], Pt[j, y[idx_lex]]
        ok = np.isfinite(b_); return float(np.mean((a_[ok] > b_[ok]) + 0.5 * (a_[ok] == b_[ok])))
    if which == 'bal_top1':
        pm = P[idx_lex].copy(); pm[:, 0] = -1; t1 = pm.argmax(1) == y[idx_lex]
        return float(np.mean([t1[y[idx_lex] == c].mean() for c in np.unique(y[idx_lex])]))


def paired(za, zb, rng, n=2000):
    ka, kb = [str(k) for k in za['key']], [str(k) for k in zb['key']]
    ib = {k: i for i, k in enumerate(kb)}; common = [(i, ib[k]) for i, k in enumerate(ka) if k in ib]
    ia = np.array([c[0] for c in common]); jb = np.array([c[1] for c in common])
    stems = np.array([CLUSTER(ka[i].split('@')[0]) for i in ia]); us, inv = np.unique(stems, return_inverse=True)
    rows = defaultdict(list)
    for k, g in enumerate(inv): rows[g].append(k)
    rows = [np.array(rows[g]) for g in range(len(us))]
    out = {}
    for w in ('det_auc', 'pair_win', 'bal_top1'):
        def f(sel):
            sa, sb = ia[sel], jb[sel]
            la, lb = za['lex'][sa], zb['lex'][sb]
            return (stat(za, sa[la], sa[~la], w) - stat(zb, sb[lb], sb[~lb], w))
        full = f(np.arange(len(ia)))
        bs = []
        for _ in range(n):
            sel = np.concatenate([rows[g] for g in rng.integers(0, len(rows), len(rows))]); bs.append(f(sel))
        out[w] = dict(diff=float(full), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
    out['n_common_events'] = int(len(ia)); out['n_clips'] = int(len(us))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--scores', required=True); ap.add_argument('--rec-prefix', required=True)
    ap.add_argument('--split', default='test'); ap.add_argument('--pairs', default=''); ap.add_argument('--out', required=True)
    ap.add_argument('--cluster', default='clip', choices=['clip', 'recording'],
                    help='paired bootstrap unit (11-06): clip (default, as before) or recording (utterances of one take resampled together)')
    a = ap.parse_args()
    global CLUSTER
    if a.cluster == 'recording': CLUSTER = lambda stem: re.sub(r'_utt_\d+$', '', stem)
    d = Path(a.scores).expanduser(); rng = np.random.default_rng(0)
    files = sorted(d.glob(f'{a.rec_prefix}*--*--{a.split}.json'))
    by = defaultdict(dict)
    for f in files:
        rec, src, _ = f.name[:-5].split('--'); by[src][rec] = json.loads(f.read_text())
    tab = {}
    for src, rs in sorted(by.items()):
        tab[src] = {k: dict(mean=float(np.mean([r[k] for r in rs.values()])), sd=float(np.std([r[k] for r in rs.values()])),
                            n_rec=len(rs)) for k in KEYS if all(k in r for r in rs.values())}
        tab[src]['per_class'] = {c: float(np.mean([r['per_class'][c] for r in rs.values()])) for c in next(iter(rs.values()))['per_class']}
        tab[src]['n_events'] = next(iter(rs.values()))['n']
    res = dict(scores=str(d), rec_prefix=a.rec_prefix, cluster=a.cluster, table=tab, pairs={})
    for pr in [p for p in a.pairs.split(',') if p]:
        x, y = pr.split(':'); per = {}
        for rec in sorted(set(by.get(x, {})) & set(by.get(y, {}))):
            fa, fb = d / f'{rec}--{x}--{a.split}.probs.npz', d / f'{rec}--{y}--{a.split}.probs.npz'
            if fa.exists() and fb.exists(): per[rec] = paired(np.load(fa), np.load(fb), rng)
        res['pairs'][pr] = per
    Path(a.out).expanduser().write_text(json.dumps(res, indent=1))
    print(f"{'source':22s} " + ' '.join(f'{k:>16s}' for k in ('det_auc', 'pair_win', 'bal_top1', 'srr')))
    for src, t in tab.items():
        print(f'{src:22s} ' + ' '.join(f"{t[k]['mean']:.3f}+-{t[k]['sd']:.3f}".rjust(16) for k in ('det_auc', 'pair_win', 'bal_top1', 'srr')))
    for pr, per in res['pairs'].items():
        for rec, v in per.items():
            print(pr, rec, {w: (round(v[w]['diff'], 3), [round(c, 3) for c in v[w]['ci95']]) for w in ('det_auc', 'pair_win', 'bal_top1')})


if __name__ == '__main__':
    main()

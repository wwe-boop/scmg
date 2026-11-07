"""Multi-seed pair_win (11-06 per_clip protocol): every source has several inference-seed score files
<scores>/<rec>--<source>-s<k>--dev.probs.npz; pair_win of a source = mean over seeds and recognisers; paired difference A - B
(e.g. full - no-text) with a recording-clustered bootstrap that resamples recordings and keeps all seeds of a recording together.
Reported overall and per class (deictic / iconic / metaphoric), per recogniser and averaged.
    python scripts/semantic/sem_multiseed.py --scores <dir> --rec-prefix rec-sem-hos-seed --seeds 0 1 2 --out res.json \
        --sources lane-full lane-notext ... --pairs lane-full:lane-notext,...
"""
import argparse, json, re
from pathlib import Path
import numpy as np

CLS = {1: 'deictic', 2: 'iconic', 3: 'metaphoric'}


SPLIT = 'dev'


def load(d, rec, src, seed):
    z = np.load(Path(d) / f'{rec}--{src}-s{seed}--{SPLIT}.probs.npz')
    P, Pt, y, lex, key = z['P'], z['Pt'], z['y'], z['lex'], z['key']
    li = np.where(lex)[0]; a = P[li, y[li]]; b = Pt[np.arange(len(li)), y[li]]
    win = np.where(np.isfinite(b), (a > b) + 0.5 * (a == b), np.nan)
    stem = np.array([str(k).split('@')[0] for k in key[li]])
    return dict(win=win, y=y[li], rec=np.array([re.sub(r'_utt_\d+$', '', s) for s in stem]), key=np.array([str(k) for k in key[li]]))


def pw(D, sel=None):
    w = np.concatenate([d['win'] if sel is None else d['win'][sel(d)] for d in D]); return float(np.nanmean(w))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--scores', required=True); ap.add_argument('--rec-prefix', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2]); ap.add_argument('--sources', nargs='+', required=True)
    ap.add_argument('--pairs', default=''); ap.add_argument('--split', default='dev'); ap.add_argument('--out', required=True); ap.add_argument('--n-boot', type=int, default=2000)
    a = ap.parse_args(); global SPLIT; SPLIT = a.split; recs = sorted({p.name.split('--')[0] for p in Path(a.scores).glob(f'{a.rec_prefix}*--*.probs.npz')})
    data = {r: {s: [load(a.scores, r, s, k) for k in a.seeds] for s in a.sources} for r in recs}
    res = dict(seeds=a.seeds, recognisers=recs, table={}, pairs={})
    for s in a.sources:
        row = {'all': float(np.mean([pw(data[r][s]) for r in recs]))}
        for c, n in CLS.items(): row[n] = float(np.mean([pw(data[r][s], lambda d, c=c: d['y'] == c) for r in recs]))
        res['table'][s] = row
    rng = np.random.default_rng(0)
    for pr in [p for p in a.pairs.split(',') if p]:
        x, y = pr.split(':'); out = {}
        for part, sel in [('all', None)] + [(n, (lambda d, c=c: d['y'] == c)) for c, n in CLS.items()]:
            diffs, cis = [], []
            for r in recs:
                A, Bd = data[r][x], data[r][y]
                allrec = np.unique(np.concatenate([d['rec'] for d in A + Bd]))
                def sums(D):                                   # per recording: (sum of wins, count) pooled over seeds
                    S = np.zeros(len(allrec)); N = np.zeros(len(allrec))
                    for d in D:
                        m = np.isfinite(d['win']) & (np.ones(len(d['win']), bool) if sel is None else sel(d))
                        ix = np.searchsorted(allrec, d['rec'][m]); np.add.at(S, ix, d['win'][m]); np.add.at(N, ix, 1)
                    return S, N
                Sa, Na = sums(A); Sb, Nb = sums(Bd)
                full = Sa.sum() / Na.sum() - Sb.sum() / Nb.sum()
                W = rng.multinomial(len(allrec), np.full(len(allrec), 1 / len(allrec)), size=a.n_boot).astype(float)
                bs = (W @ Sa) / np.maximum(W @ Na, 1) - (W @ Sb) / np.maximum(W @ Nb, 1)
                diffs.append(float(full)); cis.append([float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
            out[part] = dict(diff=float(np.mean(diffs)), per_rec=diffs, ci95_per_rec=cis, all_ci_above_0=all(c[0] > 0 for c in cis))
        res['pairs'][pr] = out
        print(pr, {k: (round(v['diff'], 4), [[round(c, 3) for c in cc] for cc in v['ci95_per_rec']]) for k, v in out.items()}, flush=True)
    for s, row in res['table'].items(): print(f'{s:24s}', ' '.join(f'{k} {v:.3f}' for k, v in row.items()))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

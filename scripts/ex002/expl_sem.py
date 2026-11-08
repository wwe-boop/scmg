"""Explicit semantics -> appropriate gesture (11-09; exp/text-path docs/criteria/EXPLICIT_SEM_1109.md, registered before any data).
  build : exemplars of every category-word occurrence (train + dev recordings of data-v36_1), W = onset +- 0.6 s, dominant hand,
          standardised D; class-balanced logistic regression (C by 5-fold recording-grouped CV on train); dev-GT sanity with the
          registered drop rule (one refit); per-category Gaussians (shrinkage 0.1) + P(right hand) for the deployable sampler.
  plan  : deployable tokens for the given recordings (onset - 0.2 s, greedy 1.2 s, one sampled descriptor per token) + the scored
          events (all kept-category word occurrences).
  eval  : appropriateness of generated runs at the scored events; given - none with the recording bootstrap CI.
    SCMG_J=34 python scripts/ex002/expl_sem.py build --sem SEM_DIR --out OUT
    SCMG_J=34 python scripts/ex002/expl_sem.py plan --expl OUT --cis 0,4,...,36 --out PLANDIR
    SCMG_J=34 python scripts/ex002/expl_sem.py eval --expl OUT --plans PLANDIR --gt GTDIR --out O.json none=D given=D ..."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'semantic'))
from scmg import sem_desc as SD
from sem_bank import V, EV, recordings, rec_words

FPS = 50
LEX = {'SELF': ('self', "i me my mine myself i'm i've i'll i'd".split() + '我 自己'.split()),
       'YOU': ('you', "you your yours yourself yourselves you're you've you'll you'd".split() + '你们 你 您'.split()),
       'POINT': (('this', 'there'), 'this here these there those'.split() + '这 那'.split()),
       'NUMBER': ('number', 'two three four five six seven eight nine ten twenty thirty first second third both'.split() + '两 三 四 五 六 七 八 九 十 第'.split()),
       'BIG': ('big', 'big huge large giant massive enormous wide whole entire'.split() + '巨大 大 整个'.split()),
       'SMALL': ('small', 'little small tiny bit few narrow slightly'.split() + '小 一点'.split())}
CATS = list(LEX)


def category(w, c):
    for k, (src, words) in LEX.items():
        srcs = src if isinstance(src, tuple) else (src,)
        if c in srcs and w in words: return k
    return None


def occurrences(recs, q, mu, sd):
    rows, W, H = [], [], []
    for stem, its in recs.items():
        Q = None
        for w, c, on, off, iid, f0 in rec_words(its):
            k = category(w, c)
            if k is None: continue
            if Q is None: Q = np.concatenate([np.asarray(q[x['offset']:x['offset'] + x['frames']], np.float32) for x in its])
            cf = int(round(on * FPS))
            if cf - SD.NW // 2 < 0 or cf + SD.NW // 2 >= len(Q): continue
            win = SD.window(Q, on); W.append(win); H.append(SD.dominant_hand(win)); rows.append(dict(stem=stem, word=w, cat=k, onset=float(on)))
    if not rows: return rows, np.zeros((0, SD.ND))
    raw = SD.np_desc(np.stack(W), np.array(H)); D = np.concatenate([(raw[:, :SD.NS] - mu) / sd, raw[:, SD.NS:]], 1)
    for r, h in zip(rows, H): r['hand'] = int(h)
    return rows, D


def fit_clf(D, y, grp, cats):
    from sklearn.linear_model import LogisticRegression
    ug = np.unique(grp); rng = np.random.default_rng(0); fold = dict(zip(ug, rng.permutation(len(ug)) % 5)); fid = np.array([fold[g] for g in grp])
    cv = {}
    for C in (0.01, 0.1, 1, 10):
        acc = []
        for k in range(5):
            m = LogisticRegression(C=C, class_weight='balanced', max_iter=3000).fit(D[fid != k], y[fid != k])
            p = m.predict(D[fid == k]); yy = y[fid == k]
            acc.append(np.mean([np.mean(p[yy == c] == c) for c in np.unique(yy)]))
        cv[C] = float(np.mean(acc))
    C = max(cv, key=cv.get)
    return LogisticRegression(C=C, class_weight='balanced', max_iter=3000).fit(D, y), cv, C


def rates(pred, y, K):
    return {c: float(np.mean(pred[y == c] == c)) for c in range(K) if (y == c).any()}


def build(a):
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    st = json.loads((Path(a.sem) / 'SEM_BANK.json').read_text()); mu, sd = np.asarray(st['mu']), np.asarray(st['sd'])
    items = json.loads((Path(V) / 'train.json').read_text()); sp = json.loads((Path(V) / 'SPLIT.json').read_text())
    q = np.load(Path(V) / 'train_q.npy', mmap_mode='r')
    tr_rows, Dtr = occurrences(recordings(items, set(sp['train_ids'])), q, mu, sd)
    dv_rows, Ddv = occurrences(recordings(items, set(sp['val_ids'])), q, mu, sd)
    np.savez(O / 'EXPL_EXEMPLARS.npz', Dtr=Dtr, Ddv=Ddv)
    (O / 'EXPL_ROWS.json').write_text(json.dumps(dict(train=tr_rows, dev=dv_rows)))
    rng = np.random.default_rng(20261109); res = dict(counts=dict(train={c: sum(r['cat'] == c for r in tr_rows) for c in CATS},
                                                                  dev={c: sum(r['cat'] == c for r in dv_rows) for c in CATS}), rounds=[])
    kept = [c for c in CATS if res['counts']['train'][c] > 0 and res['counts']['dev'][c] > 0]
    for rnd in (1, 2):
        ix = {c: i for i, c in enumerate(kept)}; K = len(kept)
        mtr = np.array([r['cat'] in ix for r in tr_rows]); mdv = np.array([r['cat'] in ix for r in dv_rows])
        ytr = np.array([ix[r['cat']] for r in tr_rows if r['cat'] in ix]); ydv = np.array([ix[r['cat']] for r in dv_rows if r['cat'] in ix])
        grp = np.array([r['stem'] for r in tr_rows if r['cat'] in ix])
        clf, cv, C = fit_clf(Dtr[mtr], ytr, grp, kept); pdv = clf.predict(Ddv[mdv])
        own = {kept[c]: v for c, v in rates(pdv, ydv, K).items()}
        sh = [np.mean(list(rates(pdv, rng.permutation(ydv), K).values())) for _ in range(20)]
        chance = 1.0 / K; fail = [c for c in kept if own.get(c, 0) < chance + 0.20]
        res['rounds'].append(dict(round=rnd, kept=list(kept), C=C, cv_bal_acc=cv, chance=chance, own=own, balanced=float(np.mean(list(own.values()))),
                                  shuffled=float(np.mean(sh)), shuffled_ok=bool(abs(np.mean(sh) - chance) <= 0.05), failed=fail))
        if rnd == 1 and fail: kept = [c for c in kept if c not in fail]; continue
        break
    last = res['rounds'][-1]
    valid = len(last['kept']) >= 2 and not last['failed'] and last['shuffled_ok']
    if len(res['rounds']) == 2 and last['failed']: valid = False
    res['METRIC_VALID'] = bool(valid); res['kept'] = last['kept'] if valid else []
    if valid:
        ix = {c: i for i, c in enumerate(res['kept'])}; mtr = np.array([r['cat'] in ix for r in tr_rows])
        ytr = np.array([ix[r['cat']] for r in tr_rows if r['cat'] in ix]); grp = np.array([r['stem'] for r in tr_rows if r['cat'] in ix])
        clf, _, C = fit_clf(Dtr[mtr], ytr, grp, res['kept'])
        gauss = {}
        for c, i in ix.items():
            Dc = Dtr[[r['cat'] == c for r in tr_rows]]; hc = np.array([r['hand'] for r in tr_rows if r['cat'] == c])
            gauss[c] = dict(mean=Dc.mean(0).tolist(), cov=(np.cov(Dc.T) + 0.1 * np.eye(SD.ND)).tolist(), p_right=float(hc.mean()), n=int(len(Dc)))
        (O / 'EXPL_MODEL.json').write_text(json.dumps(dict(cats=res['kept'], coef=clf.coef_.tolist(), intercept=clf.intercept_.tolist(), C=C, gauss=gauss)))
    (O / 'EXPL_SANITY.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def plan(a):
    E = Path(a.expl); M = json.loads((E / 'EXPL_MODEL.json').read_text()); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    cis = [int(x) for x in a.cis.split(',')]; D = json.loads(Path(EV).read_text()); inv = {v: k for k, v in D['meta']['dev_ci'].items()}
    items = json.loads((Path(V) / 'train.json').read_text()); sp = json.loads((Path(V) / 'SPLIT.json').read_text()); recs = recordings(items, set(sp['val_ids']))
    toks, scored = [], []
    for ci in cis:
        stem = inv[ci]; last = -1e9
        for w, c, on, off, iid, f0 in sorted(rec_words(recs[stem]), key=lambda x: x[2]):
            k = category(w, c)
            if k not in M['cats']: continue
            scored.append(dict(ci=ci, clip=stem, word=w, cat=k, onset=float(on)))
            t = on - 0.2
            if t - last < 1.2: continue
            g = M['gauss'][k]; rng = np.random.default_rng(int(hashlib.sha1(f"{stem}:{t:.3f}:expl".encode()).hexdigest()[:12], 16))
            d = rng.multivariate_normal(np.asarray(g['mean']), np.asarray(g['cov'])); d[SD.NS:] /= max(np.linalg.norm(d[SD.NS:]), 1e-6)
            hand = int(rng.random() < g['p_right'])
            toks.append(dict(ci=ci, clip=stem, kind='sem', t=float(t), word=w, cat=k, hand=hand, desc=d.tolist(), shuffle=dict(hand=hand, desc=d.tolist())))
            last = t
    (O / 'PLAN_expl.json').write_text(json.dumps(dict(events=toks, n_events=len(toks), cis=cis, rule='expl_sem.py plan')))
    (O / 'SCORED_expl.json').write_text(json.dumps(dict(events=scored)))
    print(json.dumps(dict(tokens=len(toks), scored=len(scored), per_cat={c: sum(e['cat'] == c for e in scored) for c in M['cats']})))


def evaluate(a):
    E = Path(a.expl); M = json.loads((E / 'EXPL_MODEL.json').read_text()); cats = M['cats']; K = len(cats)
    W_, b_ = np.asarray(M['coef']), np.asarray(M['intercept']); st = json.loads((Path(a.sem) / 'SEM_BANK.json').read_text())
    mu, sd = np.asarray(st['mu']), np.asarray(st['sd'])
    ev = json.loads((Path(a.plans) / 'SCORED_expl.json').read_text())['events']; runs = dict(s.split('=', 1) for s in a.specs)
    runs = {'GT': None, **runs}; res = dict(cats=cats, n_events=len(ev)); per = {}
    for lab, d in runs.items():
        hit, ys, recs, post = [], [], [], []
        for e in ev:
            f = Path(a.gt) / f"gt_{e['ci']}.npy" if d is None else Path(d) / f"{e['ci']}.npy"
            if not f.exists(): continue
            y = np.load(f); c = int(round(e['onset'] * FPS))
            if c - SD.NW // 2 < 0 or c + SD.NW // 2 >= len(y): continue
            w = SD.window(y, e['onset']); raw = SD.np_desc(w, SD.dominant_hand(w)); Dd = np.concatenate([(raw[:SD.NS] - mu) / sd, raw[SD.NS:]])
            z = W_ @ Dd + b_; p = np.exp(z - z.max()); p /= p.sum(); i = cats.index(e['cat'])
            hit.append(float(np.argmax(p) == i)); ys.append(i); recs.append(e['clip']); post.append(float(p[i]))
        hit, ys, recs = np.array(hit), np.array(ys), np.array(recs)
        bal = lambda m: float(np.mean([hit[m & (ys == c)].mean() for c in range(K) if (m & (ys == c)).any()]))
        per[lab] = (hit, ys, recs)
        res[lab] = dict(appropriateness=bal(np.ones(len(hit), bool)), per_cat={cats[c]: float(hit[ys == c].mean()) for c in range(K) if (ys == c).any()},
                        mean_posterior_own=float(np.mean(post)), n=int(len(hit)))
    if 'given' in per and 'none' in per:
        hg, yg, rg = per['given']; hn, yn, rn = per['none']; ur = np.unique(rg); rng = np.random.default_rng(0); bs = []
        def bal_d(idx):
            m = np.zeros(len(hg), bool); m[idx] = True
            dd = [hg[m & (yg == c)].mean() - hn[m & (yn == c)].mean() for c in range(K) if (m & (yg == c)).any()]
            return float(np.mean(dd))
        for _ in range(2000):
            pick = rng.choice(ur, len(ur)); idx = np.concatenate([np.nonzero(rg == r)[0] for r in pick]); bs.append(bal_d(idx))
        dlt = res['given']['appropriateness'] - res['none']['appropriateness']; ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        res['GATE'] = dict(rule='given - none >= +0.20 (balanced) and CI excludes 0', delta=dlt, ci95=ci, pass_=bool(dlt >= 0.20 and ci[0] > 0))
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.add_argument('--sem', required=True); b.add_argument('--out', required=True)
    p = sub.add_parser('plan'); p.add_argument('--expl', required=True); p.add_argument('--cis', required=True); p.add_argument('--out', required=True)
    e = sub.add_parser('eval'); e.add_argument('--expl', required=True); e.add_argument('--sem', required=True); e.add_argument('--plans', required=True)
    e.add_argument('--gt', required=True); e.add_argument('--out', required=True); e.add_argument('specs', nargs='+')
    a = ap.parse_args(); {'build': build, 'plan': plan, 'eval': evaluate}[a.cmd](a)

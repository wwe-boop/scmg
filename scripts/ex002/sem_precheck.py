"""P2 Round 4b pre-check (11-09; exp/text-path docs/criteria/P2_SEMCF_1109.md "Round 4b pre-check", registered before computing):
is the 16-D descriptor predictable from deployable causal inputs? Ridge / k-NN from (causal e5 of the last 5 tokens <= t_tok + 0.2,
PCA 32; lexicon one-hot; prosody mean / std over [t_tok - 2, t_tok]; hand-in-use wrist speed / height over [t_tok - 1, t_tok]) to
the standardised D; model + hyper-parameters by 5-fold recording-grouped CV on train; dev scored once vs marginal mean / marginal
sample / class mean; gate dev E <= 0.90 x E(marginal mean) with the recording bootstrap CI of the reduction above 0.
    SCMG_J=34 python scripts/ex002/sem_precheck.py --sem SEM_DIR --out PRECHECK.json"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'semantic'))
from scmg import sem_desc as SD
from scmg.ctl import LEX2SEM
from scmg.common import PROSODY_COLS
from scmg.ex002_ctl import points
from g1_lexicon import CLASSES as LEXC
from sem_bank import V, EV, recordings

B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; TC = B + '/models/text-e5-causal-v36_1'; FPS = 50
PC = sorted(PROSODY_COLS.values())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sem', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    S = Path(a.sem); Bk = np.load(S / 'SEM_BANK.npz'); dev = json.loads((S / 'DEV_EVENTS.json').read_text())['events']
    items = json.loads((Path(V) / 'train.json').read_text()); sp = json.loads((Path(V) / 'SPLIT.json').read_text())
    recs = {**recordings(items, set(sp['train_ids'])), **recordings(items, set(sp['val_ids']))}
    q = np.load(Path(V) / 'train_q.npy', mmap_mode='r'); f = np.load(Path(V) / 'train_feat.npy', mmap_mode='r')
    tf = np.load(Path(TC) / 'TEXT_FEAT.npy', mmap_mode='r'); tt = np.load(Path(TC) / 'TEXT_TIME.npy'); tix = json.loads((Path(TC) / 'TEXT_INDEX.json').read_text())
    cache = {}
    def rec(stem):
        if stem not in cache:
            its = recs[stem]; Q = np.concatenate([np.asarray(q[x['offset']:x['offset'] + x['frames']], np.float32) for x in its])
            Fp = np.concatenate([np.asarray(f[x['offset']:x['offset'] + x['frames']][:, PC], np.float32) for x in its])
            toks = []; f0 = 0
            for x in its:
                if x['id'] in tix:
                    k0, kn = tix[x['id']]; tm = tt[k0:k0 + kn] + f0 / FPS; ok = np.isfinite(tm)
                    toks.append((np.asarray(tf[k0:k0 + kn], np.float32)[ok], tm[ok]))
                f0 += x['frames']
            g = points(Q); W, Sh = g['wrist'], g['shoulder']
            sp_ = np.zeros((len(W), 2)); sp_[1:] = np.linalg.norm(np.diff(W, axis=0), axis=-1) * FPS
            cache[stem] = dict(F=Fp, toks=(np.concatenate([t[0] for t in toks]) if toks else np.zeros((0, 768), np.float32),
                                           np.concatenate([t[1] for t in toks]) if toks else np.zeros(0)), sp=sp_, hz=W[..., 2] - Sh[..., 2])
            if len(cache) > 400: cache.pop(next(iter(cache)))
        return cache[stem]
    def feats(stem, t_tok, kwcat):
        r = rec(stem); fr = lambda s: int(np.clip(round(s * FPS), 0, len(r['F']) - 1))
        e, tm = r['toks']; m = np.nonzero(tm <= t_tok + 0.2)[0][-5:]
        e5 = e[m].mean(0) if len(m) else np.zeros(768, np.float32)
        a0, a1 = fr(t_tok - 2), max(fr(t_tok), fr(t_tok - 2) + 1); pr = r['F'][a0:a1]
        h0, h1 = fr(t_tok - 1), max(fr(t_tok), fr(t_tok - 1) + 1)
        lex = np.zeros(len(LEXC), np.float32); lex[LEXC.index(kwcat) if kwcat in LEXC else 0] = 1
        return e5, np.concatenate([lex, pr.mean(0), pr.std(0), r['sp'][h0:h1].mean(0), r['hz'][h0:h1].mean(0)])
    Xtr = [feats(str(s), float(t), str(c)) for s, t, c in zip(Bk['stem'], Bk['t_tok'], Bk['cat'])]
    Xdv = [feats(e['stem'], e['t_tok'], e['cat']) for e in dev]
    Ytr = Bk['D'].astype(np.float64); Ydv = np.array([e['desc'] for e in dev], np.float64)
    grp_tr = np.array([str(s) for s in Bk['stem']]); grp_dv = np.array([e['stem'] for e in dev])
    E5tr = np.stack([x[0] for x in Xtr]); E5dv = np.stack([x[0] for x in Xdv])
    mu5 = E5tr.mean(0); U, s_, Vt = np.linalg.svd(E5tr - mu5, full_matrices=False); P32 = Vt[:32].T
    Z = lambda E5, X: np.concatenate([(E5 - mu5) @ P32, np.stack([x[1] for x in X])], 1)
    Ztr, Zdv = Z(E5tr, Xtr), Z(E5dv, Xdv); zm, zs = Ztr.mean(0), Ztr.std(0) + 1e-6; Ztr, Zdv = (Ztr - zm) / zs, (Zdv - zm) / zs
    Ef = lambda P, Y: np.sum((P - Y) ** 2, -1) / SD.ND
    def ridge(Xa, Ya, Xb, al):
        A = np.c_[Xa, np.ones(len(Xa))]; R = al * np.eye(A.shape[1]); R[-1, -1] = 0
        W = np.linalg.solve(A.T @ A + R, A.T @ Ya); return np.c_[Xb, np.ones(len(Xb))] @ W
    def knn(Xa, Ya, Xb, k):
        out = []
        for i in range(0, len(Xb), 512):
            d = ((Xb[i:i + 512, None] - Xa[None]) ** 2).sum(-1); idx = np.argsort(d, 1)[:, :k]; out.append(Ya[idx].mean(1))
        return np.concatenate(out)
    ug = np.unique(grp_tr); rng = np.random.default_rng(0); fold = dict(zip(ug, rng.permutation(len(ug)) % 5))
    fid = np.array([fold[g] for g in grp_tr]); cv = {}
    for name, fn, hs in (('ridge', ridge, (0.1, 1, 10, 100, 1000)), ('knn', knn, (5, 20, 50))):
        for h in hs:
            e = []
            for k in range(5):
                tr, va = fid != k, fid == k
                e.append(Ef(fn(Ztr[tr], Ytr[tr], Ztr[va], h), Ytr[va]))
            cv[f'{name}:{h}'] = float(np.concatenate(e).mean())
    cv_mm = float(np.concatenate([Ef(Ytr[fid != k].mean(0)[None], Ytr[fid == k]) for k in range(5)]).mean())
    best = min(cv, key=cv.get); nm, h = best.split(':'); h = float(h) if nm == 'ridge' else int(h)
    Pdv = (ridge if nm == 'ridge' else knn)(Ztr, Ytr, Zdv, h)
    e_mod = Ef(Pdv, Ydv); e_mm = Ef(Ytr.mean(0)[None], Ydv)
    e_ms = np.mean([Ef(Ytr[rng.integers(len(Ytr), size=len(Ydv))], Ydv) for _ in range(20)], 0)
    cm = {c: Ytr[Bk['cls'] == c].mean(0) for c in np.unique(Bk['cls'])}
    e_cls = Ef(np.stack([cm.get(LEX2SEM.get(e['cat'], ''), Ytr.mean(0)) for e in dev]), Ydv)
    ud = np.unique(grp_dv); bs = []
    for _ in range(2000):
        pick = rng.choice(ud, len(ud)); m = np.concatenate([np.nonzero(grp_dv == g)[0] for g in pick])
        bs.append(1 - e_mod[m].mean() / e_mm[m].mean())
    red = 1 - e_mod.mean() / e_mm.mean(); ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    r2 = 1 - ((Pdv - Ydv) ** 2).sum(0) / ((Ydv - Ytr.mean(0)) ** 2).sum(0)
    wc = {SD.NAMES[j]: float(np.mean([Ytr[Bk['cls'] == c, j].var() * (Bk['cls'] == c).mean() for c in cm]) * len(cm) / Ytr[:, j].var()) for j in range(SD.ND)}
    res = dict(n_train=len(Ytr), n_dev=len(Ydv), cv=cv, cv_marginal_mean=cv_mm, chosen=best,
               dev=dict(E_model=float(e_mod.mean()), E_marginal_mean=float(e_mm.mean()), E_marginal_sample=float(e_ms.mean()), E_class_mean_lexicon=float(e_cls.mean())),
               reduction_vs_marginal_mean=float(red), reduction_ci95=ci, r2_dev={SD.NAMES[j]: float(r2[j]) for j in range(SD.ND)},
               within_class_over_total_var_train=wc, GATE=dict(rule='E_model <= 0.90 E_marginal_mean and CI low > 0', pass_=bool(red >= 0.10 and ci[0] > 0)))
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

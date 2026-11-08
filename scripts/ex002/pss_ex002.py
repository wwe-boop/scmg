"""11-08 EX002 metric M1: paired stroke-suppression yield (PSS), docs/criteria/EX002_INTERRUPT_1107.md "Metric M1" (pre-registered 5663868).
    python pss_ex002.py calib --out P.json [--vs-pct 50 --skip 0.3]          (half A only)
    python pss_ex002.py validate --params P.json --out V.json                (gates 1-3: half B, fresh, GT in speech, mismatched ref)
    python pss_ex002.py score --params P.json --out S.json label=<dir>:<U dir> ...   (models: executed U[:f] + y[f:], ref = U)"""
import argparse, json, sys, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import yield_v4_ex002 as Y
B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; X = B + '/ex002-interrupt-1107'; FPS = 50; END = 107; PRE = 92


def speed(q):
    p = Y.pts(q); w = p[:, :2]; return np.linalg.norm(np.diff(w, axis=0), axis=-1).max(1) * FPS   # index k = frame k + 1


def strokes(s, a, b, vs):
    """onsets in frames [a, b) (speed index k <-> frame k + 1): s[k-1] < vs <= s[k] and s[k:k+5] >= vs."""
    n = 0
    for k in range(max(a - 1, 1), min(b - 1, len(s))):
        if s[k - 1] < vs <= s[k] and (s[k:k + 5] >= vs).all(): n += 1
    return n


def rate(s, a, b, vs): return strokes(s, a, b, vs) / ((b - a) / FPS)


def human(seg):
    out = []
    for st, q, off in Y.tail_windows(seg):
        if off < 50: out.append((st, None, None, off)); continue
        out.append((st, speed(q), max(0, off - PRE), off))
    return out


def pss(r_post, r_ref, rmin): return 1 - r_post / max(r_ref, rmin)


def calib(a):
    H = [h for h in human(X + '/tails/seg_halfA') if h[1] is not None]
    pk = [s[max(a0 - 1, 0):off - 1].max() for _, s, a0, off in H]; vs = float(np.percentile(pk, a.vs_pct))
    rr = [rate(s, a0, off, vs) for _, s, a0, off in H]; rmin = 1 / ((END - 15) / FPS); sk = int(round(a.skip * FPS))   # amendment 11-08: one stroke per post window
    p10 = float(np.percentile(rr, 10)); zero = float(np.mean(np.array(rr) == 0))
    passA = float(np.mean([pss(rate(s, off + sk, off + END, vs), rate(s, a0, off, vs), rmin) >= 0.5 for _, s, a0, off in H]))
    par = dict(vs=vs, vs_pct=a.vs_pct, rmin=rmin, A_rate_p10=p10, A_zero_rate_share=zero, skip=a.skip, n_A=len(H), A_pass=passA, source='half-A tails (seg_halfA)')
    Path(a.out).write_text(json.dumps(par, indent=1)); print('PSS_CALIB', json.dumps(par))


def validate(a):
    par = json.loads(Path(a.params).read_text()); vs, rmin, sk = par['vs'], par['rmin'], int(round(par['skip'] * FPS)); res = {}
    for lab, seg in (('tails_B', X + '/tails/seg_halfB'), ('fresh', X + '/tails_fresh/seg')):
        H = human(seg); ex = sum(h[1] is None for h in H); H = [h for h in H if h[1] is not None]
        ok = [pss(rate(s, off + sk, off + END, vs), rate(s, a0, off, vs), rmin) >= 0.5 for _, s, a0, off in H]
        res[lab] = dict(n=len(ok), excluded_short_pre=ex, pass_rate=float(np.mean(ok)))
        if lab == 'fresh':
            by = {}
            for (st, *_), o in zip(H, ok): by.setdefault('_'.join(st.split('_')[:2]), []).append(o)
            res[lab]['per_speaker'] = {k: dict(n=len(v), pass_rate=float(np.mean(v))) for k, v in by.items()}
        if lab == 'tails_B':                                    # gate 3: mismatched reference (derangement by sort order)
            n = len(H); mm = [pss(rate(H[i][1], H[i][3] + sk, H[i][3] + END, vs), rate(H[(i + 1) % n][1], H[(i + 1) % n][2], H[(i + 1) % n][3], vs), rmin) >= 0.5 for i in range(n)]
            res['mismatched_ref_B'] = dict(n=n, pass_rate=float(np.mean(mm)), delta_pp=float(100 * (np.mean(mm) - np.mean(ok))))
        print(lab, json.dumps(res[lab]), flush=True)
    ints = json.loads(Path(X + '/interrupts_dev_ex002.json').read_text())['interrupts']; E = B + '/ex002-sfscope-1108/enc-s0'; ok = []
    gts = {}
    for r in ints:
        f = r['f_i']; g = gts.setdefault(r['ci'], np.load(f'{E}/psync/gt_{r["ci"]}.npy').astype(np.float64))
        if len(g) < f + END or f < PRE: continue
        s = speed(g); ok.append(pss(rate(s, f + sk, f + END, vs), rate(s, f - PRE, f, vs), rmin) >= 0.5)
    res['GT_speech'] = dict(n=len(ok), pass_rate=float(np.mean(ok)))
    print('GT_speech', json.dumps(res['GT_speech']), 'mismatched', json.dumps(res['mismatched_ref_B']), flush=True)
    res['gates'] = dict(g1=res['tails_B']['pass_rate'] >= 0.85 and res['fresh']['pass_rate'] >= 0.85, g2=res['GT_speech']['pass_rate'] <= 0.15,
                        g3=abs(res['mismatched_ref_B']['delta_pp']) <= 10)
    res['gates']['all'] = all(res['gates'].values()); res['params'] = par
    Path(a.out).write_text(json.dumps(res, indent=1)); print('PSS_VALIDATE', json.dumps(res['gates']))


def score(a):
    par = json.loads(Path(a.params).read_text()); vs, rmin, sk = par['vs'], par['rmin'], int(round(par['skip'] * FPS))
    ints = json.loads(Path(X + '/interrupts_dev_ex002.json').read_text())['interrupts']; res = {}
    for spec in a.rows:
        lab, rest = spec.split('=', 1); d, ud = rest.split(':'); ok = []
        for r in ints:
            fn = Path(d) / f"{r['ci']}@{r['item']}@{r['frac']:.1f}.npy"
            if not fn.exists(): continue
            f = r['f_i']; y = np.load(fn).astype(np.float64); U = np.load(f"{ud}/{r['ci']}.npy").astype(np.float64)
            z = np.concatenate([U[:f], y[f:]])[:f + END]
            if len(z) < f + END or len(U) < f + END: continue
            ok.append(pss(rate(speed(z), f + sk, f + END, vs), rate(speed(U[:f + END]), f + sk, f + END, vs), rmin) >= 0.5)
        res[lab] = dict(n=len(ok), pss_pass=float(np.mean(ok)) if ok else None); print(lab, json.dumps(res[lab]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('calib'); p.add_argument('--out', required=True); p.add_argument('--vs-pct', type=float, default=50); p.add_argument('--skip', type=float, default=0.3)
    p = sp.add_parser('validate'); p.add_argument('--params', required=True); p.add_argument('--out', required=True)
    p = sp.add_parser('score'); p.add_argument('--params', required=True); p.add_argument('--out', required=True); p.add_argument('rows', nargs='+')
    a = ap.parse_args(); dict(calib=calib, validate=validate, score=score)[a.cmd](a)

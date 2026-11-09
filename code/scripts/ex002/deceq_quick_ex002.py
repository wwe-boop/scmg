"""11-08 EX002 design D2 quick lane (docs/criteria/EX002_DECEQ_1108.md): fit the causal polyphase equaliser on TRAIN round-trips
(2000 data-v36_1 train items, seed 0; 10 % train-internal hold-out picks lambda), evaluate on the first 10 dev recordings (ci 0-9):
round-trip L1 (normalised), per-phase arm-acceleration profile (round-trips, enc-s0 offline s0 generations, GT), generation window p95
ratio to GT; controls K = 0 (affine) and phase-agnostic. Kill: round-trip L1 worse than r3, or round-trip profile max/min > 1.15.
    python deceq_quick_ex002.py --out DIR [--n 2000]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import dec_eq as D
M = os.path.expanduser('~/bulk/cpfs-mirror'); B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'
VIEW = M + '/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1'; ST = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/STATS_v36_1.json'
VAE = M + '/CKPT/zehan/gesture-scmg/trained/vae-r3/final.pt'; E = B + '/ex002-sfscope-1108/enc-s0'
ARM = (list(range(4, 11)), list(range(11, 18)))
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=2000); a = ap.parse_args()
O = Path(a.out); O.mkdir(parents=True, exist_ok=True); dev = 'cuda'
st = json.load(open(ST)); qm = torch.tensor(st['q_mean'], device=dev).float(); qs = torch.tensor(st['q_std'], device=dev).float()
from scmg.vae import MotionVAE
ck = torch.load(VAE, map_location='cpu'); vae = MotionVAE(**ck['cfg']).to(dev); vae.load_state_dict(ck['ema']); vae.eval()


def rt(y):
    T = len(y); Tp = -(-T // 4) * 4; x = np.concatenate([y, np.repeat(y[-1:], Tp - T, 0)]) if Tp > T else y
    with torch.no_grad():
        xn = (torch.as_tensor(x, device=dev).float() - qm) / qs; mu, _ = vae.encode(xn[None]); r = vae.decode(mu)[0]
    return (r[:T] * qs + qm).cpu().numpy().astype(np.float64)


def l1n(y, g): return float(np.mean(np.abs((y - g) / st_qs)))


st_qs = np.array(st['q_std'])


def acc(y): return np.max([np.linalg.norm(np.diff(y[:, jj], n=2, axis=0), axis=1) * 2500. for jj in ARM], 0)   # index i = frame i+1


def profile(ys):
    s = np.zeros(4); c = np.zeros(4)
    for y in ys:
        aa = acc(y); ph = (np.arange(len(aa)) + 1) % 4; np.add.at(s, ph, aa); np.add.at(c, ph, 1)
    m = s / c; m = m / m.mean(); return [round(float(v), 4) for v in m], float(m.max() / m.min())


def winp95(ys):
    pk = []
    for y in ys:
        aa = acc(y); pk += [aa[i:i + 11].max() for i in range(0, len(aa) - 10, 5)]
    return float(np.percentile(pk, 95))


# ---- train round-trips (train split only)
items = json.load(open(VIEW + '/train.json')); tr = set(json.load(open(VIEW + '/SPLIT.json'))['train_ids'])
q = np.load(VIEW + '/train_q.npy', mmap_mode='r'); rng = np.random.default_rng(0)
cand = [it for it in items if it['id'] in tr and it['frames'] >= 64]; pick = [cand[i] for i in rng.choice(len(cand), a.n, replace=False)]
pairs = []
for it in pick:
    g = np.asarray(q[it['offset']:it['offset'] + it['frames']], np.float64); pairs.append((rt(g), g))
nh = len(pairs) // 10; fitp, hold = pairs[nh:], pairs[:nh]
res = dict(n_fit=len(fitp), n_hold=len(hold), train_ids_only=True, lam_grid={})
base_hold = float(np.mean([l1n(x, g) for x, g in hold]))
for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
    eq = D.fit(fitp, K=7, lam=lam); res['lam_grid'][lam] = float(np.mean([l1n(D.apply(eq, x), g) for x, g in hold]))
lam = min(res['lam_grid'], key=res['lam_grid'].get); res.update(lam=lam, hold_l1_r3=base_hold)
eqs = dict(poly=D.fit(fitp, K=7, lam=lam, mode='poly'), affine=D.fit(fitp, K=0, lam=lam, mode='poly'), agnostic=D.fit(fitp, K=7, lam=lam, mode='agnostic'))
for k, eq in eqs.items(): D.save(eq, O / f'EQ_{k}.npz')
print('FIT', json.dumps(res), flush=True)
# ---- dev evaluation (first 10 dev recordings)
gt = [np.load(f'{E}/offline/gt_{ci}.npy').astype(np.float64) for ci in range(10)]
gen = [np.load(f'{E}/offline/s0/{ci}.npy').astype(np.float64) for ci in range(10)]
rts = [rt(g) for g in gt]
g_prof, g_mm = profile(gt); gw = winp95(gt)
rows = dict(GT=dict(profile=g_prof, max_over_min=g_mm, win_p95=gw))
for lab, f in [('r3', None)] + list(eqs.items()):
    R = [x if f is None else D.apply(f, x) for x in rts]; G = [x if f is None else D.apply(f, x) for x in gen]
    rp, rmm = profile(R); gp, gmm = profile(G)
    rows[lab] = dict(rt_l1=float(np.mean([l1n(x, g) for x, g in zip(R, gt)])), rt_profile=rp, rt_max_over_min=rmm,
                     gen_profile=gp, gen_max_over_min=gmm, gen_win_p95=winp95(G), gen_win_p95_ratio_GT=winp95(G) / gw, rt_win_p95_ratio_GT=winp95(R) / gw)
    print(lab, json.dumps(rows[lab]), flush=True)
p = rows['poly']; kill = (p['rt_l1'] > rows['r3']['rt_l1']) or (p['rt_max_over_min'] > 1.15)
res.update(dev=rows, kill=bool(kill), kill_reasons=dict(rt_l1_worse=p['rt_l1'] > rows['r3']['rt_l1'], profile_not_flat=p['rt_max_over_min'] > 1.15))
(O / 'QUICK.json').write_text(json.dumps(res, indent=1)); print('DECEQ_QUICK', 'KILL' if kill else 'PASS', json.dumps(res['kill_reasons']))

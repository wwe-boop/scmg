"""11-07 motion-end head (criteria/end_head_1107.json, written before the run). Read-only binary head on the planner output of
the frozen streaming model; decides the retraction time T_r of the R0causal blend after an interruption.
  tails  --view <tail view (labelled, hos)> --tails <gaps_train_tail> --human human_ref.json --out tails.json
         interrupt list (t_i = speech offset, frac 1.0) of every tail item + the GT label of each tail (kind, yield time)
  train  --tails tails.json --plans <dir of infer_retract --record-plan outputs> --out head.pt [--steps 2000]
         y_k = 1 iff the GT hands are at rest height (R_sil + 5 cm) from the chunk end on through the tail end (low: 1,
         hold: 0); only chunks ending inside the tail; 2-layer MLP, BCE, Adam 1e-3, batch 512; recordings split 90 / 10
  apply  --head head.pt --plans <dir> --interrupts I.json --out tr.json [--thr 0.5 --tmin 0.6 --tmax 1.8]
         T_r_hat = end of the first post-interruption chunk with p >= thr, minus t_i, clamped; tmax if never
Run with PYTHONPATH=<ECHO-G official>/src (tails: icommon FK)."""
import argparse, json, sys, zlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))

ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['tails', 'train', 'apply'])
ap.add_argument('--view'); ap.add_argument('--tails'); ap.add_argument('--human'); ap.add_argument('--out', required=True)
ap.add_argument('--plans'); ap.add_argument('--steps', type=int, default=2000); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--head'); ap.add_argument('--interrupts'); ap.add_argument('--thr', type=float, default=0.5)
ap.add_argument('--tmin', type=float, default=0.6); ap.add_argument('--tmax', type=float, default=1.8)
a = ap.parse_args()


class Head(torch.nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.GELU(), torch.nn.Linear(256, 1))

    def forward(self, h): return self.net(h)[..., 0]


if a.cmd == 'tails':
    import icommon as I
    items = json.loads((Path(a.view) / 'train.json').read_text())
    rest_z = np.array(json.loads(Path(a.human).read_text())['rest']['R_sil']['hand_z'])
    rows, lab = [], {}
    for it in items:
        t = it.get('tail')
        if not t: continue
        d = np.load(f"{a.tails}/seg/{t['id']}.npz"); rr = d['robot_repr'].astype(np.float64); off = int(d['off'])
        ty, need = I.yield_time(I.fk(rr), off, rest_z, horizon=len(rr) - off)
        kind = 'low' if not need else ('retract' if ty is not None else 'hold')
        if kind != t['kind']: raise ValueError(f"{it['id']}: kind {kind} != view {t['kind']}")
        t_i = t['speech_frames'] / 50.; f_i = int(round(t_i * 30))
        rows.append(dict(stem=it['id'], frac=1.0, f_i=f_i, t_i=t_i, n30=int(it['echog']['frames30'])))
        lab[it['id']] = dict(kind=kind, t_yield=0. if kind == 'low' else ty, tail_s=t['frames'] / 50., t_off=t_i, rec=it.get('conv_key') or it['id'])
    Path(a.out).write_text(json.dumps(dict(interrupts=rows, labels=lab, rule='end_head.py tails (criteria/end_head_1107.json)')))
    from collections import Counter
    print('tails', len(rows), Counter(v['kind'] for v in lab.values()))

elif a.cmd == 'train':
    T = json.loads(Path(a.tails).read_text()); lab = T['labels']
    X, Y, G = [], [], []
    for stem, L in lab.items():
        f = Path(a.plans) / f'{stem}@1.0.plan.npz'
        if not f.exists(): continue
        z = np.load(f); tau = z['t_end'] - L['t_off']
        use = (tau > 0) & (tau <= L['tail_s'] + 1e-9)
        if L['kind'] == 'hold': y = np.zeros(len(tau))
        else: y = (tau >= L['t_yield'] - 1e-9).astype(np.float64)
        X.append(z['h'][use]); Y.append(y[use]); G += [L['rec']] * int(use.sum())
    X = torch.as_tensor(np.concatenate(X)); Y = torch.as_tensor(np.concatenate(Y)).float()
    val = torch.as_tensor([zlib.crc32(g.encode()) % 10 == 0 for g in G])
    print(f'chunks {len(Y)} positives {float(Y.mean()):.3f} val {int(val.sum())}', flush=True)
    torch.manual_seed(a.seed); mu, sd = X[~val].mean(0), X[~val].std(0) + 1e-6
    hd = Head(X.shape[1]); opt = torch.optim.Adam(hd.parameters(), 1e-3); Xt, Yt = (X[~val] - mu) / sd, Y[~val]
    g = torch.Generator().manual_seed(a.seed)
    for s in range(a.steps):
        b = torch.randint(0, len(Yt), (512,), generator=g)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(hd(Xt[b]), Yt[b]); opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0 or s == a.steps - 1: print(f'step {s} L_End {float(loss):.4f}', flush=True)
    from sklearn.metrics import roc_auc_score
    with torch.no_grad(): pv = torch.sigmoid(hd((X[val] - mu) / sd)).numpy(); pt = torch.sigmoid(hd(Xt)).numpy()
    yv = Y[val].numpy()
    info = dict(n=len(Y), pos=float(Y.mean()), n_val=int(val.sum()), auc_train=float(roc_auc_score(Yt.numpy(), pt)),
                auc_val=float(roc_auc_score(yv, pv)) if 0 < yv.mean() < 1 else None, acc_val=float(((pv >= .5) == (yv > .5)).mean()))
    print(json.dumps(info), flush=True)
    torch.save(dict(state=hd.state_dict(), mu=mu, sd=sd, d=X.shape[1], info=info, steps=a.steps), a.out)

else:
    ck = torch.load(a.head, weights_only=False); hd = Head(ck['d']); hd.load_state_dict(ck['state']); hd.eval()
    rows = json.loads(Path(a.interrupts).read_text())['interrupts']; out = {}; nf = 0
    for r in rows:
        n = f"{r['stem']}@{r['frac']:.1f}"; f = Path(a.plans) / f'{n}.plan.npz'
        if not f.exists(): continue
        z = np.load(f)
        with torch.no_grad(): p = torch.sigmoid(hd((torch.as_tensor(z['h']) - ck['mu']) / ck['sd'])).numpy()
        rel = z['t_end'] - r['t_i']; hit = np.nonzero((p >= a.thr) & (rel > 0))[0]
        tr = float(np.clip(rel[hit[0]], a.tmin, a.tmax)) if len(hit) else a.tmax; nf += len(hit) > 0
        out[n] = dict(tr=tr, fired=bool(len(hit)), p=[round(float(x), 4) for x in p], t_rel=[round(float(x), 3) for x in rel])
    trs = np.array([v['tr'] for v in out.values()])
    meta = dict(n=len(out), fired=nf, tr_mean=float(trs.mean()), tr_p10_50_90=np.percentile(trs, [10, 50, 90]).tolist(), thr=a.thr, tmin=a.tmin, tmax=a.tmax)
    Path(a.out).write_text(json.dumps(dict(meta=meta, tr=out))); print(json.dumps(meta))

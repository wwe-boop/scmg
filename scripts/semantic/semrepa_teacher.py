"""SemREPA teacher: train on the G1 hos TRAIN ids, then the dev sanity gates (docs/criteria/SEMREPA_1109.md section 1).
  train : frame labels from the BEAT2 sem annotation (g1_build_sem.py mapping: echog.recording + start_sec; per frame the
          latest-starting deictic / iconic / metaphoric segment containing it, else beat_align -> beat, else none; no sem file ->
          masked), e5 window target (mean of the item's e5 tokens with time in [c - 0.8, c + 0.8] s, L2-normalised), seed 7,
          AdamW 5e-4 / wd 0.05, batch 512, 6000 steps, OneCycle 5 %, amplitude x U(0.85, 1.15) + 0.01 noise; final checkpoint.
  sanity: dev 400 items, data-sem-hos dev events: S1 GT vs mismatched-motion pair accuracy (gate >= 0.60), S2 semantic AUC
          (gate >= 0.60); reported: 5-class balanced patch accuracy, S1 with the e5 target swapped instead.
    python scripts/semantic/semrepa_teacher.py train --out T.pt [--win 80] [--e5-loss nce]   (nce = v2, amendment 2)
    python scripts/semantic/semrepa_teacher.py sanity --ckpt T.pt --out SANITY.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from scmg.semrepa import Teacher, CH, PATCH, NCLS, CLASSES

MIR = '/home/xr-25b7/bulk/cpfs-mirror'
VIEW = MIR + '/CKPT/zehan/gesture-scmg/views/echog-g1-v1-hos'
TEXT = MIR + '/dataset/zehan/gesture/scmg/text-e5-echog-g1-v1'
SEM = '/home/xr-25b7/bulk/dsh-beat2-sem/data/beat2/sem_en'
EVENTS = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/semantic-baseline-1105/g1/data-sem-hos/EVENTS.json'
FPS, HALF_E5 = 50, 0.8
CID = {'beat': 1, 'deictic': 2, 'iconic': 3, 'metaphoric': 4}


def load_items(split):
    sp = json.loads((Path(VIEW) / 'SPLIT.json').read_text()); ids = set(sp['train_ids' if split == 'train' else 'val_ids'])
    items = [x for x in json.loads((Path(VIEW) / 'train.json').read_text()) if x['id'] in ids]
    return sorted(items, key=lambda x: x['id'])


def build(items):
    from g1_build_sem import read_sem
    q = np.load(Path(VIEW) / 'train_q.npy', mmap_mode='r')
    sem = {p.stem: read_sem(p) for p in Path(SEM).glob('*.txt')}
    tf = np.load(Path(TEXT) / 'TEXT_FEAT.npy', mmap_mode='r'); tt = np.load(Path(TEXT) / 'TEXT_TIME.npy'); tix = json.loads((Path(TEXT) / 'TEXT_INDEX.json').read_text())
    Q, Y, off, L, tok = [], [], [], [], []; o = 0; nosem = 0
    for it in items:
        n = it['frames']; x = np.asarray(q[it['offset']:it['offset'] + n][:, CH], np.float32)
        y = np.zeros(n, np.int64); rec = it['echog']['recording']; s0 = float(it['echog']['start_sec']); fr = np.arange(n) / FPS
        if rec in sem:
            segs = sorted([dict(g, a=g['a'] - s0, b=g['b'] - s0) for g in sem[rec]], key=lambda g: g['a'])
            for g in segs:
                if g['kind'] == 'beat': y[(fr >= g['a']) & (fr <= g['b']) & (y == 0)] = 1
            for g in segs:                                             # latest-starting semantic segment wins (sorted by a)
                if g['kind'] != 'beat': y[(fr >= g['a']) & (fr <= g['b'])] = CID[g['kind']]
        else: y[:] = -1; nosem += 1
        k0, kn = tix[it['id']]; tm = tt[k0:k0 + kn]; ok = np.isfinite(tm)
        tok.append((np.asarray(tf[k0:k0 + kn][ok], np.float32), tm[ok]))
        Q.append(x); Y.append(y); off.append(o); L.append(n); o += n
    return dict(Q=np.concatenate(Q), Y=np.concatenate(Y), off=np.array(off), L=np.array(L), tok=tok, nosem=nosem, ids=[x['id'] for x in items])


MU = None                                            # v2 (amendment 2): mean train e5 token feature, subtracted before normalising


def e5_target(tok, t):
    f, tm = tok; m = np.abs(tm - t) <= HALF_E5
    if not m.any(): return None
    v = f[m].mean(0)
    if MU is not None: v = v - MU
    return v / max(np.linalg.norm(v), 1e-6)


def train_mu():
    D = build(load_items('train')); f = np.concatenate([t[0] for t in D['tok']]); return f.mean(0).astype(np.float32), D


def windows(D, ii, cc, win):
    idx = np.clip(cc[:, None] - win // 2 + np.arange(win)[None], 0, D['L'][ii][:, None] - 1) + D['off'][ii][:, None]
    return idx


def train(a):
    global MU
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = 'cuda'
    if a.e5_loss == 'nce': MU, D = train_mu()
    else: D = build(load_items('train'))
    win = a.win; n = win // PATCH
    lab = D['Y'][D['Y'] >= 0]; cnt = np.bincount(lab, minlength=NCLS).astype(np.float64)
    cw = (cnt.max() / np.maximum(cnt, 1)) ** 0.5; cw = cw / (cw * cnt).sum() * cnt.sum()
    std = D['Q'].std(0) + 1e-3                      # the window-mean-removed scale is close to the raw per-channel std
    Qd = torch.tensor(D['Q'], device=dev); Yd = torch.tensor(D['Y'], device=dev)
    model = Teacher(std, win=win).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=dev), label_smoothing=0.05, ignore_index=-1)
    p_item = D['L'] / D['L'].sum(); log = []; t0 = time.time()
    for step in range(a.steps):
        ii = rng.choice(len(D['L']), a.bs, p=p_item); cc = (rng.random(a.bs) * D['L'][ii]).astype(np.int64)
        idx = torch.as_tensor(windows(D, ii, cc, win), device=dev)
        x = Qd[idx]; x = x * torch.empty(a.bs, 1, 1, device=dev).uniform_(0.85, 1.15) + 0.01 * torch.randn_like(x)
        y = Yd[idx[:, PATCH // 2::PATCH]]                                                    # label of each patch's centre frame
        tg = [e5_target(D['tok'][i], c / FPS) for i, c in zip(ii, cc)]
        em = torch.tensor([t is not None for t in tg], device=dev)
        te = torch.tensor(np.stack([t if t is not None else np.zeros(768, np.float32) for t in tg]), device=dev)
        out = model(x)
        lc = crit(out['logits'].reshape(-1, NCLS), y.reshape(-1))
        if a.e5_loss == 'nce':                     # v2: symmetric in-batch InfoNCE, temperature 0.07
            zi = F.normalize(out['e5'][em], dim=-1); zt = te[em]; lg = zi @ zt.T / 0.07; lab_ = torch.arange(len(zi), device=dev)
            le = 0.5 * (F.cross_entropy(lg, lab_) + F.cross_entropy(lg.T, lab_))
        else: le = ((1 - F.cosine_similarity(out['e5'], te, dim=-1)) * em).sum() / em.sum().clamp(min=1)
        loss = lc + a.e5_w * le
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if step % 250 == 0 or step == a.steps - 1:
            log.append(dict(step=step, cls=round(lc.item(), 4), e5=round(le.item(), 4), s=round(time.time() - t0, 1))); print(json.dumps(log[-1]), flush=True)
    torch.save(dict(model=model.state_dict(), std=std.tolist(), arch=dict(win=win), seed=a.seed, e5_loss=a.e5_loss, e5_w=a.e5_w,
                    e5_mu=None if MU is None else MU.tolist(), steps=a.steps, class_weights=cw.tolist(),
                    train_counts=cnt.tolist(), n_items=len(D['L']), n_nosem=D['nosem'], rule=__doc__.split('\n')[1]), a.out)
    Path(a.out).with_suffix('.log.json').write_text(json.dumps(dict(log=log, args=vars(a)), indent=1))


def auc(pos, neg):
    s = np.concatenate([pos, neg]); r = np.argsort(np.argsort(s, kind='mergesort'), kind='mergesort') + 1.
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def sanity(a):
    global MU
    dev = 'cuda'; s = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    if s.get('e5_mu') is not None: MU = np.asarray(s['e5_mu'], np.float32)
    model = Teacher(s['std'], **s['arch']).to(dev); model.load_state_dict(s['model']); model.eval(); win = model.win
    items = load_items('dev'); D = build(items); pos = {k: i for i, k in enumerate(D['ids'])}
    E = [e for e in json.loads(Path(EVENTS).read_text())['events'] if e['split'] == 'dev' and e['stem'] in pos]
    Qd = torch.tensor(D['Q'], device=dev)
    def run(ii, cc):
        out = []
        for j in range(0, len(ii), 1024):
            idx = torch.as_tensor(windows(D, ii[j:j + 1024], cc[j:j + 1024], win), device=dev); o = model(Qd[idx])
            out.append({k: v.float().cpu() for k, v in o.items()})
        return {k: torch.cat([o[k] for o in out]) for k in out[0]}
    sem = [e for e in E if e['cls'] != 'beat']; bg = [e for e in E if e['cls'] == 'beat']
    si = np.array([pos[e['stem']] for e in sem]); sc = np.round(np.array([e['t'] for e in sem]) * FPS).astype(np.int64)
    tg = [e5_target(D['tok'][i], e['t']) for i, e in zip(si, sem)]; ok = np.array([t is not None for t in tg])
    rng = np.random.default_rng(20261109); perm = rng.permutation(len(sem))
    for _ in range(50):                                                     # derangement across clips
        bad = [k for k in range(len(sem)) if sem[perm[k]]['stem'] == sem[k]['stem']]
        if not bad: break
        for k in bad: j = int(rng.integers(len(sem))); perm[k], perm[j] = perm[j], perm[k]
    own = run(si, sc); mis = run(si[perm], sc[perm])
    te = torch.tensor(np.stack([t if t is not None else np.zeros(768, np.float32) for t in tg]))
    c_own = F.cosine_similarity(own['e5'], te, dim=-1).numpy(); c_mis = F.cosine_similarity(mis['e5'], te, dim=-1).numpy()
    c_tsw = F.cosine_similarity(own['e5'], te[perm], dim=-1).numpy()
    okp = ok & ok[perm]
    S1 = float(np.mean((c_own > c_mis)[ok])); S1t = float(np.mean((c_own > c_tsw)[okp]))
    bi = np.array([pos[e['stem']] for e in bg]); bc = np.round(np.array([e['t'] for e in bg]) * FPS).astype(np.int64)
    psem = lambda o: torch.softmax(o['logits'], -1)[..., 2:].sum(-1).mean(1).numpy()
    S2 = auc(psem(own), psem(run(bi, bc)))
    # balanced patch accuracy over dev windows every 16 frames
    ii, cc = [], []
    for i, n in enumerate(D['L']):
        if D['Y'][D['off'][i]] < 0: continue
        for c in range(8, n, 16): ii.append(i); cc.append(c)
    ii, cc = np.array(ii), np.array(cc); o = run(ii, cc)
    y = D['Y'][windows(D, ii, cc, win)[:, PATCH // 2::PATCH]].reshape(-1); p = o['logits'].argmax(-1).numpy().reshape(-1)
    per = {CLASSES[k]: float((p[y == k] == k).mean()) for k in range(NCLS) if (y == k).any()}
    r = dict(ckpt=a.ckpt, n_sem_events=len(sem), n_with_e5=int(ok.sum()), n_bg=len(bg),
             S1_gt_vs_mismatched_motion=S1, S1_gate=S1 >= 0.60, S2_semantic_auc=S2, S2_gate=S2 >= 0.60,
             S1_e5_target_swapped=S1t, cos_own_mean=float(c_own[ok].mean()), cos_mismatched_mean=float(c_mis[ok].mean()),
             patch_bal_acc=float(np.mean(list(per.values()))), patch_per_class=per, chance_bal=1 / len(per),
             TEACHER_PASS=bool(S1 >= 0.60 and S2 >= 0.60))
    Path(a.out).write_text(json.dumps(r, indent=1)); print(json.dumps(r, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train'); t.add_argument('--out', required=True); t.add_argument('--win', type=int, default=80)
    t.add_argument('--steps', type=int, default=6000); t.add_argument('--bs', type=int, default=512); t.add_argument('--lr', type=float, default=5e-4)
    t.add_argument('--seed', type=int, default=7); t.add_argument('--e5-loss', default='cos', choices=('cos', 'nce'))
    t.add_argument('--e5-w', type=float, default=1.0, help='0 = class-only teacher T3 (amendment 3)')
    s = sub.add_parser('sanity'); s.add_argument('--ckpt', required=True); s.add_argument('--out', required=True)
    a = ap.parse_args(); train(a) if a.cmd == 'train' else sanity(a)

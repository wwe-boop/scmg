"""Train the EX002 semantic-gesture recogniser (semantic-recognizer-v1) on build_intent_dataset.py windows.

Input  per window: q [T=160, 34] rad (50 Hz) -> features on GPU: z-scored joints (STATS_v36_1), FK positions of
       elbows / wrists / hand bases / finger tips (12 bodies, scmg.kinematics), the two hand-base orientations (2 columns
       of R each), plus first differences of everything (velocities).
Model  1-D conv encoder (4 residual blocks, 3x temporal downsampling) -> 2-layer transformer -> mean+max pooling ->
       logits over classes (0 = none, 1.. = intent keys of INDEX.json meta.classes).
Train  on the train groups; 10 % of the train groups are an internal validation split (model selection + detection
       threshold); the test groups (by clip/pair) are only scored once at the end on real ground-truth motion.
Aug    random centre jitter (+-JIT frames), time-scale 0.85-1.15, joint noise, per-window joint offsets.
    python scripts/train_intent_recognizer.py --index <cache>/INDEX.json --windows <cache>/WINDOWS_f16.npy --out <dir>
"""
import argparse, hashlib, json, math, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scmg.common import P
from scmg.kinematics import Kinematics

STATS = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/STATS_v36_1.json'
PROXY = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/sphere_proxy_vox4e.npz'
BODIES = [19, 22, 24, 28, 34, 38, 48, 51, 53, 57, 63, 67]   # L elbow, L wrist(link7), L hand base, L 3 finger tips; same R
HANDS = [24, 53]
ORIGIN = (0.092, 0.0, 1.206)                                  # shoulder centre at rest (m)
W = 160


def _local(path, cache_dir='~/bulk/semrec-cache'):
    p = Path(os.path.expanduser(cache_dir)) / Path(path).name
    return str(p) if p.exists() else P(path)


class Featurizer(nn.Module):
    def __init__(self, stats=STATS, proxy=PROXY):
        super().__init__()
        st = json.load(open(_local(stats) if os.path.exists(_local(stats)) else P(stats)))
        self.register_buffer('qm', torch.tensor(st['q_mean'], dtype=torch.float32))
        self.register_buffer('qs', torch.tensor(st['q_std'], dtype=torch.float32))
        self.kin = Kinematics(_local(proxy))
        self.register_buffer('origin', torch.tensor(ORIGIN, dtype=torch.float32))
        self.dim = 2 * (34 + 3 * len(BODIES) + 6 * len(HANDS))

    @torch.no_grad()
    def forward(self, q):                                          # q [B, T, 34] rad -> [B, T, D]
        B, T, _ = q.shape
        R, p = self.kin.fk(q.reshape(B * T, 34).float())
        pos = ((p[:, BODIES] - self.origin) * 4.0).reshape(B, T, -1)      # ~unit scale
        rot = R[:, HANDS][..., :, :2].reshape(B, T, -1)                     # 2 columns of each hand-base rotation
        x = torch.cat([(q - self.qm) / self.qs, pos, rot], -1)
        v = torch.diff(x, dim=1, prepend=x[:, :1]) * 10.0
        return torch.cat([x, v], -1)


class Block(nn.Module):
    def __init__(self, c, k=5, d=1, p=0.1):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, k, padding=d * (k // 2), dilation=d); self.n1 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv1d(c, c, k, padding=k // 2); self.n2 = nn.GroupNorm(8, c); self.dp = nn.Dropout(p)

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x))); h = self.dp(h); h = self.n2(self.c2(h))
        return F.gelu(x + h)


class Recognizer(nn.Module):
    def __init__(self, din, n_cls, c=192, p=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv1d(din, c, 5, padding=2), nn.GroupNorm(8, c), nn.GELU())
        self.blocks = nn.ModuleList([Block(c, d=1, p=p), Block(c, d=2, p=p), Block(c, d=1, p=p), Block(c, d=2, p=p)])
        self.down = nn.ModuleList([nn.Conv1d(c, c, 4, stride=2, padding=1) for _ in range(3)])
        self.pos = nn.Parameter(torch.zeros(1, W // 8 + 4, c))
        layer = nn.TransformerEncoderLayer(c, 4, 2 * c, dropout=p, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.LayerNorm(2 * c), nn.Dropout(p), nn.Linear(2 * c, n_cls))

    def forward(self, f):                                          # f [B, T, D]
        x = self.inp(f.transpose(1, 2))
        for i, b in enumerate(self.blocks):
            x = b(x)
            if i < 3: x = self.down[i](x)
        x = x.transpose(1, 2); x = self.tf(x + self.pos[:, :x.shape[1]])
        return self.head(torch.cat([x.mean(1), x.amax(1)], -1))


def load_recognizer(ckpt, device='cuda'):
    s = torch.load(ckpt, map_location='cpu', weights_only=False)
    fz = Featurizer().to(device)
    m = Recognizer(fz.dim, len(s['classes']), **s.get('arch', {})).to(device)
    m.load_state_dict(s['model']); m.eval()
    return m, fz, s


def time_scale(x, scale):                                          # x [B, T, J], scale [B] -> resampled around centre
    B, T, J = x.shape
    t = torch.arange(T, device=x.device, dtype=torch.float32)[None] - (T - 1) / 2
    src = (t * scale[:, None] + (T - 1) / 2).clamp(0, T - 1)
    i0 = src.floor().long().clamp(max=T - 2); w = (src - i0)[..., None]
    g = lambda i: torch.gather(x, 1, i[..., None].expand(B, T, J))
    return g(i0) * (1 - w) + g(i0 + 1) * w


def crop(X, idx, shift, M):                                        # X [N, W+2M, J] -> [B, W, J] with per-sample shift
    ar = torch.arange(W, device=X.device)[None] + M + shift[:, None]
    return torch.gather(X[idx], 1, ar[..., None].expand(-1, -1, X.shape[-1]))


def auc(pos, neg):
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    s = np.concatenate([pos, neg]); r = np.empty(len(s)); o = np.argsort(s, kind='mergesort'); r[o] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def predict(model, fz, Xw, bs=512, shift=0):
    """Xw [N, W(+2M), 34] rad tensor on device (centre crop if longer) -> softmax probs [N, C] (numpy)."""
    out = []
    for i in range(0, len(Xw), bs):
        x = Xw[i:i + bs].float()
        if x.shape[1] > W:
            M = (x.shape[1] - W) // 2; x = x[:, M + shift:M + shift + W]
        out.append(torch.softmax(model(fz(x)).float(), -1).cpu())
    return torch.cat(out).numpy()


def metrics(prob, y, kind, classes, thr=None):
    pos = kind == 'pos'; C = len(classes)
    pi = prob[pos].copy(); yi = y[pos]
    pm = pi.copy(); pm[:, 0] = -1                                   # recognition among intent classes (none masked)
    rank = (pm > pm[np.arange(len(yi)), yi][:, None]).sum(1)
    top1 = rank == 0; top5 = rank < 5
    rec = {classes[c]: float(top1[yi == c].mean()) for c in range(1, C) if (yi == c).any()}
    det = 1 - prob[:, 0]
    r = dict(n_pos=int(pos.sum()), n_neg_syn=int((kind == 'neg_syn').sum()), n_neg_human=int((kind == 'neg_human').sum()),
             top1=float(top1.mean()), top5=float(top5.mean()), balanced_acc=float(np.mean(list(rec.values()))),
             top1_with_none=float((pi.argmax(1) == yi).mean()),
             top1_excl_seg177=float(top1[yi != classes.index('seg.177')].mean()) if 'seg.177' in classes else None,
             auc_vs_all_neg=auc(det[pos], det[~pos]), auc_vs_neg_syn=auc(det[pos], det[kind == 'neg_syn']),
             auc_vs_neg_human=auc(det[pos], det[kind == 'neg_human']), per_class_recall=rec,
             majority_baseline_top1=float((yi == np.bincount(yi).argmax()).mean()))
    if thr is not None:
        r.update(det_threshold=float(thr), tpr=float((det[pos] >= thr).mean()), fpr_syn=float((det[kind == 'neg_syn'] >= thr).mean()),
                 fpr_human=float((det[kind == 'neg_human'] >= thr).mean()))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True); ap.add_argument('--windows', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--bs', type=int, default=256); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--width', type=int, default=192); ap.add_argument('--jit', type=int, default=20); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-frac', type=float, default=0.1); ap.add_argument('--fpr', type=float, default=0.01)
    ap.add_argument('--eval-only', action='store_true', help='skip training; recalibrate + score the saved model.pt')
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    I = json.load(open(a.index)); meta = I['meta']; rows = I['rows']; classes = meta['classes']; C = len(classes); M = meta['margin_frames']
    y = np.array([r['y'] for r in rows]); kind = np.array([r['kind'] for r in rows]); split = np.array([r['split'] for r in rows])
    gval = np.array([int(hashlib.sha1(('val:' + r['group']).encode()).hexdigest()[:8], 16) / 0xffffffff < a.val_frac for r in rows])
    tr = (split == 'train') & ~gval; va = (split == 'train') & gval; te = split == 'test'
    print('train', tr.sum(), 'val', va.sum(), 'test', te.sum(), 'classes', C, flush=True)
    X = torch.from_numpy(np.load(a.windows)).to(dev)               # f16 [N, W+2M, 34] (~1 GB on GPU)
    fz = Featurizer().to(dev)
    model = Recognizer(fz.dim, C, c=a.width).to(dev)
    print('params', sum(p.numel() for p in model.parameters()) / 1e6, 'M', flush=True)
    cnt = np.bincount(y[tr], minlength=C).astype(np.float64)
    cw = (cnt.max() / np.maximum(cnt, 1)) ** 0.5; cw = cw / (cw * cnt).sum() * cnt.sum()
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=dev), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    tri = np.where(tr)[0]; steps = a.epochs * (len(tri) // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    yt = torch.tensor(y, device=dev)
    best, best_ep, log = -1, -1, []
    for ep in range(0 if a.eval_only else a.epochs):
        model.train(); t0 = time.time(); perm = np.random.permutation(tri); tl = 0
        for b in range(len(tri) // a.bs):
            idx = torch.tensor(perm[b * a.bs:(b + 1) * a.bs], device=dev); B = len(idx)
            shift = torch.randint(-a.jit, a.jit + 1, (B,), device=dev)
            # crop wider then time-scale, then centre-crop W
            full = X[idx].float()
            sc = torch.empty(B, device=dev).uniform_(0.85, 1.15)
            full = time_scale(full, sc)
            ar = torch.arange(W, device=dev)[None] + M + shift[:, None]
            x = torch.gather(full, 1, ar[..., None].expand(-1, -1, 34))
            x = x + 0.01 * torch.randn_like(x) + 0.03 * torch.randn(B, 1, 34, device=dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = crit(model(fz(x)).float(), yt[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += loss.item()
        model.eval()
        pv = predict(model, fz, X[torch.tensor(np.where(va)[0], device=dev)])
        mv = metrics(pv, y[va], kind[va], classes)
        score = mv['balanced_acc'] + mv['auc_vs_all_neg']
        log.append(dict(ep=ep, loss=tl / max(1, len(tri) // a.bs), val_top1=mv['top1'], val_bal=mv['balanced_acc'], val_auc=mv['auc_vs_all_neg'], s=time.time() - t0))
        print(json.dumps(log[-1]), flush=True)
        if score > best:
            best, best_ep = score, ep
            torch.save(dict(model=model.state_dict(), classes=classes, arch=dict(c=a.width), epoch=ep, window_frames=W, fps=50,
                            meta={k: meta[k] for k in ('version', 'window_frames', 'margin_frames', 'neg_gap_s', 'class_vocab_id')}), out / 'model.pt')
    m, fz, s = load_recognizer(out / 'model.pt', dev)
    pv = predict(m, fz, X[torch.tensor(np.where(va)[0], device=dev)])
    dv = 1 - pv[:, 0]; negv = dv[kind[va] != 'pos']; posv = dv[kind[va] == 'pos']
    # detection threshold: midway between the (1 - fpr) quantile of val negatives and the fpr quantile of val positives
    # (scores are near-saturated: negatives ~0.46, positives ~1.0 -- a pure negative quantile sits inside the negative mode)
    thr = float(0.5 * (np.quantile(negv, 1 - a.fpr) + np.quantile(posv, a.fpr)))
    res = dict(best_epoch=best_ep, val=metrics(pv, y[va], kind[va], classes, thr))
    tei = torch.tensor(np.where(te)[0], device=dev)
    pt = predict(m, fz, X[tei]); res['test'] = metrics(pt, y[te], kind[te], classes, thr)
    # timing robustness on ground truth: window centre off by +-0.2 / +-0.4 s
    for sh in (-20, -10, 10, 20):
        p2 = predict(m, fz, X[tei], shift=sh); r2 = metrics(p2, y[te], kind[te], classes, thr)
        res[f'test_shift_{sh * 20}ms'] = {k: r2[k] for k in ('top1', 'top5', 'balanced_acc', 'auc_vs_all_neg', 'tpr')}
    ck = torch.load(out / 'model.pt', weights_only=False); ck['det_threshold'] = thr; ck['det_threshold_rule'] = f'midpoint(val neg q{1 - a.fpr:.3f}, val pos q{a.fpr:.3f})'; torch.save(ck, out / 'model.pt')
    if a.eval_only and (out / 'METRICS.json').exists(): log = json.loads((out / 'METRICS.json').read_text()).get('train_log', log); best_ep = torch.load(out / 'model.pt', weights_only=False)['epoch']
    res['best_epoch'] = best_ep; res['train_log'] = log; res['args'] = vars(a); res['counts'] = dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum()))
    (out / 'METRICS.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    np.save(out / 'TEST_PROBS.npy', pt.astype(np.float16))
    t = res['test']
    print(json.dumps({k: t[k] for k in t if k != 'per_class_recall'}, indent=1))


if __name__ == '__main__':
    main()

"""G1 semantic-gesture recogniser + semantic-recognition-rate scorer (semantic-baseline-1105).

Task: from a 2.4 s window of G1 motion (30 fps, waist + both arms = channels 22..38 of the 39-d ECHO-G representation,
clip-mean pose removed, + velocities) centred on a word onset, predict the word's lexical semantic category
(g1_lexicon.py; 14 categories + 'none').  Trained on ground truth of the train split only; dev = model selection;
test (ECHO-G val_common) = scored once per motion source (GT / ECHO-G / ours ...).

  train:  python scripts/semantic/g1_recognizer.py train --data <dir> --out <dir>
  score:  python scripts/semantic/g1_recognizer.py score --data <dir> --ckpt <dir>/model.pt --name echog \
              --pt-dir <dir with <stem>.pt> [--split test] --out res.json      (no --pt-dir = ground truth)

Reported per source (events = lexicon words of the split, window inside the clip):
  top1 / bal_top1      recognition of the word's category among the 14 categories ('none' masked)
  shuf_spk_bal_top1    same windows, labels permuted within speaker (category prior + speaker style only)
  time_bal_top1        same label, window moved to a random OTHER time of the same clip (>= 1 s away, no word of
                       that category within 0.6 s): what the clip's overall motion style gives without timing
  srr = bal_top1 - time_bal_top1         <- primary number: time-locked semantic recognition
  pair_win             P(p_c(true time) > p_c(other time)) over events (0.5 = no time-locked information)
  det_auc              1 - p_none at lexicon words vs at background (non-lexicon) words
"""
import argparse, json, math, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

FPS, HALF = 30, 1.2
W = int(round(2 * HALF * FPS))          # 72 frames
CH = list(range(22, 39))                # waist (3) + left arm (7) + right arm (7)
JIT = 6


def configure(meta):
    """motion space from the dataset meta (G1 default: 30 fps, channels 22..38; EX002: 50 fps, 34 joints)."""
    global FPS, W, CH
    FPS = int(meta.get('fps', 30)); CH = list(meta.get('channels', range(22, 39))); W = int(round(2 * HALF * FPS))


def auc(pos, neg):
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    s = np.concatenate([pos, neg]); _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    o = np.argsort(s, kind='mergesort'); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    r = (np.bincount(inv, weights=r) / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Block(nn.Module):
    def __init__(self, c, d=1, p=0.1):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 5, padding=2 * d, dilation=d); self.n1 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv1d(c, c, 5, padding=2); self.n2 = nn.GroupNorm(8, c); self.dp = nn.Dropout(p)

    def forward(self, x):
        return F.gelu(x + self.n2(self.c2(self.dp(F.gelu(self.n1(self.c1(x)))))))


class Recognizer(nn.Module):
    def __init__(self, din, n_cls, c=128, p=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv1d(din, c, 5, padding=2), nn.GroupNorm(8, c), nn.GELU())
        self.blocks = nn.ModuleList([Block(c, 1, p), Block(c, 2, p), Block(c, 1, p)])
        self.down = nn.ModuleList([nn.Conv1d(c, c, 4, stride=2, padding=1) for _ in range(2)])
        self.pos = nn.Parameter(torch.zeros(1, W // 4 + 2, c))
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(c, 4, 2 * c, dropout=p, batch_first=True, norm_first=True), 2)
        self.head = nn.Sequential(nn.LayerNorm(2 * c), nn.Dropout(p), nn.Linear(2 * c, n_cls))

    def forward(self, f):
        x = self.inp(f.transpose(1, 2))
        for i, b in enumerate(self.blocks):
            x = b(x)
            if i < 2: x = self.down[i](x)
        x = self.tf(x.transpose(1, 2) + self.pos[:, :x.shape[-1]])
        return self.head(torch.cat([x.mean(1), x.amax(1)], -1))


class Feat:
    """window [B, W, 17] (clip-mean removed) -> z-scored pose + velocity [B, W, 34]."""
    def __init__(self, mean, std, dev):
        self.m = torch.tensor(mean, device=dev); self.s = torch.tensor(std, device=dev)

    def __call__(self, x):
        z = x / self.s
        v = torch.diff(z, dim=1, prepend=z[:, :1]) * 5.0
        return torch.cat([z, v], -1)


def clip_table(data):
    z = np.load(Path(data) / 'MOTION.npz')
    q = z['q'][:, CH].astype(np.float32)
    stems = [str(s) for s in z['stems']]; off = z['off']; L = z['len']
    idx = {s: i for i, s in enumerate(stems)}
    # clip-mean removal (speaker posture offset) -- per clip, computed on the whole clip
    for i in range(len(stems)):
        a, n = off[i], L[i]; q[a:a + n] -= q[a:a + n].mean(0, keepdims=True)
    return q, stems, off, L, idx


def load_source(pt_dir, stems_needed, gt_len, npy_pattern=None, ci_of=None):
    """generator outputs <stem>.pt (robot_repr [T, 39], 30 fps) or <npy_pattern.format(ci=...)> ([T, 34] rad, EX002)
    -> flat clip-mean-removed array aligned like clip_table; also the true (unpadded) source length per stem."""
    qs, off, L = [], [], []; o = 0; bad = Counter(); real = {}
    for s in stems_needed:
        if npy_pattern:
            x = np.load(Path(pt_dir) / npy_pattern.format(ci=ci_of[s])).astype(np.float32)
        else:
            d = torch.load(Path(pt_dir) / f'{s}.pt', map_location='cpu', weights_only=False)
            x = np.asarray(d['robot_repr'], np.float32)[:int(d.get('real_num_frames', len(d['robot_repr'])))]
            if abs(d.get('fps', 30.0) - FPS) > 1e-3: raise ValueError(f'{s}: fps {d.get("fps")}')
        real[s] = len(x)
        if len(x) != gt_len[s]:
            bad['len_mismatch'] += 1
            x = x[:gt_len[s]] if len(x) > gt_len[s] else np.concatenate([x, np.repeat(x[-1:], gt_len[s] - len(x), 0)])
        x = x[:, CH] - x[:, CH].mean(0, keepdims=True)
        qs.append(x); off.append(o); L.append(len(x)); o += len(x)
    return np.concatenate(qs), {s: i for i, s in enumerate(stems_needed)}, np.array(off), np.array(L), bad, real


def gather(Q, off, L, ci, t, shift=None):
    """Q torch [N, C]; ci, t numpy -> windows [B, W, C] centred on frame round(t*FPS)(+shift), edge-held inside the clip."""
    c = np.round(np.asarray(t) * FPS).astype(np.int64)
    if shift is not None: c = c + shift
    s = c - W // 2
    idx = np.clip(s[:, None] + np.arange(W)[None], 0, L[ci][:, None] - 1) + off[ci][:, None]
    return Q[torch.as_tensor(idx, device=Q.device)]


@torch.no_grad()
def predict(model, feat, Q, off, L, ci, t, bs=2048):
    out = []
    for i in range(0, len(ci), bs):
        x = gather(Q, off, L, ci[i:i + bs], t[i:i + bs])
        out.append(torch.softmax(model(feat(x)).float(), -1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, model.head[-1].out_features))


def other_times(E, words, L, rng, min_gap=1.0, cls_gap=0.6):
    """for each event: a random other time in the same clip (window inside the clip, >= min_gap s from the event,
    no word of the same category within cls_gap s); NaN when the clip has no such time."""
    out = np.full(len(E), np.nan)
    for k, e in enumerate(E):
        dur = L[e['_ci']] / FPS
        cand = np.arange(HALF, dur - HALF + 1e-6, 0.1)
        same = np.array([w[2] for w in words[e['stem']] if w[1] == e['cls']])
        ok = np.abs(cand - e['t']) >= min_gap
        if len(same): ok &= np.abs(cand[:, None] - same[None]).min(1) >= cls_gap
        if ok.any(): out[k] = rng.choice(cand[ok])
    return out


def metrics(P, y, classes, spk=None, Ptime=None, rng=None, none_P=None):
    """P [E, C] probs at lexicon events (y >= 1); none_P [B, C] at background words."""
    K = [c for c in range(1, len(classes)) if (y == c).any()]
    def rec(P, y):
        pm = P.copy(); pm[:, 0] = -1
        t1 = pm.argmax(1) == y
        per = {classes[c]: float(t1[y == c].mean()) for c in K if (y == c).any()}
        return float(t1.mean()), float(np.mean(list(per.values()))), per
    t1, bal, per = rec(P, y)
    r = dict(n=int(len(y)), n_classes=len(K), chance_bal=1.0 / len(K), top1=t1, bal_top1=bal, per_class=per,
             majority_top1=float((y == np.bincount(y).argmax()).mean()))
    if spk is not None and rng is not None:
        sh = []
        for _ in range(30):
            yp = y.copy()
            for s in np.unique(spk):
                m = np.where(spk == s)[0]; yp[m] = y[rng.permutation(m)]
            sh.append(rec(P, yp)[1])
        r['shuf_spk_bal_top1'] = float(np.mean(sh))
    if Ptime is not None:
        ok = np.isfinite(Ptime[:, 0])
        r['time_n'] = int(ok.sum())
        r['time_bal_top1'] = rec(Ptime[ok], y[ok])[1]
        r['bal_top1_on_time_subset'] = rec(P[ok], y[ok])[1]
        r['srr'] = r['bal_top1_on_time_subset'] - r['time_bal_top1']
        a_, b_ = P[ok, y[ok]], Ptime[ok, y[ok]]
        r['pair_win'] = float(np.mean((a_ > b_) + 0.5 * (a_ == b_)))
    if none_P is not None and len(none_P):
        r['det_auc'] = auc(1 - P[:, 0], 1 - none_P[:, 0])
    return r


def load_events(data):
    D = json.loads((Path(data) / 'EVENTS.json').read_text()); return D


def train(a):
    dev = 'cuda'; torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.default_rng(a.seed)
    D = load_events(a.data); configure(D['meta']); classes = D['meta']['classes']; cid = {c: i for i, c in enumerate(classes)}
    if a.target == 'spk':                  # pipeline sanity control: same windows, label = speaker (should be easy)
        classes = ['none'] + sorted({e['spk'] for e in D['events']}); cid = {c: i for i, c in enumerate(classes)}
        D['events'] = [dict(e, cls=e['spk']) for e in D['events']]
    q, stems, off, L, idx = clip_table(a.data)
    trm = np.concatenate([np.arange(off[idx[s]], off[idx[s]] + L[idx[s]]) for s in stems if D['split'][s] == 'train'])
    std = q[trm].std(0) + 1e-3
    Q = torch.tensor(q, device=dev); feat = Feat(np.zeros(len(CH)), std, dev)
    if a.sel_frac > 0:                     # selection set = dev + a take-level slice of train (dev alone is small)
        import hashlib
        take = lambda st: st.rsplit('_utt_', 1)[0]
        sel = lambda st: int(hashlib.sha1(('sel:' + take(st)).encode()).hexdigest()[:8], 16) / 0xffffffff < a.sel_frac
        D['events'] = [dict(e, split='dev') if e['split'] == 'train' and sel(e['stem']) else e for e in D['events']]
    ev = {sp: [e for e in D['events'] if e['split'] == sp and (a.lang == 'all' or e['lang'] == a.lang)] for sp in ('train', 'dev', 'test')}
    arr = {sp: (np.array([idx[e['stem']] for e in E]), np.array([e['t'] for e in E]), np.array([cid[e['cls']] for e in E])) for sp, E in ev.items()}
    ci, t, y = arr['train']; C = len(classes)
    cnt = np.bincount(y, minlength=C).astype(np.float64)
    cw = (cnt.max() / np.maximum(cnt, 1)) ** a.cw_pow; cw = cw / (cw * cnt).sum() * cnt.sum()
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=dev), label_smoothing=0.05)
    model = Recognizer(2 * len(CH), C, c=a.width, p=a.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    nb = len(y) // a.bs; sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * nb, pct_start=0.05)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    best, log = -1, []
    dci, dt, dy = arr['dev']; lex = dy > 0
    for ep in range(a.epochs):
        model.train(); perm = rng.permutation(len(y)); tl = 0; t0 = time.time()
        for b in range(nb):
            j = perm[b * a.bs:(b + 1) * a.bs]
            sh = rng.integers(-JIT, JIT + 1, len(j))
            x = gather(Q, off, L, ci[j], t[j], shift=sh)
            sc = torch.empty(len(j), 1, 1, device=dev).uniform_(0.85, 1.15)
            x = x * sc + 0.01 * torch.randn_like(x)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = crit(model(feat(x)).float(), torch.as_tensor(y[j], device=dev))
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += loss.item()
        model.eval()
        P = predict(model, feat, Q, off, L, dci, dt)
        m = metrics(P[lex], dy[lex], classes, none_P=P[~lex])
        log.append(dict(ep=ep, loss=tl / nb, dev_top1=m['top1'], dev_bal=m['bal_top1'], dev_det_auc=m.get('det_auc'), s=round(time.time() - t0, 1)))
        print(json.dumps(log[-1]), flush=True)
        if m['bal_top1'] > best:
            best = m['bal_top1']
            torch.save(dict(model=model.state_dict(), classes=classes, std=std.tolist(), arch=dict(c=a.width, p=a.dropout), epoch=ep,
                            ch=CH, W=W, fps=FPS, args=vars(a)), out / 'model.pt')
    (out / 'TRAIN_LOG.json').write_text(json.dumps(dict(log=log, args=vars(a), cw=cw.tolist(), train_counts=cnt.tolist()), indent=1))


def load_model(ck, dev='cuda'):
    s = torch.load(ck, map_location='cpu', weights_only=False)
    assert s['ch'] == CH and s['fps'] == FPS and s['W'] == W, 'recogniser / dataset motion space differ'
    m = Recognizer(2 * len(CH), len(s['classes']), **s['arch']).to(dev); m.load_state_dict(s['model']); m.eval()
    return m, Feat(np.zeros(len(CH)), np.array(s['std'], np.float32), dev), s


def score(a):
    dev = 'cuda'; rng = np.random.default_rng(0)
    D = load_events(a.data); configure(D['meta']); classes = D['meta']['classes']; cid = {c: i for i, c in enumerate(classes)}
    model, feat, s = load_model(a.ckpt, dev)
    q, stems, off, L, idx = clip_table(a.data)
    E = [e for e in D['events'] if e['split'] == a.split and (a.lang == 'all' or e['lang'] == a.lang)]
    need = sorted({e['stem'] for e in E}); bad = {}
    if a.pt_dir:
        gl = {st: int(L[idx[st]]) for st in need}
        q, idx, off, L, bad, real = load_source(a.pt_dir, need, gl, a.npy_pattern, D['meta'].get('dev_ci'))
        n0 = len(E); E = [e for e in E if e['t'] + HALF <= real[e['stem']] / FPS]      # window must lie inside the generated span
        bad['events_beyond_generated_span'] = n0 - len(E)
    elif a.gen_len_from:                   # ground truth restricted to the same spans as a generator export dir
        real = {st: len(np.load(Path(a.gen_len_from) / a.npy_pattern.format(ci=D['meta']['dev_ci'][st]), mmap_mode='r')) for st in need}
        n0 = len(E); E = [e for e in E if e['t'] + HALF <= real[e['stem']] / FPS]; bad = {'events_beyond_generated_span': n0 - len(E)}
    Q = torch.tensor(q, device=dev)
    for e in E: e['_ci'] = idx[e['stem']]
    ci = np.array([e['_ci'] for e in E]); t = np.array([e['t'] for e in E]); y = np.array([cid[e['cls']] for e in E])
    spk = np.array([e['spk'] for e in E])
    P = predict(model, feat, Q, off, L, ci, t)
    lex = y > 0
    tt = other_times([e for e, l in zip(E, lex) if l], D['words'], L, np.random.default_rng(1))
    Pt = np.full((lex.sum(), len(classes)), np.nan)
    ok = np.isfinite(tt)
    Pt[ok] = predict(model, feat, Q, off, L, ci[lex][ok], tt[ok])
    r = metrics(P[lex], y[lex], classes, spk=spk[lex], Ptime=Pt, rng=rng, none_P=P[~lex])
    r.update(source=a.name, pt_dir=a.pt_dir, split=a.split, lang=a.lang, ckpt=a.ckpt, n_stems=len(need), load_issues=dict(bad),
             recogniser_epoch=s['epoch'])
    pc = P[lex].copy(); pc[:, 0] = -1
    r['pred_hist'] = Counter(classes[i] for i in pc.argmax(1)).most_common()
    if a.save_probs: np.savez(Path(a.out).with_suffix('.probs.npz'), P=P, Pt=Pt, y=y, lex=lex, key=np.array([f"{e['stem']}@{e['t']}" for e in E]))
    txt = json.dumps(r, indent=1, ensure_ascii=False)
    if a.out: Path(a.out).write_text(txt)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ('per_class', 'pred_hist')}))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train'); t.add_argument('--data', required=True); t.add_argument('--out', required=True)
    t.add_argument('--epochs', type=int, default=30); t.add_argument('--bs', type=int, default=256); t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--width', type=int, default=128); t.add_argument('--dropout', type=float, default=0.2); t.add_argument('--seed', type=int, default=0)
    t.add_argument('--cw-pow', type=float, default=0.5); t.add_argument('--lang', default='all', choices=('all', 'en', 'zh'))
    t.add_argument('--target', default='sem', choices=('sem', 'spk'))
    t.add_argument('--sel-frac', type=float, default=0.0, help='also move this fraction of train takes into the selection set')
    s = sub.add_parser('score'); s.add_argument('--data', required=True); s.add_argument('--ckpt', required=True); s.add_argument('--name', default='gt')
    s.add_argument('--pt-dir'); s.add_argument('--split', default='test'); s.add_argument('--lang', default='all', choices=('all', 'en', 'zh'))
    s.add_argument('--out'); s.add_argument('--save-probs', action='store_true')
    s.add_argument('--npy-pattern', help="EX002 exports, e.g. 'gen_Hinf_{ci}.npy' (ci from EVENTS meta dev_ci)")
    s.add_argument('--gen-len-from', help='GT only: keep the events inside the spans of this export dir (with --npy-pattern)')
    a = ap.parse_args()
    train(a) if a.cmd == 'train' else score(a)


if __name__ == '__main__':
    main()

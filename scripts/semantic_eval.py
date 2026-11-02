"""Semantic-gesture recall of generated EX002 motion (ConvoFusion semantic recall / BEAT SRGR analogue).

Two sub-commands.

  build-manifest  pick N held-out semsynth clips (recogniser TEST split, but from the generator TRAIN view: there are no
                  semsynth clips in the val view) with >= 1 intent event of a recognised class, class-stratified; writes
                  <out>/MANIFEST.json, cond_<ci>.npz (WindowDataset.sample(ci, 0) of the whole clip, nch*16 frames, as
                  scmg/evaluate.py / scripts/text_probe.py build it) and gt_<ci>.npy (ground-truth q, rad).
      python scripts/semantic_eval.py build-manifest --out <dir> --n 200
  score           recognise the intended class in each event window of gen_<ci>.npy ([T, 34] rad, 50 Hz, frame 0 =
                  manifest start_frame) and report top-1 / top-5 recall of the intended class (none masked), class-
                  balanced top-1, detection rate (1 - p_none >= threshold calibrated at 10 % FPR on real negatives),
                  false-positive rate on the clip's no-intent windows, and a SHUFFLED-INTENT baseline (labels permuted
                  across events): recall - shuffled = how much the motion follows THIS clip's text rather than the class prior.
      python scripts/semantic_eval.py score --gen-dir <dir with gen_<ci>.npy> --manifest <dir>/MANIFEST.json --out res.json
      (--prefix gt_ --gen-dir <manifest dir> scores the ground truth of the same clips = ceiling)

Window: 160 frames (3.2 s) centred on the ground-truth gesture core (t_on/t_off); --search S also takes the best
centre within +-S s (tolerates generators that place the gesture slightly early / late).
"""
import argparse, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_intent_recognizer import load_recognizer, W

REC = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/semantic-recognizer-v1'
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'


def _rec_file(name, rec):
    loc = Path(os.path.expanduser('~/bulk/semrec-cache/final')) / name
    return str(loc) if loc.exists() else str(Path(rec) / name)


def windows(q, centres):
    """q [T, 34] rad -> [len(centres), W, 34]; frames c-W/2 .. c+W/2-1, edge-held outside [0, T)."""
    T = len(q)
    idx = np.clip(np.asarray(centres)[:, None] + np.arange(-W // 2, W // 2)[None], 0, T - 1)
    return q[idx]


def build_manifest(a):
    from scmg.common import CHUNK_FRAMES, P
    from scmg.data import View, WindowDataset, VIEW, clip_lm_capacity
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    I = json.load(open(_rec_file('INDEX.json', a.rec)))
    classes = I['meta']['classes']; cidx = {k: i for i, k in enumerate(classes)}
    st = json.load(open(P(D + '/STATS_v36_1.json')))
    v = View(VIEW, 'train', D + '/text-e5-v36_1', st, mmap=True, tok_cache=D + '/tok-qwen25-v36_1'); v.glitch = None
    by_id = {}
    for ci, c in enumerate(v.clips):
        for it in c['items']: by_id[it['id']] = ci
    # held-out semsynth clips with recognisable events
    pos = defaultdict(list)
    for r in I['rows']:
        if r['split'] == 'test' and r['kind'] == 'pos': pos[r['clip']].append(r)
    rng = np.random.default_rng(a.seed)
    per_cls = defaultdict(list)
    for cid, rs in pos.items():
        rare = max(rs, key=lambda r: cidx[r['key']])['key']        # the clip's rarest class (classes sorted by frequency)
        per_cls[rare].append(cid)
    for k in per_cls: rng.shuffle(per_cls[k])
    order = sorted(per_cls, key=lambda k: cidx[k], reverse=True)
    chosen = []
    while len(chosen) < a.n and any(per_cls[k] for k in order):  # round-robin over classes, rarest first
        for k in order:
            if per_cls[k] and len(chosen) < a.n: chosen.append(per_cls[k].pop())
    entries = []
    for cid in chosen:
        info = I['clips'][cid]; ci = by_id[info['items'][0]]
        c = v.clips[ci]; L = c['frames']; nch = L // CHUNK_FRAMES; T = nch * CHUNK_FRAMES
        assert c['start'] == info['a'] and L == info['L'], (cid, c['start'], info['a'])
        s = WindowDataset(v, T, max_tokens=256, max_ids=clip_lm_capacity(v, ci)).sample(ci, 0, np.random.default_rng(ci))
        np.savez(out / f'cond_{ci}.npz', **{k: np.asarray(x) for k, x in s.items()})
        gt = np.asarray(v.q[c['start']:c['start'] + T], np.float32); np.save(out / f'gt_{ci}.npy', gt)
        evs = [dict(key=r['key'], cls=cidx[r['key']], t_on=r['t_on'], t_off=r['t_off'], centre_frame=r['c'],
                    tok_item=r.get('tok_item'), tok_pos=r.get('tok_pos')) for r in sorted(pos[cid], key=lambda r: r['c']) if r['c'] < T]
        negs = [r['c'] for r in I['rows'] if r['clip'] == cid and r['kind'] == 'neg_syn' and r['c'] < T]
        entries.append(dict(file=f'gen_{ci}.npy', ci=ci, clip_id=cid, item_ids=info['items'], start_frame=0, frames=T,
                            chunks=nch, text=info['text'], rhythm_source=info.get('rhythm'), events=evs, neg_centres=negs,
                            other_events=[e for e in info['events'] if e['key'] not in cidx]))
    man = dict(version='semantic-eval-manifest-v1', recognizer=a.rec, view=VIEW, view_split='train', fps=50, window_frames=W,
               classes=classes, note=('clips are in the recogniser TEST split but in the generator TRAIN view (no semsynth '
                                      'clip exists in the val view): this measures text-sensitivity, not generalisation'),
               cond_keys_text_probe=['text', 'ttime', 'tmask', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c',
                                     'spk', 'spk_valid', 'source', 'ref', 'ref_valid'],
               text_causal_env=os.environ.get('SCMG_TEXT_CAUSAL'), seed=a.seed, entries=entries)
    (out / 'MANIFEST.json').write_text(json.dumps(man, ensure_ascii=False, indent=1, default=int))
    cnt = Counter(e['key'] for x in entries for e in x['events'])
    print(json.dumps(dict(clips=len(entries), events=sum(cnt.values()), classes=len(cnt), neg=sum(len(x['neg_centres']) for x in entries),
                          per_class=cnt.most_common()), ensure_ascii=False))


def score(a):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m, fz, s = load_recognizer(a.ckpt or _rec_file('model.pt', a.rec), dev)
    classes = s['classes']; thr = s['det_threshold'] if a.threshold is None else a.threshold
    man = json.load(open(a.manifest)); assert man['classes'] == classes, 'manifest / recogniser class tables differ'
    gd = Path(a.gen_dir); S = int(round(a.search * 50))
    shifts = list(range(-S, S + 1, 5)) if S > 0 else [0]
    ev_rows, neg_scores, missing = [], [], []
    for e in man['entries']:
        f = gd / e['file'].replace('gen_', a.prefix, 1)
        if not f.exists(): missing.append(e['file']); continue
        q = np.load(f).astype(np.float32)
        assert q.ndim == 2 and q.shape[1] == 34, (f, q.shape)
        st = e.get('start_frame', 0)
        cs = [ev['centre_frame'] - st for ev in e['events']]
        if not cs: continue
        cand = np.array([[c + sh for sh in shifts] for c in cs])               # [E, S]
        x = torch.as_tensor(windows(q, cand.reshape(-1)), device=dev)
        with torch.no_grad():
            p = torch.softmax(m(fz(x)).float(), -1).cpu().numpy().reshape(len(cs), len(shifts), -1)
        for k, ev in enumerate(e['events']):
            pk = p[k]; y = ev['cls']
            j = int(pk[:, y].argmax()) if S > 0 else 0                          # best centre for the intended class
            pr = pk[j].copy(); pr0 = pk[shifts.index(0)] if 0 in shifts else pk[0]
            ev_rows.append(dict(file=e['file'], key=ev['key'], y=y, prob=pr, prob_centre=pr0, shift_s=shifts[j] / 50, all=pk))
        ncs = [c - st for c in e.get('neg_centres', []) if 0 <= c - st < len(q)]
        if ncs:
            with torch.no_grad():
                pn = torch.softmax(m(fz(torch.as_tensor(windows(q, np.array(ncs)), device=dev))).float(), -1).cpu().numpy()
            neg_scores += (1 - pn[:, 0]).tolist()
    if not ev_rows: raise SystemExit(f'no generated files found in {gd} (prefix {a.prefix}); missing {len(missing)}')
    P = np.stack([r['prob'] for r in ev_rows]); P0 = np.stack([r['prob_centre'] for r in ev_rows]); Y = np.array([r['y'] for r in ev_rows])

    def recall(P, Y):
        pm = P.copy(); pm[:, 0] = -1
        rank = (pm > pm[np.arange(len(Y)), Y][:, None]).sum(1)
        t1 = rank == 0
        per = {classes[c]: float(t1[Y == c].mean()) for c in sorted(set(Y.tolist()))}
        return dict(top1=float(t1.mean()), top5=float((rank < 5).mean()), balanced_top1=float(np.mean(list(per.values()))),
                    detection_rate=float(((1 - P[:, 0]) >= thr).mean()), per_class_top1=per)
    res = dict(gen_dir=str(gd), prefix=a.prefix, n_files=len(man['entries']) - len(missing), missing=len(missing),
               n_events=len(Y), det_threshold=thr, search_s=a.search)
    res['centre'] = recall(P0, Y)
    if S > 0: res['search'] = recall(P, Y)
    rng = np.random.default_rng(0); sh = []
    for _ in range(50):                                   # shuffled-intent baseline: same windows, labels permuted
        sh.append(recall(P0, rng.permutation(Y))['top1'])
    res['centre']['shuffled_top1'] = float(np.mean(sh))
    res['centre']['text_following'] = res['centre']['top1'] - res['centre']['shuffled_top1']
    if S > 0:                                             # search variant of the baseline: best shift for the permuted label
        PA = np.stack([r['all'] for r in ev_rows])        # [E, S, C]
        def search_pick(Yp):
            j = PA[np.arange(len(Yp)), :, Yp].argmax(1); return PA[np.arange(len(Yp)), j]
        res['search']['shuffled_top1'] = float(np.mean([recall(search_pick(yp), yp)['top1'] for yp in (rng.permutation(Y) for _ in range(20))]))
        res['search']['text_following'] = res['search']['top1'] - res['search']['shuffled_top1']
    res['false_positive_rate'] = float(np.mean(np.array(neg_scores) >= thr)) if neg_scores else None
    res['n_neg_windows'] = len(neg_scores)
    pc = P0.copy(); pc[:, 0] = -1
    res['predicted_class_hist'] = Counter(classes[i] for i in pc.argmax(1)).most_common(12)
    res['intended_class_hist'] = Counter(classes[i] for i in Y).most_common(12)
    if a.per_event:
        res['events'] = [dict(file=r['file'], key=r['key'], p_intended=float(r['prob_centre'][r['y']]),
                              pred=classes[int(np.argmax(np.r_[-1, r['prob_centre'][1:]]))], p_none=float(r['prob_centre'][0]),
                              best_shift_s=r['shift_s']) for r in ev_rows]
    txt = json.dumps(res, indent=1, ensure_ascii=False)
    if a.out: Path(a.out).write_text(txt)
    c = res['centre']
    print(json.dumps(dict(n_events=len(Y), top1=round(c['top1'], 4), top5=round(c['top5'], 4), balanced_top1=round(c['balanced_top1'], 4),
                          shuffled_top1=round(c['shuffled_top1'], 4), text_following=round(c['text_following'], 4),
                          detection_rate=round(c['detection_rate'], 4), fpr=res['false_positive_rate'],
                          search_top1=round(res['search']['top1'], 4) if S > 0 else None, missing=len(missing))))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build-manifest'); b.add_argument('--out', required=True); b.add_argument('--n', type=int, default=200)
    b.add_argument('--seed', type=int, default=0); b.add_argument('--rec', default=REC)
    s = sub.add_parser('score'); s.add_argument('--gen-dir', required=True); s.add_argument('--manifest', required=True)
    s.add_argument('--out'); s.add_argument('--rec', default=REC); s.add_argument('--ckpt'); s.add_argument('--prefix', default='gen_')
    s.add_argument('--search', type=float, default=0.5); s.add_argument('--threshold', type=float); s.add_argument('--per-event', action='store_true')
    a = ap.parse_args()
    build_manifest(a) if a.cmd == 'build-manifest' else score(a)


if __name__ == '__main__':
    main()

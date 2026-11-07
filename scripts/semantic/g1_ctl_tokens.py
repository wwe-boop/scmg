"""Control-token labels for G1 (11-05): group 0 = one token per BEAT2 human-annotated deictic / iconic / metaphoric / beat
segment, values measured on the ground-truth motion with the official FK; group 1 = one token per g1_lexicon word (category,
at the word onset, no values: what a lexicon inserts from text alone). Definitions: scmg/ctl.py. Train AND dev (val_ids)
items of the view; the view's test/ (ECHO-G val_common) is never read. Amplitude bin edges = tertiles of the TRAIN tokens,
per class.
    python scripts/semantic/g1_ctl_tokens.py --view <hos view> --out <dir> [--sem <BEAT2 sem_en dir>]
Outputs CTL.json {item id: [token, ...]} (token: type, cls, level, t, a, b, apex, hand, target, wrist, ray, dist, amp,
amp_bin, kw, t_src, split) and PREP.json (split sha256, edges, counts, keyword-anchor rate).
"""
import argparse, hashlib, json, sys, zlib
from collections import Counter
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from scmg import ctl
from g1_build_sem import read_sem, SEM
from g1_lexicon import words, load_align

FPS = 50
MIN_SPAN = 0.3            # s: segment part inside the clip


def token_time(ws, g, kw):
    """keyword occurrence nearest to the segment middle within [a - 1, b + 1]; else first word with onset in [a - 0.2, b];
    else a. ws = [(word, cat, onset, offset)]."""
    mid = 0.5 * (g['a'] + g['b'])
    if kw:
        c = [w for w in ws if w[0].lower().strip("'") == kw.strip("'") and g['a'] - 1 <= w[2] <= g['b'] + 1]
        if c: return min(c, key=lambda w: abs(w[2] - mid))[2], 'keyword'
    c = [w for w in ws if g['a'] - 0.2 <= w[2] <= g['b']]
    if c: return min(c, key=lambda w: w[2])[2], 'first_word'
    return g['a'], 'segment_start'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--sem', default=SEM); ap.add_argument('--align-view', default=None, help='view with alignments/ (default --view)')
    a = ap.parse_args(); torch.set_num_threads(4)
    v = Path(a.view); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); av = Path(a.align_view or a.view)
    sp = json.loads((v / 'SPLIT.json').read_text()); tr, va = set(sp['train_ids']), set(sp['val_ids'])
    items = [it for it in json.loads((v / 'train.json').read_text()) if it['id'] in tr or it['id'] in va]
    q = np.load(v / 'train_q.npy', mmap_mode='r')
    sem = {p.stem: read_sem(p) for p in Path(a.sem).glob('*.txt')}
    raw, st = {}, Counter()
    for n_it, it in enumerate(items):
        rec = it['echog']['recording']; split = 'train' if it['id'] in tr else 'dev'
        s0 = float(it['echog']['start_sec']); n = int(it['frames']); dur = n / FPS
        ws = words(it['text'], load_align(av, it['id']), it['lang'])
        toks = [dict(type=ctl.TYPE_ID['lex_' + w[1]], cls='lex_' + w[1], t=round(w[2], 3), a=round(w[2], 3), b=round(w[3], 3),
                     word=w[0], split=split) for w in ws if w[1] != 'none' and 0 <= w[2] < dur]
        st[f'lex_{split}'] += len(toks)
        if rec not in sem:
            st[f'no_sem_{split}'] += 1
            if toks: raw[it['id']] = toks
            continue
        segs = []
        for g in sem[rec]:
            ga, gb = max(0., g['a'] - s0), min(dur, g['b'] - s0)
            if gb - ga >= MIN_SPAN: segs.append(dict(g, a=ga, b=gb))
        geo = ctl.geometry_np(q[it['offset']:it['offset'] + n, 10:39]) if segs else None
        for k, g in enumerate(segs):
            m = ctl.measure(geo, round(g['a'] * FPS), round(g['b'] * FPS))
            if m is None: continue
            ty = ctl.TYPES[g['kind']]; t, src = token_time(ws, g, g['kw'])
            tok = dict(type=ty, cls=g['kind'], level=g['level'], t=round(float(t), 3), a=round(g['a'], 3), b=round(g['b'], 3),
                       apex=round(m['apex'] / FPS, 3), kw=g['kw'], t_src=src, split=split,
                       amp=m['amp_hand'] if ty == ctl.POINT else m['amp_max'])
            if ty == ctl.POINT:
                d = float(np.random.default_rng(zlib.crc32(f"{it['id']}:{k}".encode())).uniform(*ctl.DIST))
                tok.update(hand=m['hand'], wrist=[round(x, 4) for x in m['wrist']], ray=[round(x, 4) for x in m['ray']], dist=round(d, 3),
                           target=[round(w + d * r, 4) for w, r in zip(m['wrist'], m['ray'])])
            toks.append(tok); st[f'{g["kind"]}_{split}'] += 1; st[f't_src_{src}'] += 1
        if toks: raw[it['id']] = toks
        if n_it % 2000 == 0: print(n_it, len(items), dict(st), flush=True)
    edges = {}
    for c in ctl.SEM_CLASSES:
        ty = ctl.TYPE_ID[c]
        x = [t['amp'] for ts in raw.values() for t in ts if t['type'] == ty and t['split'] == 'train']
        edges[ty] = [float(e) for e in np.quantile(x, [1 / 3, 2 / 3])]
    for ts in raw.values():
        for t in ts:
            if 'amp' in t: t['amp_bin'] = ctl.amp_bin(t['amp'], edges[t['type']]); t['amp'] = round(t['amp'], 4)
    (out / 'CTL.json').write_text(json.dumps(raw))
    bins = {f'{s}_{c}': Counter(t['amp_bin'] for ts in raw.values() for t in ts if t['split'] == s and t['cls'] == c)
            for s in ('train', 'dev') for c in ctl.SEM_CLASSES}
    prep = dict(view=str(v), split_sha256=hashlib.sha256((v / 'SPLIT.json').read_bytes()).hexdigest(), sem=a.sem,
                amp_edges={ctl.TYPE_NAMES[k]: e for k, e in edges.items()}, type_names=ctl.TYPE_NAMES, box=ctl.BOX, dist=ctl.DIST, stats=dict(st),
                items=dict(train=sum(1 for k in raw if k in tr), dev=sum(1 for k in raw if k in va)), bins={k: dict(v_) for k, v_ in bins.items()},
                definitions='scmg/ctl.py')
    (out / 'PREP.json').write_text(json.dumps(prep, indent=1, default=int)); print(json.dumps(prep, indent=1, default=int))


if __name__ == '__main__':
    main()

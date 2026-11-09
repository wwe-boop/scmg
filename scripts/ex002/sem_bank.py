"""P2 sem_cf bank + tokens + oracle dev events + metric validation (11-09; exp/text-path docs/criteria/P2_SEMCF_1109.md §1-2).
Events: EVENTS.json (semantic-baseline-1105/ex002-sem/data), cls deictic / iconic / metaphoric; split train -> bank + training
tokens, split test (= the 43 dev recordings of data-v36_1 val) -> oracle dev events. Recording time = the items of a recording
concatenated in offset order (View clips). Window W = [t - 0.6, t + 0.6] (scmg/sem_desc.py), hand = larger wrist-path diameter in W.
Token time = onset of the keyword's aligned occurrence nearest to t within t +- 1 s, minus 0.2 s; no occurrence: t - 0.2 s.
Word onsets: data-v36_1 alignments/<item dir>/alignment.json through scripts/semantic/g1_lexicon.words (item time + item offset).
Outputs (OUT): SEM_BANK.npz (raw [N, 16], hand, cls, kw, cat, stem, t, t_tok), SEM_BANK.json (mu / sd of the 13 scalars over the
bank, counts), tokens/CTL.json + PREP.json (train items; type SEM; t / a / b item-relative; hand; desc = standardised D),
DEV_EVENTS.json (oracle: ci, t, t_tok, hand, desc, kw, cls), METRIC_VALIDATION.json (V1 / V2 / shifted).
    SCMG_J=34 python scripts/ex002/sem_bank.py --out OUT"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'semantic'))
from scmg import sem_desc as SD
from scmg.ctl import SEM
from g1_lexicon import words as lex_words, EN_MAP

B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; C = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'
V = C + '/train-v1-20261101/views/data-v36_1'; EV = B + '/semantic-baseline-1105/ex002-sem/data/EVENTS.json'
import os
V = os.environ.get('SCMG_EXPL_VIEW', V)   # 11-09 (default unset = data-v36_1): the test-only view of EVAL_PROTOCOL v1
FPS = 50; CLS = ('deictic', 'iconic', 'metaphoric')


def recordings(items, ids):
    rec = {}
    for x in items:
        if x['id'] in ids: rec.setdefault(x['clip'], []).append(x)
    return {k: sorted(v, key=lambda x: x['offset']) for k, v in rec.items()}


def rec_words(its):
    """recording-relative (word, cat, onset, offset) of all items; item of each word."""
    out = []; f0 = 0
    for it in its:
        try: al = json.loads((Path(V) / it['dir'] / 'alignment.json').read_text())
        except FileNotFoundError: al = []
        for w, c, a, b in lex_words(it['text'], al, it['lang']): out.append((w.lower().strip("'"), c, a + f0 / FPS, b + f0 / FPS, it['id'], f0))
        f0 += it['frames']
    return out


def tok_time(e, ws):
    kw = (e.get('kw') or '').lower().strip("'")
    cand = [w for w in ws if kw and w[0] == kw and abs(w[2] - e['t']) <= 1.0]
    if cand: return min(cand, key=lambda w: abs(w[2] - e['t']))[2] - 0.2, True
    return e['t'] - 0.2, False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args(); O = Path(a.out); (O / 'tokens').mkdir(parents=True, exist_ok=True)
    items = json.loads((Path(V) / 'train.json').read_text()); sp = json.loads((Path(V) / 'SPLIT.json').read_text())
    tr, va = set(sp['train_ids']), set(sp['val_ids']); q = np.load(Path(V) / 'train_q.npy', mmap_mode='r')
    D = json.loads(Path(EV).read_text()); dev_ci = D['meta']['dev_ci']
    recs = {'train': recordings(items, tr), 'test': recordings(items, va)}
    rows = {'train': [], 'test': []}; nokw = {'train': 0, 'test': 0}; skip = 0
    for e in D['events']:
        if e['cls'] not in CLS: continue
        its = recs[e['split']].get(e['stem'])
        if not its: skip += 1; continue
        Q = np.concatenate([np.asarray(q[x['offset']:x['offset'] + x['frames']], np.float32) for x in its])
        c = int(round(e['t'] * FPS))
        if c - SD.NW // 2 < 0 or c + SD.NW // 2 >= len(Q): skip += 1; continue
        w = SD.window(Q, e['t']); h = SD.dominant_hand(w); ws = rec_words(its); tt, hit = tok_time(e, ws)
        nokw[e['split']] += int(not hit)
        kw = (e.get('kw') or '').lower().strip("'")
        rows[e['split']].append(dict(stem=e['stem'], cls=e['cls'], kw=kw, cat=EN_MAP.get(kw, 'none'), t=float(e['t']), t_tok=float(tt), hand=h,
                                     raw=SD.np_desc(w, h).tolist(), its=[(x['id'], x['frames']) for x in its], ci=dev_ci.get(e['stem'])))
    R = np.array([r['raw'] for r in rows['train']], np.float32); mu = R[:, :SD.NS].mean(0); sd = R[:, :SD.NS].std(0) + 1e-6
    std = lambda raw: np.concatenate([(np.asarray(raw[:SD.NS]) - mu) / sd, raw[SD.NS:]]).astype(np.float32)
    np.savez(O / 'SEM_BANK.npz', raw=R, D=np.stack([std(r['raw']) for r in rows['train']]), hand=np.array([r['hand'] for r in rows['train']], np.int8),
             cls=np.array([r['cls'] for r in rows['train']]), kw=np.array([r['kw'] for r in rows['train']]), cat=np.array([r['cat'] for r in rows['train']]),
             stem=np.array([r['stem'] for r in rows['train']]), t=np.array([r['t'] for r in rows['train']]), t_tok=np.array([r['t_tok'] for r in rows['train']]))
    meta = dict(mu=mu.tolist(), sd=sd.tolist(), names=SD.NAMES, n_train=len(rows['train']), n_dev=len(rows['test']), skipped=skip,
                no_keyword_occurrence=nokw, rule=__doc__.split('\n')[0])
    (O / 'SEM_BANK.json').write_text(json.dumps(meta, indent=1))
    # training tokens (train items): the item that contains the token time; a / b = W
    ctl = {}
    for r in rows['train']:
        f0 = 0
        for iid, fr in r['its']:
            if f0 / FPS <= r['t_tok'] < (f0 + fr) / FPS:
                d0 = f0 / FPS
                ctl.setdefault(iid, []).append(dict(type=SEM, t=r['t_tok'] - d0, a=r['t'] - SD.HALF - d0, b=r['t'] + SD.HALF - d0, hand=r['hand'],
                                                    desc=std(r['raw']).tolist(), split='train', cls=r['cls'], kw=r['kw']))
                break
            f0 += fr
    (O / 'tokens' / 'CTL.json').write_text(json.dumps(ctl))
    (O / 'tokens' / 'PREP.json').write_text(json.dumps(dict(split_sha256=hashlib.sha256((Path(V) / 'SPLIT.json').read_bytes()).hexdigest(),
                                                             n_items=len(ctl), n_tokens=sum(len(v) for v in ctl.values()), rule='sem_bank.py')))
    dev = [dict(ci=r['ci'], stem=r['stem'], cls=r['cls'], kw=r['kw'], cat=r['cat'], t=r['t'], t_tok=r['t_tok'], hand=r['hand'], desc=std(r['raw']).tolist())
           for r in rows['test']]
    (O / 'DEV_EVENTS.json').write_text(json.dumps(dict(events=dev, rule='oracle dev events (P2_SEMCF_1109.md §2)')))
    # metric validation on the dev GT (V1 own = 0, V2 mismatched, shifted +-1 s)
    Dd = np.stack([e['desc'] for e in dev]); rng = np.random.default_rng(20261109); p = rng.permutation(len(dev))
    for _ in range(100):
        bad = [k for k in range(len(dev)) if dev[p[k]]['stem'] == dev[k]['stem']]
        if not bad: break
        for k in bad: j = int(rng.integers(len(dev))); p[k], p[j] = p[j], p[k]
    E = lambda x, y: float(np.mean(np.sum((x - y) ** 2, -1) / SD.ND))
    sh = []
    for r, e in zip(rows['test'], dev):
        its = recs['test'][r['stem']]; Q = np.concatenate([np.asarray(q[x['offset']:x['offset'] + x['frames']], np.float32) for x in its])
        for s in (-1.0, 1.0):
            if 0 <= (r['t'] + s) * FPS - SD.NW // 2 and (r['t'] + s) * FPS + SD.NW // 2 < len(Q):
                sh.append(np.sum((std(SD.np_desc(SD.window(Q, r['t'] + s), r['hand'])) - np.asarray(e['desc'])) ** 2) / SD.ND)
    val = dict(V1_own=E(Dd, Dd), V2_mismatched=E(Dd, Dd[p]), shifted_1s=float(np.mean(sh)), n=len(dev), n_shifted=len(sh),
               same_recording_after_derangement=int(sum(dev[p[k]]['stem'] == dev[k]['stem'] for k in range(len(dev)))))
    (O / 'METRIC_VALIDATION.json').write_text(json.dumps(val, indent=1)); print(json.dumps(meta | val, indent=1)[:3000])


if __name__ == '__main__':
    main()

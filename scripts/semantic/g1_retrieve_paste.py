"""'Retrieve and paste' semantic baseline (11-06, module 2 of SCMG_IMPROVE_DESIGN_1104 §7.42, no training): at every token of the
chosen dev token set, take a TRAIN-split BEAT2 annotated semantic segment (ground-truth G1 motion of a training speaker) and
paste its waist + arm channels onto a base prediction (cross-fade), then write evaluator-ready files.
  bank   : train annotation tokens of class deictic / iconic / metaphoric (CTL.json), keyed by (class, lexicon category of
           the annotated keyword); snippet = item frames [t_src - 0.6 s, b_src + 0.3 s] (t_src = the token time = its keyword)
  query  : --mode auto -> every dev lexical token whose category maps (scmg.ctl.LEX2SEM) to a semantic class; --mode gt -> the
           dev annotation tokens (oracle class + time). Candidate = same (class, keyword category), else same class; seeded
           choice per (stem, token). Greedy in time: a paste never overlaps the previous one (+ 0.2 s gap).
  paste  : snippet frame of t_src aligned to the query time; channels 22..38 (waist + both arms); blend weight ramps 0 -> 1 -> 0
           over 0.2 s at both ends; the snippet is offset so that its first frame matches the base pose (no jump).
Caveat (reported): the bank and the semantic recognisers both come from the train-split speakers (not speaker-disjoint) -- unless
--bank-split SPLIT_RK.json --bank-half K (11-07): bank from the K speakers only, scored with the R-half recognisers.
    python scripts/semantic/g1_retrieve_paste.py --ctl <dir> --view <hos view> --base <pred dir with q50> --out <dir> --mode auto
"""
import argparse, json, sys, zlib
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from scmg import ctl, g1
from g1_lexicon import EN_MAP

FPS = 50; CH = list(range(22, 39)); PRE, POST, FADE, GAP = 0.6, 0.3, 0.2, 0.2


def main():
    ap = argparse.ArgumentParser()
    for k in ('ctl', 'view', 'base', 'out'): ap.add_argument('--' + k, required=True)
    ap.add_argument('--mode', default='auto', choices=['auto', 'gt'])
    ap.add_argument('--bank-split', default=None, help="11-07: SPLIT_RK.json of g1_speaker_split.py; with --bank-half K the bank "
                    "holds only that half's speakers (score with the OTHER half's recognisers: speaker-disjoint bank / recogniser)")
    ap.add_argument('--bank-half', default='K'); a = ap.parse_args()
    keep_spk = set(json.loads(Path(a.bank_split).read_text())[a.bank_half]) if a.bank_split else None
    spk_of = lambda stem: '_'.join(stem.split('_')[1:3])
    raw = json.loads((Path(a.ctl) / 'CTL.json').read_text())
    items = {it['id']: it for it in json.loads((Path(a.view) / 'train.json').read_text())}
    q = np.load(Path(a.view) / 'train_q.npy', mmap_mode='r')
    sem = {ctl.TYPE_ID[c]: c for c in ('deictic', 'iconic', 'metaphoric')}
    bank = {}
    for s, ts in raw.items():
        if not ts or ts[0]['split'] != 'train': continue
        if keep_spk is not None and spk_of(s) not in keep_spk: continue
        rec = items[s]['echog']['recording']
        for t in ts:
            if t['type'] not in sem: continue
            cat = EN_MAP.get((t.get('kw') or '').strip("'"), 'none')
            bank.setdefault((sem[t['type']], cat), []).append((s, rec, t['t'], t['b']))
            bank.setdefault((sem[t['type']], '*'), []).append((s, rec, t['t'], t['b']))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); log = []
    for s, ts in sorted(raw.items()):
        if not ts or ts[0]['split'] != 'dev': continue
        f = Path(a.base) / f'{s}.q50.npy'
        if not f.exists(): continue
        y = np.load(f).astype(np.float64); T = len(y)
        if a.mode == 'auto':
            qs = [(t['t'], ctl.LEX2SEM[t['cls'][4:]], t['cls'][4:]) for t in ts if t['cls'].startswith('lex_') and t['cls'][4:] in ctl.LEX2SEM]
        else:
            qs = [(t['t'], sem[t['type']], EN_MAP.get((t.get('kw') or '').strip("'"), 'none')) for t in ts if t['type'] in sem]
        last_end = -1e9; n = 0
        for i, (tq, cls, cat) in enumerate(sorted(qs)):
            cand = bank.get((cls, cat)) or bank.get((cls, '*'))
            if not cand: continue
            src, rec, ts_, bs_ = cand[np.random.default_rng(zlib.crc32(f'{s}:{i}'.encode())).integers(len(cand))]
            it = items[src]; f0 = max(0, int(round((ts_ - PRE) * FPS))); f1 = min(it['frames'], int(round((bs_ + POST) * FPS)))
            snip = np.asarray(q[it['offset'] + f0:it['offset'] + f1, :], np.float64)[:, CH]
            d0 = int(round(tq * FPS)) - (int(round(ts_ * FPS)) - f0)          # target frame of the snippet's first frame
            if d0 < last_end + GAP * FPS: continue
            lo, hi = max(0, d0), min(T, d0 + len(snip))
            if hi - lo < FADE * FPS * 2: continue
            seg = snip[lo - d0:hi - d0]; seg = seg - seg[0] + y[lo, CH]          # start from the base pose
            w = np.ones(hi - lo); r = int(FADE * FPS); w[:r] = np.linspace(0, 1, r); w[-r:] = np.linspace(1, 0, r)
            y[lo:hi, CH] = (1 - w[:, None]) * y[lo:hi, CH] + w[:, None] * seg
            last_end = hi; n += 1
        b = torch.load(Path(a.base) / f'{s}.pt', map_location='cpu', weights_only=False)
        n30 = int(b['real_num_frames']); r30 = g1.to30(y, n30).astype(np.float32)
        gen = dict(b['generator'], retrieve_paste=dict(mode=a.mode, base=str(a.base), pastes=n, bank_split=a.bank_split,
                                                       bank_half=a.bank_half if a.bank_split else None))
        torch.save(dict(b, robot_repr=torch.from_numpy(r30), generator=gen), out / f'{s}.pt')
        np.save(out / f'{s}.q50.npy', y.astype(np.float32)); log.append(dict(stem=s, pastes=n, queries=len(qs)))
    (out / 'PASTE_LOG.json').write_text(json.dumps(dict(n_clips=len(log), pastes=sum(x['pastes'] for x in log),
                                                        queries=sum(x['queries'] for x in log), clips=log)))
    print('PASTE_DONE', len(log), sum(x['pastes'] for x in log))


if __name__ == '__main__':
    main()

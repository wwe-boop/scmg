"""EX002 semantic events from the BEAT2 human semantic-gesture annotation, on the 43-clip EX002 dev set (semantic-baseline-1105).

Same label source as g1_build_sem.py (beat_english_v2.0.0/sem: beat_align / deictic / iconic / metaphoric segments), mapped
onto the EX002 view (data-v36_1, 34 joints, rad, 50 Hz).  A view clip = speech items concatenated in offset order; item frame
i <-> take time item.clip_start_s + t0 + i/50 (t0 = 0 for beat2-en, NN*60 for beat2-enadd ...-wNN; scmg/beat2 sem prep rule).
  train = beat2_en / beat2_en_additional clips of the view TRAIN split that have a sem file (recogniser training only)
  dev   = the first 43 clips of the view VAL split = scmg.evaluate dev set (gen_H*_<ci>.npy / gt_<ci>.npy exports, ci = 0..42);
          only clips whose take has a sem file (English takes) contribute events.  Never in any generator's training.
Events: midpoint of each semantic segment (window [mid - 1.2 s, mid + 1.2 s] inside the clip and inside one item's time span);
background 'beat' = random times inside beat_align segments >= 1 s from every semantic segment (<= 3 per item).
    python scripts/semantic/ex002_build_sem.py --out ~/bulk/semantic-baseline-1105/ex002-sem/data
"""
import argparse, json, re, sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_build_sem import read_sem, SEM, CLASSES

VIEW = '/home/xr-25b7/bulk/cpfs-mirror/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1'
FPS, HALF, BEAT_GAP, BEAT_PER_ITEM, NDEV = 50, 1.2, 1.0, 3, 43


def rec_of(clip):
    m = re.match(r'beat2-enadd-(.+)-w(\d+)$', clip)
    if m: return m.group(1), 60.0 * int(m.group(2))
    m = re.match(r'beat2-en-(.+)$', clip)
    return (m.group(1), 0.0) if m else (None, None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--view', default=VIEW)
    ap.add_argument('--sem', default=SEM); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(a.seed)
    v = Path(a.view); items = json.loads((v / 'train.json').read_text()); sp = json.loads((v / 'SPLIT.json').read_text())
    tr, va = set(sp['train_ids']), set(sp['val_ids'])
    q = np.load(v / 'train_q.npy', mmap_mode='r')
    sem = {p.stem: read_sem(p) for p in Path(a.sem).glob('*.txt')}
    groups = {'train': OrderedDict(), 'dev': OrderedDict()}
    for it in items:                                   # scmg.data.View groups by 'clip' in item order
        if it['id'] in va: groups['dev'].setdefault(it['clip'], []).append(it)
        elif it['id'] in tr and it.get('data_source') in ('beat2_en', 'beat2_en_additional'): groups['train'].setdefault(it['clip'], []).append(it)
    dev_clips = list(groups['dev'])[:NDEV]
    st = Counter(); ev = []; split = {}; stems = []; Q = []; offs = []; lens = []; o = 0; dev_ci = {}; words = {}
    for spn in ('dev', 'train'):
        names = dev_clips if spn == 'dev' else list(groups['train'])
        for ci, c in enumerate(names):
            xs = groups[spn][c]; start = xs[0]['offset']; F = sum(x['frames'] for x in xs)
            if any(xs[i + 1]['offset'] != xs[i]['offset'] + xs[i]['frames'] for i in range(len(xs) - 1)): st['noncontig_' + spn] += 1; continue
            rec, t0 = rec_of(c)
            if spn == 'train' and rec not in sem: st['train_no_sem'] += 1; continue
            if spn == 'dev': dev_ci[c] = ci
            lab = 'test' if spn == 'dev' else 'train'      # dev-43 = the scored split ('test'); selection uses a train slice
            stems.append(c); split[c] = lab; Q.append(np.asarray(q[start:start + F], np.float32)); offs.append(o); lens.append(F); o += F
            words[c] = []
            if rec not in sem: st['dev_no_sem'] += 1; continue
            segs = sem[rec]; dur = F / FPS
            for it in xs:
                f0 = (it['offset'] - start) / FPS; ts = it['clip_start_s'] + t0; te = ts + it['frames'] / FPS
                loc = [dict(g, a=g['a'] - ts + f0, b=g['b'] - ts + f0) for g in segs if g['b'] > ts and g['a'] < te]
                ss = [g for g in loc if g['kind'] != 'beat']
                for g in ss:
                    mid = 0.5 * (g['a'] + g['b'])
                    if not (f0 <= mid <= f0 + it['frames'] / FPS) or mid - HALF < 0 or mid + HALF > dur: st['dropped_' + spn] += 1; continue
                    ev.append(dict(stem=c, split=lab, cls=g['kind'], level=g['level'], t=round(mid, 3), kw=g['kw'], spk=it['speaker'], lang='en'))
                cand = np.arange(max(HALF, f0), min(dur - HALF, f0 + it['frames'] / FPS) + 1e-6, 0.1)
                if len(cand):
                    inb = np.zeros(len(cand), bool)
                    for g in loc:
                        if g['kind'] == 'beat': inb |= (cand >= g['a']) & (cand <= g['b'])
                    for g in ss: inb &= (cand < g['a'] - BEAT_GAP) | (cand > g['b'] + BEAT_GAP)
                    for t in rng.permutation(cand[inb])[:BEAT_PER_ITEM]:
                        ev.append(dict(stem=c, split=lab, cls='beat', level=None, t=round(float(t), 3), spk=it['speaker'], lang='en'))
    np.savez(out / 'MOTION.npz', q=np.concatenate(Q), stems=np.array(stems), off=np.array(offs), len=np.array(lens))
    cnt = {s: Counter(e['cls'] for e in ev if e['split'] == s) for s in ('train', 'test')}
    meta = dict(version='ex002-beat2sem-v1', classes=CLASSES, fps=FPS, half_window_s=HALF, channels=list(range(34)), counts=cnt, stats=st,
                dev_ci=dev_ci, view=str(v), sem=a.sem)
    (out / 'EVENTS.json').write_text(json.dumps(dict(meta=meta, split=split, events=ev, words=words,
                                                     spk={e['stem']: e['spk'] for e in ev}), ensure_ascii=False))
    print(json.dumps(meta, indent=1, default=int)[:3000])


if __name__ == '__main__':
    main()

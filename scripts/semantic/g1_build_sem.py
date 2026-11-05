"""G1 semantic events from the BEAT2 human semantic-gesture annotation (beat_english_v2.0.0/sem, semantic-baseline-1105).

BEAT2 annotators segmented every take into beat_align / deictic / iconic / metaphoric (+ l/m/h relevance, keyword) gestures.
The ECHO-G G1 segments are utterance cuts of BEAT2 takes (echog.recording + start_sec/end_sec), so each annotation maps to
clip time t - start_sec.  Kept: semantic segments (deictic / iconic / metaphoric) whose midpoint window
[mid - 1.2 s, mid + 1.2 s] lies inside the clip; background 'beat' = random times inside beat_align segments >= 1 s from
every semantic segment (<= 3 per clip).  Keyword check: fraction of semantic events whose annotated keyword is one of the
clip's aligned words within the segment +- 1 s (validates the time mapping).
Reuses MOTION.npz of g1_build.py (same stems / splits).
    python scripts/semantic/g1_build_sem.py --data ~/bulk/semantic-baseline-1105/g1/data --out ~/bulk/semantic-baseline-1105/g1/data-sem
"""
import argparse, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

SEM = '/home/xr-25b7/bulk/dsh-beat2-sem/data/beat2/sem_en'   # HF haiyangliu0/BEAT2 beat_english_v2.0.0/sem (1,628 takes)
VIEW = '/home/xr-25b7/bulk/cpfs-mirror/dataset/zehan/gesture/echog-g1-v1'
FPS, HALF, BEAT_GAP, BEAT_PER_CLIP = 30, 1.2, 1.0, 3
CLASSES = ['beat', 'deictic', 'iconic', 'metaphoric']


def read_sem(p):
    out = []
    for line in Path(p).read_text(errors='replace').splitlines():
        f = line.split('\t')
        if len(f) < 4: continue
        m = re.match(r'\d+_(beat_align|deictic|iconic|metaphoric)(?:_([lmh]))?', f[0].strip())
        if not m: continue
        try: a, b = float(f[1]), float(f[2])
        except ValueError: continue
        out.append(dict(kind='beat' if m.group(1) == 'beat_align' else m.group(1), level=m.group(2), a=a, b=b,
                        kw=(f[5].strip().lower() if len(f) > 5 else '')))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--sem', default=SEM); ap.add_argument('--view', default=VIEW); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(a.seed)
    D = json.loads((Path(a.data).expanduser() / 'EVENTS.json').read_text())
    z = np.load(Path(a.data).expanduser() / 'MOTION.npz'); L = dict(zip([str(s) for s in z['stems']], z['len']))
    v = Path(a.view)
    meta = {x['id']: x for x in json.loads((v / 'train.json').read_text())}
    meta.update({x['id']: x for x in json.loads((v / 'test' / 'train.json').read_text())})
    sem = {p.stem: read_sem(p) for p in Path(a.sem).glob('*.txt')}
    st = Counter(); ev = []; kwhit = Counter()
    for s, sp in D['split'].items():
        it = meta[s]; rec = it['echog']['recording']; s0 = float(it['echog']['start_sec'])
        if rec not in sem: st[f'no_sem_{sp}'] += 1; continue
        st[f'clips_with_sem_{sp}'] += 1
        dur = L[s] / FPS; ws = D['words'][s]
        segs = [dict(g, a=g['a'] - s0, b=g['b'] - s0) for g in sem[rec]]
        segs = [g for g in segs if g['b'] > 0 and g['a'] < dur]
        semseg = [g for g in segs if g['kind'] != 'beat']
        for g in semseg:
            mid = 0.5 * (g['a'] + g['b'])
            if not (mid - HALF >= 0 and mid + HALF <= dur): st['edge_dropped'] += 1; continue
            ok = any(w[0].lower().strip("'") == g['kw'].strip("'") and w[2] <= g['b'] + 1 and w[3] >= g['a'] - 1 for w in ws) if g['kw'] else None
            if g['kw']: kwhit['hit' if ok else 'miss'] += 1
            ev.append(dict(stem=s, split=sp, cls=g['kind'], level=g['level'], t=round(mid, 3), a=round(g['a'], 3), b=round(g['b'], 3),
                           kw=g['kw'], kw_found=ok, lang=it['lang'], spk=it['speaker']))
        cand = np.arange(HALF, dur - HALF + 1e-6, 0.1)
        if len(cand):
            inb = np.zeros(len(cand), bool)
            for g in segs:
                if g['kind'] == 'beat': inb |= (cand >= g['a']) & (cand <= g['b'])
            for g in semseg: inb &= (cand < g['a'] - BEAT_GAP) | (cand > g['b'] + BEAT_GAP)
            for t in rng.permutation(cand[inb])[:BEAT_PER_CLIP]:
                ev.append(dict(stem=s, split=sp, cls='beat', level=None, t=round(float(t), 3), lang=it['lang'], spk=it['speaker']))
    cnt = {sp: Counter(e['cls'] for e in ev if e['split'] == sp) for sp in ('train', 'dev', 'test')}
    m = dict(version='g1-beat2sem-v1', classes=CLASSES, fps=FPS, half_window_s=HALF, counts=cnt, stats=st,
             keyword_check=dict(kwhit, frac=kwhit['hit'] / max(1, sum(kwhit.values()))), sem=a.sem,
             levels={sp: Counter(f"{e['cls']}_{e['level']}" for e in ev if e['split'] == sp and e['cls'] != 'beat') for sp in ('train', 'dev', 'test')})
    (out / 'EVENTS.json').write_text(json.dumps(dict(meta=dict(D['meta'], **m), split=D['split'], lang=D['lang'], spk=D['spk'],
                                                     events=ev, words=D['words']), ensure_ascii=False))
    mo = out / 'MOTION.npz'
    if not mo.exists(): mo.symlink_to(Path(a.data).expanduser().resolve() / 'MOTION.npz')
    print(json.dumps(m, indent=1, default=int))


if __name__ == '__main__':
    main()

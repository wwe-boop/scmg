"""Frame-level g_k labels for G1 training from the BEAT2 human semantic-gesture annotation (11-05, semantic head).

One int16 per row of <view>/train_q.npy (50 Hz), the format scmg.data.View(gest_frames=...) feeds as batch['gest']:
  -1 unknown (every row that is not a TRAIN item of THIS view's SPLIT.json, takes without annotation, need_cut / gaps)
   0 none (nogesture / habit segments)   1 deictic   2 iconic   3 metaphoric   4 beat (beat_align)   (5, 6 unused; n_sem = 7)
Item frame i <-> take time echog.start_sec + i / 50 of take echog.recording (the same mapping as g1_build_sem.py, verified
there: annotated keywords hit the aligned words 62 % at the true time vs 3 % 5 s off).  Only split train_ids get labels: the
view's val_ids (dev / held-out-speaker dev), the speakers listed in --exclude-speakers and the ECHO-G test set never do.
    python scripts/semantic/g1_gest_frames.py --view <hos view> --out <dir> [--exclude-speakers english_2_scott english_6_carla]
Outputs GEST_FRAMES.npy, PREP.json (rows, split sha256, class histogram, labelled items / speakers) and LABELLED_IDS.txt.
"""
import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_build_sem import read_sem, SEM

CLS = {'deictic': 1, 'iconic': 2, 'metaphoric': 3, 'beat': 4}
FPS = 50


def read_raw(p):
    """all annotation rows incl. nogesture / habit / need_cut: [(kind, a, b)]"""
    out = []
    for line in Path(p).read_text(errors='replace').splitlines():
        f = line.split('\t')
        if len(f) < 3: continue
        k = f[0].strip()
        try: a, b = float(f[1]), float(f[2])
        except ValueError: continue
        kind = next((c for c in ('beat_align', 'deictic', 'iconic', 'metaphoric', 'nogesture', 'habit', 'need_cut') if c in k), None)
        if kind: out.append((kind, a, b))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--sem', default=SEM); ap.add_argument('--exclude-speakers', nargs='*', default=[])
    ap.add_argument('--test-view', default=None, help='view whose items must never be labelled (default <view>/test)')
    a = ap.parse_args(); v = Path(a.view); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sp = json.loads((v / 'SPLIT.json').read_text()); tr = set(sp['train_ids']); val = set(sp['val_ids'])
    items = json.loads((v / 'train.json').read_text())
    tv = Path(a.test_view) if a.test_view else v / 'test'
    test_ids = {x['id'] for x in json.loads((tv / 'train.json').read_text())} if (tv / 'train.json').exists() else set()
    rows = int(np.load(v / 'train_q.npy', mmap_mode='r').shape[0])
    g = np.full(rows, -1, np.int16)
    sem = {p.stem: read_raw(p) for p in Path(a.sem).glob('*.txt')}
    excl = set(a.exclude_speakers); st = Counter(); labelled = []; spk = Counter()
    for it in items:
        i = it['id']
        if i not in tr: st['not_train'] += 1; continue
        if i in val or i in test_ids: raise SystemExit(f'{i} is in train_ids AND val/test')
        if '_'.join(i.split('_')[:3]) in excl or it.get('speaker') in excl: st['excluded_speaker'] += 1; continue
        rec = it['echog']['recording']
        if rec not in sem: st['no_annotation'] += 1; continue
        s0 = float(it['echog']['start_sec']); n = int(it['frames']); o = int(it['offset'])
        t = s0 + np.arange(n) / FPS; lab = np.full(n, -1, np.int16)
        for kind, a0, b0 in sem[rec]:
            m = (t >= a0) & (t < b0)
            if not m.any(): continue
            lab[m] = CLS.get('beat' if kind == 'beat_align' else kind, 0 if kind in ('nogesture', 'habit') else -1)
        g[o:o + n] = lab; labelled.append(i); spk[it['speaker']] += 1; st['labelled_items'] += 1
    hist = {int(k): int(c) for k, c in zip(*np.unique(g, return_counts=True))}
    np.save(out / 'GEST_FRAMES.npy', g)
    (out / 'LABELLED_IDS.txt').write_text('\n'.join(labelled) + '\n')
    prep = dict(view=str(v), rows=rows, split_sha256=hashlib.sha256((v / 'SPLIT.json').read_bytes()).hexdigest(),
                classes=dict(unknown=-1, none=0, deictic=1, iconic=2, metaphoric=3, beat=4), hist=hist, stats=st,
                labelled_speakers=dict(spk), exclude_speakers=sorted(excl), sem=a.sem, n_test_ids=len(test_ids))
    (out / 'PREP.json').write_text(json.dumps(prep, indent=1))
    print(json.dumps(prep, indent=1))


if __name__ == '__main__':
    main()

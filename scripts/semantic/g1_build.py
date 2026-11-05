"""Build the G1 semantic-recognition dataset (semantic-baseline-1105).

Motion: the OFFICIAL ECHO-G 30 fps 39-d ground truth (motion_39d_30fps tar, the same files ECHO-G's evaluator reads), so
train / dev / test ground truth and every generator output (30 fps .pt) share one frame rate and representation.
Splits (no stem in two splits):
  train = echog-g1-v1 SPLIT train_ids  (ECHO-G train_drop minus the 28-take dev holdout)
  dev   = echog-g1-v1 SPLIT val_ids    (306 segments, take-level holdout of train_drop; recogniser model selection only)
  test  = ECHO-G val_common            (3,242 segments, 3 unseen English speakers; never in our view, ECHO-G's test set)
Events: every lexicon word (g1_lexicon.py) whose 2.4 s window [onset - 1.2 s, onset + 1.2 s] lies inside the clip;
background ('none'): non-lexicon words >= 0.6 s from every lexicon word, same window rule, <= 3 per clip.
    python scripts/semantic/g1_build.py --out ~/bulk/semantic-baseline-1105/g1/data
Outputs: MOTION.npz (q float32 [N_frames, 39], per-stem offset / length), EVENTS.json (stems, splits, events, words).
"""
import argparse, io, json, sys, tarfile
from collections import Counter
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_lexicon import CLASSES, load_align, words

VIEW = '/home/xr-25b7/bulk/cpfs-mirror/dataset/zehan/gesture/echog-g1-v1'          # local mirror of the CPFS view
TAR = '/home/xr-25b7/bulk/echog-dataset/data/motion_39d_30fps-00000-of-00001.tar'   # official ECHO-G release
FPS, HALF, NONE_GAP, NONE_PER_CLIP = 30, 1.2, 0.6, 3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--view', default=VIEW)
    ap.add_argument('--tar', default=TAR); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    v = Path(a.view)
    sp = json.loads((v / 'SPLIT.json').read_text()); tr, dv = set(sp['train_ids']), set(sp['val_ids'])
    items = {x['id']: dict(x, _dir=v) for x in json.loads((v / 'train.json').read_text())}
    test = {x['id']: dict(x, _dir=v / 'test') for x in json.loads((v / 'test' / 'train.json').read_text())}
    assert not set(items) & set(test)
    split = {s: ('train' if s in tr else 'dev') for s in items if s in tr or s in dv}
    split.update({s: 'test' for s in test}); meta = {**items, **test}
    mot = {}
    with tarfile.open(a.tar) as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith('.pt'): continue
            s = Path(m.name).stem
            if s not in split: continue
            d = torch.load(io.BytesIO(tf.extractfile(m).read()), map_location='cpu', weights_only=False)
            q = d['robot_repr'] if isinstance(d, dict) else d
            mot[s] = np.asarray(q, np.float32)[:int(d.get('real_num_frames', len(q))) if isinstance(d, dict) else None]
    st = Counter(); stems = sorted(mot); offs, lens, ev, wl = [], [], [], {}
    o = 0
    for s in stems:
        it = meta[s]; L = len(mot[s]); offs.append(o); lens.append(L); o += L
        try:
            ws = words(it['text'], load_align(it['_dir'], s), it['lang'])
        except FileNotFoundError:
            st['no_alignment'] += 1; ws = []
        wl[s] = [(w, c, round(t0, 3), round(t1, 3)) for w, c, t0, t1 in ws]
        dur = L / FPS
        inside = lambda t0: t0 - HALF >= 0 and t0 + HALF <= dur
        lex = [x for x in ws if x[1] != 'none']
        for w, c, t0, t1 in lex:
            if inside(t0): ev.append(dict(stem=s, split=split[s], cls=c, word=w, t=t0, t_off=t1, lang=it['lang'], spk=it['speaker']))
            else: st['edge_dropped'] += 1
        lt = np.array([x[2] for x in lex]) if lex else np.zeros(0)
        cand = [x for x in ws if x[1] == 'none' and inside(x[2]) and (len(lt) == 0 or np.abs(lt - x[2]).min() >= NONE_GAP)]
        for k in rng.permutation(len(cand))[:NONE_PER_CLIP]:
            w, c, t0, t1 = cand[k]
            ev.append(dict(stem=s, split=split[s], cls='none', word=w, t=t0, t_off=t1, lang=it['lang'], spk=it['speaker']))
    Q = np.concatenate([mot[s] for s in stems])
    np.savez(out / 'MOTION.npz', q=Q, stems=np.array(stems), off=np.array(offs), len=np.array(lens))
    cnt = {sp_: Counter(e['cls'] for e in ev if e['split'] == sp_) for sp_ in ('train', 'dev', 'test')}
    meta_out = dict(version='g1-semantic-v1', fps=FPS, half_window_s=HALF, none_gap_s=NONE_GAP, classes=CLASSES,
                    n_stems={k: sum(1 for s in stems if split[s] == k) for k in ('train', 'dev', 'test')},
                    missing_motion=sorted(set(split) - set(mot))[:20], n_missing_motion=len(set(split) - set(mot)),
                    counts=cnt, stats=st, view=str(v), tar=a.tar)
    (out / 'EVENTS.json').write_text(json.dumps(dict(meta=meta_out, split={s: split[s] for s in stems},
                                                     lang={s: meta[s]['lang'] for s in stems}, spk={s: meta[s]['speaker'] for s in stems},
                                                     events=ev, words=wl), ensure_ascii=False))
    print(json.dumps(meta_out, ensure_ascii=False, indent=1, default=int))


if __name__ == '__main__':
    main()

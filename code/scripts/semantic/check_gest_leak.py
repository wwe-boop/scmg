"""Leak check of a GEST_FRAMES.npy track (g1_gest_frames.py): every labelled row (>= 0) must belong to a train_ids item of the
training view, and NO row of any evaluation item may carry a label.  Evaluation items checked:
  * the training view's val_ids (held-out-speaker dev, english_2_scott / english_6_carla)
  * every item of the held-out speakers (also the ones outside the 400-item dev sample)
  * the source view's val_ids (take-level dev, echog-g1-dev-assets) and the dev-asset stem lists
  * ECHO-G val_common (test view items; also by stem name against the training view's items)
--labels-only (test side, full view): those items may be in train_ids as motion but must not carry labels; the
view's own val_ids and val_common must still be outside train_ids.
Exit status 0 = PASS (prints LEAK_CHECK PASS + counts), 1 = FAIL.
    python scripts/semantic/check_gest_leak.py --gest <dir> --view <hos view> --src-view <echog-g1-v1> \
        --dev-stems <hos-assets>/dev_stems.txt <dev-assets>/dev_stems.txt --heldout-speakers english_2_scott english_6_carla
"""
import argparse, json, sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--gest', required=True); ap.add_argument('--view', required=True)
    ap.add_argument('--src-view', required=True); ap.add_argument('--dev-stems', nargs='*', default=[])
    ap.add_argument('--heldout-speakers', nargs='*', default=[])
    ap.add_argument('--labels-only', action='store_true', help='full view: held-out-speaker / dev-list items may be train motion, never labelled')
    a = ap.parse_args(); v, sv = Path(a.view), Path(a.src_view)
    g = np.load(Path(a.gest) / 'GEST_FRAMES.npy', mmap_mode='r')
    items = json.loads((v / 'train.json').read_text()); sp = json.loads((v / 'SPLIT.json').read_text())
    ssp = json.loads((sv / 'SPLIT.json').read_text())
    test = {x['id'] for x in json.loads((sv / 'test' / 'train.json').read_text())}
    tr = set(sp['train_ids']); held = set(a.heldout_speakers)
    dev_stems = set()
    for f in a.dev_stems: dev_stems |= set(Path(f).read_text().split())
    forbidden = {'hos_val': set(sp['val_ids']), 'src_val': set(ssp['val_ids']), 'dev_stem_lists': dev_stems,
                 'heldout_speaker_items': {x['id'] for x in items if '_'.join(x['id'].split('_')[:3]) in held}, 'val_common': test}
    fails = []; lab_rows = 0; lab_items = 0; covered = np.zeros(len(g), bool)
    for x in items:
        o, n = int(x['offset']), int(x['frames']); seg = np.asarray(g[o:o + n]); covered[o:o + n] = True
        has = bool((seg >= 0).any())
        if has:
            lab_items += 1; lab_rows += int((seg >= 0).sum())
            if x['id'] not in tr: fails.append(('labelled_not_train', x['id']))
            for name, s in forbidden.items():
                if x['id'] in s: fails.append((f'labelled_{name}', x['id']))
    stray = int((np.asarray(g)[~covered] >= 0).sum())
    if stray: fails.append(('labelled_rows_outside_items', stray))
    overlap = {k: len(s & tr) for k, s in forbidden.items()}
    for k, n in overlap.items():                      # --labels-only (test-side full view): these items are trained on as
        if n and (not a.labels_only or k in ('hos_val', 'src_val', 'val_common')):   # motion, but must carry no label
            fails.append((f'train_ids_contain_{k}', n))
    rep = dict(labelled_items=lab_items, labelled_rows=lab_rows, forbidden_sizes={k: len(s) for k, s in forbidden.items()},
               train_overlap=overlap, n_fail=len(fails), fails=fails[:20])
    print(json.dumps(rep, indent=1))
    print('LEAK_CHECK', 'PASS' if not fails else 'FAIL')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()

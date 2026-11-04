"""Held-out-SPEAKER G1 view (11-05). The ECHO-G test set (val_common) has 3 speakers that never occur in training, while the
take-level dev split shares all 13 of its speakers with training -- signed jerk even flips sign between the two. This view
removes whole speakers from training so that inference settings / decoders can be chosen on unseen speakers, like the test.

Every data file of the source view is symlinked (rows, text caches and statistics stay valid); only SPLIT.json changes:
train_ids = source train_ids without the held-out speakers, val_ids = a seeded sample of their items. The ground-truth
30 fps motion and audio of those items are copied from the full ECHO-G dataset into <assets>/ for the official evaluator.

    python scripts/echog_g1/make_hos_view.py --src <view> --out <new view> --assets <dir> --dataset <ds root> \
        --speakers english_2_scott english_6_carla --n-dev 400
"""
import argparse, json, os, shutil, hashlib
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True); ap.add_argument('--out', required=True); ap.add_argument('--assets', required=True)
ap.add_argument('--dataset', required=True, help='full ECHO-G dataset root with motion_39d_30fps/ and audio/')
ap.add_argument('--speakers', nargs='+', required=True); ap.add_argument('--n-dev', type=int, default=400); ap.add_argument('--seed', type=int, default=20261105)
a = ap.parse_args()
src, out, assets, ds = Path(a.src), Path(a.out), Path(a.assets), Path(a.dataset)
spk = lambda i: '_'.join(i.split('_')[:3])
sp = json.loads((src / 'SPLIT.json').read_text())
held = set(a.speakers)
missing = held - {spk(i) for i in sp['train_ids']}
if missing: raise SystemExit(f'speakers not in the source train split: {sorted(missing)}')
train = [i for i in sp['train_ids'] if spk(i) not in held]
pool = sorted(i for i in sp['train_ids'] if spk(i) in held)
rng = np.random.default_rng(a.seed); dev = sorted(rng.choice(pool, size=min(a.n_dev, len(pool)), replace=False).tolist())
test_spk = {spk(i) for i in sp.get('test_ids', [])}
out.mkdir(parents=True, exist_ok=False)
for f in src.iterdir():
    if f.name != 'SPLIT.json': os.symlink(f.resolve() if f.is_symlink() else f, out / f.name)
new = dict(train_ids=train, val_ids=dev, rule=f'held-out-speaker view of {src}: train = source train_ids minus speakers {sorted(held)} '
           f'({len(pool)} items); val = {len(dev)} of their items (numpy default_rng({a.seed}) choice of the sorted list). Source rule: {sp.get("rule", "")}')
if 'test_ids' in sp: new['test_ids'] = sp['test_ids']
(out / 'SPLIT.json').write_text(json.dumps(new))
(assets / 'motion_39d_30fps').mkdir(parents=True, exist_ok=True); (assets / 'audio').mkdir(parents=True, exist_ok=True)
for i in dev:
    shutil.copy2(ds / 'motion_39d_30fps' / f'{i}.pt', assets / 'motion_39d_30fps' / f'{i}.pt')
    wavs = sorted((ds / 'audio').glob(f'{i}.*'))
    if len(wavs) != 1: raise SystemExit(f'{i}: expected one audio file, found {wavs}')
    shutil.copy2(wavs[0], assets / 'audio' / wavs[0].name)
(assets / 'dev_stems.txt').write_text('\n'.join(dev) + '\n')
rec = dict(view=str(out), speakers=sorted(held), train_items=len(train), heldout_items=len(pool), dev_items=len(dev),
           source_train_items=len(sp['train_ids']), seed=a.seed, split_sha256=hashlib.sha256((out / 'SPLIT.json').read_bytes()).hexdigest())
(out / 'HOS.json').write_text(json.dumps(rec, indent=1)); (assets / 'HOS.json').write_text(json.dumps(rec, indent=1))
print(json.dumps(rec))

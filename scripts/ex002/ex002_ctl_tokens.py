"""EX002 control-token labels from the ground-truth motion (11-07, EX002 semantic line; criteria docs/criteria/EX002_CTL.md).
Train items of data-v36_1 (all sources except the synthetic semsynth). Per item, commands at random times (seeded per item,
>= 1.5 s apart, >= 0.5 s from the item ends), alternating:
  AIM    (type deictic = scmg.ctl.POINT): hand = the hand whose wrist is farther from its rest position at t; target = wrist +
         d * forearm ray (elbow -> wrist) at t, d ~ U[1.5, 3] m, base frame; apex = t (the command time is the apex).
  FINGER (type beat with a 'fingers' value): extended-finger counts (left, right) at t (scmg/ex002_ctl.py finger_counts).
Also writes PREP.json (view SPLIT sha256, target pools per hand for the commanded dev evaluation, count distribution) and
dev_stems.txt (the 380 val items). Dev commands are NOT written here (ex002_ctl_commands.py makes them, from the train pools).
    SCMG_J=34 python scripts/ex002/ex002_ctl_tokens.py --view <data-v36_1> --out <dir>
"""
import argparse, hashlib, json, sys, zlib
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import ctl, ex002_ctl as E

FPS = 50; GAP = 1.5; EDGE = 0.5


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
    V = Path(a.view); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((V / 'META.json').read_text()); up = np.asarray(meta['ex002']['upper'], np.float32)
    rest = np.asarray(meta['ex002']['rest34'], np.float32)
    split = json.loads((V / 'SPLIT.json').read_text()); items = {x['id']: x for x in json.loads((V / 'train.json').read_text())}
    q = np.load(V / 'train_q.npy', mmap_mode='r')
    rw = E.points(rest[None])['wrist'][0]                                       # [2, 3] rest wrists
    train = [i for i in split['train_ids'] if not items[i]['source'].startswith('semsynth')]
    if a.limit: train = train[:a.limit]
    raw = {}; pools = {0: [], 1: []}; cnt = np.zeros((2, 4), np.int64); n_aim = n_fing = 0
    for k, iid in enumerate(train):
        it = items[iid]; T = it['frames']
        if T / FPS < 2 * EDGE + 0.5: continue
        rng = np.random.default_rng(zlib.crc32(iid.encode()))
        ts = np.arange(EDGE + rng.uniform(0, GAP), T / FPS - EDGE, GAP)
        if not len(ts): continue
        fr = np.minimum((ts * FPS).round().astype(int), T - 1)
        Q = np.asarray(q[it['offset'] + fr], np.float32)
        geo = E.points(Q); cnts = E.finger_counts(Q, up); toks = []
        for j, (t, f) in enumerate(zip(ts, fr)):
            if (j + k) % 2 == 0:
                h = int(np.argmax(np.linalg.norm(geo['wrist'][j] - rw, axis=-1)))
                r = geo['wrist'][j, h] - geo['elbow'][j, h]; r = r / max(np.linalg.norm(r), 1e-9)
                tgt = geo['wrist'][j, h] + rng.uniform(1.5, 3.0) * r
                toks.append(dict(type=ctl.POINT, cls='deictic', t=float(t), a=float(t - 0.3), b=float(t + 0.3), apex=float(t),
                                 hand=h, target=[float(x) for x in tgt], src='aim', split='train'))
                pools[h].append([float(x) for x in tgt]); n_aim += 1
            else:
                c = [int(cnts[j, 0]), int(cnts[j, 1])]
                toks.append(dict(type=ctl.TYPE_ID['beat'], cls='beat', t=float(t), a=float(t - 0.2), b=float(t + 0.2), fingers=c,
                                 src='finger', split='train'))
                cnt[0, c[0]] += 1; cnt[1, c[1]] += 1; n_fing += 1
        raw[iid] = toks
        if k % 5000 == 0: print(f'{k}/{len(train)}', flush=True)
    (out / 'CTL.json').write_text(json.dumps(raw))
    (out / 'dev_stems.txt').write_text('\n'.join(split['val_ids']) + '\n')
    prep = dict(split_sha256=hashlib.sha256((V / 'SPLIT.json').read_bytes()).hexdigest(), view=str(V), n_items=len(raw), n_aim=n_aim,
                n_finger=n_fing, ext_frac=E.EXT_FRAC, count_dist=(cnt / cnt.sum(1, keepdims=True)).round(4).tolist(),
                amp_edges={c: [0.1, 0.2] for c in ('deictic', 'iconic', 'metaphoric', 'beat')}, rule=__doc__.split('\n')[1])
    (out / 'PREP.json').write_text(json.dumps(prep, indent=1))
    np.savez_compressed(out / 'TARGET_POOLS.npz', left=np.asarray(pools[0], np.float32), right=np.asarray(pools[1], np.float32))
    print(json.dumps({k: v for k, v in prep.items() if k != 'rule'}))


if __name__ == '__main__':
    main()

"""Module 10 (a) human hand-shape library (11-08; criteria "Module 10"), TRAIN split only: for each hand h and count n, up to 2000 train
frames (every 5th frame of train items, seeded choice 1115) whose frozen finger-count detector count (ex002_ctl.finger_counts) is n;
stored as the 8 finger joints of that hand -> tokens/HAND_LIBRARY_TRAIN.npz (keys 'h_n'). Also writes the independent joint-angle check
(angle_agreement): per finger, extended / curled by the nearest of the HAND_SHAPES all-extended (h_3) / all-curled (h_0) template
values over that finger's flexion joints; agreement = share of entries whose angle-classified extended count equals n.
    SCMG_J=34 python hand_library.py OUT.npz"""
import json, sys
import numpy as np
from scmg import ex002_ctl as E
C = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'; V = C + '/train-v1-20261101/views/data-v36_1'
T = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002/tokens'
SLOTS = {f: [j - 18 for j in E.FINGERS[0][f]] for f in ('thumb', 'index', 'third')}       # flexion joints inside the 8-joint block


def angle_count(sh, h, tmpl):
    ext, cur = np.asarray(tmpl[f'{h}_3']), np.asarray(tmpl[f'{h}_0']); n = np.zeros(len(sh), int)
    for f, s in SLOTS.items():
        n += (np.linalg.norm(sh[:, s] - ext[s], axis=1) < np.linalg.norm(sh[:, s] - cur[s], axis=1)).astype(int)
    return n


def main():
    meta = json.load(open(V + '/META.json')); up = np.asarray(meta['ex002']['upper'], np.float32)
    train = set(json.load(open(V + '/SPLIT.json'))['train_ids']); items = [x for x in json.load(open(V + '/train.json')) if x['id'] in train]
    q = np.load(V + '/train_q.npy', mmap_mode='r'); idx = np.concatenate([np.arange(x['offset'], x['offset'] + x['frames'], 5) for x in items])
    Q = np.asarray(q[np.sort(idx)], np.float32); cnt = E.finger_counts(Q, up)
    tmpl = json.load(open(T + '/HAND_SHAPES.json'))['shapes']; rng = np.random.default_rng(1115); out, rep = {}, {}
    for h, (lo, hi) in ((0, (18, 26)), (1, (26, 34))):
        for n in range(4):
            cand = np.nonzero(cnt[:, h] == n)[0]
            sel = np.sort(rng.choice(cand, size=min(2000, len(cand)), replace=False)) if len(cand) else cand
            sh = Q[sel, lo:hi]; out[f'{h}_{n}'] = sh
            rep[f'{h}_{n}'] = dict(candidates=int(len(cand)), entries=int(len(sel)), angle_agreement=float((angle_count(sh, h, tmpl) == n).mean()) if len(sel) else None)
    np.savez(sys.argv[1], **out); json.dump(dict(report=rep, frames_scanned=int(len(Q)), seed=1115), open(sys.argv[1].replace('.npz', '.json'), 'w'), indent=1)
    print(json.dumps(rep))


if __name__ == '__main__':
    main()

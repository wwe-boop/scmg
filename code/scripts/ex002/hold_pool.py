"""9c hold-motion pools (11-08; criteria "Module 9c"; TRAIN split only, never dev / APEX scores). For each pool entry: the wrist
displacement dw [36, 3] (m, robot frame) relative to the window start over a 0.7 s window (= the 9b / 9c aim hold [t - 0.1, t + 0.6]).
  point   : 3000 train-split EX002 pointing tokens (CTL.json src 'aim'; seeded sample, seed 1114), window [t - 0.1, t + 0.6], token's hand
  control : the same number of windows from train items, start uniform (seed 1114) inside the item's speech span, at least 1.0 s away
            from every pointing token of that item, random hand -> "any human gesturing motion", not pointing holds
Writes tokens/HOLD_POOL_TRAIN.npz (point_dw, point_hand, control_dw, control_hand) and a JSON of the wrist-speed statistics.
    SCMG_J=34 python hold_pool.py OUT.npz"""
import json, sys
import numpy as np
from scmg import ex002_ctl as E
C = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'; V = C + '/train-v1-20261101/views/data-v36_1'
T = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002/tokens'; FPS = 50; N = 36; A0 = -0.1
items = {x['id']: x for x in json.load(open(V + '/train.json'))}; train = set(json.load(open(V + '/SPLIT.json'))['train_ids'])
q = np.load(V + '/train_q.npy', mmap_mode='r'); ctl = json.load(open(T + '/CTL.json'))


def dw_of(it, f0, h):
    y = np.asarray(q[it['offset'] + f0:it['offset'] + f0 + N], np.float32); w = E.points(y)['wrist'][:, h]; return (w - w[0]).astype(np.float32)


pd, ph, aims, toks = [], [], {}, []
for k, ts in sorted(ctl.items()):
    if k not in train: continue
    for t in ts:
        if t.get('src') != 'aim' or t['split'] != 'train': continue
        aims.setdefault(k, []).append(t['t']); toks.append((k, t))
rng = np.random.default_rng(1114)
for i in rng.choice(len(toks), size=min(3000, len(toks)), replace=False):
    k, t = toks[i]; it = items[k]; f0 = int(round((t['t'] + A0) * FPS))
    if f0 < 0 or f0 + N > it['frames']: continue
    pd.append(dw_of(it, f0, int(t['hand']))); ph.append(int(t['hand']))
cand = [k for k in sorted(train) if k in items and items[k].get('speech') and items[k]['frames'] > N + 2]
cd, ch = [], []
while len(cd) < len(pd):
    k = cand[rng.integers(len(cand))]; it = items[k]; s0, s1 = (int(round(float(x) * FPS)) for x in it['speech']); s1 = min(s1, it['frames']) - N
    if s1 <= s0: continue
    f0 = int(rng.integers(s0, s1 + 1)); tc = (f0 + N / 2) / FPS
    if any(abs(tc - ta) < 1.0 + N / 2 / FPS for ta in aims.get(k, [])): continue
    h = int(rng.integers(2)); cd.append(dw_of(it, f0, h)); ch.append(h)
pd, cd = np.stack(pd), np.stack(cd)
sp = lambda D: np.linalg.norm(np.diff(D, axis=1), axis=-1).mean(1) * FPS
st = {lab: dict(n=len(D), speed_median=float(np.median(sp(D))), speed_q25=float(np.percentile(sp(D), 25)), speed_q75=float(np.percentile(sp(D), 75)),
                disp_end_median=float(np.median(np.linalg.norm(D[:, -1], axis=-1)))) for lab, D in (('point', pd), ('control', cd))}
np.savez(sys.argv[1], point_dw=pd, point_hand=np.array(ph, np.int8), control_dw=cd, control_hand=np.array(ch, np.int8))
json.dump(dict(stats=st, point_tokens_total=len(toks), window=[A0, A0 + (N - 1) / FPS], frames=N, seed=1114, ctl=T + '/CTL.json'), open(sys.argv[1].replace('.npz', '.json'), 'w'), indent=1)
print(json.dumps(st))

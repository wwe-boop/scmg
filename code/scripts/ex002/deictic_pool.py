"""9c' point pool + hold statistics from REAL human pointing (11-08; criteria "Module 9c'"): the train-split BEAT2 deictic events of
EVENTS.json (semantic-baseline-1105/ex002-sem/data; 1060), hand = the hand with the larger wrist-path diameter in [t - 1, t + 1] s (the
APEX.json rule), window [t - 0.1, t + 0.6] s at the annotated time t (36 frames) -> point_dw / point_hand; the control pool is copied
unchanged from HOLD_POOL_TRAIN.npz. Also: hold statistics as hold_stats.py (along-ray push-out, perpendicular drift, wrist speed) on
[t - 0.3, t + 0.6] and [t - 0.1, t + 0.6]; and the dev-overlap check (no event stem may be a dev item / clip).
    SCMG_J=34 python deictic_pool.py OUT_POOL.npz OUT_STATS.json"""
import json, sys
import numpy as np
from scmg import ex002_ctl as E
B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; C = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'; V = C + '/train-v1-20261101/views/data-v36_1'
T = B + '/control-token-1105/ex002/tokens'; FPS = 50; N = 36
items = {x['id']: x for x in json.load(open(V + '/train.json'))}; sp = json.load(open(V + '/SPLIT.json')); train, val = set(sp['train_ids']), set(sp['val_ids'])
clips = {}
for x in items.values(): clips.setdefault(x['clip'], []).append(x)
val_clips = {items[i]['clip'] for i in val if i in items}
q = np.load(V + '/train_q.npy', mmap_mode='r')
ev = [e for e in json.load(open(B + '/semantic-baseline-1105/ex002-sem/data/EVENTS.json'))['events'] if e['split'] == 'train' and e['cls'] == 'deictic']
overlap = sum(1 for e in ev if e['stem'] in val_clips or any(x['id'] in val for x in clips.get(e['stem'], [])))
pd, ph, rows = [], [], {'onset': [], 'hold9b': []}
for e in ev:
    its = sorted(clips.get(e['stem'], []), key=lambda x: x['offset'])
    if not its or any(x['id'] not in train for x in its): continue
    Qc = np.concatenate([np.asarray(q[x['offset']:x['offset'] + x['frames']], np.float32) for x in its]); c = int(round(e['t'] * FPS))
    f0, f1 = max(0, c - FPS), min(len(Qc), c + FPS)
    if f1 - f0 < 20: continue
    a0, a1 = c - int(round(0.3 * FPS)), c + int(round(0.6 * FPS))
    if a0 < 0 or a1 >= len(Qc): continue
    g = E.points(Qc[max(0, min(f0, a0)):max(f1, a1 + 1)]); off = max(0, min(f0, a0))
    seg = g['wrist'][f0 - off:f1 - off]; h = int(np.argmax([np.linalg.norm(seg[:, k].max(0) - seg[:, k].min(0)) for k in (0, 1)]))
    w, el = g['wrist'][:, h], g['elbow'][:, h]
    for lab, s0 in (('onset', a0), ('hold9b', c - int(round(0.1 * FPS)))):
        i0, i1 = s0 - off, a1 - off; ww = w[i0:i1 + 1]; r = ww[0] - el[i0]; r = r / max(np.linalg.norm(r), 1e-9); d = (ww - ww[0]) @ r
        rows[lab].append(dict(A=float(d.max()), tau=float(np.argmax(d) / FPS), d_end=float(d[-1]), speed=float(np.linalg.norm(np.diff(ww, axis=0), axis=1).mean() * FPS),
                              perp_end=float(np.linalg.norm((ww[-1] - ww[0]) - d[-1] * r))))
    s0 = c - int(round(0.1 * FPS)) - off; ww = w[s0:s0 + N]
    if len(ww) == N: pd.append((ww - ww[0]).astype(np.float32)); ph.append(h)
st = {lab: {m: dict(median=float(np.median([x[m] for x in v])), q25=float(np.percentile([x[m] for x in v], 25)), q75=float(np.percentile([x[m] for x in v], 75)))
            for m in ('A', 'tau', 'd_end', 'speed', 'perp_end')} | {'n': len(v)} for lab, v in rows.items()}
old = np.load(T + '/HOLD_POOL_TRAIN.npz'); pd = np.stack(pd)
np.savez(sys.argv[1], point_dw=pd, point_hand=np.array(ph, np.int8), control_dw=old['control_dw'], control_hand=old['control_hand'])
spd = np.linalg.norm(np.diff(pd, axis=1), axis=-1).mean(1) * FPS
json.dump(dict(events_train_deictic=len(ev), dev_overlap=overlap, point_n=len(pd), point_speed=dict(median=float(np.median(spd)), q25=float(np.percentile(spd, 25)), q75=float(np.percentile(spd, 75))),
               hold_stats=st), open(sys.argv[2], 'w'), indent=1)
print('dev_overlap', overlap, 'point_n', len(pd), json.dumps({lab: {m: {k: round(x, 3) for k, x in v[m].items()} for m in ('A', 'speed', 'perp_end')} for lab, v in st.items()}))

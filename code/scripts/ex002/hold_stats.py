"""9c source statistics (11-08, before any 9c run): human wrist motion along the pointing ray during the hold window, from TRAIN-split
human pointing events (EX002 CTL.json tokens src 'aim', split 'train' = the annotation source of the plans; motion = data-v36_1 train_q).
Window = the 9c hold relative to the token time t: [t - 0.3, t + 0.6] s (onset a = t - 0.3 .. end of the 9b hold); also [t - 0.1, t + 0.6].
Per event (hand = token hand): r = unit forearm ray (elbow -> wrist) at the window start; d(tau) = (w(tau) - w(start)) . r;
A = max d, tau* = time of the max from the window start, d_end = d at the window end; mean wrist speed; perpendicular drift at the end.
    SCMG_J=34 python hold_stats.py OUT.json"""
import json, sys
import numpy as np
from scmg import ex002_ctl as E
C = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'; V = C + '/train-v1-20261101/views/data-v36_1'
T = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002/tokens'; FPS = 50
items = {x['id']: x for x in json.load(open(V + '/train.json'))}; train = set(json.load(open(V + '/SPLIT.json'))['train_ids'])
q = np.load(V + '/train_q.npy', mmap_mode='r'); ctl = json.load(open(T + '/CTL.json'))
ev = [(k, t) for k, ts in sorted(ctl.items()) for t in ts if t.get('src') == 'aim' and t['split'] == 'train' and k in train]
rng = np.random.default_rng(1113); sel = rng.choice(len(ev), size=min(3000, len(ev)), replace=False)
R = {}
for lab, (a0, a1) in (('onset', (-0.3, 0.6)), ('hold9b', (-0.1, 0.6))):
    rows = []
    for i in sel:
        k, tok = ev[i]; it = items[k]; f0 = int(round((tok['t'] + a0) * FPS)); f1 = int(round((tok['t'] + a1) * FPS))
        if f0 < 0 or f1 >= it['frames']: continue
        y = np.asarray(q[it['offset'] + f0:it['offset'] + f1 + 1], np.float32); g = E.points(y); h = int(tok['hand'])
        w, el = g['wrist'][:, h], g['elbow'][:, h]; r = w[0] - el[0]; r = r / max(np.linalg.norm(r), 1e-9)
        d = (w - w[0]) @ r; perp = np.linalg.norm((w[-1] - w[0]) - d[-1] * r)
        sp = np.linalg.norm(np.diff(w, axis=0), axis=1) * FPS
        rows.append(dict(A=float(d.max()), tau=float(np.argmax(d) / FPS), d_end=float(d[-1]), speed=float(sp.mean()), perp_end=float(perp), key=k, t=float(tok['t'])))
    st = {m: dict(median=float(np.median([x[m] for x in rows])), q25=float(np.percentile([x[m] for x in rows], 25)), q75=float(np.percentile([x[m] for x in rows], 75)))
          for m in ('A', 'tau', 'd_end', 'speed', 'perp_end')}
    R[lab] = dict(window=[a0, a1], n=len(rows), stats=st, samples=[[x['A'], x['tau'], x['d_end']] for x in rows])
    print(lab, len(rows), json.dumps({m: {kk: round(vv, 3) for kk, vv in v.items()} for m, v in st.items()}), flush=True)
R['source'] = dict(ctl=T + '/CTL.json', events_total=len(ev), sampled=int(len(sel)), seed=1113, view=V)
json.dump(R, open(sys.argv[1], 'w'))

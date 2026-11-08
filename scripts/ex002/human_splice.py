"""M2 (11-08; criteria "M2: human-splice control (ii)_H", REPORT-ONLY): FGD v2 of the dev GT with train-split human windows spliced in at the
plan's commands (insert9c machinery), against the GT. Pools (TRAIN only): real deictic windows (EVENTS.json; APEX-rule hand; [t-0.1, t+0.6]),
hindsight-token windows (CTL 'aim'; validity check b), raised presenting windows per (hand, count) ([c-0.4, c+0.4], detector count = n,
wrist >= shoulder - 0.20 m); both hands' finger joints from HAND_LIBRARY_TRAIN.npz at finger commands.
    SCMG_J=34 python human_splice.py --out M2.json [--seeds 0,1,2]"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from scmg import ex002_ctl as E
from ex002_keyframe import insert9c, ARM, HOLD
from ex002_ctl_run import finger_override, tokens_for
B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; C = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'; V = C + '/train-v1-20261101/views/data-v36_1'
X = B + '/control-token-1105/ex002'; T = X + '/tokens'; FPS = 50; FGD_CODE = '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/fgd-ex002-v2-20261102/code'


def pools(seed=1117):
    items = {x['id']: x for x in json.load(open(V + '/train.json'))}; train = set(json.load(open(V + '/SPLIT.json'))['train_ids'])
    meta = json.load(open(V + '/META.json'))['ex002']; up = np.asarray(meta['upper'], np.float32)
    clips = {}
    for x in items.values(): clips.setdefault(x['clip'], []).append(x)
    q = np.load(V + '/train_q.npy', mmap_mode='r'); P = {'deictic': {0: [], 1: []}, 'hindsight': {0: [], 1: []}, 'present': {}}
    for e in json.load(open(B + '/semantic-baseline-1105/ex002-sem/data/EVENTS.json'))['events']:
        if e['split'] != 'train' or e['cls'] != 'deictic': continue
        its = sorted(clips.get(e['stem'], []), key=lambda x: x['offset'])
        if not its or any(x['id'] not in train for x in its): continue
        Qc = np.concatenate([np.asarray(q[x['offset']:x['offset'] + x['frames']], np.float32) for x in its]); c = int(round(e['t'] * FPS))
        f0, f1 = c - 5, c + 31
        if f0 < 0 or f1 > len(Qc) or c - FPS < 0 or c + FPS > len(Qc): continue
        g = E.points(Qc[c - FPS:c + FPS]); h = int(np.argmax([np.linalg.norm(g['wrist'][:, k].max(0) - g['wrist'][:, k].min(0)) for k in (0, 1)]))
        P['deictic'][h].append(Qc[f0:f1])
    ctl = json.load(open(T + '/CTL.json')); rng = np.random.default_rng(seed); toks = [(k, t) for k, ts in sorted(ctl.items()) if k in train for t in ts if t.get('src') == 'aim']
    for i in rng.choice(len(toks), size=min(3000, len(toks)), replace=False):
        k, t = toks[i]; it = items[k]; f0 = int(round((t['t'] - 0.1) * FPS))
        if f0 < 0 or f0 + 36 > it['frames']: continue
        P['hindsight'][int(t['hand'])].append(np.asarray(q[it['offset'] + f0:it['offset'] + f0 + 36], np.float32))
    tr = [x for x in items.values() if x['id'] in train and x['frames'] > 60]
    for j in rng.choice(len(tr), size=min(4000, len(tr)), replace=False):
        it = tr[j]; Qi = np.asarray(q[it['offset']:it['offset'] + it['frames']], np.float32); cs = np.arange(20, len(Qi) - 21, 25)
        if not len(cs): continue
        g = E.points(Qi[cs]); cnt = E.finger_counts(Qi[cs], up)
        for k_, c in enumerate(cs):
            for h in (0, 1):
                if g['wrist'][k_, h, 2] >= g['shoulder'][k_, h, 2] - 0.20:
                    P['present'].setdefault((h, int(cnt[k_, h])), []).append(Qi[c - 20:c + 21])
    return P


def pick(pool, key, n):
    return int(hashlib.sha1(key.encode()).hexdigest()[:12], 16) % n


def splice(gt, evs, P, seed, which, lib, shapes, a95):
    y = gt.astype(np.float32).copy()
    for e in evs:
        key = f"{e['clip']}:{e['t']:.3f}:M2:{seed}"
        if e['kind'] == 'aim':
            pool = P[which][e['hand']]; Q = pool[pick(pool, key, len(pool))]; insert9c(y, Q, e['hand'], e['t'], 'aim', a95)
        else:
            hp = 0 if e['fingers'][0] > e['fingers'][1] else 1; pool = P['present'].get((hp, int(e['fingers'][hp])), [])
            if pool: insert9c(y, pool[pick(pool, key, len(pool))], hp, e['t'], 'finger', a95)
    return finger_override(y, tokens_for(evs, 'given'), shapes, ramp=0.3, profile='minjerk', library=lib).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--seeds', default='0,1,2'); a = ap.parse_args()
    sys.path.insert(0, FGD_CODE)
    from model import Embedder
    from config import WEIGHTS
    from scmg.evaluate import frechet
    emb = [Embedder(WEIGHTS / f'human_seed{k}' / 'best.pt') for k in (0, 1)]
    a95 = json.load(open(T + '/ARM_ACC_P95.json'))['a95']; lib = dict(np.load(T + '/HAND_LIBRARY_TRAIN.npz')); shapes = json.load(open(T + '/HAND_SHAPES.json'))['shapes']
    P = pools(); print('pools', {k: ({h: len(v) for h, v in d.items()} if k != 'present' else {str(kk): len(v) for kk, v in d.items()}) for k, d in P.items()}, flush=True)
    G = X + '/m9b-enc/offline/gen'; R = {}
    def fgd(ys, gts): return float(np.mean([frechet(np.concatenate([e.features(g) for g in gts]), np.concatenate([e.features(y) for y in ys])) for e in emb]))
    for pn in ('sparse', 'dense'):
        Pl = json.load(open(X + f'/PLAN_dev_m9b_{pn}.json')); by = {}
        for e in Pl['events']: by.setdefault(e['ci'], []).append(e)
        cis = sorted(int(f.stem[3:]) for f in Path(G).glob('gt_*.npy')); gts = [np.load(f'{G}/gt_{ci}.npy') for ci in cis]
        if pn == 'sparse': R['a_zero_insertions'] = fgd(gts, gts)
        for which in (('deictic', 'hindsight') if pn == 'sparse' else ('deictic',)):
            v = [fgd([splice(g, by.get(ci, []), P, s, which, lib, shapes, a95) for ci, g in zip(cis, gts)], gts) for s in [int(x) for x in a.seeds.split(',')]]
            R[f'{pn}_{which}'] = dict(seeds=v, mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.)
            print(pn, which, R[f'{pn}_{which}'], flush=True)
    R['checks'] = dict(a=R['a_zero_insertions'] < 1e-6, b=R['sparse_hindsight']['mean'] <= R['sparse_deictic']['mean'],
                       c=R['dense_deictic']['mean'] > 1.5 * R['sparse_deictic']['mean'])
    print(json.dumps(R)); Path(a.out).write_text(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()

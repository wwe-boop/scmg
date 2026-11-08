"""EX002 module 8 present-count (11-08; criteria docs/criteria/EX002_CTL.md "Module 8"; default off: only used through its own plans).
  target : presentation direction per hand from TRAIN GT frames (data-v36_1 train split) whose hand is raised (wrist z >= shoulder z -
           0.20 m) with >= 1 extended finger (curl fraction < 0.15): median unit FOREARM direction elbow -> wrist (the POINT ray); target = rest shoulder +
           2.25 m x that direction (the POINT target convention) -> PRESENT.json
  plan   : copy of a plan with, at every finger command, one extra AIM event per hand toward its presentation target at the same time
           (flag present=True; the shuffle copy keeps the same presentation targets) -> generation plan (evaluate keeps the original)
  eval   : per finger command (+-0.4 s): share of frames with wrist z >= shoulder z - 0.20 m (both hands), palm-normal angle to the
           forward direction +x (normal = cross(index base - wrist, third-finger base - wrist); sign-agnostic: min(a, 180 - a)), medians
    SCMG_J=34 python ex002_present.py target --out PRESENT.json | plan --plan P --present PRESENT.json --out P2 | eval --plan P --out R label=dir ..."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
FPS = 50; V = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1'
RAISE = 0.20


PALM_SIGN = {0: -1.0, 1: 1.0}   # 11-08: palm = sign * cross(index_base - wrist, third_base - wrist) (left: -n, right: +n; checked: curling
                                 # moves the index tip toward it, and at rest both palms face the torso midline)


def third_bodies():
    from scmg.common import P
    from scmg import ex002_ctl as E
    z = np.load(P(E.NPZ)); bj = np.asarray(z['body_joint']); b = lambda j: int(np.nonzero(bj == j)[0][0])
    return b(24), b(32)


def geo_all(q):
    from scmg import ex002_ctl as E
    import torch
    g = E.points(q); K = E.kin('cpu'); tl, tr = third_bodies()
    with torch.no_grad(): _, p = K.fk(torch.as_tensor(np.asarray(q, np.float32)))
    g['third_base'] = np.stack([p[:, tl].numpy(), p[:, tr].numpy()], 1); return g


def target(a):
    from scmg import ex002_ctl as E
    meta = json.loads((Path(a.view) / 'META.json').read_text())['ex002']; up = np.asarray(meta['upper'], np.float32)
    items = json.loads((Path(a.view) / 'train.json').read_text()); tr = set(json.loads((Path(a.view) / 'SPLIT.json').read_text())['train_ids'])
    Q = np.load(Path(a.view) / 'train_q.npy', mmap_mode='r'); rng = np.random.default_rng(20261108)
    hum = [it for it in items if it['id'] in tr and not str(it.get('data_source', '')).startswith('semsynth')]
    dirs = [[], []]
    for i in rng.choice(len(hum), min(a.n_items, len(hum)), replace=False):
        it = hum[i]; q = np.asarray(Q[it['offset']:it['offset'] + it['frames']]); g = E.points(q); cf = E.curl_frac(q, up)   # [T, 2, 3]
        for h in (0, 1):
            ok = (g['wrist'][:, h, 2] >= g['shoulder'][:, h, 2] - RAISE) & ((cf[:, h] < E.EXT_FRAC).sum(-1) >= 1)
            v = g['wrist'][ok, h] - g['elbow'][ok, h]; dirs[h].append(v / np.linalg.norm(v, axis=-1, keepdims=True))
    rest = E.points(np.zeros((1, 34), np.float32))['shoulder'][0]; out = {}
    for h in (0, 1):
        D = np.concatenate(dirs[h]); md = np.median(D, 0); md /= np.linalg.norm(md)
        out[str(h)] = dict(direction=md.tolist(), target=(rest[h] + 2.25 * md).tolist(), frames=int(len(D)))
    Path(a.out).write_text(json.dumps(dict(rule=__doc__.split('\n')[2].strip(), raise_m=RAISE, n_items=a.n_items, hands=out), indent=1)); print(json.dumps(out))


def wrist(a):
    """11-08 module 8b: median wrist joint angles (arm_joint5-7 = q34 [8, 9, 10] left / [15, 16, 17] right) of the same raised-hand,
    >= 1 extended-finger train-GT frames (library presentation pose) -> PRESENT_WRIST.json"""
    from scmg import ex002_ctl as E
    meta = json.loads((Path(a.view) / 'META.json').read_text())['ex002']; up = np.asarray(meta['upper'], np.float32)
    items = json.loads((Path(a.view) / 'train.json').read_text()); tr = set(json.loads((Path(a.view) / 'SPLIT.json').read_text())['train_ids'])
    Q = np.load(Path(a.view) / 'train_q.npy', mmap_mode='r'); rng = np.random.default_rng(20261108)
    hum = [it for it in items if it['id'] in tr and not str(it.get('data_source', '')).startswith('semsynth')]; J = {0: [8, 9, 10], 1: [15, 16, 17]}; got = [[], []]
    for i in rng.choice(len(hum), min(a.n_items, len(hum)), replace=False):
        it = hum[i]; q = np.asarray(Q[it['offset']:it['offset'] + it['frames']]); g = E.points(q); cf = E.curl_frac(q, up)
        for h in (0, 1):
            ok = (g['wrist'][:, h, 2] >= g['shoulder'][:, h, 2] - RAISE) & ((cf[:, h] < E.EXT_FRAC).sum(-1) >= 1); got[h].append(q[ok][:, J[h]])
    out = {str(h): dict(joints=J[h], angles=np.median(np.concatenate(got[h]), 0).tolist(), frames=int(sum(len(x) for x in got[h]))) for h in (0, 1)}
    Path(a.out).write_text(json.dumps(dict(rule='median arm_joint5-7 of raised-hand (wrist >= shoulder - 0.20 m), >= 1 extended finger train-GT frames', hands=out), indent=1)); print(json.dumps(out))


def present_hand(f):
    return 0 if f[0] > f[1] else 1                                   # the hand with the larger commanded count; ties -> right


def plan_one(a):
    """11-08 module 8b: ONE AIM per finger command (present_hand) toward its presentation target."""
    P = json.loads(Path(a.plan).read_text()); pr = json.loads(Path(a.present).read_text())['hands']; ev = []
    for e in P['events']:
        ev.append(e)
        if e['kind'] == 'finger':
            h = present_hand(e['fingers']); tg = pr[str(h)]['target']
            ev.append(dict(ci=e['ci'], clip=e['clip'], kind='aim', t=e['t'], hand=h, target=tg, present=True, shuffle=dict(hand=h, target=tg)))
    P2 = dict(P, events=ev, n_events=len(ev), present_from=a.plan, present_rule='one aim, present_hand'); Path(a.out).write_text(json.dumps(P2)); print('PLAN1', len(P['events']), '->', len(ev))


def plan(a):
    P = json.loads(Path(a.plan).read_text()); pr = json.loads(Path(a.present).read_text())['hands']; ev = []
    for e in P['events']:
        ev.append(e)
        if e['kind'] == 'finger':
            for h in (0, 1):
                tg = pr[str(h)]['target']; ev.append(dict(ci=e['ci'], clip=e['clip'], kind='aim', t=e['t'], hand=h, target=tg, present=True, shuffle=dict(hand=h, target=tg)))
    P2 = dict(P, events=ev, n_events=len(ev), present_from=a.plan); Path(a.out).write_text(json.dumps(P2)); print('PLAN', len(P['events']), '->', len(ev))


def evaluate(a):
    P = json.loads(Path(a.plan).read_text()); by = {}
    for e in P['events']:
        if e['kind'] == 'finger' and not e.get('present'): by.setdefault(e['ci'], []).append(e)
    res = {}
    for spec in a.specs:
        lab, d = spec.split('=', 1); hi, ang, hip, angp, angs, angps = [], [], [], [], [], []
        for f in sorted(Path(d).glob('*.npy'), key=lambda p: int(p.stem)):
            ci = int(f.stem); y = np.load(f); n = len(y)
            if ci not in by: continue
            g = geo_all(y)
            for e in by[ci]:
                fe = int(round(e['t'] * FPS)); f0, f1 = max(0, fe - 20), min(n, fe + 21)
                if f0 >= f1: continue
                for h in (0, 1):
                    hi.append(float((g['wrist'][f0:f1, h, 2] >= g['shoulder'][f0:f1, h, 2] - RAISE).mean()))
                    nrm = np.cross(g['index_base'][f0:f1, h] - g['wrist'][f0:f1, h], g['third_base'][f0:f1, h] - g['wrist'][f0:f1, h])
                    c = np.abs(nrm[:, 0]) / np.maximum(np.linalg.norm(nrm, axis=-1), 1e-9); ang.append(float(np.median(np.degrees(np.arccos(np.clip(c, 0, 1))))))
                    # 11-08 fix: SIGNED palm direction (palm = PALM_SIGN[h] * normal; curling moves the fingertips toward it) vs forward +x
                    cs = PALM_SIGN[h] * nrm[:, 0] / np.maximum(np.linalg.norm(nrm, axis=-1), 1e-9); angs.append(float(np.median(np.degrees(np.arccos(np.clip(cs, -1, 1))))))
                    if h == present_hand(e['fingers']): hip.append(hi[-1]); angp.append(ang[-1]); angps.append(angs[-1])
        grp = lab.rsplit('_s', 1)[0]; res.setdefault(grp, dict(hi=[], ang=[], hip=[], angp=[], angs=[], angps=[]))
        res[grp]['hi'] += hi; res[grp]['ang'] += ang; res[grp]['hip'] += hip; res[grp]['angp'] += angp; res[grp]['angs'] += angs; res[grp]['angps'] += angps
    out = {k: dict(hand_raised_share_mean=float(np.mean(v['hi'])), events_hands=len(v['hi']), events_hands_raised_ge80=float(np.mean(np.array(v['hi']) >= .8)),
                   palm_forward_angle_median=float(np.median(v['ang'])), present_hand_raised_share_mean=float(np.mean(v['hip'])),
                   present_hand_raised_ge80=float(np.mean(np.array(v['hip']) >= .8)), present_hand_palm_angle_median=float(np.median(v['angp'])),
                   present_hand_palm_angle_signed_median=float(np.median(v['angps'])), present_hand_palm_forward_share=float(np.mean(np.array(v['angps']) <= 45.))) for k, v in res.items()}
    for k, v in out.items(): print(k, json.dumps({kk: round(vv, 3) for kk, vv in v.items()}))
    Path(a.out).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('target'); t.add_argument('--out', required=True); t.add_argument('--view', default=V); t.add_argument('--n-items', type=int, default=4000)
    p = sub.add_parser('plan'); p.add_argument('--plan', required=True); p.add_argument('--present', required=True); p.add_argument('--out', required=True)
    w = sub.add_parser('wrist'); w.add_argument('--out', required=True); w.add_argument('--view', default=V); w.add_argument('--n-items', type=int, default=4000)
    p1 = sub.add_parser('plan1'); p1.add_argument('--plan', required=True); p1.add_argument('--present', required=True); p1.add_argument('--out', required=True)
    e = sub.add_parser('eval'); e.add_argument('--plan', required=True); e.add_argument('--out', required=True); e.add_argument('specs', nargs='+')
    a = ap.parse_args(); dict(target=target, plan=plan, eval=evaluate, wrist=wrist, plan1=plan_one)[a.cmd](a)


if __name__ == '__main__':
    main()

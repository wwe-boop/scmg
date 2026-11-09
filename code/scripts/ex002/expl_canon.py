"""Explicit semantics, canonical path (11-09; exp/text-path docs/criteria/EXPLICIT_SEM_1109.md amendment 1, registered before the
data-driven sanity result).
  plan     : category words of the given dev recordings -> commands (greedy 1.2 s): SELF / YOU / POINT = aim commands (targets in the
             base frame, free hand = lower wrist speed over the previous 1 s of the none run, ties right), NUMBER = finger command on
             the free hand, BIG / SMALL = bimanual keyframes. --proto tts: t = onset - 0.2 s; psync: t = onset.
             Writes PLAN_<proto>.json (all commands, ex002_ctl_run format) and TOKPLAN_<proto>.json (aim + finger only, for bench).
  keyframe : BIG / SMALL keyframes into a run dir: per hand IK (present-style wrist position, ex002_keyframe.IK), held 0.4 s
             (t - 0.2 .. t + 0.2), quintic ramps to / from the run's own state (insert9b rule, train a95).
  validate : the geometric metrics on synthetic positives (the commanded keyframe IK-inserted into the GT) vs negatives (GT).
  eval     : per-category success of runs at the plan's commands; given - none with the recording bootstrap CI.
Targets (m, base frame): sternum ref = mid-shoulder of the rest pose - 0.10 z + 0.10 x = (0.192, 0, 1.106); YOU (1.5, 0, 1.3);
this / here / these (0.8, 0, 0.6); there / those (1.5, +-0.8, 1.2) on the free hand's side; BIG wrists (0.45, +-0.45, 1.15);
SMALL wrists (0.40, +-0.08, 1.10).
Metrics: SELF min over [t - 0.3, t + 0.9] of |commanded wrist - sternum ref| <= 0.15 m; YOU / POINT Score_b >= 0.5 (forearm ray at
the apex-detector frame in t +- 1 s, the ex002_ctl_run rule); NUMBER both-hands count hit (majority over t +- 0.2 s);
BIG max wrist separation in [t - 0.3, t + 0.9] >= the none run's + 0.15 m; SMALL min separation <= 0.20 m with both wrists above
0.9 m.
    SCMG_J=34 python scripts/ex002/expl_canon.py plan --cis ... --proto tts --none NONE_DIR --out DIR
    SCMG_J=34 python scripts/ex002/expl_canon.py keyframe --plan DIR/PLAN_tts.json --run IN_DIR --out OUT_DIR
    SCMG_J=34 python scripts/ex002/expl_canon.py validate --plan DIR/PLAN_tts.json --gt GTDIR --out V.json
    SCMG_J=34 python scripts/ex002/expl_canon.py eval --plan DIR/PLAN_tts.json --none NONE_DIR --out O.json given=D ..."""
import argparse, json, sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'semantic'))
from sem_bank import V, EV, recordings, rec_words
from expl_sem import category

FPS = 50; X = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002'
STERNUM = np.array([0.192, 0.0, 1.106]); YOU = [1.5, 0.0, 1.3]; HERE = [0.8, 0.0, 0.6]
NUM = dict(two=2, three=3, four=3, five=3, six=3, seven=3, eight=3, nine=3, ten=3, twenty=3, thirty=3, first=1, second=2, third=3, both=2,
           **{'两': 2, '三': 3, '四': 3, '五': 3, '六': 3, '七': 3, '八': 3, '九': 3, '十': 3, '第': 1})
WRIST = dict(BIG=(0.45, 0.45, 1.15), SMALL=(0.40, 0.08, 1.10))


def free_hand(y, t):
    from scmg import ex002_ctl as E
    f1 = int(np.clip(round(t * FPS), 2, len(y) - 1)); f0 = max(0, f1 - FPS); w = E.points(y[f0:f1 + 1])['wrist']
    sp = np.linalg.norm(np.diff(w, axis=0), axis=-1).mean(0)
    return 1 if sp[1] <= sp[0] else 0


def plan(a):
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True); cis = [int(x) for x in a.cis.split(',')]
    items = json.loads((Path(V) / 'train.json').read_text()); sp = json.loads((Path(V) / 'SPLIT.json').read_text())
    if os.environ.get('SCMG_EXPL_VIEW'):     # 11-09 (default off): test-only view; ci = View.clips order (first appearance of each clip among test items)
        ids = set(sp['test_ids']); recs = recordings(items, ids); inv = {}
        for it in items:
            if it['id'] in ids and it['clip'] not in inv.values(): inv[len(inv)] = it['clip']
    else:
        D = json.loads(Path(EV).read_text()); inv = {v: k for k, v in D['meta']['dev_ci'].items()}
        recs = recordings(items, set(sp['val_ids']))
    ev = []
    for ci in cis:
        stem = inv[ci]; y = np.load(Path(a.none) / f'{ci}.npy'); last = -1e9; used = set()
        ws = sorted(rec_words(recs[stem]), key=lambda x: x[2])
        for wi, (w, c, on, off, iid, f0) in enumerate(ws):
            k = category(w, c)
            if k is None: continue
            if k == 'SELF':                       # amendment 3: object / possessive emphasis only
                nxt = ws[wi + 1][0] if wi + 1 < len(ws) and ws[wi + 1][4] == iid else ''
                if not (w in ('me', 'my', 'myself', 'mine', '自己') or (w == '我' and nxt == '的')): continue
            t = on - 0.2 if a.proto == 'tts' else on
            if iid in used or t - last < 3.0 or t < 1.0 or t > len(y) / FPS - 1.0: continue     # amendment 3: 1 per sentence, 3 s
            used.add(iid)
            h = free_hand(y, t); e = dict(ci=ci, clip=stem, cat=k, word=w, onset=float(on), t=float(t), hand=h)
            if k in ('SELF', 'YOU', 'POINT'):
                tgt = STERNUM.tolist() if k == 'SELF' else YOU if k == 'YOU' else HERE if w in ('this', 'here', 'these', '这') else [1.5, 0.8 if h == 0 else -0.8, 1.2]
                e.update(kind='aim', target=list(map(float, tgt)))
            elif k == 'NUMBER':
                n = NUM.get(w, 1); e.update(kind='finger', fingers=[n, 0] if h == 0 else [0, n])
            else: e.update(kind=k.lower())
            e['shuffle'] = {kk: e[kk] for kk in ('target', 'hand', 'fingers') if kk in e}
            ev.append(e); last = t
    meta = dict(cis=cis, proto=a.proto, rule='expl_canon.py plan', per_cat={c: sum(e['cat'] == c for e in ev) for c in ('SELF', 'YOU', 'POINT', 'NUMBER', 'BIG', 'SMALL')})
    (O / f'PLAN_{a.proto}.json').write_text(json.dumps(dict(events=ev, n_events=len(ev), **meta)))
    tok = [e for e in ev if e['kind'] in ('aim', 'finger')]
    (O / f'TOKPLAN_{a.proto}.json').write_text(json.dumps(dict(events=tok, n_events=len(tok), **meta))); print(json.dumps(meta))


CHEST = (0.25, 0.05, 1.10)          # amendment 2: SELF chest keyframe (x, toward the midline, z)


def ik_pose(ik, q0, cat, hand=None):
    """amendment 2: present-style wrist position, orientation off (palm_fix, or_scale 0), 3 chained rounds x 30 iterations.
    BIG / SMALL: both hands (WRIST[cat]); SELF: the given hand to CHEST."""
    import torch
    q = np.asarray(q0, np.float32).copy(); info = []
    tg = [(h, (WRIST[cat][0], WRIST[cat][1] if h == 0 else -WRIST[cat][1], WRIST[cat][2])) for h in (0, 1)] if cat in WRIST else \
        [(hand, (CHEST[0], CHEST[1] if hand == 0 else -CHEST[1], CHEST[2]))]
    for h, (x, y, z) in tg:
        for _ in range(3):
            s = ik.geo(torch.as_tensor(q, dtype=torch.float64))['shoulder'][h].numpy()
            off = (x - s[0], (y - s[1]) * (1 if h == 0 else -1), z - s[2])            # present_off = (forward, OUTWARD, dz)
            q, inf = ik.solve(q, h, 'present', present_off=off, it=30, palm_fix=True, or_scale=0.0)
        info.append(inf)
    return q, info


def insert_key(y, t, cat, ik, a95, hand=None):
    from ex002_keyframe import insert9b, HOLD
    HOLD.setdefault('spread', (0.2, 0.2))
    f = int(np.clip(round(t * FPS), 0, len(y) - 1)); qk, info = ik_pose(ik, y[f], cat, hand); yi = y.copy(); tm = []
    for h in ((0, 1) if cat in WRIST else (hand,)): tm.append(insert9b(yi, qk, h, t, 'spread', a95))
    return yi, dict(ik=info, timing=tm)


def insert_point(y, e, ik, a95):
    """amendment 4: aim IK (ray toward the target + wrist at shoulder + 0.45 m toward it), 3 chained rounds x 30 iterations, hold 0.4 s
    over [t - 0.1, t + 0.3], quintic ramps (insert9b)."""
    import torch
    from ex002_keyframe import insert9b, HOLD
    HOLD.setdefault('point4', (0.1, 0.3))
    f = int(np.clip(round(e['t'] * FPS), 0, len(y) - 1)); q = y[f].copy(); h = e['hand']; tg = np.asarray(e['target'], np.float64)
    for _ in range(3):
        s = ik.geo(torch.as_tensor(q, dtype=torch.float64))['shoulder'][h].numpy(); wp = s + (tg - s) / np.linalg.norm(tg - s) * 0.45
        q, info = ik.solve(q, h, 'aim', target=tg.tolist(), wrist_pos=wp.tolist(), it=30)
    yi = y.copy(); tm = insert9b(yi, q, h, e['t'], 'point4', a95)
    return yi, dict(ik=info, timing=tm)


def keyframe(a):
    from ex002_keyframe import IK, A95
    P = json.loads(Path(a.plan).read_text())['events']; ik = IK(); a95 = json.loads(Path(A95).read_text())['a95']
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True); log = []
    for f in sorted(Path(a.run).glob('[0-9]*.npy'), key=lambda p: int(p.stem)):
        ci = int(f.stem); y = np.load(f).astype(np.float32)
        if a.point_only:                                                  # amendment 4: YOU / POINT point keyframes only
            for e in [e for e in P if e['ci'] == ci and e['cat'] in ('YOU', 'POINT')]:
                y, info = insert_point(y, e, ik, a95); log.append(dict(ci=ci, t=e['t'], cat=e['cat'], ik=info['ik']))
        else:
          for e in [e for e in P if e['ci'] == ci and (e['kind'] in ('big', 'small') or (a.self_chest and e['cat'] == 'SELF'))]:
            y, info = insert_key(y, e['t'], e['cat'], ik, a95, e['hand']); log.append(dict(ci=ci, t=e['t'], cat=e['cat'], ik=info['ik']))
        np.save(O / f.name, y.astype(np.float32))
    for g in Path(a.run).parent.glob('gt_*.npy'):
        d = O.parent / g.name
        if not d.exists(): d.symlink_to(g)
    (O / 'KEYFRAMES.json').write_text(json.dumps(log)); print('KEYFRAMES', len(log))


def success(y, e, ynone=None, det=None):
    from scmg import ex002_ctl as E, ctl
    n = len(y); fe = int(round(e['t'] * FPS)); geo = E.points(y); a0, a1 = max(0, fe - int(0.3 * FPS)), min(n, fe + int(0.9 * FPS) + 1)
    if e['cat'] == 'SELF':
        return float(np.linalg.norm(geo['wrist'][a0:a1, e['hand']] - STERNUM, axis=-1).min() <= 0.15)
    if e['cat'] in ('YOU', 'POINT'):
        h = e['hand']; f0, f1 = max(0, fe - FPS), min(n, fe + FPS); ah = det.apex(geo, h, f0, f1); rr = E.rays(geo, ah, h)
        sb, th = ctl.score_b(rr['forearm'][0], rr['forearm'][1], e['target'])          # amendment 2: angle + arm extension
        ext = np.linalg.norm(geo['wrist'][ah, h] - geo['shoulder'][ah, h]); return float(th <= 15.0 and ext >= 0.40)
    if e['cat'] == 'NUMBER':
        up = np.asarray(json.loads((Path(V) / 'META.json').read_text())['ex002']['upper'], np.float32)
        cnt = E.finger_counts(y, up)[max(0, fe - 10):min(n, fe + 11)]
        maj = [int(np.bincount(cnt[:, h], minlength=4).argmax()) for h in (0, 1)]; return float(maj == list(e['fingers']))
    sep = np.linalg.norm(geo['wrist'][a0:a1, 0] - geo['wrist'][a0:a1, 1], axis=-1)
    if e['cat'] == 'BIG':
        gn = E.points(ynone[a0:a1])['wrist']; return float(sep.max() >= np.linalg.norm(gn[:, 0] - gn[:, 1], axis=-1).max() + 0.15)
    low = geo['wrist'][a0:a1, :, 2].min(1) > 0.9
    return float(((sep <= 0.20) & low).any())


def detector():
    from ex002_apex import Detector
    return Detector(json.loads(Path(X + '/tokens/APEX.json').read_text()))


def validate(a):
    """synthetic positives: the commanded pose IK-inserted into the GT (aim: wrist on the line shoulder -> target at 0.45 m for
    YOU / POINT, at the sternum ref + 0.05 x for SELF; finger: the module-10 template override; BIG / SMALL: the keyframe);
    negatives: the GT. A metric passes if accuracy = (TP + TN) / 2N >= 0.90."""
    from ex002_keyframe import IK, A95
    import torch
    from ex002_ctl_run import finger_override
    P = json.loads(Path(a.plan).read_text())['events']; ik = IK(); a95 = json.loads(Path(A95).read_text())['a95']; det = detector()
    HS = json.loads(Path(X + '/tokens/HAND_SHAPES.json').read_text())['shapes']; res = {}
    for e in P:
        gt = np.load(Path(a.gt) / f"gt_{e['ci']}.npy").astype(np.float32); f = int(round(e['t'] * FPS))
        if e['cat'] == 'SELF':
            pos, _ = insert_key(gt, e['t'], 'SELF', ik, a95, e['hand'])
        elif e['kind'] == 'aim':
            g = ik.geo(torch.as_tensor(gt[f], dtype=torch.float64)); s = g['shoulder'][e['hand']].numpy(); tg = np.asarray(e['target'])
            wp = STERNUM + np.array([0.05, 0, 0]) if e['cat'] == 'SELF' else s + (tg - s) / np.linalg.norm(tg - s) * 0.45
            qk, _ = ik.solve(gt[f], e['hand'], 'aim', target=(tg if e['cat'] != 'SELF' else STERNUM + np.array([-0.3, 0, 0])).tolist(), wrist_pos=wp.tolist(), it=30)
            from ex002_keyframe import insert9b
            pos = gt.copy(); insert9b(pos, qk, e['hand'], e['t'], 'aim', a95)
        elif e['kind'] == 'finger':
            pos = finger_override(gt, [dict(t=e['t'], fingers=e['fingers'], clip=e['clip'])], HS)
        else:
            pos, _ = insert_key(gt, e['t'], e['cat'], ik, a95)
        sp_ = success(pos, e, gt, det); sn = success(gt, e, gt, det)
        res.setdefault(e['cat'], []).append((sp_, sn))
    out = {c: dict(n=len(v), tp=float(np.mean([x[0] for x in v])), fp=float(np.mean([x[1] for x in v])),
                   acc=float((np.sum([x[0] for x in v]) + np.sum([1 - x[1] for x in v])) / (2 * len(v)))) for c, v in res.items()}
    for c in out: out[c]['pass'] = out[c]['acc'] >= 0.90
    Path(a.out).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


def readable(a):
    """amendment 6 report: per NUMBER command, seconds within [t - 1, t + 2] with the presenting hand raised (wrist >= shoulder -
    0.20 m) AND its signed palm angle <= 45 deg (ex002_present conventions)."""
    from ex002_present import geo_all, PALM_SIGN, RAISE, present_hand
    P = [e for e in json.loads(Path(a.plan).read_text())['events'] if e['cat'] == 'NUMBER']; res = {}
    for spec in a.specs:
        lab, d = spec.split('=', 1); dur = []
        for e in P:
            y = np.load(Path(d) / f"{e['ci']}.npy"); g = geo_all(y); h = present_hand(e['fingers']); fe = int(round(e['t'] * FPS))
            f0, f1 = max(0, fe - FPS), min(len(y), fe + 2 * FPS); w, s_ = g['wrist'][f0:f1, h], g['shoulder'][f0:f1, h]
            nrm = np.cross(g['index_base'][f0:f1, h] - w, g['third_base'][f0:f1, h] - w)
            cs = PALM_SIGN[h] * nrm[:, 0] / np.maximum(np.linalg.norm(nrm, axis=-1), 1e-9); ang = np.degrees(np.arccos(np.clip(cs, -1, 1)))
            dur.append(float(((w[:, 2] >= s_[:, 2] - RAISE) & (ang <= 45.)).sum() / FPS))
        res[lab] = dict(median_s=float(np.median(dur)), iqr_s=[float(np.percentile(dur, 25)), float(np.percentile(dur, 75))], n=len(dur))
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def evaluate(a):
    """--plan / --none / every spec dir may be comma lists of equal length (one entry per seed): events are pooled over them and
    the bootstrap resamples recordings (all seeds of a recording together)."""
    plans = a.plan.split(','); nones = a.none.split(','); det = detector()
    runs = {'none': nones, **{k: v.split(',') for k, v in (s.split('=', 1) for s in a.specs)}}
    P, src = [], []
    for i, pl in enumerate(plans):
        for e in json.loads(Path(pl).read_text())['events']: P.append(e); src.append(i)
    hits = {}
    for lab, ds in runs.items():
        h = []
        for e, i in zip(P, src):
            y = np.load(Path(ds[i]) / f"{e['ci']}.npy"); yn = np.load(Path(nones[i]) / f"{e['ci']}.npy")
            h.append(success(y, e, yn, det))
        hits[lab] = np.array(h)
    cats = sorted({e['cat'] for e in P}); recs = np.array([e['clip'] for e in P]); cat = np.array([e['cat'] for e in P]); res = dict(n=len(P))
    from scmg import ex002_ctl as E_
    for lab, ds in runs.items():                    # report-only (main 11-09): BIG wrist height relative to the shoulder (max in the window)
        hz = []
        for e, i in zip(P, src):
            if e['cat'] != 'BIG': continue
            y = np.load(Path(ds[i]) / f"{e['ci']}.npy"); fe = int(round(e['t'] * FPS)); g = E_.points(y[max(0, fe - 15):fe + 46])
            hz.append(float((g['wrist'][..., 2] - g['shoulder'][..., 2]).max()))
        if hz: res.setdefault('BIG_wrist_minus_shoulder_max_m', {})[lab] = dict(mean=float(np.mean(hz)), max=float(np.max(hz)), n=len(hz))
    for lab, h in hits.items(): res[lab] = {c: dict(rate=float(h[cat == c].mean()), n=int((cat == c).sum())) for c in cats}
    rng = np.random.default_rng(0); ur = np.unique(recs)
    for lab in runs:
        if lab == 'none': continue
        g = {}
        for c in cats:
            m = cat == c; d = hits[lab][m].mean() - hits['none'][m].mean(); bs = []
            for _ in range(2000):
                pick = rng.choice(ur, len(ur)); idx = np.concatenate([np.nonzero((recs == r) & m)[0] for r in pick])
                if len(idx): bs.append(hits[lab][idx].mean() - hits['none'][idx].mean())
            ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else [None, None]
            g[c] = dict(delta=float(d), ci95=ci, pass_=bool(d >= 0.20 and ci[0] is not None and ci[0] > 0))
        res[f'GATE_{lab}'] = g
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan'); p.add_argument('--cis', required=True); p.add_argument('--proto', choices=['tts', 'psync'], required=True)
    p.add_argument('--none', required=True); p.add_argument('--out', required=True)
    k = sub.add_parser('keyframe'); k.add_argument('--plan', required=True); k.add_argument('--run', required=True); k.add_argument('--out', required=True)
    k.add_argument('--point-only', action='store_true', help='amendment 4: only the YOU / POINT IK point keyframes (applied on --run)')
    k.add_argument('--self-chest', action='store_true', help='amendment 2 SELF variant (b): chest keyframe instead of the aim token')
    v = sub.add_parser('validate'); v.add_argument('--plan', required=True); v.add_argument('--gt', required=True); v.add_argument('--out', required=True)
    e = sub.add_parser('eval'); e.add_argument('--plan', required=True); e.add_argument('--none', required=True); e.add_argument('--out', required=True); e.add_argument('specs', nargs='+')
    r = sub.add_parser('readable'); r.add_argument('--plan', required=True); r.add_argument('--out', required=True); r.add_argument('specs', nargs='+')
    a = ap.parse_args(); {'plan': plan, 'keyframe': keyframe, 'validate': validate, 'eval': evaluate, 'readable': readable}[a.cmd](a)

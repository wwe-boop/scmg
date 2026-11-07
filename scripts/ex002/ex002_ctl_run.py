"""EX002 commanded-event controllability (11-07; criteria docs/criteria/EX002_CTL.md).
  plan  : commands for every dev clip (380 val clips of data-v36_1): events at random times (seeded per clip, >= 1.5 s apart, >= 1 s
          from the ends), alternating AIM (hand uniform, target drawn from that hand's TRAIN target pool) and FINGER (counts per hand
          uniform 0..3); shuffle = every event's value replaced by the value of an event of the same kind in another clip (seeded).
  infer : generate the dev clips with modes none / given / shuffle (tokens of the plan; per_clip seed = seed + crc32(clip); offline
          H = inf, 10 steps, cfg 2.0, the EX002 evaluation defaults) -> <out>/<mode>_s<seed>/<ci>.npy (50 Hz, 34 joints) + gt_<ci>.npy
  eval  : per event: forearm-ray (OFFICIAL 2610.03105) and index-ray (ours) Score_a / Score_b / STG against the GIVEN and the ORIGINAL
          command (a_ref = command time, apex detector APEX.json, window +-1 s), finger count hit against given / original
          (counts at t +- 0.2 s, majority); FKD and FGD v2 (ours, validated) of the whole clips vs GT.
    SCMG_J=34 python scripts/ex002/ex002_ctl_run.py plan --ctl <tokens dir> --view V --out plan.json
    SCMG_J=34 python scripts/ex002/ex002_ctl_run.py infer --ckpt C --vae V --stats S --text-cache T --tok-cache K --plan plan.json --mode given --seed 0 --out D
    SCMG_J=34 python scripts/ex002/ex002_ctl_run.py eval --plan plan.json --ctl <tokens dir> --out res.json label=<dir> ...
"""
import argparse, json, os, sys, time, zlib
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
FPS = 50


def plan(a):
    from scmg import ctl
    V = Path(a.view); split = json.loads((V / 'SPLIT.json').read_text()); items = {x['id']: x for x in json.loads((V / 'train.json').read_text())}
    pools = np.load(Path(a.ctl) / 'TARGET_POOLS.npz'); pools = [pools['left'], pools['right']]
    from scmg.data import View
    st = json.loads(Path(a.stats).read_text()); view = View(a.view, 'val', a.text_cache, st, mmap=True)
    ev = []
    for ci, c in enumerate(view.clips):
        dur = c['frames'] / FPS; rng = np.random.default_rng(zlib.crc32(f"{c['clip']}".encode()) + 7)
        ts = np.arange(1.0 + rng.uniform(0, 1.5), dur - 1.0, 1.5 + rng.uniform(0, 0.5))
        for j, t in enumerate(ts):
            if j % 2 == 0:
                h = int(rng.integers(2)); tgt = pools[h][rng.integers(len(pools[h]))]
                ev.append(dict(ci=ci, clip=c['clip'], kind='aim', t=float(t), hand=h, target=[float(x) for x in tgt]))
            else:
                ev.append(dict(ci=ci, clip=c['clip'], kind='finger', t=float(t), fingers=[int(rng.integers(4)), int(rng.integers(4))]))
    rng = np.random.default_rng(20261107)
    for kind, keys in (('aim', ('hand', 'target')), ('finger', ('fingers',))):
        idx = [i for i, e in enumerate(ev) if e['kind'] == kind]; perm = list(rng.permutation(idx))
        for _ in range(10):                                           # derangement across clips
            bad = [k for k, (i, j) in enumerate(zip(idx, perm)) if ev[i]['ci'] == ev[j]['ci']]
            if not bad: break
            for k in bad:
                s = int(rng.integers(len(perm))); perm[k], perm[s] = perm[s], perm[k]
        for i, j in zip(idx, perm): ev[i]['shuffle'] = {k: ev[j][k] for k in keys}
    out = dict(n_clips=len(view.clips), n_events=len(ev), n_aim=sum(e['kind'] == 'aim' for e in ev), events=ev, rule=__doc__.split('\n')[1])
    Path(a.out).write_text(json.dumps(out)); print({k: v for k, v in out.items() if k != 'events'})


def tokens_for(evs, mode):
    from scmg import ctl
    toks = []
    for e in evs:
        val = e if mode == 'given' else dict(e, **e['shuffle'])
        if e['kind'] == 'aim': toks.append(dict(type=ctl.POINT, t=e['t'], target=val['target'], hand=val['hand']))
        else: toks.append(dict(type=ctl.TYPE_ID['beat'], t=e['t'], fingers=val['fingers']))
    return toks


def finger_override(y, toks, shapes, hold=0.4, ramp=0.15):
    """y [T, 34] 50 Hz: at every finger command the 8 finger joints of each hand are blended to the canonical hand shape of the
    commanded count (HAND_SHAPES.json: median of train frames with exactly that count) over [t - hold, t + hold], cosine ramps of
    `ramp` s outside it; arms / torso untouched (CBF-QP runs afterwards)."""
    y = y.copy(); T = len(y); fr = np.arange(T) / FPS
    for t in toks:
        if t.get('fingers') is None: continue
        d = np.abs(fr - t['t']); w = np.clip((hold + ramp - d) / ramp, 0, 1); w = 0.5 - 0.5 * np.cos(np.pi * w)
        for h, (lo, hi) in ((0, (18, 26)), (1, (26, 34))):
            tgt = np.asarray(shapes[f"{h}_{int(t['fingers'][h])}"], np.float32)
            y[:, lo:hi] = (1 - w[:, None]) * y[:, lo:hi] + w[:, None] * tgt[None]
    return y


def infer(a):
    from scmg.common import CHUNK_FRAMES
    from scmg.data import View, WindowDataset
    from scmg.evaluate import load
    from scmg.infer import generate
    from scmg.model import INF
    from scmg.ctl import token_value
    dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    st = json.loads(Path(a.stats).read_text()); m, vae = load(a.ckpt, a.vae, dev)
    view = View(a.view, 'val', a.text_cache, st, mmap=True, tok_cache=a.tok_cache)
    P = json.loads(Path(a.plan).read_text()); by = {}
    for e in P['events']: by.setdefault(e['ci'], []).append(e)
    qm, qs = np.asarray(st['q_mean'], np.float32), np.asarray(st['q_std'], np.float32)
    si, sn = (int(x) for x in a.shard.split('/')); t0 = time.time()
    for ci in range(len(view.clips))[si::sn]:
        o = out / f'{ci}.npy'
        if o.exists(): continue
        c = view.clips[ci]; L = c['frames']; nch = L // CHUNK_FRAMES
        if nch < 3: continue
        s = WindowDataset(view, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
        cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c') if k in s}
        if a.mode != 'none' and by.get(ci):
            toks = tokens_for(by[ci], a.mode); tv = [token_value(t) for t in toks]
            cond.update(ctl_t=np.array([t['t'] for t in toks], np.float32), ctl_type=np.array([x[0] for x in tv], np.int64),
                        ctl_v=np.stack([x[1] for x in tv]), ctl_vm=np.stack([x[2] for x in tv]))
        seed = (a.seed + zlib.crc32(c['clip'].encode())) % (2 ** 63)
        cmd, _, _ = generate(m, vae, s['feat'], cond, st, nch, H=INF, steps=a.steps, seed=seed, standby=qm, ctx_t=m.ctx_t, cfg_w=a.cfg_w)
        y = cmd[:L].astype(np.float32)
        if a.finger_override and a.mode != 'none':      # 11-07 module 2 (default off): joint-space hand-shape override after decoding
            y = finger_override(y, tokens_for(by.get(ci, []), a.mode), json.loads(Path(a.finger_override).read_text())['shapes'])
        np.save(o, y)
        g = out.parent / f'gt_{ci}.npy'
        if not g.exists(): np.save(g, (s['q'] * qs + qm)[:L].astype(np.float32))
        if ci % 40 == 0: print(f'{ci}/{len(view.clips)} {time.time() - t0:.0f}s', flush=True)
    print('EX002_CTL_INFER_DONE', a.mode, a.seed)


def evaluate(a):
    from scmg import ex002_ctl as E, ctl
    from ex002_apex import Detector
    from scmg.evaluate import kin_feats, frechet
    meta = json.loads((Path(a.view) / 'META.json').read_text()); up = np.asarray(meta['ex002']['upper'], np.float32)
    det = Detector(json.loads((Path(a.ctl) / 'APEX.json').read_text()))
    P = json.loads(Path(a.plan).read_text()); by = {}
    for e in P['events']: by.setdefault(e['ci'], []).append(e)
    sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/fgd-ex002-v2-20261102/code')
    emb = None
    try:
        from model import Embedder
        from config import WEIGHTS
        emb = [Embedder(WEIGHTS / f'human_seed{k}' / 'best.pt') for k in (0, 1)]       # as score_dirs.py (frozen FGD v2)
    except Exception as ex: print('FGD v2 embedder unavailable:', ex)
    res = {}
    for spec in a.specs:
        lab, d = spec.split('=', 1); d = Path(d); gt_dir = d.parent
        acc = {k: [] for k in ('sa_g', 'sb_g', 'stg_g', 'sa_o', 'sb_o', 'stg_o', 'stgi_g', 'stgi_o', 'fh_g', 'fh_o', 'fl_g', 'fr_g')}
        Fg, Ft, gens, gts = [], [], [], []
        for f in sorted(d.glob('*.npy'), key=lambda p: int(p.stem)):
            ci = int(f.stem); y = np.load(f); gt = np.load(gt_dir / f'gt_{ci}.npy'); n = min(len(y), len(gt)); y, gt = y[:n], gt[:n]
            Fg += kin_feats(y); Ft += kin_feats(gt); gens.append(y); gts.append(gt)
            geo = E.points(y); cnt = E.finger_counts(y, up)
            for e in by.get(ci, []):
                fe = int(round(e['t'] * FPS))
                if fe >= n: continue
                own, shuf = dict(e), dict(e, **e['shuffle'])
                # 'g' = the command this run was given (shuffle runs: the shuffled value; none / given runs: the plan's own value),
                # 'o' = the other one (shuffle runs: the original; none / given runs: the shuffled value = a random-command baseline)
                for tag, v in (('g', shuf if a.shuffled_run else own), ('o', own if a.shuffled_run else shuf)):
                    if e['kind'] == 'aim':
                        h = v['hand']; f0, f1 = max(0, fe - FPS), min(n, fe + FPS); ah = det.apex(geo, h, f0, f1)
                        sa = ctl.score_a(ah / FPS, e['t'])
                        rr = E.rays(geo, ah, h)
                        sb, _ = ctl.score_b(rr['forearm'][0], rr['forearm'][1], v['target'])
                        sbi, _ = ctl.score_b(rr['index'][0], rr['index'][1], v['target'])
                        acc['sa_' + tag].append(sa); acc['sb_' + tag].append(sb); acc['stg_' + tag].append(sa * sb); acc['stgi_' + tag].append(sa * sbi)
                    else:
                        w = cnt[max(0, fe - 10):min(n, fe + 11)]
                        maj = [int(np.bincount(w[:, h], minlength=4).argmax()) for h in (0, 1)]
                        acc['fh_' + tag].append(float(maj == list(v['fingers'])))
                        if tag == 'g': acc['fl_g'].append(float(maj[0] == v['fingers'][0])); acc['fr_g'].append(float(maj[1] == v['fingers'][1]))
        r = {k: (float(np.mean(x)) if x else None) for k, x in acc.items()}
        r['n_aim'] = len(acc['stg_g']); r['n_finger'] = len(acc['fh_g']); r['FKD'] = float(frechet(Fg, Ft)) if len(Fg) > 110 else None
        if emb:
            r['FGD_v2_seeds'] = [float(frechet(np.concatenate([e_.features(g) for g in gts]), np.concatenate([e_.features(y) for y in gens]))) for e_ in emb]
            r['FGD_v2'] = float(np.mean(r['FGD_v2_seeds']))
        res[lab] = r; print(lab, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


def main():
    C = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    for n_ in ('plan', 'infer', 'eval'):
        p = sub.add_parser(n_); p.add_argument('--view', default=C + '/train-v1-20261101/views/data-v36_1'); p.add_argument('--out', required=True)
        p.add_argument('--stats', default=C + '/scmg/STATS_v36_1.json'); p.add_argument('--text-cache', default=C + '/scmg/text-e5-v36_1')
    sub.choices['plan'].add_argument('--ctl', required=True)
    p = sub.choices['infer']
    for k in ('ckpt', 'vae', 'plan'): p.add_argument('--' + k, required=True)
    p.add_argument('--tok-cache', default=C + '/scmg/tok-qwen25-v36_1'); p.add_argument('--mode', choices=['none', 'given', 'shuffle'], required=True)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--steps', type=int, default=10); p.add_argument('--cfg-w', type=float, default=2.0)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--finger-override', default=None, help='HAND_SHAPES.json: joint-space hand-shape override at finger commands (module 2, default off)')
    p = sub.choices['eval']; p.add_argument('--plan', required=True); p.add_argument('--ctl', required=True)
    p.add_argument('--shuffled-run', action='store_true', help='the runs were generated with the SHUFFLED values (mode shuffle)')
    p.add_argument('specs', nargs='+')
    a = ap.parse_args(); dict(plan=plan, infer=infer, eval=evaluate)[a.cmd](a)


if __name__ == '__main__':
    main()

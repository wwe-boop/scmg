"""11-07 EX002 interruption benchmark (3b): BEAT2 post-speech tails retargeted to EX002 three-finger (34 joints, 50 Hz) with the
frozen release06 retargeter (smplx2ex002_3f, BEAT source, near=False), the read-only route of ex002-echogeval-1105 retarget_worker:
whole BEAT2 recording (H.load_beat2(npz, 0.0, None, verts=True)) -> retarget(seq, fps, near=False)['q_full'] [T50, 37] -> 50 Hz frame
window -> learn_index (34). Run with the byte-identical environment (MUJOCO_GL=disable, PYTHONHASHSEED=0,
NPY_DISABLE_CPU_FEATURES=AVX512F,AVX2,FMA3, single-thread BLAS, CPU only).
  verify : retarget the recordings of N train items and compare the item windows with their data-v36_1 rows (train_q)
  tails  : every selected candidate of human_gaps.py --select-only (candidates.json): window [t_off - pre, t_off + post] ->
           <out>/seg/<id>.npz (q34 [T, 34] float32 rad, off = offset frame (50 Hz), fps 50) + index.<i>of<n>.jsonl; retarget
           rejections are logged and kept out (no backfill)
    python ex002_retarget_tails.py verify --view V --n 5 --out O
    python ex002_retarget_tails.py tails --cands candidates.json --beat2 B --out O --shard i/n [--pre 1.5 --post 3.0]"""
import argparse, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'disable'); os.environ.setdefault('PYTHONHASHSEED', '0')
PKG = Path('/mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture-dlc-20260928/smplx2ex002-r2-20261022/candidates/codex-dlc-release06-beat-nearoff')
sys.path.insert(0, str(PKG))
LDIR = dict(chinese='beat_chinese_v2.0.0', english='beat_english_v2.0.0', zh='beat_chinese_v2.0.0', en='beat_english_v2.0.0')
BEAT2 = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/beat2-addl-r1'


def q_full(npz):
    from smplx2ex002_3f import human as H, retarget
    d = H.load_beat2(str(npz), 0.0, None, verts=True)
    return retarget(d, d['fps'], near=False)['q_full'].astype(np.float32)


def learn(q37, view):
    li = json.loads((Path(view) / 'META.json').read_text())['ex002']['learn_index']
    return q37[:, li]


def verify(a):
    v = Path(a.view); items = json.loads((v / 'train.json').read_text()); sp = set(json.loads((v / 'SPLIT.json').read_text())['train_ids'])
    q = np.load(v / 'train_q.npy', mmap_mode='r')
    its = sorted([it for it in items if it['id'] in sp and str(it.get('source', '')).startswith('beat2') and '-w' not in it['clip']], key=lambda it: it['id'])   # -wNN clips: window-relative times
    pick = [its[i] for i in np.random.default_rng(0).choice(len(its), a.n, replace=False)]
    res = []
    for it in pick:
        rec = it['clip'].split('-', 2)[2]; npz = Path(a.beat2, LDIR[it['lang']], 'smplxflame_30', f'{rec}.npz')
        t0 = time.time(); Q = learn(q_full(npz), a.view)
        f0 = int(round(float(it['clip_start_s']) * 50)); n = int(it['frames'])
        mine = Q[f0:f0 + n]; view = np.asarray(q[it['offset']:it['offset'] + n], np.float32)
        m = min(len(mine), len(view)); d = np.abs(mine[:m] - view[:m])
        inner = d[10:m - 10] if m > 20 else d
        r = dict(id=it['id'], frames=n, got=len(mine), max=float(d.max()), mean=float(d.mean()), inner_max=float(inner.max()), inner_mean=float(inner.mean()),
                 bitwise=bool(np.array_equal(mine[:m], view[:m])), seconds=round(time.time() - t0, 1))
        res.append(r); print(json.dumps(r), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / 'VERIFY.json').write_text(json.dumps(res, indent=1))


def tails(a):
    C = json.loads(Path(a.cands).read_text()); sel = C['selected']
    si, sn = (int(x) for x in a.shard.split('/'))
    recs = sorted({(c['lang'], c['recording']) for c in sel}); mine = [r for k, r in enumerate(recs) if k % sn == si]
    out = Path(a.out); (out / 'seg').mkdir(parents=True, exist_ok=True); fo = open(out / f'index.{si}of{sn}.jsonl', 'a'); t0 = time.time()
    for k, (lang, rec) in enumerate(mine):
        cs = [c for c in sel if (c['lang'], c['recording']) == (lang, rec)]
        if all((out / 'seg' / f"{c['id']}.npz").exists() for c in cs): continue
        try: Q = learn(q_full(Path(a.beat2, LDIR[lang], 'smplxflame_30', f'{rec}.npz')), a.view)
        except Exception as e:
            fo.write(json.dumps(dict(recording=rec, lang=lang, rejected=repr(e)[:300], n=len(cs))) + '\n'); fo.flush(); continue
        for c in cs:
            s = int(round((c['t_off'] - a.pre) * 50)); off = int(round(c['t_off'] * 50)) - s; e = min(len(Q), int(round((c['t_off'] + c['post']) * 50)))
            if s < 0 or e - s <= off: continue
            seg = Q[s:e]
            tmp = out / 'seg' / f"{c['id']}.tmp.npz"; np.savez(tmp, q34=seg, off=off, fps=50); os.replace(tmp, out / 'seg' / f"{c['id']}.npz")
            fo.write(json.dumps(dict(c, seg_frames=len(seg), off=off)) + '\n'); fo.flush()
        print(f'{k + 1}/{len(mine)} {rec} {len(cs)} tails {time.time() - t0:.0f}s', flush=True)
    print('TAILS_DONE', si, sn)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    for n_ in ('verify', 'tails'):
        p = sub.add_parser(n_); p.add_argument('--view', default='/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1')
        p.add_argument('--beat2', default=BEAT2); p.add_argument('--out', required=True)
    sub.choices['verify'].add_argument('--n', type=int, default=5)
    p = sub.choices['tails']; p.add_argument('--cands', required=True); p.add_argument('--shard', default='0/1')
    p.add_argument('--pre', type=float, default=1.5)
    a = ap.parse_args(); dict(verify=verify, tails=tails)[a.cmd](a)


if __name__ == '__main__':
    main()

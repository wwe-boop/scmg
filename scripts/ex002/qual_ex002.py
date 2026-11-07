"""EX002 qualitative set (11-08, user: "需要确保实际的效果"; coordinator rule written before rendering, see SELECTION_RULE below).
Source: the fresh-plan module-5b confirmation (control-token-1105/ex002/m5b-confirm/fresh, DLC A800 generations, seed 3; dev only).
  select : fixed rule -> <out>/SELECT.json
  prep   : 10 s windows -> q37 npz per panel, recording audio wav, per-frame draw primitives (target marker, forearm ray, index ray,
           finger counts) in the kinematics frame, Isaac jobs.json for scripts/ex002/render_qual_isaac.py
  sheets : 3x4 contact sheet PNG per clip (frames around the command; evenly spaced for no-command / GT clips)
  manifest: MANIFEST.json (clip, mode, seed, event time, metric values)
    SCMG_J=34 python scripts/ex002/qual_ex002.py select|prep|sheets|manifest --out OUT
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
FPS = 50
B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; X = B + '/control-token-1105/ex002'; R = X + '/m5b-confirm/fresh'
FD, FS = X + '/PLAN_dev_fresh_dense.json', X + '/PLAN_dev_fresh_sparse.json'
SEED = 3; WIN = 10.0; QWIN = 6.0
SELECTION_RULE = ('seed 3 (first fresh-plan inference seed); events whose 10 s window [t - 5, t + 5] lies inside the clip; ranked by '
                  'sha1("<clip>:<t:.3f>") ascending; first k with distinct clips: 6 aim events (given / none / shuffle), 6 finger events '
                  '(override vs plain); plus the 2 aim events with the lowest forearm STG of the 5b given run (any clip); 4 sparse-plan '
                  'windows of 10 s (start on a 1 s grid) with no command within the window +-1 s, ranked by sha1("<clip>:<start>"), '
                  'distinct clips (none vs 5b + override). AMENDED before any rendering (11-08 ~02:00): the sparse plan has a command every ~6.5 s, so '
                  'no 10 s window is command-free (0 windows; 66 of 6 s with no margin) -> quiet clips = 6 s command-free windows (no '
                  'margin), first 4 by sha1, distinct clips; panels none / 5b + override / unblended lane-4 given (under 5b these frames '
                  'equal the no-token generation by construction except the blend ramps, so the unblended model is shown too); 2 GT windows: clips ranked by sha1("<clip>"), the centred 10 s')
h = lambda s: hashlib.sha1(s.encode()).hexdigest()


def load_plan(p):
    P = json.loads(Path(p).read_text()); by = {}
    for e in P['events']: by.setdefault(e['ci'], []).append(e)
    return P, by


def clip_len(ci): return len(np.load(f'{R}/../fresh/gen/none_s{SEED}/{ci}.npy')) / FPS


def stg_events(d, by, ci_list):
    from scmg import ex002_ctl as E, ctl
    from ex002_apex import Detector
    det = Detector(json.loads((Path(X) / 'tokens/APEX.json').read_text())); out = []
    for ci in ci_list:
        y = np.load(f'{d}/{ci}.npy'); n = len(y); geo = E.points(y)
        for e in by.get(ci, []):
            if e['kind'] != 'aim': continue
            fe = int(round(e['t'] * FPS))
            if fe >= n: continue
            hh = e['hand']; ah = det.apex(geo, hh, max(0, fe - FPS), min(n, fe + FPS)); sa = ctl.score_a(ah / FPS, e['t']); rr = E.rays(geo, ah, hh)
            sb, _ = ctl.score_b(rr['forearm'][0], rr['forearm'][1], e['target']); sbi, _ = ctl.score_b(rr['index'][0], rr['index'][1], e['target'])
            out.append(dict(ci=ci, t=e['t'], stg=sa * sb, stg_index=sa * sbi, apex_t=ah / FPS))
    return out


def select(a):
    PD, byd = load_plan(FD); PS, bys = load_plan(FS); cis = sorted(byd)
    L = {ci: clip_len(ci) for ci in cis}
    ok = lambda e: e['t'] - 5 >= 0 and e['t'] + 5 <= L[e['ci']]
    def first(evs, k):
        got, seen = [], set()
        for e in sorted(evs, key=lambda e: h(f"{e['clip']}:{e['t']:.3f}")):
            if e['ci'] in seen: continue
            got.append(e); seen.add(e['ci'])
            if len(got) == k: break
        return got
    aim = first([e for e in PD['events'] if e['kind'] == 'aim' and ok(e)], 6)
    fin = first([e for e in PD['events'] if e['kind'] == 'finger' and ok(e)], 6)
    st = stg_events(f'{R}/dense/given_s{SEED}', byd, cis)
    worst = sorted([s for s in st if s['t'] - 5 >= 0 and s['t'] + 5 <= L[s['ci']]], key=lambda s: s['stg'])[:2]
    worst_ev = [next(e for e in PD['events'] if e['ci'] == w['ci'] and abs(e['t'] - w['t']) < 1e-9) for w in worst]
    quiet = []
    for ci in sorted(bys):
        ts = [e['t'] for e in bys[ci]]
        for s0 in range(0, int(L[ci] - QWIN) + 1):
            if all(t < s0 or t > s0 + QWIN for t in ts): quiet.append(dict(ci=ci, clip=bys[ci][0]['clip'], start=float(s0)))
    qs, seen = [], set()
    for q in sorted(quiet, key=lambda q: h(f"{q['clip']}:{int(q['start'])}")):
        if q['ci'] in seen: continue
        qs.append(q); seen.add(q['ci'])
        if len(qs) == 4: break
    clips = {e['ci']: e['clip'] for e in PD['events']}
    gt = [dict(ci=ci, clip=clips[ci], start=max(0., L[ci] / 2 - 5)) for ci in sorted(clips, key=lambda c: h(clips[c]))[:2]]
    S = dict(rule=SELECTION_RULE, seed=SEED, source=R, aim=aim, finger=fin, worst_aim=worst_ev, worst_stg=worst, quiet=qs, gt=gt, n_aim_scored=len(st))
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / 'SELECT.json').write_text(json.dumps(S, indent=1))
    print('SELECT', len(aim), len(fin), len(worst_ev), len(qs), len(gt), [round(w['stg'], 3) for w in worst])


def q37(q34):
    from traj_io import JOINT_ORDER, MIMIC_A, MIMIC_B, LIFT_FIXED
    meta = json.loads(Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1/META.json').read_text())['ex002']
    names = meta['names34']; out = np.zeros((len(q34), 37))
    for i, n in enumerate(JOINT_ORDER):
        if n in names: out[:, i] = q34[:, names.index(n)]
    out[:, 0] = LIFT_FIXED
    for s in ('left', 'right'):
        out[:, JOINT_ORDER.index(f'{s}_1_4_joint')] = MIMIC_A * out[:, JOINT_ORDER.index(f'{s}_1_3_joint')] + MIMIC_B
    return out


def rec_audio(ci):
    """recording PCM (16 kHz int16) = the clip's items' audio in offset order (bench_ex002 rec_inputs rule)."""
    from scmg.data import View
    from scmg import causal_feat as CF
    C = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'
    st = json.loads(Path(C + '/scmg/STATS_v36_1.json').read_text())
    v = View(C + '/train-v1-20261101/views/data-v36_1', 'val', C + '/scmg/text-e5-v36_1', st, mmap=True); c = v.clips[ci]
    its = sorted(c['items'], key=lambda it: it['offset']); pcm = []
    for it in its:
        ma = it['manifest_audio'] if isinstance(it['manifest_audio'], dict) else eval(it['manifest_audio'])
        p, _ = CF.clip_audio(ma['path']); pcm.append(CF.pad_pcm(p, it['frames']))
    return np.concatenate(pcm).astype(np.int16)


def prim(y, target=None, hand=None, counts=None, win=None, upper=None):
    """per-frame primitives (kinematics frame): rays from elbow->wrist and index base->tip (1.5 m), target marker in its window."""
    from scmg import ex002_ctl as E
    g = E.points(y); fr = []
    cnt = E.finger_counts(y, upper) if counts is not None else None
    for k in range(len(y)):
        t = k / FPS; d = dict(seg=[], pts=[], txt=[])
        on = win is not None and win[0] <= t <= win[1]
        for hh in ((hand,) if hand is not None else ()):
            w, el = g['wrist'][k, hh], g['elbow'][k, hh]; u = (w - el) / max(np.linalg.norm(w - el), 1e-9)
            d['seg'].append([w.tolist(), (w + 1.5 * u).tolist(), [255, 80, 40]])
            ib, it_ = g['index_base'][k, hh], g['index_tip'][k, hh]; u2 = (it_ - ib) / max(np.linalg.norm(it_ - ib), 1e-9)
            d['seg'].append([it_.tolist(), (it_ + 1.5 * u2).tolist(), [40, 160, 255]])
            if on and target is not None: d['pts'].append([list(map(float, target)), [255, 220, 0], 14])
        if counts is not None:
            d['txt'].append(f"fingers L {int(cnt[k, 0])} R {int(cnt[k, 1])}" + (f"   command L {counts[0]} R {counts[1]}" if on else ''))
        fr.append(d)
    return fr, [g['wrist'][0, 0].tolist(), g['wrist'][0, 1].tolist()]


def prep(a):
    import soundfile as sf
    O = Path(a.out); S = json.loads((O / 'SELECT.json').read_text()); jobs = []; man = []
    upper = np.asarray(json.loads(Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1/META.json').read_text())['ex002']['upper'], np.float32)
    G = f'{R}/gen'; aud = {}
    def window(ci, t0, w=WIN):
        T = len(np.load(f'{G}/none_s{SEED}/{ci}.npy')); f0 = max(0, int(round(t0 * FPS))); return f0, min(T, f0 + int(w * FPS))
    def audio(ci, f0, f1, name):
        if ci not in aud: aud[ci] = rec_audio(ci)
        p = O / 'audio' / f'{name}.wav'; p.parent.mkdir(parents=True, exist_ok=True); sf.write(p, aud[ci][f0 * 320:f1 * 320], 16000, subtype='PCM_16'); return str(p)
    def panel(name, lab, src, f0, f1, **kw):
        y = np.load(src)[f0:f1]; p = O / 'traj' / f'{name}__{lab}.npz'; p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, Q37=q37(y), fps=50.0); fr, ref = prim(y, upper=upper, **kw); return dict(path=str(p), kind='q37', label=lab), dict(frames=fr, ref_wrist=ref)
    def add(name, title, f0, f1, ci, panels, kind, ev):
        trajs, draws = zip(*panels); dp = O / 'draw' / f'{name}.json'; dp.parent.mkdir(parents=True, exist_ok=True)
        dp.write_text(json.dumps(dict(robots=list(draws))))
        jobs.append(dict(trajs=list(trajs), audio=audio(ci, f0, f1, name), out=str(O / 'video' / f'{name}.mp4'), title=title, views=['front', 'q34'], draw=str(dp)))
        man.append(dict(name=name, kind=kind, ci=ci, clip=ev.get('clip'), seed=SEED, window_s=[f0 / FPS, f1 / FPS], event=ev, panels=[t['label'] for t in trajs]))
    for kind, evs in (('aim', S['aim']), ('aim_worst', S['worst_aim'])):
        for i, e in enumerate(evs):
            ci = e['ci']; f0, f1 = window(ci, e['t'] - 5); t = e['t'] - f0 / FPS; win = (t - 0.3, t + 0.9); sh = e['shuffle']
            name = f'{kind}{i}_{e["clip"]}_t{e["t"]:.2f}'
            add(name, f'{kind} {e["clip"]} t={e["t"]:.2f}s hand={"LR"[e["hand"]]} (seed 3, fresh plan)', f0, f1, ci,
                [panel(name, 'given (5b)', f'{R}/dense/given_s{SEED}/{ci}.npy', f0, f1, target=e['target'], hand=e['hand'], win=win),
                 panel(name, 'none', f'{G}/none_s{SEED}/{ci}.npy', f0, f1, target=e['target'], hand=e['hand'], win=win),
                 panel(name, 'shuffle (5b)', f'{R}/dense/shuffle_s{SEED}/{ci}.npy', f0, f1, target=sh['target'], hand=sh['hand'], win=win)], kind, e)
    for i, e in enumerate(S['finger']):
        ci = e['ci']; f0, f1 = window(ci, e['t'] - 5); t = e['t'] - f0 / FPS; win = (t - 0.4, t + 0.4); name = f'finger{i}_{e["clip"]}_t{e["t"]:.2f}'
        add(name, f'finger {e["clip"]} t={e["t"]:.2f}s command L{e["fingers"][0]} R{e["fingers"][1]} (seed 3)', f0, f1, ci,
            [panel(name, 'override (5b + module 2)', f'{R}/dense/fo_given_s{SEED}/{ci}.npy', f0, f1, counts=e['fingers'], win=win),
             panel(name, 'plain (5b)', f'{R}/dense/given_s{SEED}/{ci}.npy', f0, f1, counts=e['fingers'], win=win)], 'finger', e)
    for i, q in enumerate(S['quiet']):
        ci = q['ci']; f0, f1 = window(ci, q['start'], QWIN); name = f'quiet{i}_{q["clip"]}_s{int(q["start"])}'
        add(name, f'no command in window {q["clip"]} {q["start"]:.0f}-{q["start"] + QWIN:.0f}s (sparse plan, seed 3)', f0, f1, ci,
            [panel(name, 'none', f'{G}/none_s{SEED}/{ci}.npy', f0, f1), panel(name, '5b + override', f'{R}/sparse/fo_given_s{SEED}/{ci}.npy', f0, f1),
             panel(name, 'unblended lane 4 given', f'{G}/s_given_s{SEED}/{ci}.npy', f0, f1)], 'quiet', q)
    for i, g in enumerate(S['gt']):
        ci = g['ci']; f0, f1 = window(ci, g['start']); name = f'gt{i}_{g["clip"]}'
        add(name, f'GT {g["clip"]} {f0 / FPS:.1f}-{f1 / FPS:.1f}s', f0, f1, ci, [panel(name, 'GT (human, retargeted)', f'{G}/gt_{ci}.npy', f0, f1)], 'gt', g)
    (O / 'jobs.json').write_text(json.dumps(jobs, indent=1)); (O / 'MANIFEST.partial.json').write_text(json.dumps(man, indent=1)); print('PREP', len(jobs))


def sheets(a):
    import subprocess
    O = Path(a.out); man = json.loads((O / 'MANIFEST.partial.json').read_text()); (O / 'sheets').mkdir(exist_ok=True)
    for m in man:
        v = O / 'video' / f"{m['name']}.mp4"
        if not v.exists(): continue
        if m['kind'] in ('aim', 'aim_worst', 'finger'):
            t = m['event']['t'] - m['window_s'][0]; ts = np.linspace(t - 1.0, t + 1.5, 12)
        else: ts = np.linspace(0.3, m['window_s'][1] - m['window_s'][0] - 0.3, 12)
        fr = [int(round(x * FPS)) for x in ts]; sel = '+'.join(f'eq(n\\,{f})' for f in fr)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(v), '-vf', f"select='{sel}',scale=iw/2:-1,tile=3x4", '-frames:v', '1', '-vsync', '0',
                        str(O / 'sheets' / f"{m['name']}.png")], check=True)
        m['sheet_frames_s'] = [round(float(x), 2) for x in ts]
    (O / 'MANIFEST.partial.json').write_text(json.dumps(man, indent=1)); print('SHEETS done')


def manifest(a):
    O = Path(a.out); man = json.loads((O / 'MANIFEST.partial.json').read_text()); S = json.loads((O / 'SELECT.json').read_text())
    PD, byd = load_plan(FD); PS, bys = load_plan(FS)
    from scmg import ex002_ctl as E
    upper = np.asarray(json.loads(Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1/META.json').read_text())['ex002']['upper'], np.float32)
    for m in man:
        ci = m['ci']; met = {}
        if m['kind'] in ('aim', 'aim_worst'):
            for lab, d in (('given_5b', f'{R}/dense/given_s{SEED}'), ('none', f'{R}/gen/none_s{SEED}')):
                s = [x for x in stg_events(d, byd, [ci]) if abs(x['t'] - m['event']['t']) < 1e-9][0]; met[lab] = s
        if m['kind'] == 'finger':
            fe = int(round(m['event']['t'] * FPS))
            for lab, d in (('override', f'{R}/dense/fo_given_s{SEED}'), ('plain', f'{R}/dense/given_s{SEED}')):
                cnt = E.finger_counts(np.load(f'{d}/{ci}.npy'), upper)[max(0, fe - 10):fe + 11]
                maj = [int(np.bincount(cnt[:, hh], minlength=4).argmax()) for hh in (0, 1)]; met[lab] = dict(counts=maj, hit=maj == list(m['event']['fingers']))
        m['metrics'] = met
    (O / 'MANIFEST.json').write_text(json.dumps(dict(rule=S['rule'], source=R, seed=SEED, clips=man), indent=1)); print('MANIFEST', len(man))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['select', 'prep', 'sheets', 'manifest']); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd in ('prep',): sys.path.insert(0, '/home/xr-25b7/projects/gesture/codex/render-isaac')
    dict(select=select, prep=prep, sheets=sheets, manifest=manifest)[a.cmd](a)


if __name__ == '__main__':
    main()

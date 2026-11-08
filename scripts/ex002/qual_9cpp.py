"""EX002 module 9c'' render check (11-08; coordinator: the same two events as the 9b review -- present0 clip 10 t = 10.30 s and
aim0 clip 32 t = 20.56 s; panels tok5c / 9c' / 9c'' T-only + palm fix). Formerly: EX002 module 9b qualitative renders (11-08; rule fixed before rendering, committed before the m9b-enc numbers): fresh plan 1112 dense,
offline, seed 0, base enc-s0; the first aim event and the first finger (present) event by sha1("<clip>:<t:.3f>") whose 10 s window lies inside
the clip, plus the second aim event (same rule as module 9); panels token-only (5b + override) / module 9b (IK on 5b plain) / module 9b on 5c
(m9b5c, from the 5c job; the panel is left out only if that job has not produced the clip); overlays as qual_ex002 (yellow command direction, red forearm ray, blue index
ray, finger counts).  python qual_m9.py --out OUT   (then render_qual_isaac.py --jobs OUT/jobs.json, sheets via qual_ex002.py sheets)"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/xr-25b7/projects/gesture/codex/render-isaac')
X = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002'; R = sys.argv[sys.argv.index('--rend') + 1] if '--rend' in sys.argv else '/home/xr-25b7/bulk/qual-1108/ql9cpp/rend'; R5 = R; FD = X + '/PLAN_dev_m9b_dense.json'
h = lambda s: hashlib.sha1(s.encode()).hexdigest(); FPS = 50


def main():
    from qual_ex002 import q37, prim
    import soundfile as sf
    from qual_ex002 import rec_audio
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--rend', default=R); a = ap.parse_args(); O = Path(a.out)
    for d in ('traj', 'draw', 'audio', 'video'): (O / d).mkdir(parents=True, exist_ok=True)
    P = json.loads(Path(FD).read_text()); L = {}
    def ok(e):
        if e['ci'] not in L: L[e['ci']] = len(np.load(f"{R}/tok5c/{e['ci']}.npy")) / FPS
        return e['t'] - 5 >= 0 and e['t'] + 5 <= L[e['ci']]
    rank = lambda kind: [e for e in sorted([e for e in P['events'] if e['kind'] == kind], key=lambda e: h(f"{e['clip']}:{e['t']:.3f}")) if ok(e)]
    ev = {(e['ci'], round(e['t'], 2)): e for e in P['events']}; pick = [('present0', ev[(10, 10.30)]), ('aim0', ev[(32, 20.56)])]
    upper = np.asarray(json.loads(Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1/META.json').read_text())['ex002']['upper'], np.float32)
    jobs, man = [], []
    for name, e in pick:
        ci = e['ci']; f0 = int(round((e['t'] - 5) * FPS)); f1 = f0 + 10 * FPS; t = e['t'] - f0 / FPS; panels, draws = [], []
        for lab, d, root in (('token-only (5c)', 'tok5c', R), ("9c' (deictic pool)", 'm9cp', R), ("9c'' T + palm fix", 'm9cpTf', R)):
            y = np.load(f'{root}/{d}/{ci}.npy')[f0:f1]; tp = O / 'traj' / f'{name}__{d}.npz'; np.savez(tp, Q37=q37(y), fps=50.0)
            if e['kind'] == 'aim': fr, ref = prim(y, target=e['target'], hand=e['hand'], win=(t - 0.4, t + 0.9), upper=upper)
            else:
                hh = 0 if e['fingers'][0] > e['fingers'][1] else 1; fr, ref = prim(y, counts=e['fingers'], win=(t - 0.7, t + 0.7), upper=upper)
            panels.append(dict(path=str(tp), kind='q37', label=lab)); draws.append(dict(frames=fr, ref_wrist=ref))
        dp = O / 'draw' / f'{name}.json'; dp.write_text(json.dumps(dict(robots=draws)))
        ap_ = O / 'audio' / f'{name}.wav'; sf.write(ap_, rec_audio(ci)[f0 * 320:f1 * 320], 16000, subtype='PCM_16')
        title = f"{name} {e['clip']} t={e['t']:.2f}s " + (f"hand={'LR'[e['hand']]}" if e['kind'] == 'aim' else f"fingers L{e['fingers'][0]} R{e['fingers'][1]}") + '  | yellow = command, red = forearm ray, blue = index ray'
        jobs.append(dict(trajs=panels, audio=str(ap_), out=str(O / 'video' / f'{name}.mp4'), title=title, views=['front', 'q34'], draw=str(dp)))
        man.append(dict(name=name, kind=e['kind'], ci=ci, clip=e['clip'], seed=0, event_t=e['t'], window_s=[f0 / FPS, f1 / FPS], panels=[p['label'] for p in panels], event=e))
    (O / 'jobs.json').write_text(json.dumps(jobs, indent=1)); (O / 'MANIFEST.partial.json').write_text(json.dumps(man, indent=1)); print('QUAL_9CPP_PREP', len(jobs))


if __name__ == '__main__':
    main()

"""Explicit semantics canonical path: render check (11-09; exp/text-path docs/criteria/EXPLICIT_SEM_1109.md §4 + amendment 1; rule fixed
and committed before rendering). Words "me", "you", "three", "big": for each, the first command of PLAN_tts.json with that word by
sha1("<clip>:<onset:.3f>:explrender"); a word with no command is replaced by the first command of its category (NUMBER / BIG /
SELF / YOU) by the same hash, stated in the title. Panels: GT / none / given (canonical, TTS-sync row); for SELF also given with the
chest keyframe. Front + 3/4 view. Overlay: the target marker (yellow, aim commands), the commanded hand's wrist (yellow dot)
in [t - 0.3, t + 0.9], and text: category, word, command, success of the panel.
    python qual_expl.py --run RUN_DIR --out OUT"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'semantic')); sys.path.insert(0, '/home/xr-25b7/projects/gesture/codex/render-isaac')
FPS = 50; h = lambda s: hashlib.sha1(s.encode()).hexdigest()
P2 = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002/p2-quick-1109'


def main():
    from qual_ex002 import q37, rec_audio
    from scmg import ex002_ctl as E
    import soundfile as sf
    import expl_canon as C
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    R = Path(a.run); O = Path(a.out)
    for d in ('traj', 'draw', 'audio', 'video'): (O / d).mkdir(parents=True, exist_ok=True)
    P = json.loads((R / 'plans' / 'PLAN_tts.json').read_text())['events']; det = C.detector()
    rank = lambda xs: sorted(xs, key=lambda e: h(f"{e['clip']}:{e['onset']:.3f}:explrender"))
    pick = []
    for w, cat in (('me', 'SELF'), ('you', 'YOU'), ('three', 'NUMBER'), ('big', 'BIG')):
        xs = rank([e for e in P if e['word'] == w]); sub = False
        if not xs: xs = rank([e for e in P if e['cat'] == cat]); sub = True
        if xs: pick.append((w, sub, xs[0]))
    jobs, man = [], []
    for n, (w, sub, e) in enumerate(pick):
        ci = e['ci']; f0 = max(0, int(round((e['t'] - 5) * FPS))); f1 = f0 + 10 * FPS; t = e['t'] - f0 / FPS
        rows = [('GT', f'{P2}/psync/gt_{ci}.npy'), ('none', f'{P2}/psync/none_s0/{ci}.npy'), ('given (canonical)', f'{R}/final/tts/{ci}.npy')]
        if e['cat'] == 'SELF': rows.append(('given (chest keyframe)', f'{R}/final/tts_chest/{ci}.npy'))
        ynone = np.load(f'{P2}/psync/none_s0/{ci}.npy')
        panels, draws, info = [], [], []
        for lab, path in rows:
            yfull = np.load(path); f1_ = min(f1, len(yfull)); y = yfull[f0:f1_]
            ok = C.success(yfull, e, ynone, det); tp = O / 'traj' / f'ex{n}__p{len(panels)}.npz'; np.savez(tp, Q37=q37(y), fps=50.0)
            g = E.points(y); fr = []
            cmd = (f"aim {e['cat']} target ({e['target'][0]:.2f},{e['target'][1]:+.2f},{e['target'][2]:.2f}) hand {'LR'[e['hand']]}" if e['kind'] == 'aim' else
                   f"fingers L{e['fingers'][0]} R{e['fingers'][1]}" if e['kind'] == 'finger' else f"keyframe {e['cat']}")
            for k in range(len(y)):
                tk = k / FPS; d = dict(seg=[], pts=[], txt=[])
                if -0.3 <= tk - t <= 0.9:
                    d['pts'].append([g['wrist'][k, e['hand']].tolist(), [255, 220, 0], 10])
                    if e['kind'] == 'aim': d['pts'].append([e['target'], [255, 220, 0], 14])
                d['txt'].append(f"{e['cat']} '{e['word']}' t={e['t']:.2f}s | {cmd}"); d['txt'].append(f"success: {'YES' if ok else 'no'}")
                fr.append(d)
            panels.append(dict(path=str(tp), kind='q37', label=lab)); draws.append(dict(frames=fr, ref_wrist=[g['wrist'][0, 0].tolist(), g['wrist'][0, 1].tolist()]))
            info.append(dict(panel=lab, success=ok))
        dp = O / 'draw' / f'ex{n}.json'; dp.write_text(json.dumps(dict(robots=draws)))
        ap_ = O / 'audio' / f'ex{n}.wav'; au = rec_audio(ci)[f0 * 320:f1 * 320]; sf.write(ap_, au, 16000, subtype='PCM_16')
        title = f"explicit '{w}'" + (f" -> substituted by {e['cat']} '{e['word']}'" if sub else '') + f" | {e['clip']} onset {e['onset']:.2f}s, command t {e['t']:.2f}s (TTS-sync)"
        jobs.append(dict(trajs=panels, audio=str(ap_), out=str(O / 'video' / f'ex{n}.mp4'), title=title, views=['front', 'q34'], draw=str(dp)))
        man.append(dict(name=f'ex{n}', word=w, substituted=sub, event=e, panels=info))
    for k in sorted({len(j['trajs']) for j in jobs}):                 # the batch renderer needs equal panel counts per batch
        (O / f'jobs{k}.json').write_text(json.dumps([j for j in jobs if len(j['trajs']) == k], indent=1))
    (O / 'jobs.json').write_text(json.dumps(jobs, indent=1)); (O / 'MANIFEST.json').write_text(json.dumps(man, indent=1, default=str)); print('QUAL_EXPL_PREP', len(jobs))


if __name__ == '__main__':
    main()

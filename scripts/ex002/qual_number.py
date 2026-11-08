"""Explicit semantics: NUMBER hand close-up (main 11-09, legibility check). Rule fixed and committed before rendering: the full run's
seed-0 TTS-sync plan; the first NUMBER command by sha1("<clip>:<onset:.3f>:explrender"); panels GT / none / given; views front +
hands close-up; text: the commanded counts and the measured counts (majority over t +- 0.2 s) of each panel.
    python qual_number.py --run FULL_RUN_DIR --out OUT"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'semantic')); sys.path.insert(0, '/home/xr-25b7/projects/gesture/codex/render-isaac')
FPS = 50; ENC = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/ex002-sfscope-1108/enc-s0/psync'
V = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1'


def main():
    from qual_ex002 import q37, rec_audio
    from scmg import ex002_ctl as E
    import soundfile as sf
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--a5', action='store_true', help='amendment 5: the "three" (count-3) pick + the "first" pick, + the a5 panel'); a = ap.parse_args()
    R, O = Path(a.run), Path(a.out)
    for d in ('traj', 'draw', 'audio', 'video'): (O / d).mkdir(parents=True, exist_ok=True)
    P = [e for e in json.loads((R / 'plans' / 's0' / 'PLAN_tts.json').read_text())['events'] if e['cat'] == 'NUMBER']
    rank = lambda xs: sorted(xs, key=lambda e: hashlib.sha1(f"{e['clip']}:{e['onset']:.3f}:explrender".encode()).hexdigest())
    picks = [('first', rank(P)[0])]
    if a.a5:
        c3 = rank([e for e in P if max(e['fingers']) == 3]); picks = ([('three', c3[0])] if c3 else []) + picks
    up = np.asarray(json.loads((Path(V) / 'META.json').read_text())['ex002']['upper'], np.float32); jobs, man = [], []
    for name, e in picks:
        ci = e['ci']; f0 = max(0, int(round((e['t'] - 3) * FPS))); f1 = f0 + 6 * FPS
        rows = [('GT', f'{ENC}/gt_{ci}.npy'), ('none', f'{ENC}/s0/{ci}.npy'), ('given (canonical)', f'{R}/final/tts/s0/{ci}.npy')]
        if a.a5: rows.append(('given (amendment 5: present)', f'{R}/a5/tts_s0/{ci}.npy'))
        panels, draws, info = [], [], []
        for lab, path in rows:
            yf = np.load(path); y = yf[f0:min(f1, len(yf))]; fe = int(round(e['t'] * FPS))
            cnt = E.finger_counts(yf, up)[max(0, fe - 10):fe + 11]; maj = [int(np.bincount(cnt[:, h], minlength=4).argmax()) for h in (0, 1)]
            tp = O / 'traj' / f'{name}__p{len(panels)}.npz'; np.savez(tp, Q37=q37(y), fps=50.0)
            fr = [dict(seg=[], pts=[], txt=[f"NUMBER '{e['word']}' command L{e['fingers'][0]} R{e['fingers'][1]} at t={e['t']:.2f}s", f"measured L{maj[0]} R{maj[1]}"]) for _ in range(len(y))]
            panels.append(dict(path=str(tp), kind='q37', label=lab)); draws.append(dict(frames=fr, ref_wrist=[[0, 0, 0], [0, 0, 0]])); info.append(dict(panel=lab, measured=maj))
        dp = O / 'draw' / f'{name}.json'; dp.write_text(json.dumps(dict(robots=draws)))
        ap_ = O / 'audio' / f'{name}.wav'; sf.write(ap_, rec_audio(ci)[f0 * 320:f1 * 320], 16000, subtype='PCM_16')
        jobs.append(dict(trajs=panels, audio=str(ap_), out=str(O / 'video' / f'{name}.mp4'), title=f"NUMBER '{e['word']}' {e['clip']} onset {e['onset']:.2f}s (TTS-sync, seed 0) | hands close-up", views=['front', 'hands'], draw=str(dp)))
        man.append(dict(name=name, event=e, panels=info))
    (O / 'jobs.json').write_text(json.dumps(jobs, indent=1)); (O / 'MANIFEST.json').write_text(json.dumps(man, indent=1)); print('QUAL_NUMBER_PREP', [(m['name'], m['event']['word'], m['panels']) for m in man])


if __name__ == '__main__':
    main()

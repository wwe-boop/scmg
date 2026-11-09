"""11-08 qualitative EX002 interruption set (user: "需要确保实际的效果"): Isaac Sim renders (codex/render-isaac/render_ex002_isaac.py,
the EX002 default renderer) of the selected sweep base czm-s12000-lr1e-4-w0.1 at dev interruptions, 3 robots side by side, audio silent
from t_i, red INTERRUPT marker from t_i, + 3x4 contact sheet (12 frames over [t_i - 0.5 s, t_i + 2.0 s]).
Selection (fixed, RULE.json before rendering): interrupts_dev_ex002.json keys ci@item@frac sorted by sha1, first N.
Panels: B3 = the model's in-stream continuation (frames before f_i = its uninterrupted psync output: executed prefix);
        B2 = waist + arms critically damped from the frame before f_i to the human rest pose (T_r 1.8 s, icommon_ex002.retract_after);
        GT = the dev GT (the human keeps speaking; reference only). R0causal has no EX002 port yet (not shown).
Window [t_i - 3 s, t_i + 2.14 s] at 50 Hz. Per-clip metrics (icommon_ex002 + retargeter WBC clearance): react, v3 ratio, yield, unsafe frames.
    python scripts/ex002/qual_ex002_int.py prep --out O [--n 4]    (then render_ex002_isaac.py --jobs O/jobs.json, then)    ... post --out O"""
import argparse, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve(); sys.path.insert(0, str(HERE.parents[2])); sys.path.insert(0, str(HERE.parent))
B = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'); SEL = B / 'ex002-sweep-1107/czm-s12000-lr1e-4-w0.1'
INT = B / 'ex002-interrupt-1107/interrupts_dev_ex002.json'
VIEW = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1')
FF = os.environ.get('RENDER_FFMPEG', '/home/xr-25b7/bulk/pydeps/render/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v9.0.2')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
sha = lambda s: hashlib.sha1(s.encode()).hexdigest()
PRE, POST = 150, 107


def to37(q, li):
    o = np.zeros((len(q), 37)); o[:, 0] = 0.32; o[:, li] = q
    for m_, s_ in ((22, 21), (31, 30)): o[:, m_] = 1.389831 * o[:, s_] + 0.072772
    return o


def prep(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ints = json.load(open(INT))['interrupts']; key = lambda r: f"{r['ci']}@{r['item']}@{r['frac']:.1f}"
    pool = [r for r in ints if (SEL / 'int_psync_s0' / f'{key(r)}.npy').exists()]
    sel = sorted(pool, key=lambda r: sha(key(r)))[:a.n]
    hr = json.load(open(a.human + '.json')); rest = np.array(hr['rest']['R_sil']['joints'])
    rule = dict(set='ex002_interrupt', rule='interrupts_dev_ex002.json with the selected-base continuation present; sorted by sha1(ci@item@frac); first N',
                n=a.n, pool=len(pool), base=str(SEL), human_ref=a.human, T_r=1.8, selected=[dict(key=key(r), t_i=r['t_i'], f_i=r['f_i'], clip=r['clip']) for r in sel],
                note='R0causal has no EX002 port yet; panels B3 / B2 / GT')
    (out / 'RULE.json').write_text(json.dumps(rule, indent=1))
    import icommon_ex002 as I
    import bench_ex002 as BX
    from scmg import causal_feat as CF
    from scmg.data import View
    st = json.loads((VIEW.parent.parent.parent / 'scmg/STATS_v36_1.json').read_text()) if False else json.loads(Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/STATS_v36_1.json').read_text())
    view = View(str(VIEW), 'val', '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/text-e5-v36_1', st, mmap=True)
    li = json.loads((VIEW / 'META.json').read_text())['ex002']['learn_index']
    import soundfile as sf
    jobs = []
    for r in sel:
        k = key(r); f, ci = int(r['f_i']), int(r['ci']); f0, f1 = max(0, f - PRE), f + POST
        U = np.load(SEL / 'psync/s0' / f'{ci}.npy').astype(np.float64); Y = np.load(SEL / 'int_psync_s0' / f'{k}.npy').astype(np.float64)
        G = np.load(SEL / 'psync' / f'gt_{ci}.npy').astype(np.float64)
        B3 = np.concatenate([U[:f], Y[f:]])[f0:f1]; B2 = I.retract_after(U, f, rest, 1.8)[f0:f1]; GT = G[f0:f1]
        d = out / k.replace('@', '_'); d.mkdir(exist_ok=True)
        for lab, q in (('B3', B3), ('B2', B2), ('GT', GT)): np.savez(d / f'{lab}.npz', Q37=to37(q, li), fps=np.float64(50))
        pcm, wave, words, w0 = BX.rec_inputs(view, view.clips[ci], CF, a.words_dir)
        x = np.asarray(pcm, np.float32) / 32768.; x = x[int(f0 / 50 * 16000):int(f1 / 50 * 16000)].copy()
        x[int((f - f0) / 50 * 16000):] = 0.                                                # the model's input after t_i: silence
        sf.write(d / 'audio.wav', x, 16000)
        jobs.append(dict(trajs=[dict(path=str(d / f'{lab}.npz'), kind='q37', label=lab) for lab in ('B3', 'B2', 'GT')], audio=str(d / 'audio.wav'),
                         out=str(d / 'raw.mp4'), title=f'EX002 interruption {k}  (t_i at {3.0 if f0 > 0 else f / 50:.2f} s)', views=['front'], no_preview=True))
        np.save(d / 'meta.npy', np.array([f0, f, f1]))
    (out / 'jobs.json').write_text(json.dumps(jobs, indent=1)); print('PREP', len(jobs))


def post(a):
    out = Path(a.out); rule = json.loads((out / 'RULE.json').read_text()); man = dict(rule=rule, clips={})
    sys.path.insert(0, '/home/xr-25b7/projects/gesture/scmg-wt-safeproj/scripts/echog_g1')
    import qual_g1 as Q
    import icommon_ex002 as I
    os.environ.setdefault('MUJOCO_GL', 'disable')
    sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/gesture-dlc-20260928/smplx2ex002-r2-20261022/candidates/codex-dlc-release06-beat-nearoff')
    from smplx2ex002_3f import robot as RB
    wbc = RB.EX002(); hr = json.load(open(rule['human_ref'] + '.json')); rest_z = np.array(hr['rest']['R_sil']['hand_z'])
    li = json.loads((VIEW / 'META.json').read_text())['ex002']['learn_index']
    for s in rule['selected']:
        d = out / s['key'].replace('@', '_'); f0, f, f1 = np.load(d / 'meta.npy').tolist(); rel = (f - f0) / 50.
        raw, dst = d / 'raw.mp4', out / f"{s['key'].replace('@', '_')}.mp4"
        subprocess.run([FF, '-y', '-loglevel', 'error', '-i', str(raw), '-vf', f"drawbox=x=0:y=0:w=iw:h=ih:color=red@0.9:t=8:enable='gte(t,{rel:.3f})',"
                        f"drawtext=fontfile={FONT}:text='INTERRUPT':x=(w-text_w)/2:y=h-60:fontsize=36:fontcolor=red:box=1:boxcolor=white@0.7:enable='gte(t,{rel:.3f})'",
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-c:a', 'copy', str(dst)], check=True)
        ts = list(np.linspace(rel - 0.5, rel + 2.0, 12)); Q.contact_sheet(dst, ts, out / f"{s['key'].replace('@', '_')}_sheet.png", [f't-t_i {t - rel:+.2f} s' for t in ts])
        mets = {}
        q = {lab: np.load(d / f'{lab}.npz')['Q37'][:, li] for lab in ('B3', 'B2', 'GT')}; fl = f - f0
        for lab, x in q.items():
            P = I.fk(x); m = dict(unsafe_wbc_frames_pct=float(100 * np.mean([not wbc.clearance(z)[0] for z in x[fl:]])))
            ty, need = I.yield_time(P, fl, rest_z); m.update(yield_s=ty, needed=bool(need), peak_jerk=I.peak_jerk(P, fl - I.PRE, fl + I.WIN))
            if lab != 'GT':
                PU = I.fk(q['GT'] if False else np.load(SEL / 'psync/s0' / f"{s['key'].split('@')[0]}.npy").astype(np.float64)[f0:f1])
                m['v3_ratio'] = I.v3_ratio(P, PU, fl)
            mets[lab] = m
        man['clips'][s['key']] = dict(video=dst.name, sheet=f"{s['key'].replace('@', '_')}_sheet.png", t_i_in_video=rel, metrics=mets)
        print('POST', s['key'], flush=True)
    (out / 'MANIFEST.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('prep'); p.add_argument('--out', required=True); p.add_argument('--n', type=int, default=4)
    p.add_argument('--human', required=True); p.add_argument('--words-dir', default=str(B / 'ex002-causal-1107/words'))
    q = sp.add_parser('post'); q.add_argument('--out', required=True)
    a = ap.parse_args(); prep(a) if a.cmd == 'prep' else post(a)

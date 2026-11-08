"""11-08 EX002 review set for the current pipeline (coordinator: the user reviews the videos). Isaac Sim renders (codex/render-isaac
render_ex002_isaac.py, the EX002 default renderer), views front + q34 (45 deg), word-timed subtitles burned in, 3x4 / 4x3 contact sheets.
  base : 6 dev recordings (3 zh + 3 en; recording names ranked by sha1, first 3 per language), the centred 10 s window; two videos per
         recording: offline and psync, each GT / sibag-s2 / enc-s0 (per_clip inference seed 0 bench outputs).
  int  : 4 dev interruptions (2 zh + 2 en; keys ci@item@frac ranked by sha1, first 2 per language, all rows present), window
         [t_i - 3 s, t_i + 2.14 s], audio silent from t_i, red INTERRUPT marker from t_i; executed streams through 3c-4b + ISC
         (half-B reference for the post methods): B3 + ISC / R0causal + ISC / Funnel + ISC / L2diag + ISC / GT tail (a half-B human
         post-speech tail, its speech offset aligned to t_i; tails ranked by sha1(id), the k-th for the k-th clip).
  RULE.json is written (and committed) by `select` before any rendering.
    python scripts/ex002/qual_ex002_enc.py select|prep|post --out O"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve(); sys.path.insert(0, str(HERE.parents[2])); sys.path.insert(0, str(HERE.parent))
B = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'); X = B / 'ex002-interrupt-1107'
VIEW = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1')
SIB = B / 'ex002-lively-1108/diag-sibag-s2'; ENC = B / 'ex002-sfscope-1108/enc-s0'
PU = Path('/home/xr-25b7/bulk/causal-train-1105/runs/ex002-proj4b-enc/psync/s0')
INTROWS = dict(B3_ISC=X / 'eval-enc-s0/SAFE_ISC_B3', R0causal_ISC=X / 'eval-enc-s0-halfB/SAFE_ISC_R0causal',
               Funnel_ISC=X / 'eval-enc-s0-halfB/SAFE_ISC_Funnel', L2diag_ISC=X / 'eval-enc-s0-halfB/SAFE_ISC_L2diag')
FF = os.environ.get('RENDER_FFMPEG', '/home/xr-25b7/bulk/pydeps/render/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v9.0.2')
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
sha = lambda s: hashlib.sha1(s.encode()).hexdigest()
PRE, POST, WIN = 150, 107, 500


def to37(q, li):
    o = np.zeros((len(q), 37)); o[:, 0] = 0.32; o[:, li] = q
    for m_, s_ in ((22, 21), (31, 30)): o[:, m_] = 1.389831 * o[:, s_] + 0.072772
    return o


def view():
    from scmg.data import View
    st = json.loads(Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/STATS_v36_1.json').read_text())
    return View(str(VIEW), 'val', '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/text-e5-v36_1', st, mmap=True)


def select(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); v = view()
    cl = [(ci, c['clip'], c['frames']) for ci, c in enumerate(v.clips)]
    base = []
    for lang in ('zh', 'en'):
        xs = sorted([x for x in cl if x[1].split('-')[1] == lang], key=lambda x: sha(x[1]))[:3]
        for ci, c, n in xs: f0 = max(0, (n - WIN) // 2); base.append(dict(ci=ci, clip=c, lang=lang, f0=int(f0), f1=int(min(n, f0 + WIN))))
    ints = json.loads((X / 'interrupts_dev_ex002.json').read_text())['interrupts']; key = lambda r: f"{r['ci']}@{r['item']}@{r['frac']:.1f}"
    ok = [r for r in ints if all((d / f'{key(r)}.npy').exists() for d in INTROWS.values())]
    sel = []
    for lang in ('zh', 'en'):
        sel += sorted([r for r in ok if r['clip'].split('-')[1] == lang], key=lambda r: sha(key(r)))[:2]
    tails = sorted([p.stem for p in (X / 'tails/seg_halfB').glob('*.npz')], key=sha)
    rule = dict(base=dict(rule='dev recordings of data-v36_1 val; per language (zh / en) names ranked by sha1(name), first 3; the centred 10 s '
                               '(f0 = (frames - 500) // 2); panels GT / sibag-s2 / enc-s0, per_clip seed 0, offline and psync videos', clips=base),
                int=dict(rule='interrupts_dev_ex002.json with all four rows present; per language keys ranked by sha1(ci@item@frac), first 2; '
                              'window [f_i - 150, f_i + 107]; GT tail = half-B tails ranked by sha1(id), k-th for the k-th clip, offset aligned to t_i',
                         rows=list(INTROWS), selected=[dict(key=key(r), clip=r['clip'], ci=r['ci'], t_i=r['t_i'], f_i=r['f_i'], tail=tails[k]) for k, r in enumerate(sel)]),
                views=['front', 'q34'], subtitles='word-timed (alignment words of the recording; each word shown from its start to its end + 0.3 s)')
    (out / 'RULE.json').write_text(json.dumps(rule, indent=1, ensure_ascii=False)); print('SELECT', len(base), len(sel))


def prep(a):
    import soundfile as sf
    import bench_ex002 as BX
    from scmg import causal_feat as CF
    out = Path(a.out); rule = json.loads((out / 'RULE.json').read_text()); v = view()
    li = json.loads((VIEW / 'META.json').read_text())['ex002']['learn_index']; jb, ji = [], []
    for c in rule['base']['clips']:
        ci, f0, f1 = c['ci'], c['f0'], c['f1']; pcm, wave, words, w0 = BX.rec_inputs(v, v.clips[ci], CF, a.words_dir)
        for P in ('offline', 'psync'):
            d = out / 'base' / f"{c['clip']}_{P}"; d.mkdir(parents=True, exist_ok=True)
            src = dict(GT=SIB / P / f'gt_{ci}.npy', sibag_s2=SIB / P / 's0' / f'{ci}.npy', enc_s0=ENC / P / 's0' / f'{ci}.npy')
            for lab, p in src.items(): np.savez(d / f'{lab}.npz', Q37=to37(np.load(p).astype(np.float64)[f0:f1], li), fps=np.float64(50))
            x = np.asarray(pcm, np.float32) / 32768.; sf.write(d / 'audio.wav', x[int(f0 / 50 * 16000):int(f1 / 50 * 16000)], 16000)
            (d / 'words.json').write_text(json.dumps([dict(w=w['text'], s=w['start_time'] - f0 / 50, e=w['end_time'] - f0 / 50) for w in words
                                                     if f0 / 50 <= w['start_time'] < f1 / 50], ensure_ascii=False))
            jb.append(dict(trajs=[dict(path=str(d / f'{lab}.npz'), kind='q37', label=lab) for lab in src], audio=str(d / 'audio.wav'),
                           out=str(d / 'raw.mp4'), title=f"EX002 {c['clip']} {P}  GT | sibag-s2 | enc-s0", views=['front', 'q34'], no_preview=True))
    for k, s in enumerate(rule['int']['selected']):
        ci, f = s['ci'], s['f_i']; f0, f1 = max(0, f - PRE), f + POST; pcm, wave, words, w0 = BX.rec_inputs(v, v.clips[ci], CF, a.words_dir)
        d = out / 'int' / s['key'].replace('@', '_'); d.mkdir(parents=True, exist_ok=True); U = np.load(PU / f'{ci}.npy').astype(np.float64)
        labs = []
        for lab, dd in INTROWS.items():
            y = np.load(dd / f"{s['key']}.npy").astype(np.float64); z = np.concatenate([U[:f], y[f:]])[f0:f1]
            np.savez(d / f'{lab}.npz', Q37=to37(z, li), fps=np.float64(50)); labs.append(lab)
        z_ = np.load(X / 'tails/seg' / f"{s['tail']}.npz"); t = z_['q34'].astype(np.float64); off = int(z_['off'])
        pre = t[max(0, off - (f - f0)):off]; pre = np.concatenate([np.repeat(pre[:1], (f - f0) - len(pre), 0), pre]) if len(pre) < f - f0 else pre
        post = t[off:off + POST]; post = np.concatenate([post, np.repeat(post[-1:], POST - len(post), 0)]) if len(post) < POST else post
        np.savez(d / 'GT_tail.npz', Q37=to37(np.concatenate([pre, post]), li), fps=np.float64(50)); labs.append('GT_tail')
        x = np.asarray(pcm, np.float32) / 32768.; x = x[int(f0 / 50 * 16000):int(f1 / 50 * 16000)].copy(); x[int((f - f0) / 50 * 16000):] = 0.
        sf.write(d / 'audio.wav', x, 16000)
        (d / 'words.json').write_text(json.dumps([dict(w=w['text'], s=w['start_time'] - f0 / 50, e=w['end_time'] - f0 / 50) for w in words
                                                 if f0 / 50 <= w['start_time'] < f / 50], ensure_ascii=False))
        np.save(d / 'meta.npy', np.array([f0, f, f1]))
        ji.append(dict(trajs=[dict(path=str(d / f'{lab}.npz'), kind='q37', label=lab) for lab in labs], audio=str(d / 'audio.wav'), out=str(d / 'raw.mp4'),
                       title=f"EX002 interruption {s['key']}  B3+ISC | R0causal+ISC | Funnel+ISC | L2diag+ISC | GT tail", views=['front', 'q34'], no_preview=True))
    (out / 'jobs_base.json').write_text(json.dumps(jb, indent=1, ensure_ascii=False)); (out / 'jobs_int.json').write_text(json.dumps(ji, indent=1, ensure_ascii=False))
    print('PREP', len(jb), len(ji))


def burn(raw, dst, words, t_int=None):
    esc = lambda s: s.replace('\\', '\\\\').replace(':', '\\:').replace("'", "’").replace('%', '\\%')
    fl = [f"drawtext=fontfile={FONT}:text='{esc(w['w'])}':x=(w-text_w)/2:y=h-110:fontsize=44:fontcolor=white:box=1:boxcolor=black@0.6:"
          f"enable='between(t,{max(0, w['s']):.2f},{w['e'] + 0.3:.2f})'" for w in words]
    if t_int is not None:
        fl += [f"drawbox=x=0:y=0:w=iw:h=ih:color=red@0.9:t=8:enable='gte(t,{t_int:.3f})'",
               f"drawtext=fontfile={FONT}:text='INTERRUPT':x=(w-text_w)/2:y=40:fontsize=40:fontcolor=red:box=1:boxcolor=white@0.7:enable='gte(t,{t_int:.3f})'"]
    vf = ','.join(fl) if fl else 'null'
    subprocess.run([FF, '-y', '-loglevel', 'error', '-i', str(raw), '-vf', vf, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-c:a', 'copy', str(dst)], check=True)


def post(a):
    sys.path.insert(0, '/home/xr-25b7/projects/gesture/scmg-wt-safeproj/scripts/echog_g1')
    import qual_g1 as Q
    out = Path(a.out); rule = json.loads((out / 'RULE.json').read_text()); man = dict(rule=rule, base={}, int={})
    for sub in ('video', 'sheets'): (out / sub).mkdir(exist_ok=True)
    for c in rule['base']['clips']:
        for P in ('offline', 'psync'):
            n = f"{c['clip']}_{P}"; d = out / 'base' / n; dst = out / 'video' / f'{n}.mp4'
            burn(d / 'raw.mp4', dst, json.loads((d / 'words.json').read_text()))
            T = (c['f1'] - c['f0']) / 50.; ts = list(np.linspace(0.4, T - 0.4, 12))
            Q.contact_sheet(dst, ts, out / 'sheets' / f'{n}_sheet.png', [f'{t:.1f} s' for t in ts])
            man['base'][n] = dict(video=f'video/{n}.mp4', sheet=f'sheets/{n}_sheet.png', panels=['GT', 'sibag-s2', 'enc-s0'], protocol=P, lang=c['lang'])
    for s in rule['int']['selected']:
        n = s['key'].replace('@', '_'); d = out / 'int' / n; f0, f, f1 = np.load(d / 'meta.npy').tolist(); rel = (f - f0) / 50.
        dst = out / 'video' / f'int_{n}.mp4'; burn(d / 'raw.mp4', dst, json.loads((d / 'words.json').read_text()), rel)
        ts = [rel + x for x in (-0.4, -0.1, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.05)]
        Q.contact_sheet(dst, ts, out / 'sheets' / f'int_{n}_sheet.png', [f't-t_i {t - rel:+.2f} s' for t in ts])
        man['int'][s['key']] = dict(video=f'video/int_{n}.mp4', sheet=f'sheets/int_{n}_sheet.png', t_i_in_video=rel,
                                    panels=list(INTROWS) + ['GT_tail (' + s['tail'] + ')'])
    (out / 'MANIFEST.json').write_text(json.dumps(man, indent=1, ensure_ascii=False)); print('POST_DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    for n in ('select', 'prep', 'post'):
        p = sp.add_parser(n); p.add_argument('--out', required=True); p.add_argument('--words-dir', default=str(B / 'ex002-causal-1107/words'))
    a = ap.parse_args(); dict(select=select, prep=prep, post=post)[a.cmd](a)

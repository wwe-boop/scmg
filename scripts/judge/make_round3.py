"""Judge round 3 (11-09, coordinator; EVAL_PROTOCOL §4): the CURRENT EX002 pipeline. Builder = round 2 (make_round2.py layouts, fixed
subtitles); A / B sides from random.Random(20263108 + pair index); KEY.json is written OUTSIDE the sheet directory (--key) and is not sent.
Pair sets (dev only, offline, per_clip seed 0; 24 pairs, half English / half Chinese):
  base (10): enc-s0 no-token vs sibag-s2 no-token, round-1 layout (3 x 4 front frames + subtitles); question: natural.
  ctl  (10): tok10ab vs tok5c on command windows (plan PLAN_dev_m9b_dense, segment [t - 3, t + 3]), round-2 control layout (front +
             second view, hand close-ups in the command window, command overlay); questions: natural, command. Per language aim and
             finger commands alternate in sha1 rank order.
  valid (4): 2 GT vs same GT (1 en + 1 zh; expect a tie) + 2 order-swapped duplicates (of the first base and the first ctl pair).
  (pair set 3, tok10ab + 9c'' vs tok10ab, is added after the 9c'' port exists, with the same rule and seed.)
Selection (committed before rendering): 6 s segments on a 0.5 s grid with >= 3 s of aligned words (fixed English word timing), ranked by
sha1("r3:<type>:<clip>:<start>"); segments of one set never overlap within a clip; control segments ranked by sha1("r3:ctl:<clip>:<t>").
    SCMG_J=34 python scripts/judge/make_round3.py --out OUT --key KEYDIR [--stage all|prep|render|compose]"""
import argparse, hashlib, json, random, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_pairs as M
FPS, SEG = M.FPS, M.SEG
h = lambda s: hashlib.sha1(s.encode()).hexdigest()
ZH_FUNC = set('的了是我你他她它在和也就都这那有个们吧呢啊吗嗯着么')


X3 = M.X
M.SYS.update(encnone=X3 + '/m9b-enc/offline/gen/none_s0', sibnone=X3 + '/m78b-sibag/offline/none_s0',
             tok10ab=X3 + '/m10-enc/offline/tok10ab/d_given_s0', tok5c=X3 + '/m5c-enc/offline/tok5c/d_given_s0')
M.GT = X3 + '/m9b-enc/offline/gen'; M.PLAN = X3 + '/PLAN_dev_m9b_dense.json'


def select(v, clips, r1=None):
    fw = {cl['ci']: M.words_of(v.clips[cl['ci']], fixed=(cl['lang'] == 'en')) for cl in clips}
    def segs(typ, lang, k):
        out = [(cl, s) for cl in clips if cl['lang'] == lang for s in cl['starts']]
        res, mine = [], {}
        for cl, s in sorted(out, key=lambda x: h(f"r3:{typ}:{x[0]['clip']}:{x[1]:.1f}")):
            if any(abs(s - u) < SEG for u in mine.get(cl['ci'], [])): continue
            res.append(dict(type=typ, ci=cl['ci'], clip=cl['clip'], lang=lang, start=s)); mine.setdefault(cl['ci'], []).append(s)
            if len(res) == k: break
        return res
    P = json.loads(Path(M.PLAN).read_text())
    def ctl(lang, k):
        cand = []
        for e in P['events']:
            cl = clips[e['ci']]
            if cl['lang'] == lang and e['t'] - 3 >= 0 and e['t'] + 3 <= cl['L']: cand.append((cl, e))
        res, mine, want, miss = [], {}, 'aim', 0
        ranked = sorted(cand, key=lambda x: h(f"r3:ctl:{x[0]['clip']}:{x[1]['t']:.3f}"))
        while len(res) < k and miss < 2:
            got = False
            for cl, e in ranked:
                if e['kind'] != want or any(abs(e['t'] - 3 - u) < SEG for u in mine.get(cl['ci'], [])): continue
                res.append(dict(type='ctl', ci=cl['ci'], clip=cl['clip'], lang=lang, start=e['t'] - 3, event=e)); mine.setdefault(cl['ci'], []).append(e['t'] - 3); got = True; break
            miss = 0 if got else miss + 1; want = 'finger' if want == 'aim' else 'aim'
        return res
    pairs = []
    for lang in ('en', 'zh'):
        for r in segs('base', lang, 5): pairs.append(dict(r, set='base', sys=('encnone', 'sibnone'), questions=['natural']))
    for lang in ('en', 'zh'):
        for r in ctl(lang, 5): pairs.append(dict(r, set='ctl', sys=('tok10ab', 'tok5c'), questions=['natural', 'command']))
    for r in segs('gtsame', 'en', 1) + segs('gtsame', 'zh', 1): pairs.append(dict(r, set='valid', sys=('gt', 'gt'), questions=['natural'], check='tie'))
    for i, p in enumerate(pairs): p['id'] = f'r{i:02d}'
    for src in ([p for p in pairs if p['set'] == 'base'][0], [p for p in pairs if p['set'] == 'ctl'][0]):
        pairs.append(dict(src, set='swap', of=src['id'], id=f'r{len(pairs):02d}'))
    return pairs, fw


def prep(a, v, pairs):
    from qual_ex002 import q37
    from scmg import ex002_ctl as E
    O = Path(a.out); (O / 'traj').mkdir(parents=True, exist_ok=True); (O / 'draw').mkdir(exist_ok=True); jobs = {}
    for p in pairs:
        if p['set'] == 'swap': continue
        f0 = int(round(p['start'] * FPS)); f1 = f0 + int(SEG * FPS)
        views = ['front'] if p['set'] != 'ctl' else (['front', 'q34', 'hands'] if p['event']['kind'] == 'aim' else ['front', 'hands'])
        for s in sorted(set(p['sys'])):
            if s == 'gt': y = np.load(f"{M.GT}/gt_{p['ci']}.npy")[f0:f1]
            elif s == 'gt_mismatch':
                m = p['mismatch']; g0 = int(round(m['start'] * FPS)); y = np.load(f"{M.GT}/gt_{m['ci']}.npy")[g0:g0 + int(SEG * FPS)]
            else: y = np.load(f"{M.SYS[s]}/{p['ci']}.npy")[f0:f1]
            tp = O / 'traj' / f"{p['id']}_{s}.npz"; np.savez(tp, Q37=q37(y), fps=50.0)
            frames = [dict(seg=[], pts=[], txt=[]) for _ in range(len(y))]
            if p['set'] == 'ctl':
                e = p['event']; t = e['t'] - p['start']; g = E.points(y)
                for k in range(len(y)):
                    tk = k / FPS
                    if e['kind'] == 'aim' and t - 0.3 <= tk <= t + 0.9:
                        sh = g['shoulder'][k, e['hand']]; tv = np.asarray(e['target']) - sh; tq = sh + tv / np.linalg.norm(tv)
                        frames[k]['seg'].append([sh.tolist(), tq.tolist(), [255, 220, 0]]); frames[k]['pts'].append([tq.tolist(), [255, 220, 0], 12])
                    if e['kind'] == 'finger' and t - 0.4 <= tk <= t + 0.4: frames[k]['txt'].append(f"command: fingers L {e['fingers'][0]} R {e['fingers'][1]}")
            dp = O / 'draw' / f"{p['id']}_{s}.json"; dp.write_text(json.dumps(dict(robots=[dict(frames=frames, ref_wrist=E.points(y[:1])['wrist'][0].tolist())])))
            jobs.setdefault(len(views), []).append(dict(trajs=[dict(path=str(tp), kind='q37', label='')], audio=None, out=str(O / 'video' / f"{p['id']}_{s}.mp4"),
                                                        title='', views=views, draw=str(dp)))
    for n, js in jobs.items(): (O / f'jobs_{n}.json').write_text(json.dumps(js, indent=1))
    return jobs


def render(a):
    O = Path(a.out); code = str(Path(__file__).resolve().parents[2])
    for jf in sorted(O.glob('jobs_*.json')):
        subprocess.run(f'cd {code} && CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES /home/xr-25b7/envs/isaacsim51/bin/python scripts/ex002/render_qual_isaac.py '
                       f'--jobs {jf} --headless --enable_cameras --no_preview --view_width 640 < /dev/null > {O}/render_{jf.stem}.log 2>&1', shell=True, check=True)


def grab(video, frames_idx, nviews):
    from PIL import Image
    out = []
    with tempfile.TemporaryDirectory() as td:
        for j, f in enumerate(frames_idx):
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(video), '-vf', f"select='eq(n\\,{f})'", '-frames:v', '1', '-vsync', '0', f'{td}/{j}.png'], check=True)
            im = Image.open(f'{td}/{j}.png').convert('RGB'); out.append([im.crop((640 * k, 0, 640 * (k + 1), im.height)) for k in range(nviews)])
    return out


_FK = {}


def hand_px(src, s, t, vid):
    """pixel position (full-res front view, 640 x 720) of the left / right hand (mid wrist..index tip) at time t, projected with the
    renderer's logged front camera (kinematics frame == Isaac frame, offset 0 measured)."""
    from scmg import ex002_ctl as E
    key = (src['id'], s)
    if key not in _FK:
        f0 = int(round(src['start'] * FPS)); f1 = f0 + int(SEG * FPS)
        y = np.load(f"{M.GT}/gt_{src['ci']}.npy")[f0:f1] if s == 'gt' else np.load(f"{M.SYS[s]}/{src['ci']}.npy")[f0:f1]
        cam = json.loads((Path(vid).parent / 'logs' / (Path(vid).stem + '.json')).read_text())['cameras']['front']
        _FK[key] = (E.points(y), np.asarray(cam['eye'][0]), np.asarray(cam['target'][0]))
    g, eye, tg = _FK[key]; k = min(int(round(t * FPS)), len(g['wrist']) - 1); TAN_H = 20.955 / 2 / 24.0; VW, H = 640, 720
    dv = (tg - eye) / np.linalg.norm(tg - eye); rt = np.cross(dv, [0, 0, 1.0]); rt /= np.linalg.norm(rt); up = np.cross(rt, dv); out = []
    for hh in (0, 1):
        pnt = (g['wrist'][k, hh] + g['index_tip'][k, hh]) / 2; rel = pnt - eye; zc = rel @ dv
        out.append((VW / 2 + (rel @ rt) / zc / TAN_H * VW / 2, H / 2 - (rel @ up) / zc / TAN_H * VW / 2))
    return out


def compose(a, v, pairs, fw):
    from PIL import Image, ImageDraw, ImageFont
    O = Path(a.out); (O / 'pairs').mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22, index=2)
    bold = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 22, index=2)
    big = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 34, index=2)
    key, meta = {}, []
    def subtitle(ws, ta, lang, kw=None):
        act = [w for (a0, a1, w) in ws if a0 - 0.15 <= ta <= a1 + 0.15]; return act
    for i, p in enumerate(pairs):
        src = p if p['set'] != 'swap' else [q for q in pairs if q['id'] == p['of']][0]
        s1, s2 = src['sys']; flip = random.Random(20263108 + i).random() < 0.5
        if p['set'] == 'swap': flip = not key[p['of']]['flip']
        A, Bs = (s2, s1) if flip else (s1, s2); key[p['id']] = dict(A=A, B=Bs, flip=flip, set=p['set'])
        ws = fw[src['ci']]; sep = '' if src['lang'] == 'zh' else ' '; sheets = []
        for lab, s in (('A', A), ('B', Bs)):
            vid = O / 'video' / f"{src['id']}_{s}.mp4"
            if src['set'] == 'ctl':
                nv = 3 if src['event']['kind'] == 'aim' else 2; ts = np.linspace(0.25, SEG - 0.25, 12); fr = grab(vid, [min(int(round(t * FPS)), 299) for t in ts], nv)
                e = src['event']; tc = e['t'] - src['start']; win = (tc - 0.3, tc + 0.9) if e['kind'] == 'aim' else (tc - 0.4, tc + 0.4)
                CW, CH, SUB = 960, 540, 64; sh = Image.new('RGB', (2 * CW, 50 + 6 * (CH + SUB)), (255, 255, 255)); d = ImageDraw.Draw(sh)
                d.text((10, 6), f"{p['id']}  system {lab}", font=big, fill=(0, 0, 0))
                for j, (t, ims) in enumerate(zip(ts, fr)):
                    x0, y0 = (j % 2) * CW, 50 + (j // 2) * (CH + SUB)
                    sh.paste(ims[0].resize((480, 540)), (x0, y0)); sh.paste(ims[1].resize((480, 540)), (x0 + 480, y0))
                    if win[0] <= t <= win[1]:            # both-hands close-ups: 90 px crops of the full-res front view around each hand, x200/90
                        for hh, (cu, cv) in enumerate(hand_px(src, s, t, vid)):
                            box = (int(cu - 45), int(cv - 45), int(cu + 45), int(cv + 45)); ins = ims[0].crop(box).resize((200, 200))
                            yy = y0 + 4 + hh * 206; sh.paste(ins, (x0 + 4, yy)); d.rectangle([x0 + 4, yy, x0 + 204, yy + 200], outline=(255, 0, 0), width=3)
                            d.text((x0 + 10, yy + 4), 'LR'[hh] + ' hand', font=font, fill=(255, 0, 0))
                    act = subtitle(ws, src['start'] + t, src['lang']); d.rectangle([x0, y0 + CH, x0 + CW - 1, y0 + CH + SUB - 1], fill=(240, 240, 240))
                    d.text((x0 + 8, y0 + CH + 4), f't={t:.2f}s' + ('   [command window]' if win[0] <= t <= win[1] else ''), font=font, fill=(200, 0, 0) if win[0] <= t <= win[1] else (90, 90, 90))
                    d.text((x0 + 8, y0 + CH + 32), (sep.join(act) if act else '—')[:60], font=font, fill=(0, 0, 0))
            elif src['set'] == 'dense':
                kws = src['keywords']; CW, CH, SUB = 240, 270, 56
                sh = Image.new('RGB', (8 * CW, 50 + len(kws) * (CH + SUB + 30)), (255, 255, 255)); d = ImageDraw.Draw(sh)
                d.text((10, 6), f"{p['id']}  system {lab}", font=big, fill=(0, 0, 0))
                for r, (ka, kw) in enumerate(kws):
                    tk = ka - src['start']; ts = np.linspace(tk - 1, tk + 1, 8); fr = grab(vid, [max(0, min(int(round(t * FPS)), 299)) for t in ts], 1)
                    y0 = 50 + r * (CH + SUB + 30); d.text((8, y0), f'keyword: {kw}  (t = {tk:.2f} s)', font=bold, fill=(200, 0, 0))
                    for j, (t, ims) in enumerate(zip(ts, fr)):
                        x0 = j * CW; sh.paste(ims[0].resize((CW, CH)), (x0, y0 + 30))
                        act = subtitle(ws, src['start'] + t, src['lang']); d.rectangle([x0, y0 + 30 + CH, x0 + CW - 1, y0 + 30 + CH + SUB - 1], fill=(240, 240, 240))
                        d.text((x0 + 4, y0 + 30 + CH + 2), f'{t - tk:+.2f}s', font=font, fill=(90, 90, 90)); xx = x0 + 4
                        for w in act[:4]:
                            col, ft = ((200, 0, 0), bold) if w == kw else ((0, 0, 0), font)
                            d.text((xx, y0 + 30 + CH + 28), w + sep, font=ft, fill=col); xx += d.textlength(w + sep, font=ft)
            else:
                ts = np.linspace(0.25, SEG - 0.25, 12); fr = grab(vid, [min(int(round(t * FPS)), 299) for t in ts], 1)
                FW, FH, SUB = 480, 540, 64; sh = Image.new('RGB', (4 * FW, 50 + 3 * (FH + SUB)), (255, 255, 255)); d = ImageDraw.Draw(sh)
                d.text((10, 6), f"{p['id']}  system {lab}", font=big, fill=(0, 0, 0))
                for j, (t, ims) in enumerate(zip(ts, fr)):
                    x0, y0 = (j % 4) * FW, 50 + (j // 4) * (FH + SUB); sh.paste(ims[0].resize((FW, FH)), (x0, y0))
                    act = subtitle(ws, src['start'] + t, src['lang']); d.rectangle([x0, y0 + FH, x0 + FW - 1, y0 + FH + SUB - 1], fill=(240, 240, 240))
                    d.text((x0 + 8, y0 + FH + 4), f't={t:.2f}s', font=font, fill=(90, 90, 90)); d.text((x0 + 8, y0 + FH + 32), (sep.join(act) if act else '—')[:40], font=font, fill=(0, 0, 0))
            sheets.append(sh)
        full = Image.new('RGB', (sheets[0].width, sheets[0].height * 2 + 20), (180, 180, 180)); full.paste(sheets[0], (0, 0)); full.paste(sheets[1], (0, sheets[0].height + 20))
        full.save(O / 'pairs' / f"{p['id']}.png")
        meta.append(dict(id=p['id'], set=p['set'], lang=src['lang'], clip=src['clip'], segment_s=[src['start'], src['start'] + SEG], questions=src['questions'],
                         keywords=src.get('keywords'), r1_id=src.get('r1_id'),
                         control_event=({k: src['event'][k] for k in ('kind', 't', 'hand', 'fingers') if k in src['event']} if 'event' in src else None)))
    Path(a.key).mkdir(parents=True, exist_ok=True); (Path(a.key) / 'KEY.json').write_text(json.dumps(dict(note='do not open before the judgements are written', key=key, pairs_full=pairs), indent=1, default=str))
    (O / 'PAIRS.json').write_text(json.dumps(dict(rule=__doc__, questions=dict(natural='① 更自然、更像人 (A / B / 平)', content='② 与字幕内容更相符 (A / B / 平)',
                                                                            command='③ 更好地执行了指令 (A / B / 平)', frozen='⑤ 哪个看起来更欠动 / 僵住 (A / B / 平)'), pairs=meta), ensure_ascii=False, indent=1))
    print('PAIRS', len(meta))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--key', required=True)
    ap.add_argument('--stage', default='all', choices=['all', 'prep', 'render', 'compose'])
    a = ap.parse_args(); v = M.view_clips(); assert M.check_words(v); clips = M.select(v)[0]; pairs, fw = select(v, clips)
    if a.stage in ('all', 'prep'): prep(a, v, pairs)
    if a.stage in ('all', 'render'): render(a)
    if a.stage in ('all', 'compose'): compose(a, v, pairs, fw)


if __name__ == '__main__':
    main()

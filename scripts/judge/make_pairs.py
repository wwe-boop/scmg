"""Blinded pair material for the main-agent judging protocol (refine-logs/EVAL_PROTOCOL.md §4; user 11-08 02:30).
For every pair: two systems on the same dev audio segment (6 s), left / right randomised with a seeded RNG, names hidden as A / B; one
PNG = sheet A above sheet B, each sheet 3 rows x 4 columns of front-view frames evenly spaced over the segment, the subtitle words active
at each frame time (word alignment of the clip's items; English words / Chinese characters, Noto Sans CJK) under every frame, the
command overlay for control pairs (yellow shoulder -> target direction + commanded finger counts during the command window) and an
interrupt marker for interruption pairs (frames at / after t_i framed in red). KEY.json (which side is which system) is written
separately and must not be opened before the judgements are written.
Renderer: EX002 = scripts/ex002/render_qual_isaac.py (Isaac, fixed front camera, no labels / titles / audio); G1 (agent A) = pass
--render-cmd with a command template that renders a jobs JSON of 1-robot clips to the given out paths (q / npz paths as written by
--emit-traj), e.g. A's MuJoCo renderer.
    SCMG_J=34 python scripts/judge/make_pairs.py ex002-round1 --out OUT            (select + prep + render + compose)
"""
import argparse, hashlib, json, random, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ex002')); sys.path.insert(0, '/home/xr-25b7/projects/gesture/codex/render-isaac')
FPS = 50; SEG = 6.0; NF = 12
B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; C = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'
V = C + '/train-v1-20261101/views/data-v36_1'; X = B + '/control-token-1105/ex002'
SYS = dict(                                                       # EX002 round-1 systems: dir of <ci>.npy (50 Hz, 34 joints), GT next to it
    czm=B + '/ex002-sweep-1107/czm-s12000-lr1e-4-w0.1/offline/s0',  # new causal-match base, no token, offline, per_clip seed 0
    sibag=X + '/lane1/ctrl/none_s0',                                # deployed sibag-s2, no token, offline, per_clip seed 0
    ctl5b=X + '/m5b-confirm/fresh/dense/fo_given_s3',               # 5b + override, given, fresh dense plan, seed 3 (sibag-s2 lane 4)
    ctlnone=X + '/m5b-confirm/fresh/gen/none_s3')                   # the same model / seed without tokens
GT = X + '/lane1/ctrl'                                            # gt_<ci>.npy
PLAN = X + '/PLAN_dev_fresh_dense.json'
RULE = ('6 s segments on a 0.5 s grid inside the clip with >= 3 s of aligned words; ranked by sha1("<type>:<clip>:<start>"); per pair '
        'type half English / half Chinese (odd counts: the extra one English); segments of one type never overlap within a clip; '
        'control pairs: segments [t - 3, t + 3] around fresh-dense-plan commands (aim and finger alternating by sha1 rank); mismatched '
        'GT: the motion of the first other clip of the same language (sha1 order) with a 6 s segment; swapped duplicates: the first 3 '
        'pairs of the czm-vs-sibag type with A / B exchanged; side assignment: random.Random(20261108 + pair index)')
h = lambda s: hashlib.sha1(s.encode()).hexdigest()


def view_clips():
    from scmg.data import View
    st = json.loads(Path(C + '/scmg/STATS_v36_1.json').read_text())
    v = View(V, 'val', C + '/scmg/text-e5-v36_1', st, mmap=True); return v


CHAR_CAP = 0.3


def words_of(c, fixed=False):
    """clip-relative words [(t0, t1, text)] from the items' alignment.json (en: char indices -> space-separated words; zh: chars).
    fixed=False is the round-1 SELECTION rule (kept bit-identical so the selected segments do not move); fixed=True (11-08 fix, used
    for the English subtitles): a word lasts from its first char onset for the sum of its char durations each capped at 0.3 s -- the
    aligner stretches the last chars of an utterance over trailing silence (e.g. 'the' spanning 0.99-18.75 s), which made one word
    cover a whole segment."""
    out = []
    for it in sorted(c['items'], key=lambda it: it['offset']):
        if it.get('text_null') in (True, 'True'): continue
        p = Path(it['part']) / it['dir'] / 'alignment.json'
        if not p.exists(): continue
        al = json.loads(p.read_text()); t0 = (it['offset'] - c['start']) / FPS; text = it['text']
        if it['lang'] == 'zh':
            out += [(t0 + a['onset_s'], t0 + a['offset_s'], a['char']) for a in al]
        else:
            by = {a['index']: a for a in al}; w, s0, s1, dur = '', None, None, 0.
            for i, ch in enumerate(text + ' '):
                if ch == ' ':
                    if w and s0 is not None: out.append((t0 + s0, t0 + (s0 + dur if fixed else s1), w))
                    w, s0, s1, dur = '', None, None, 0.
                else:
                    w += ch
                    if i in by:
                        s0 = by[i]['onset_s'] if s0 is None else s0; s1 = by[i]['offset_s']
                        dur += min(by[i]['offset_s'] - by[i]['onset_s'], CHAR_CAP)
    return out


def check_words(v):
    """unit check (11-08): for every English dev clip the fixed subtitle words joined in time order equal the clip's transcript words
    (items in offset order, text_null skipped, words of items without alignment excluded), no word lasts > 2.5 s, and onsets ascend."""
    bad = []
    for c in v.clips:
        if c['items'][0]['lang'] != 'en': continue
        ws = words_of(c, fixed=True); want = []
        for it in sorted(c['items'], key=lambda it: it['offset']):
            if it.get('text_null') in (True, 'True') or not (Path(it['part']) / it['dir'] / 'alignment.json').exists(): continue
            al = {a['index'] for a in json.loads((Path(it['part']) / it['dir'] / 'alignment.json').read_text())}; pos = 0
            for w in it['text'].split(' '):
                if w and any(i in al for i in range(pos, pos + len(w))): want.append(w)
                pos += len(w) + 1
        got = [w for _, _, w in ws]
        if got != want or any(b - a > 2.5 for a, b, _ in ws) or any(ws[k + 1][0] < ws[k][0] - 1e-6 for k in range(len(ws) - 1)): bad.append(c['clip'])
    print('CHECK_WORDS', 'OK' if not bad else f'FAIL {bad[:5]}'); return not bad


def select(v):
    clips = []
    for ci, c in enumerate(v.clips):
        L = c['frames'] / FPS; ws = words_of(c); lang = c['items'][0]['lang']
        starts = [s / 2 for s in range(0, int((L - SEG) * 2) + 1)]
        good = [s for s in starts if sum(max(0, min(b, s + SEG) - max(a, s)) for a, b, _ in ws) >= 3.0]
        clips.append(dict(ci=ci, clip=c['clip'], lang=lang, L=L, starts=good, words=ws))
    def pick(typ, n, cand=None):
        res = []; nz = n // 2; ne = n - nz
        for lang, k in (('en', ne), ('zh', nz)):
            pool = cand(lang) if cand else [(cl, s) for cl in clips if cl['lang'] == lang for s in cl['starts']]
            used = {}
            for cl, s in sorted(pool, key=lambda x: h(f"{typ}:{x[0]['clip']}:{x[1]:.1f}")):
                if any(abs(s - u) < SEG for u in used.get(cl['ci'], [])): continue
                res.append(dict(type=typ, ci=cl['ci'], clip=cl['clip'], lang=lang, start=s)); used.setdefault(cl['ci'], []).append(s)
                if len([r for r in res if r['lang'] == lang]) == k: break
        return res
    P = json.loads(Path(PLAN).read_text())
    def cmd_cand(lang):
        out = []
        for e in P['events']:
            cl = clips[e['ci']]
            if cl['lang'] == lang and e['t'] - 3 >= 0 and e['t'] + 3 <= cl['L']: out.append((cl, e['t'] - 3))
        return out
    pairs = []
    for r in pick('czm_vs_sibag', 12): pairs.append(dict(r, sys=('czm', 'sibag'), questions=['natural', 'content']))
    for r in pick('ctl_given_vs_none', 8, cmd_cand):
        ev = [e for e in P['events'] if e['ci'] == r['ci'] and abs(e['t'] - 3 - r['start']) < 1e-6][0]
        pairs.append(dict(r, sys=('ctl5b', 'ctlnone'), questions=['natural', 'content', 'command'], event=ev))
    for r in pick('gt_vs_mismatched', 6):
        same = [cl for cl in clips if cl['lang'] == r['lang'] and cl['ci'] != r['ci'] and cl['starts']]
        other = sorted(same, key=lambda cl: h(f"mm:{r['clip']}:{cl['clip']}"))[0]
        pairs.append(dict(r, sys=('gt', 'gt_mismatch'), mismatch=dict(ci=other['ci'], clip=other['clip'], start=other['starts'][0]), questions=['natural', 'content'], check='content -> GT'))
    for r in pick('gt_vs_same', 3): pairs.append(dict(r, sys=('gt', 'gt'), questions=['natural', 'content'], check='tie'))
    base = [p for p in pairs if p['type'] == 'czm_vs_sibag'][:3]
    for p in base: pairs.append(dict(p, type='swap_duplicate', of=None, swapped=True))
    for i, p in enumerate(pairs): p['id'] = f'p{i:02d}'
    for p in pairs:
        if p['type'] == 'swap_duplicate': p['of'] = [q['id'] for q in pairs if q['type'] == 'czm_vs_sibag' and q['ci'] == p['ci'] and q['start'] == p['start']][0]
    return clips, pairs


def prep(a, v, clips, pairs):
    from qual_ex002 import q37
    O = Path(a.out); (O / 'traj').mkdir(parents=True, exist_ok=True); (O / 'draw').mkdir(exist_ok=True); jobs = []
    meta = json.loads((Path(V) / 'META.json').read_text())['ex002']
    from scmg import ex002_ctl as E
    for p in pairs:
        if p['type'] == 'swap_duplicate': continue
        f0 = int(round(p['start'] * FPS)); f1 = f0 + int(SEG * FPS)
        for s in sorted(set(p['sys'])):
            if s == 'gt': y = np.load(f"{GT}/gt_{p['ci']}.npy")[f0:f1]
            elif s == 'gt_mismatch':
                m = p['mismatch']; g0 = int(round(m['start'] * FPS)); y = np.load(f"{GT}/gt_{m['ci']}.npy")[g0:g0 + int(SEG * FPS)]
            else: y = np.load(f"{SYS[s]}/{p['ci']}.npy")[f0:f1]
            tp = O / 'traj' / f"{p['id']}_{s}.npz"; np.savez(tp, Q37=q37(y), fps=50.0)
            frames = [dict(seg=[], pts=[], txt=[]) for _ in range(len(y))]
            if p['type'] == 'ctl_given_vs_none':                     # command overlay (same on both sides): direction / counts in the window
                e = p['event']; t = e['t'] - p['start']; g = E.points(y)
                for k in range(len(y)):
                    tk = k / FPS
                    if e['kind'] == 'aim' and t - 0.3 <= tk <= t + 0.9:
                        sh = g['shoulder'][k, e['hand']]; tv = np.asarray(e['target']) - sh; tq = sh + tv / np.linalg.norm(tv)
                        frames[k]['seg'].append([sh.tolist(), tq.tolist(), [255, 220, 0]]); frames[k]['pts'].append([tq.tolist(), [255, 220, 0], 12])
                    if e['kind'] == 'finger' and t - 0.4 <= tk <= t + 0.4:
                        frames[k]['txt'].append(f"command: fingers L {e['fingers'][0]} R {e['fingers'][1]}")
            dp = O / 'draw' / f"{p['id']}_{s}.json"; dp.write_text(json.dumps(dict(robots=[dict(frames=frames, ref_wrist=E.points(y[:1])['wrist'][0].tolist())])))
            jobs.append(dict(trajs=[dict(path=str(tp), kind='q37', label='')], audio=None, out=str(O / 'video' / f"{p['id']}_{s}.mp4"), title='', views=['front'], draw=str(dp)))
    (O / 'jobs.json').write_text(json.dumps(jobs, indent=1)); return jobs


def render(a, jobs):
    O = Path(a.out); cmd = a.render_cmd or ('cd {code} && CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES /home/xr-25b7/envs/isaacsim51/bin/python '
                                           'scripts/ex002/render_qual_isaac.py --jobs {jobs} --headless --enable_cameras --no_preview --view_width 640')
    subprocess.run(cmd.format(code=str(Path(__file__).resolve().parents[2]), jobs=str(O / 'jobs.json')) + f' < /dev/null > {O}/render.log 2>&1', shell=True, check=True)


def compose(a, v, clips, pairs):
    from PIL import Image, ImageDraw, ImageFont
    import tempfile
    O = Path(a.out); (O / 'pairs').mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22, index=2)   # SC
    big = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 34, index=2) if Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc').exists() else font
    FW, FH, SUB = 480, 540, 64
    key, meta = {}, []
    def frames(video, n_frames):
        ts = np.linspace(0.25, SEG - 0.25, NF); out = []
        with tempfile.TemporaryDirectory() as td:
            for j, t in enumerate(ts):
                f = min(int(round(t * FPS)), n_frames - 1)
                subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(video), '-vf', f"select='eq(n\\,{f})'", '-frames:v', '1', '-vsync', '0', f'{td}/{j}.png'], check=True)
                im = Image.open(f'{td}/{j}.png').convert('RGB'); im = im.crop((0, 0, 640, im.height)).resize((FW, FH)); out.append((t, im))
        return out
    only = set(a.only.split(',')) if a.only else None
    for i, p in enumerate(pairs):
        src = p if p['type'] != 'swap_duplicate' else [q for q in pairs if q['id'] == p['of']][0]
        s1, s2 = src['sys']; rng = random.Random(20261108 + i); flip = rng.random() < 0.5
        if p['type'] == 'swap_duplicate': flip = not key[p['of']]['flip']
        A, Bs = (s2, s1) if flip else (s1, s2)
        key[p['id']] = dict(A=A, B=Bs, flip=flip, type=p['type'])
        if only is not None and p['id'] not in only: continue
        ws = clips[src['ci']]['words'] if src['lang'] == 'zh' else words_of(v.clips[src['ci']], fixed=True); sheets = []
        for lab, s in (('A', A), ('B', Bs)):
            fr = frames(O / 'video' / f"{src['id']}_{s}.mp4", int(SEG * FPS))
            sh = Image.new('RGB', (4 * FW, 50 + 3 * (FH + SUB)), (255, 255, 255)); d = ImageDraw.Draw(sh)
            d.text((10, 6), f'{p["id"]}  system {lab}', font=big, fill=(0, 0, 0))
            for j, (t, im) in enumerate(fr):
                x0, y0 = (j % 4) * FW, 50 + (j // 4) * (FH + SUB); sh.paste(im, (x0, y0)); ta = src['start'] + t
                act = [w for (a0, a1, w) in ws if a0 - 0.15 <= ta <= a1 + 0.15]
                sep = '' if src['lang'] == 'zh' else ' '; txt = sep.join(act) if act else '—'
                d.rectangle([x0, y0 + FH, x0 + FW - 1, y0 + FH + SUB - 1], fill=(240, 240, 240))
                d.text((x0 + 8, y0 + FH + 4), f't={t:.2f}s', font=font, fill=(90, 90, 90)); d.text((x0 + 8, y0 + FH + 32), txt[:40], font=font, fill=(0, 0, 0))
            sheets.append(sh)
        full = Image.new('RGB', (4 * FW, sheets[0].height * 2 + 20), (180, 180, 180)); full.paste(sheets[0], (0, 0)); full.paste(sheets[1], (0, sheets[0].height + 20))
        full.save(O / 'pairs' / f"{p['id']}.png")
        meta.append(dict(id=p['id'], lang=src['lang'], clip=src['clip'], segment_s=[src['start'], src['start'] + SEG], questions=src['questions'],
                         control_event=({k: src['event'][k] for k in ('kind', 't', 'hand', 'fingers') if k in src['event']} if 'event' in src else None)))
    if only is not None: print('RECOMPOSED', sorted(only), '(KEY.json / PAIRS.json untouched)'); return
    (O / 'KEY.json').write_text(json.dumps(dict(note='do not open before the judgements are written', key=key, pairs_full=pairs), indent=1, default=str))
    (O / 'PAIRS.json').write_text(json.dumps(dict(rule=RULE, protocol='refine-logs/EVAL_PROTOCOL.md §4', questions=dict(
        natural='① 更自然、更像人 (A / B / 平)', content='② 与字幕内容更相符 (A / B / 平)', command='③ 更好地执行了指令 (A / B / 平)'), pairs=meta), ensure_ascii=False, indent=1))
    print('PAIRS', len(meta))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('round', choices=['ex002-round1']); ap.add_argument('--out', required=True)
    ap.add_argument('--render-cmd', default=None); ap.add_argument('--only', default=None, help='comma-separated pair ids to recompose (KEY / PAIRS untouched)'); ap.add_argument('--stage', default='all', choices=['all', 'prep', 'render', 'compose'])
    a = ap.parse_args(); v = view_clips(); assert check_words(v); clips, pairs = select(v)
    if a.stage in ('all', 'prep'): jobs = prep(a, v, clips, pairs)
    if a.stage in ('all', 'render'): render(a, json.loads((Path(a.out) / 'jobs.json').read_text()))
    if a.stage in ('all', 'compose'): compose(a, v, clips, pairs)


if __name__ == '__main__':
    main()

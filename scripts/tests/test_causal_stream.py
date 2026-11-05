"""11-05 causal streaming inputs (design v5 section 3; scmg/causal_feat.py + Streamer causal_inputs / text_mode voiced|asr).
    SCMG_J=39 CUDA_VISIBLE_DEVICES=1 python scripts/tests/test_causal_stream.py      (exit 1 on failure)
Real model: the sf0 dev lane checkpoint on the held-out-speaker D-spk clips (env CK, VAE, STATS, HOSV, TEXT, TEXTC, DS, PSTATS;
never val_common).
 a default arguments: Streamer output bitwise equal to the pre-change scmg/infer.py (git BASE, default 5f57993), offline and
   engine protocol (and the old code against itself = the determinism floor)
 b text_mode voiced: visible words == ECHO-G stream_core.visible_tokens 'voiced' (functions taken from the ECHO-G streaming
   worktree source) on random word lists and on every tick of real D-spk clips (token level, through the Streamer)
 c future-perturbation invariance: PCM after t_k replaced by noise and the words not finished by t_k deleted -> every
   condition tensor of ticks 0..k (feat window, speaker vector, visible text rows / times, visibility, ADSC state) and every
   emitted frame bitwise unchanged (psync and pasr, causal inputs, fixed seed); the whole-clip path (view feat89 / CAM++ of
   the whole perturbed clip, engine x2 text) under the same perturbation must CHANGE (the test detects leakage)
 d causal feat89 vs the view's whole-clip feat89 (normalised units): with the whole clip arrived, and as read at every tick;
   prefix CAM++ vs whole-clip cosine (1 / 2 / 4 / 8 s)"""
import ast, json, os, subprocess, sys, types
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve(); CODE = HERE.parents[2]
sys.path.insert(0, str(CODE))
import torch
from scmg.common import CHUNK_FRAMES, U80
from scmg.model import INF
from scmg import causal_feat as CF

E = os.environ
SP = E.get('SP', '/tmp')
CK, VAE, STATS, HOSV = E['CK'], E['VAE'], E['STATS'], E['HOSV']
TEXT, TEXTC, DS, PSTATS = E['TEXT'], E['TEXTC'], E['DS'], E['PSTATS']
BASE = E.get('BASE', '5f57993')
CORE = E.get('STREAM_CORE', '/home/xr-25b7/projects/gesture/scmg-wt-echog-stream/scripts/echog_stream/stream_core.py')
STEMS = E.get('TC_STEMS', 'english_2_scott_0_103_103_utt_0000,english_6_carla_0_74_74_utt_0003,english_2_scott_1_6_6_utt_0025').split(',')
N_D = int(E.get('TC_ND', '40'))
FAIL = []


def ok(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not cond: FAIL.append(name)


# ------------------------------------------------------------------ setup
os.environ['SCMG_TEXT_CAUSAL'] = TEXTC
from scmg.data import View, WindowDataset
from scmg.evaluate import load
from scmg import infer as NEW
stats = json.loads(Path(STATS).read_text())
view = View(HOSV, 'val', TEXT, stats)
idx = {c['items'][0]['id']: i for i, c in enumerate(view.clips)}
m, vae = load(CK, VAE, 'cuda')
qm = np.asarray(stats['q_mean'], np.float32)
tokz = CF.e5_tokenizer(); enc = CF.speaker_encoder(); pst = CF.load_pstats(PSTATS)
print('cuda', torch.cuda.get_device_name(0), 'CUDA_VISIBLE_DEVICES', E.get('CUDA_VISIBLE_DEVICES'), 'pid', os.getpid(), flush=True)


def old_infer():
    src = subprocess.run(['git', '-C', str(CODE), 'show', f'{BASE}:scmg/infer.py'], capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType('scmg._infer_base'); mod.__package__ = 'scmg'; mod.__file__ = f'{BASE}:scmg/infer.py'
    exec(compile(src, mod.__file__, 'exec'), mod.__dict__)
    return mod


def clip(stem):
    ci = idx[stem]; c = view.clips[ci]; it = c['items'][0]; L = c['frames']; nch = -(-L // CHUNK_FRAMES)
    s = WindowDataset(view, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
    s['ref'] = np.zeros_like(s['ref']); s['ref_valid'] = np.float32(0.)
    cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm',
                              'titime', 'tseg', 'tiseg', 'tiend', 'text_c') if k in s}
    words = CF.clip_words(Path(DS) / 'annotations/words' / f'{stem}.json', L)
    kn = view.text['index'].get(stem, [0, 0])[1] if not it.get('text_null') else 0
    assert int(s['tmask'].sum()) == kn
    pcm16, wave = CF.clip_audio(Path(DS) / 'audio' / f'{stem}.wav')
    return dict(stem=stem, it=it, L=L, nch=nch, s=s, cond=cond, words=words, kn=kn, pcm16=pcm16, wave=wave)


ENGINE = dict(H=2 * U80, text_mode='x2', x2_rate=15., x2_lag=.15, deploy_timing=True)


def run(mod, d, n_ticks=None, feat=None, cond=None, causal=None, trace=None, **kw):
    cond = cond if cond is not None else d['cond']; feat = d['s']['feat'] if feat is None else feat
    extra = {} if causal is None and trace is None else dict(causal_inputs=causal, trace=trace)
    st = mod.Streamer(m, vae, feat, cond, stats, steps=10, seed=0, standby=qm, ctx_t=m.ctx_t, cfg_w=2.5, **kw, **extra)
    out = []
    for i in range(d['nch'] if n_ticks is None else n_ticks):
        out.append(st.tick().clone())
        if n_ticks is None and i == d['nch'] - 1: out.append(st.flush().clone())
    return torch.cat(out, 0), st


def causal_of(d, pcm16=None, wave=None, words=None):
    return CF.CausalInputs(d['pcm16'] if pcm16 is None else pcm16, d['wave'] if wave is None else wave,
                           d['words'] if words is None else words, d['it']['lang'], d['L'], pst, view.fm, view.fs, enc,
                           Tpad=len(d['s']['feat']))


def voiced_cond(d, cond=None, words=None):
    c = dict(d['cond'] if cond is None else cond)
    c.update(CF.text_extras(d['it'], d['words'] if words is None else words, tokz, d['kn'], len(d['s']['tmask'])))
    return c


D = {st: clip(st) for st in STEMS}

# ------------------------------------------------------------------ a: default path bitwise unchanged
OLD = old_infer()
for name, kw in (('offline', dict(H=INF)), ('engine', ENGINE)):
    for st in STEMS[:2]:
        d = D[st]
        y_old, _ = run(OLD, d, **kw); y_old2, _ = run(OLD, d, **kw); y_new, _ = run(NEW, d, **kw)
        ok(f'a {name} {st}: new default == {BASE}', torch.equal(y_old, y_new) and torch.equal(y_old, y_old2),
           f'frames {tuple(y_new.shape)} old-vs-old equal {torch.equal(y_old, y_old2)} max|new-old| {float((y_new - y_old).abs().max()):.3g}')

# ------------------------------------------------------------------ b: voiced visibility == stream_core.visible_tokens
tree = ast.parse(Path(CORE).read_text()); ns = dict(np=np)
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('x2_arrival', 'visible_tokens'):
        exec(compile(ast.Module([node], []), CORE, 'exec'), ns)
rng = np.random.default_rng(0); nbad = ncmp = 0
for trial in range(2000):
    n = int(rng.integers(0, 40)); st_ = np.sort(rng.uniform(0, 12, n)); tt = np.stack([st_, st_ + rng.uniform(0, .8, n)], 1)
    for _ in range(5):
        now = float(rng.uniform(-0.5, 13)); ahead = float(rng.choice([0., .16, .32, rng.uniform(0, 1)]))
        rate, lag = float(rng.uniform(5, 30)), float(rng.uniform(0, .5))
        for mode in ('voiced', 'x2'):
            a_ = ns['visible_tokens'](tt, now, ahead, mode, rate, lag); b_ = CF.visible_words(tt, now, ahead, mode, rate, lag)
            ncmp += 1; nbad += int(not np.array_equal(a_, b_))
ok('b random word lists: visible_words == stream_core.visible_tokens (voiced, x2)', nbad == 0, f'{ncmp} comparisons, {nbad} mismatches')
for st in STEMS:
    d = D[st]; cv = voiced_cond(d); tr = []
    run(NEW, d, cond=cv, trace=tr, **dict(ENGINE, text_mode='voiced'))
    wt = CF.word_times(d['words']); K = len(d['s']['tmask']); bad = 0; nv = []
    for t in tr:
        now = t['k'] * CHUNK_FRAMES / 50.
        want = CF.token_visible(ns['visible_tokens'](wt, now, 2 * 4 / 50., 'voiced', 15., .15), cv['tword']) & d['s']['tmask']
        got = (t['cur_vis'][:K] | t['cur_vis'][K:]).numpy()
        bad += int(not np.array_equal(got, want)); nv.append(int(got.sum()))
    # word level: a word is visible iff its own tokens are (tokens inside one word)
    ok(f'b real clip {st}: Streamer voiced tokens == stream_core.visible_tokens on every tick', bad == 0,
       f'{len(tr)} ticks, {len(d["words"])} words, {d["kn"]} tokens, visible per tick {nv[:6]}..{nv[-3:]}')

# ------------------------------------------------------------------ c: future-perturbation invariance
def perturb(d, k, mode_lag=0.):
    """noise after t_k, words not finished by t_k deleted (their token rows removed from every text array)."""
    tk = min(d['L'], 16 * k + 8); n = tk * CF.HOP
    g = np.random.default_rng(1000 + k)
    w2 = d['wave'].copy(); w2[n:] = g.normal(0, max(1e-3, float(d['wave'].std())), len(w2) - n).astype(np.float32)
    p2 = np.clip(np.round(w2 * 32768.), -32768, 32767).astype(np.int16)
    kw_ = [i for i, w in enumerate(d['words']) if w['end_time'] <= tk / 50.]; words2 = [d['words'][i] for i in kw_]
    remap = np.full(len(d['words']) + 1, -1, np.int64); remap[kw_] = np.arange(len(kw_))
    span = CF.text_extras(d['it'], d['words'], tokz, d['kn'], len(d['s']['tmask']))['tword']
    keep = np.array([r[0] >= 0 and all(remap[j] >= 0 for j in range(r[0], r[1] + 1)) for r in span])
    c2 = dict(d['cond'])
    for kk in ('text', 'text_c'): c2[kk] = np.where(keep[:, None], d['cond'][kk], 0.).astype(np.float32)
    c2['ttime'] = np.where(keep, d['cond']['ttime'], np.nan).astype(np.float32); c2['tmask'] = d['cond']['tmask'] & keep
    c2['wtime'] = CF.word_times(words2); c2['tword'] = np.where(keep[:, None], remap[np.clip(span, 0, None)], -1)
    return tk, w2, p2, words2, c2, int((~keep & d['s']['tmask']).sum())


def same_trace(ta, tb):
    diff = []
    for x, y in zip(ta, tb):
        for key in ('feat', 'spk', 'spk_valid', 'tvis', 'sstate'):
            if (x[key] is None) != (y[key] is None) or (x[key] is not None and not torch.equal(x[key], y[key])): diff.append((x['k'], key))
        K = len(x['ttime'])
        va = x['cur_vis'][:K] | x['cur_vis'][K:]; vb = y['cur_vis'][:K] | y['cur_vis'][K:]
        ea, eb = x['ttime'][va], y['ttime'][vb]                        # untimed tokens (NaN) must stay untimed
        same_t = torch.equal(ea.isnan(), eb.isnan()) and torch.equal(ea.nan_to_num(), eb.nan_to_num())
        if not torch.equal(va, vb) or not torch.equal(x['raw'][va], y['raw'][vb]) or not same_t:
            diff.append((x['k'], 'visible text'))
    return diff


for st in STEMS:
    d = D[st]
    for k in sorted({3, d['nch'] // 2, d['nch'] - 3}):
        if not (0 < k < d['nch'] - 1): continue
        tk, w2, p2, words2, c2, ndel = perturb(d, k)
        for proto in ('voiced', 'asr'):
            kw = dict(ENGINE, text_mode=proto)
            ta, tb = [], []
            ya, _ = run(NEW, d, n_ticks=k + 1, cond=voiced_cond(d), causal=causal_of(d), trace=ta, **kw)
            yb, _ = run(NEW, d, n_ticks=k + 1, cond=c2, causal=causal_of(d, p2, w2, words2), trace=tb, **kw)
            dif = same_trace(ta, tb)
            ok(f'c causal {proto} {st} k={k} (t_k={tk / 50:.2f}s, {len(d["words"]) - len(words2)} words / {ndel} tokens deleted)',
               not dif and torch.equal(ya, yb), f'cond diffs {dif[:4]} frames equal {torch.equal(ya, yb)} ({len(ya)} frames)')
        # control: the whole-clip (pre-change) path under the same perturbation
        fa = d['s']['feat']
        fb = (np.asarray(CF.full_feat89(p2, words2, d['it']['lang'], d['L']), np.float32) - view.fm) / view.fs
        fbp = np.zeros_like(fa); fbp[:d['L']] = fb
        cb = {kk: v for kk, v in c2.items() if kk not in ('wtime', 'tword')}; cb['spk'] = CF.SpeakerPrefix(w2, enc).full(); cb['spk_valid'] = np.float32(1.)
        ta, tb = [], []
        ya, _ = run(NEW, d, n_ticks=k + 1, trace=ta, **ENGINE)
        yb, _ = run(NEW, d, n_ticks=k + 1, feat=fbp, cond=cb, trace=tb, **ENGINE)
        dif = same_trace(ta, tb)
        ok(f'c control whole-clip path {st} k={k}: perturbation detected (must differ)', bool(dif) and not torch.equal(ya, yb),
           f'differing conditions {sorted({x[1] for x in dif})} first tick {min([x[0] for x in dif], default=None)}; '
           f'feat rows<t_k max|d| {float(np.abs(fa[:tk] - fbp[:tk]).max()):.3g}; max|frames d| {float((ya - yb).abs().max()):.3g}')

# ------------------------------------------------------------------ d: causal vs whole-clip feat89 (normalised), prefix CAM++
stems = Path(E.get('TC_DEV', f'{SP}/work/dev_stems.txt')).read_text().split()[:N_D]
G = dict(mel=slice(0, 80), onset=80, prominence=81, energy=82, logf0=83, voiced=84, beat=85, beat_l=86, beat_r=87, flag=88)
fin = {g: [] for g in G}; tick = {g: [] for g in G}; front = {g: [] for g in G}; cos = {1: [], 2: [], 4: [], 8: []}
nb = dict(view=0, causal=0)
for st in stems:
    ci = idx[st]; c = view.clips[ci]; it = c['items'][0]; L = c['frames']
    whole = np.asarray(view.f[c['start']:c['start'] + L], np.float32)
    words = CF.clip_words(Path(DS) / 'annotations/words' / f'{st}.json', L); pcm16, wave = CF.clip_audio(Path(DS) / 'audio' / f'{st}.wav')
    cf = CF.CausalFeat89(pcm16, words, it['lang'], L, pst)
    nfull = lambda x: np.abs((np.asarray(x, np.float32) - whole[:len(x)]) / view.fs)
    dfin = nfull(cf.prefix(L)); nb['view'] += int((whole[:, 85] > 0).sum()); nb['causal'] += int((np.asarray(cf.prefix(L))[:, 85] > 0).sum())
    for g, sl in G.items(): fin[g].append(dfin[:, sl].reshape(len(dfin), -1))
    for k in range(-(-L // CHUNK_FRAMES)):
        tk = min(L, 16 * k + 8); dd = nfull(cf.prefix(tk))
        for g, sl in G.items():
            tick[g].append(dd[:, sl].reshape(len(dd), -1)); front[g].append(dd[max(0, tk - 8):, sl].reshape(-1, 1 if isinstance(sl, int) else 80))
    sp = CF.SpeakerPrefix(wave, enc); full = sp.full()
    for s_ in cos:
        if L / 50. >= s_:
            e, v = sp.at(int(s_ * 16000))
            if v: cos[s_].append(float(e @ full))
stat = lambda xs: (float(np.concatenate(xs).max()), float(np.concatenate(xs).mean()))
print(f'd causal vs whole-clip feat89 on {len(stems)} D-spk clips, |diff| / f_std (normalised units), max / mean:', flush=True)
for g in G:
    a1, b1 = stat(fin[g]); a2, b2 = stat(tick[g]); a3, b3 = stat(front[g])
    print(f'   {g:10s} whole-arrived max {a1:8.3f} mean {b1:.4f} | per-tick rows<t_k max {a2:8.3f} mean {b2:.4f} | '
          f'last 160 ms before t_k max {a3:8.3f} mean {b3:.4f}', flush=True)
print(f'   beats: view {nb["view"]} causal(v2, whole arrived) {nb["causal"]}', flush=True)
print('d prefix CAM++ cosine to the whole-clip embedding: ' + ', '.join(
    f'{s_} s: n={len(v)} mean {np.mean(v):.3f} p10 {np.percentile(v, 10):.3f}' for s_, v in cos.items() if v), flush=True)
ok('d computed', all(len(v) for v in fin.values()))

print('ALL_OK' if not FAIL else f'FAILED {FAIL}', flush=True)
sys.exit(1 if FAIL else 0)

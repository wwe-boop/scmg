"""11-05 训练匹配: training-matched causal streaming inputs (StageCfg causal_front / causal_spk / causal_text, View.load_causal,
SCMG.plan per target, causal_feat.front_prosody).
    SCMG_J=39 CUDA_VISIBLE_DEVICES=1 python scripts/tests/test_causal_train.py      (exit 1 on failure)
env CK, VAE, STATS, HOSV, TEXT, TEXTC, DS (audio/, annotations/words/), PSTATS, BASE (default 80c5793), SP (scratch),
TRV (train view for the loss-parity batch, default HOSV), N_DEV (dev clips precomputed here, default 6). Real model:
the sf0 dev-lane checkpoint; clips: the held-out-speaker dev split (never val_common).
 a default path: loss terms + every parameter gradient bitwise equal to the BASE code on a real training batch
   (sf0 StageCfg); the same batch with the cz_* keys added and the switches off -> bitwise equal again
 b front_prosody == prefix_feat89[:, 80:89] (f16) bitwise at every 9th frame of every dev clip (mel 0-79: prefix rows ==
   whole-clip rows); committed_beats / assign_sides copies == their sources
 c model: plan() per target with the same encoding / condition for every target == plan(); h_k does not move when the
   encodings / conditions of the other targets change; local_audio with a per-chunk encoding == the shared one
 d training == inference: encoder input of chunk k (causal_front_feat) == Streamer causal_inputs feat at tick k (psync,
   H = 160 ms) row for row; the prefix CAM++ grade of chunk k is available by t_k and equals the Streamer's vector when
   t_k is a grade time (spk_refresh 'grades': at every tick, and the rule ignores audio after t_k); every text row visible to chunk k in training has ended by t_k (and agrees with the Streamer's
   word-level psync visibility)
 e future-perturbation invariance of TRAINING: PCM after t_k -> noise, words not ended by t_k deleted (precomputed again
   on the perturbed clip), not-yet-voiced text rows given random features / later times -> every planner input of chunk k
   captured inside step_losses (audio encoding k, global condition k, text visibility + visible text rows, stream state)
   bitwise unchanged; control: the default (whole-clip) training inputs of the same perturbed clip DO change"""
import importlib.util, json, os, subprocess, sys, tempfile, types
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve(); CODE = HERE.parents[2]
sys.path.insert(0, str(CODE))
import torch
from scmg.common import CHUNK, CHUNK_FRAMES, STRIDE, U80
from scmg import causal_feat as CF

E = os.environ
SP = Path(E.get('SP', tempfile.mkdtemp(prefix='causal_train_')))
CK, VAE, STATS, HOSV = E['CK'], E['VAE'], E['STATS'], E['HOSV']
TEXT, TEXTC, DS, PSTATS = E['TEXT'], E['TEXTC'], Path(E['DS']), E['PSTATS']
BASE = E.get('BASE', '80c5793'); TRV = E.get('TRV', HOSV); N_DEV = int(E.get('N_DEV', '6'))
FAIL = []
os.environ['SCMG_TEXT_CAUSAL'] = TEXTC


def ok(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not cond: FAIL.append(name)


from scmg.data import View, WindowDataset, collate
from scmg.evaluate import load
from scmg.losses import StageCfg, step_losses, causal_front_feat, causal_spk_select
spec = importlib.util.spec_from_file_location('precompute_causal', CODE / 'scripts/echog_g1/precompute_causal.py')
PC = importlib.util.module_from_spec(spec); spec.loader.exec_module(PC)
stats = json.loads(Path(STATS).read_text())
SC = dict(p_stream=0.67, p_async=1.0, p_async_seg=0.0, self_force_p=0)                # the sf0 lane
dev = 'cuda'

# ------------------------------------------------------------------ a: default path bitwise == BASE
tree = SP / f'base_{BASE}'
if not (tree / 'scmg').exists():
    tree.mkdir(parents=True, exist_ok=True)
    subprocess.run(f'git -C {CODE} archive {BASE} scmg scripts | tar -x -C {tree}', shell=True, check=True)
vt = View(TRV, 'train', TEXT, stats)
ds_t = WindowDataset(vt, 256, max_tokens=256)
rng = np.random.default_rng(7); cis = rng.choice(len(vt.clips), 8, replace=False)
samples = [ds_t.sample(int(ci), int(rng.integers(0, max(1, vt.clips[int(ci)]['frames'] - 256))), np.random.default_rng(int(ci))) for ci in cis]
bA = collate(samples); torch.save(bA, SP / 'batchA.pt')


def probe(code, batch_path, sc, tag):
    out = SP / f'probe_{tag}.pt'
    r = subprocess.run([sys.executable, str(HERE.parent / '_causal_loss_probe.py'), str(batch_path), str(out), CK, VAE, STATS,
                        '{"adsc": true}', json.dumps(sc)], env=dict(os.environ, PYTHONPATH=str(code)), cwd=str(code),
                       capture_output=True, text=True)
    if r.returncode: print(r.stdout[-2000:], r.stderr[-3000:])
    return torch.load(out, weights_only=False) if r.returncode == 0 else None


def same(p, q):
    if p is None or q is None: return False, 'probe failed'
    lk = all(torch.equal(p['loss'][k], q['loss'][k]) for k in p['loss']) and p['loss'].keys() == q['loss'].keys()
    gk = p['grad'].keys() == q['grad'].keys() and all(torch.equal(p['grad'][k], q['grad'][k]) for k in p['grad'])
    nd = sum(1 for k in p['grad'] if k in q['grad'] and not torch.equal(p['grad'][k], q['grad'][k]))
    return lk and gk, f"loss {({k: round(float(v), 6) for k, v in p['loss'].items()})} grads {len(p['grad'])} differing {nd}"


pb, pn = probe(tree, SP / 'batchA.pt', SC, 'base'), probe(CODE, SP / 'batchA.pt', SC, 'new')
ok_, info = same(pb, pn); ok(f'a default StageCfg: loss + grads bitwise == {BASE}', ok_, info)
pn2 = probe(CODE, SP / 'batchA.pt', SC, 'new2'); ok('a determinism floor (new code twice)', same(pn, pn2)[0])

# ------------------------------------------------------------------ precompute a few dev clips (same code as the train set)
vd = View(HOSV, 'val', TEXT, stats)
idx = {c['items'][0]['id']: i for i, c in enumerate(vd.clips)}
items = {it['id']: it for it in json.loads((Path(HOSV) / 'train.json').read_text())}
tix = json.loads((Path(TEXT) / 'TEXT_INDEX.json').read_text())
stems = sorted(idx, key=lambda s: -vd.clips[idx[s]]['frames'])[:2] + sorted(idx)[:max(0, N_DEV - 2)]
PC._init('/home/xr-25b7/bulk/models-ro/campplus', PSTATS, '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/models/scmg/multilingual-e5-base')
pst = CF.load_pstats(PSTATS); enc = PC._W['enc']; tokz = PC._W['tok']


def job(stem, audio_dir=DS / 'audio', words_dir=DS / 'annotations/words'):
    it = items[stem]
    return (stem, int(it['frames']), it['lang'], str(audio_dir), str(words_dir), it.get('text', ''),
            0 if it.get('text_null') else int(tix[stem][1]))


res = [PC.one(job(s)) for s in stems]
PC.write(SP / 'cz_dev', res)
(SP / 'cz_dev' / 'META.json').write_text(json.dumps(dict(unit_cols=list(CF.UNIT_COLS), grades_s=PC.GRADES)))
vd.load_causal(SP / 'cz_dev', StageCfg().horizons)
# a (cont.): the cz_* keys in the batch with every switch off change nothing
dsd = WindowDataset(vd, 256, max_tokens=256)
smp = [dsd.sample(idx[s_], 0, np.random.default_rng(idx[s_])) for s_ in stems]
cz = vd.causal; vd.causal = None
smp0 = [dsd.sample(idx[s_], 0, np.random.default_rng(idx[s_])) for s_ in stems]; vd.causal = cz
torch.save(collate(smp), SP / 'batchC.pt'); torch.save(collate(smp0), SP / 'batchC0.pt')
ok_, info = same(probe(CODE, SP / 'batchC0.pt', SC, 'c0'), probe(CODE, SP / 'batchC.pt', SC, 'c1'))
ok('a batch with cz_* keys, switches off == batch without them (bitwise)', ok_, info)

# ------------------------------------------------------------------ b: front_prosody == prefix_feat89; copied functions
nb = nbad = 0; melbad = 0
for (stem, T, spk, val, feats, units, tend) in res:
    words = CF.clip_words(DS / 'annotations/words' / f'{stem}.json', T); pcm, _ = CF.clip_audio(DS / 'audio' / f'{stem}.wav')
    full = feats[4].astype(np.float32)
    for Tk in list(range(9, T, 9)) + [T]:
        p = np.asarray(CF.prefix_feat89(pcm, words, items[stem]['lang'], T, Tk, pst), np.float32)
        fp = CF.front_prosody(units, Tk, final=Tk >= T).astype(np.float16).astype(np.float32)
        nb += 1; nbad += int(not np.array_equal(fp, p[:, 80:89])); melbad += int(not np.array_equal(p[:, :80], full[:Tk, :80]))
ok('b front_prosody == prefix_feat89 columns 80-88 bitwise (f16)', nbad == 0, f'{nb - nbad} / {nb} stream times')
ok('b prefix mel rows == whole-clip mel rows', melbad == 0, f'{nb - melbad} / {nb}')
import inspect
src_cb = inspect.getsource(CF.fc().committed_beats); src_as = inspect.getsource(CF.F89().S2.assign_sides)
body = lambda s: [l.strip() for l in s.splitlines() if l.strip() and not l.strip().startswith(('"""', '#', 'def '))]
strip_c = lambda ls: [l.split('  #')[0].strip() for l in ls if not l.startswith('= ')]
ok('b committed_beats copy == scripts/feat_causality.py', strip_c(body(src_cb)) == strip_c(body(inspect.getsource(CF.committed_beats))))
ok('b assign_sides copy == feat89 human_stage2', strip_c(body(src_as)) == strip_c(body(inspect.getsource(CF.assign_sides))))

# ------------------------------------------------------------------ c: model, per-target planner
m, vae = load(CK, VAE, dev)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
g0 = torch.Generator(device=dev).manual_seed(0)
B, K, N, d = 2, 16, 64, m.cfg.d
lat = torch.randn(B, K, CHUNK, m.cfg.z, device=dev, generator=g0); a = torch.randn(B, N, d, device=dev, generator=g0)
Hh = torch.tensor([2 * U80, 8 * U80], device=dev); amask = m.chunk_audio_mask(K, N, Hh, dev)
text = torch.randn(B, 40, d, device=dev, generator=g0); tmask = torch.ones(B, 40, dtype=torch.bool, device=dev); tmask[1, 30:] = False
tvis = torch.rand(B, K, 40, device=dev, generator=g0) < .5; gg = torch.randn(B, d, device=dev, generator=g0)
sst = torch.randint(0, 6, (B, K), device=dev, generator=g0)
m.eval()
with torch.no_grad():
    h0 = m.plan(lat, a, amask, text, tmask, gg, tvis=tvis, sstate=sst)
    h1 = m.plan(lat, a[:, None].expand(B, K, N, d), amask, text, tmask, gg[:, None].expand(B, K, d), tvis=tvis, sstate=sst)
    ok('c plan per target (uniform inputs) == plan', torch.allclose(h0, h1, atol=1e-5, rtol=1e-5), f'max|d| {float((h0 - h1).abs().max()):.2e}')
    a4 = torch.randn(B, K, N, d, device=dev, generator=g0); g3 = torch.randn(B, K, d, device=dev, generator=g0)
    hA = m.plan(lat, a4, amask, text, tmask, g3, tvis=tvis, sstate=sst)
    k0 = 9; a4b, g3b = a4.clone(), g3.clone(); a4b[:, :k0] += 1.; a4b[:, k0 + 1:] -= 1.; g3b[:, :k0] *= 2; g3b[:, k0 + 1:] = 0
    hB = m.plan(lat, a4b, amask, text, tmask, g3b, tvis=tvis, sstate=sst)
    ok('c h_k depends only on encoding k / condition k', torch.equal(hA[:, k0], hB[:, k0]) and not torch.equal(hA, hB))
    aw0, m0 = m.local_audio(a, Hh, K); aw1, m1 = m.local_audio(a[:, None].expand(B, K, N, d), Hh, K)
    ok('c local_audio per-chunk (uniform) == shared', torch.equal(aw0, aw1) and torch.equal(m0, m1))

# ------------------------------------------------------------------ d: training inputs == Streamer causal inputs (psync)
from scmg import infer as INF_
qm = np.asarray(stats['q_mean'], np.float32)
H2 = 2 * U80; dd_feat = dd_spk = 0; nspk_eq = n_feat = 0; vis_bad = 0; vis_n = 0; agree = []
for stem in stems:
    ci = idx[stem]; L = vd.clips[ci]['frames']; nch = -(-L // CHUNK_FRAMES)
    T = 256; sm = WindowDataset(vd, T, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
    ok_cz = bool(sm['cz_ok'])
    if not ok_cz: ok(f'd {stem} covered', False); continue
    bt = {k: v.to(dev) for k, v in collate([sm]).items()}
    Kw = T // CHUNK_FRAMES
    fk = causal_front_feat(bt, torch.tensor([0], device=dev), torch.tensor([H2], device=dev), Kw, StageCfg().horizons)[0]   # [K, T, 89]
    spk_c, val_c, grade = causal_spk_select(bt, torch.tensor([H2], device=dev), Kw)
    # Streamer psync over the clip (window rows of tick k start at f0 = 0 while the window is trimmed)
    sw = WindowDataset(vd, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
    cond = {k: sw[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'text_c') if k in sw}
    cond['ref'] = np.zeros_like(cond['ref']); cond['ref_valid'] = np.float32(0.)
    words = CF.clip_words(DS / 'annotations/words' / f'{stem}.json', L)
    kn = int(sw['tmask'].sum()); cond.update(CF.text_extras(items[stem], words, tokz, kn, len(sw['tmask'])))
    pcm, wave = CF.clip_audio(DS / 'audio' / f'{stem}.wav')
    ci_ = CF.CausalInputs(pcm, wave, words, items[stem]['lang'], L, pst, vd.fm, vd.fs, enc, Tpad=len(sw['feat']), spk_refresh='grades')
    tr = []
    st = INF_.Streamer(m, vae, sw['feat'], cond, stats, H=H2, steps=2, seed=0, standby=qm, ctx_t=m.ctx_t, cfg_w=1.0,
                       text_mode='voiced', x2_rate=15., x2_lag=.15, deploy_timing=True, causal_inputs=ci_, trace=tr)
    nt = min(nch, Kw, 13)                                        # ticks whose window is the trimmed one (f0 = 0)
    for _ in range(nt): st.tick()
    tend = bt['cz_tend'][0]
    for x in tr:
        k = x['k']; tk = min(L, (k * CHUNK + H2) * STRIDE)
        if x['c0'] != 0: continue
        a_ = fk[k, :tk].cpu(); b_ = x['feat'][0, :tk]
        n_feat += 1; dd_feat = max(dd_feat, float((a_ - b_).abs().max()))
        av = int(bt['cz_spkt'][0, int(grade[0, k])]) if int(grade[0, k]) >= 0 else -1
        if av > tk: dd_spk += 1
        nspk_eq += 1                                 # spk_refresh 'grades' == the training rule at EVERY tick
        dd_spk += int(not torch.equal(spk_c[0, k].cpu().float(), x['spk'].float()) or float(val_c[0, k]) != float(x['spk_valid']))
    # causality of the grade rule: noise after frame t -> every vector the rule returns at Tk <= t is unchanged
    for t in (60, 130, 230, 420):
        if t >= L: continue
        w2 = wave.copy(); w2[t * CF.HOP:] = np.random.default_rng(t).normal(0, .1, len(w2) - t * CF.HOP).astype(np.float32)
        c2 = CF.CausalInputs(pcm, w2, words, items[stem]['lang'], L, pst, vd.fm, vd.fs, enc, Tpad=len(sw['feat']), spk_refresh='grades')
        for Tk in range(8, t + 1, 16):
            a1, a2 = ci_.spk(Tk), c2.spk(Tk); dd_spk += int(not (np.array_equal(a1[0], a2[0]) and a1[1] == a2[1]))
        now = k * CHUNK_FRAMES / 50.; Kt = len(x['ttime'])
        inf_vis = (x['cur_vis'][:Kt] | x['cur_vis'][Kt:]).numpy()
        ended = CF.ended(np.nan_to_num(tend.cpu().numpy(), nan=1e9), now + H2 * STRIDE / 50.)
        vis_bad += int((inf_vis & ~ended).sum()); vis_n += int(inf_vis.sum()); agree.append(float((inf_vis == (ended & sw['tmask'])).mean()))
ok('d causal_front_feat == Streamer causal feat at tick k (rows < t_k)', n_feat > 0 and dd_feat == 0., f'{n_feat} ticks, max|d| {dd_feat:.3g}')
ok('d prefix speaker grade available by t_k; == Streamer spk_refresh=grades vector at every tick; grade rule causal (noise after t)',
   dd_spk == 0, f'{nspk_eq} ticks compared')
ok('d Streamer psync visible rows have ended by t_k (training rule superset)', vis_bad == 0,
   f'{vis_n} visible rows; row agreement training-ended vs Streamer {np.mean(agree):.3f}')

# ------------------------------------------------------------------ e: future-perturbation invariance of training
def capture_plan(model):
    box = {}
    orig = model.plan
    def wrap(lat, a, amask, text, tmask, g, *args, **kw):
        box.update(a=a.detach().clone(), g=g.detach().clone(), text=text.detach().clone(), tmask=tmask.clone(),
                   tvis=None if kw.get('tvis') is None else kw['tvis'].clone(), sstate=None if kw.get('sstate') is None else kw['sstate'].clone())
        return orig(lat, a, amask, text, tmask, g, *args, **kw)
    model.plan = wrap
    return box, lambda: setattr(model, 'plan', orig)


SCE = dict(SC, p_adsc_stream=1.0, horizons=(2,), x2_rate=(15., 15.), x2_lag=(.15, .15))
stl = dict(qm=torch.tensor(stats['q_mean'], device=dev), qs=torch.tensor(stats['q_std'], device=dev))


def train_inputs(batch, **sw_):
    m.train(); box, undo = capture_plan(m)
    torch.manual_seed(0)
    with torch.no_grad(): step_losses(m, vae, {k: v.to(dev) for k, v in batch.items()}, StageCfg(**SCE, **sw_), stl)
    undo(); return box


tmpd = SP / 'perturb'; (tmpd / 'audio').mkdir(parents=True, exist_ok=True); (tmpd / 'words').mkdir(parents=True, exist_ok=True)
import soundfile as sf
for stem in stems[:3]:
    ci = idx[stem]; L = vd.clips[ci]['frames']
    for k in sorted({3, min(8, L // CHUNK_FRAMES - 2)}):
        tk = min(L, 16 * k + 8)
        if tk >= L - 16: continue
        pcm, wave = CF.clip_audio(DS / 'audio' / f'{stem}.wav'); g = np.random.default_rng(1000 + k)
        w2 = wave.copy(); w2[tk * CF.HOP:] = g.normal(0, max(1e-3, float(wave.std())), len(w2) - tk * CF.HOP).astype(np.float32)
        sf.write(tmpd / 'audio' / f'{stem}.wav', w2, CF.SR, subtype='FLOAT')
        W = json.loads((DS / 'annotations/words' / f'{stem}.json').read_text())
        W2 = dict(W, words=[w for w in W['words'] if CF.ended(float(w['end']), tk / 50.)])
        (tmpd / 'words' / f'{stem}.json').write_text(json.dumps(W2))
        rp = PC.one(job(stem, tmpd / 'audio', tmpd / 'words'))
        ro = [r for r in res if r[0] == stem][0]
        rp = rp[:6] + (ro[6],)                        # token rows / their end times: the text cache rows stay (text is handled below)
        PC.write(SP / 'cz_pert', [rp]); (SP / 'cz_pert' / 'META.json').write_text((SP / 'cz_dev' / 'META.json').read_text())
        ds_ = WindowDataset(vd, 256, max_tokens=256)
        vd.load_causal(SP / 'cz_dev', (2,)); sa = ds_.sample(ci, 0, np.random.default_rng(ci))
        vd.load_causal(SP / 'cz_pert', (2,), check_spk=False); sb = ds_.sample(ci, 0, np.random.default_rng(ci))
        ba, bb = collate([sa]), collate([sb])
        # the perturbed clip's own mel (the view mel is the unperturbed one): whole-clip grade of the perturbed precompute
        n_ = min(256, L); melp = (rp[4][4][:n_, :80].astype(np.float32) - vd.fm[:80]) / vd.fs[:80]
        for kk in ('feat', 'cz_feat'):
            x = bb[kk].numpy().copy(); x[0, :n_, :80] = melp; bb[kk] = torch.as_tensor(x)
        # text not voiced by t_k: random features, later times (deleted words cannot be voiced yet)
        now_end = ~CF.ended(np.nan_to_num(ba['cz_tend'][0].numpy(), nan=1e9), (k * CHUNK_FRAMES + 2 * U80 * STRIDE) / 50.)
        r2 = np.random.default_rng(5)
        for kk in ('text', 'text_c'):
            x = bb[kk].numpy().copy(); x[0, now_end] = r2.normal(size=x[0, now_end].shape); bb[kk] = torch.as_tensor(x)
        tt = bb['ttime'].numpy().copy(); tt[0, now_end] = tt[0, now_end] + 1.5; bb['ttime'] = torch.as_tensor(tt)
        te = bb['cz_tend'].numpy().copy(); te[0, now_end] = te[0, now_end] + 1.5; bb['cz_tend'] = torch.as_tensor(te)
        on = dict(causal_front=True, causal_spk=True, causal_text=True)
        A_, B_ = train_inputs(ba, **on), train_inputs(bb, **on)
        nK = A_['tvis'].shape[-1]
        vis_a = A_['tvis'][0, k]; vis_b = B_['tvis'][0, k]
        diffs = []
        if not torch.equal(A_['a'][0, k], B_['a'][0, k]): diffs.append('audio encoding k')
        if not torch.equal(A_['g'][0, k], B_['g'][0, k]): diffs.append('global condition k')
        if not torch.equal(vis_a, vis_b): diffs.append('text visibility k')
        elif not torch.equal(A_['text'][0][vis_a], B_['text'][0][vis_b]): diffs.append('visible text rows')
        if A_['sstate'] is not None and not torch.equal(A_['sstate'][0, k], B_['sstate'][0, k]): diffs.append('stream state k')
        ok(f'e training inputs of chunk k invariant to the future: {stem} k={k} t_k={tk / 50:.2f}s '
           f'({len(W["words"]) - len(W2["words"])} words deleted, {int(now_end.sum())} rows not voiced)', not diffs, f'diffs {diffs}')
        # control: default training inputs (whole-clip view feat + speaker) under the same perturbation
        fb = (np.asarray(CF.full_feat89(np.clip(np.round(w2 * 32768.), -32768, 32767).astype(np.int16),
                                        CF.clip_words(tmpd / 'words' / f'{stem}.json', L), items[stem]['lang'], L), np.float32) - vd.fm) / vd.fs
        bc = dict(bb); fbb = bc['feat'].numpy().copy(); n_ = min(256, L); fbb[0, :n_] = fb[:n_]; bc['feat'] = torch.as_tensor(fbb)
        bc['spk'] = torch.as_tensor(CF.SpeakerPrefix(w2, enc).full())[None]
        A0, C0 = train_inputs(ba), train_inputs(bc)
        ch = (not torch.equal(A0['a'][0, :(16 * k + 8) // STRIDE], C0['a'][0, :(16 * k + 8) // STRIDE])) or not torch.equal(A0['g'], C0['g'])
        ok(f'e control: default training inputs DO change under the perturbation ({stem} k={k})', ch)
print('ALL_OK' if not FAIL else f'SOME_FAIL {FAIL}')
sys.exit(1 if FAIL else 0)

"""11-07 duplex training switch StageCfg.yield_cf_p (counterfactual YIELD) -- default off, bitwise.
    SCMG_J=39 CUDA_VISIBLE_DEVICES=k python scripts/tests/test_duplex_train.py   env CK VAE STATS HOSV TEXT TEXTC CZ YTOK HUMAN DS PSTATS BASE_TREE SP
a  default: losses + every gradient bitwise equal to BASE_TREE (code before the switch) on a real causal-recipe batch
   (czm sc), and with yield_cf_p 0 given explicitly
b  live: yield_cf_p 1 on a ctl model: finite losses, yield_cf_n > 0, the losses change
c  prepared rows (yield_cf_prepare / yield_silence): YIELD token at t_y in a free slot; frames < t_y unchanged, frames >= t_y a
   damped retraction (upper joints -> R_sil, legs held); words not ended by t_y withdrawn; for chunks with t_k > t_y the encoder
   rows [t_y, t_k) are the silence row and the prosody of rows < t_y is the prefix at t_y; chunks with t_k <= t_y unchanged
d  training == inference: on real clips the training silence input of chunk k equals PostInterrupt-style CausalInputs on the
   silenced PCM with the words ended by t_y (mel rows < t_y bitwise up to the f16 ulp, prosody rows < t_y bitwise; silence rows
   from t_y + 8 frames within 1e-3: the inference mel window still covers speech samples for <= 8 frames); the speaker grade
   is the one available at t_y"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve(); CODE = HERE.parents[2]
sys.path.insert(0, str(CODE))
import torch
E = os.environ; SP = Path(E['SP']); SP.mkdir(parents=True, exist_ok=True)
os.environ['SCMG_TEXT_CAUSAL'] = E['TEXTC']
from scmg.common import CHUNK, STRIDE, U80
from scmg import causal_feat as CF
from scmg.data import View, WindowDataset, collate
from scmg.losses import StageCfg, yield_cf_prepare, yield_silence, causal_front_feat, causal_spk_select, ycf_sil
from scmg.ctl import YIELD
FAIL = []


def ok(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not cond: FAIL.append(name)


stats = json.loads(Path(E['STATS']).read_text())
SC = dict(p_stream=0.67, p_async=1.0, p_async_seg=0.0, self_force_p=0, w_safe=300., causal_front=True, causal_spk=True, causal_text=True)
hz = StageCfg().horizons
vt = View(E['HOSV'], 'train', E['TEXT'], stats); vt.load_causal(E['CZ'], hz)
it_ = iter(WindowDataset(vt, 256, max_tokens=256, seed=7)); s0 = [next(it_) for _ in range(8)]
bA = collate(s0); torch.save(bA, SP / 'bA.pt')
def probe(code, bp, cfg, sc, tag):
    out = SP / f'probe_{tag}.pt'
    r = subprocess.run([sys.executable, str(HERE.parent / '_causal_loss_probe.py'), str(bp), str(out), E['CK'], E['VAE'], E['STATS'], json.dumps(cfg), json.dumps(sc)],
                       env=dict(os.environ, PYTHONPATH=str(code)), cwd=str(code), capture_output=True, text=True)
    if r.returncode: print(r.stderr[-2500:])
    return torch.load(out, weights_only=False) if r.returncode == 0 else None
same = lambda p, q: p is not None and q is not None and all(torch.equal(p['loss'][k], q['loss'][k]) for k in p['loss']) and \
    set(p['grad']) == set(q['grad']) and all(torch.equal(p['grad'][k], q['grad'][k]) for k in p['grad'])
pb = probe(E['BASE_TREE'], SP / 'bA.pt', {'adsc': True}, SC, 'base'); pn = probe(CODE, SP / 'bA.pt', {'adsc': True}, SC, 'new')
p0 = probe(CODE, SP / 'bA.pt', {'adsc': True}, dict(SC, yield_cf_p=0.0), 'new0')
ok('a default czm losses + grads == BASE_TREE', same(pb, pn), f"{len(pn['grad']) if pn else 0} grads")
ok('a yield_cf_p 0 explicit == default', same(pn, p0))
# ctl batch with the YIELD token arrays (tail tokens: none of these clips need one)
vy = View(E['HOSV'], 'train', E['TEXT'], stats, ctl_tokens=E['YTOK']); vy.load_causal(E['CZ'], hz)
it2 = iter(WindowDataset(vy, 256, max_tokens=256, seed=7)); s1 = [next(it2) for _ in range(8)]
bY = collate(s1); torch.save(bY, SP / 'bY.pt')
CTL = {'adsc': True, 'ctl': True, 'ctl_flow': True}
SY = dict(SC, yield_cf_p=1.0, yield_human=E['HUMAN'])
pc0 = probe(CODE, SP / 'bY.pt', CTL, SC, 'ctl0'); pc1 = probe(CODE, SP / 'bY.pt', CTL, SY, 'ctl1')
ok('b live: finite, yield_cf_n > 0, losses change', pc1 is not None and all(np.isfinite(float(v)) for v in pc1['loss'].values())
   and float(pc1['loss'].get('yield_cf_n', torch.tensor(0.))) > 0 and not same(pc0, pc1),
   '' if pc1 is None else f"yield_cf_n {float(pc1['loss'].get('yield_cf_n', torch.tensor(0.)))}")
# c: direct
class M_: pass
from scmg.evaluate import load
m, vae = load(E['CK'], E['VAE'], 'cpu')
for k, v in CTL.items(): setattr(m.cfg, k, v)
m.add_modules()
st = dict(qm=torch.tensor(stats['q_mean']), qs=torch.tensor(stats['q_std']), fm_raw=stats['f_mean'], fs_raw=stats['f_std'])
b = {k: v.clone() for k, v in bY.items()}; B = b['q'].shape[0]; K = 256 // 16
torch.manual_seed(1)
stream = b['cz_ok'].bool(); H = torch.full((B,), 2 * U80)
with torch.no_grad():
    mu, _ = vae.encode(b['q']); x1 = (mu - m.lat_mean) / m.lat_std
sc = StageCfg(**SY)
b2, q2, x12, ycf = yield_cf_prepare(m, vae, b, sc, st, stream, H, K, x1, b['q'])
on = ycf['on']; ly = ycf['ly']
ok('c some samples prepared', bool(on.any()), f'{int(on.sum())} / {B}')
rest = torch.tensor(json.loads(Path(E['HUMAN']).read_text())['rest']['R_sil']['joints'])
qm, qs = st['qm'], st['qs']; good = True; info = []
for i in on.nonzero()[:, 0].tolist():
    fy = int(ly[i]) * STRIDE; ty = fy / 50.
    slot = (b2['ctl_type'][i] == YIELD).nonzero()[:, 0]
    good &= len(slot) == 1 and abs(float(b2['ctl_t'][i, slot[0]]) - ty) < 1e-5
    good &= torch.equal(q2[i, :fy], bY['q'][i, :fy])
    qp = q2[i].float() * qs + qm
    nv = int(bY['mask'][i].sum()); end = min(nv, 256) - 1
    good &= float((qp[end, 22:39] - rest[12:29]).abs().max()) < 0.05 and torch.allclose(qp[fy:end, 10:22], (bY['q'][i, fy - 1].float() * qs + qm)[10:22][None].expand(end - fy, -1), atol=1e-4)
    te = torch.nan_to_num(bY['cz_tend'][i], nan=1e9)
    good &= bool(((b2['tmask'][i] == (bY['tmask'][i] & (te <= ty + CF.WORD_EPS)))).all())
    info.append(f'ly {int(ly[i])}')
ok('c token / motion / text rewrite', good, ' '.join(info[:6]))
bi = torch.arange(B)
fk0 = causal_front_feat(b2, bi, H, K, hz); fk1 = yield_silence(fk0, b2, bi, H, K, hz, ycf)
sil = ycf_sil(st, 'cpu'); good = True
from scmg.causal_feat import FRONT_COLS
lut = {l: j for j, l in enumerate(CF.stream_lat_grid(K, hz, CHUNK, U80))}
for i in range(B):
    for k in range(K):
        tk = k * CHUNK + int(H[i])
        if not bool(on[i]) or tk <= int(ly[i]): good &= torch.equal(fk1[i, k], fk0[i, k]); continue
        fy = int(ly[i]) * STRIDE
        good &= bool((fk1[i, k, fy:tk * STRIDE] == sil).all()) and bool((fk1[i, k, tk * STRIDE:] == 0).all())
        good &= torch.equal(fk1[i, k, :fy, list(FRONT_COLS)], b2['cz_pro'][i, lut[int(ly[i])], :fy]) and torch.equal(fk1[i, k, :fy, :80], fk0[i, k, :fy, :80])
ok('c encoder rows: silence after t_y, prefix prosody at t_y before it, untouched otherwise', good)
# d: training == inference on real clips (the PostInterrupt silence rule of infer_retract)
pst = CF.load_pstats(E['PSTATS']); items = {it['id']: it for it in json.loads((Path(E['HOSV']) / 'train.json').read_text())}
dmel = dpro = dsil = 0.; n = 0; spk_ok = True; trans = 0
for i in on.nonzero()[:, 0].tolist()[:4]:
    ci_ = int(bY['clip'][i]); c = vy.clips[ci_]; stem = c['items'][0]['id']; s_ = int(bY['start'][i]); L = c['frames']
    pcm, wave = CF.clip_audio(Path(E['DS']) / 'audio' / f'{stem}.wav'); w = CF.clip_words(Path(E['DS']) / 'annotations' / 'words' / f'{stem}.json', L)
    fy = s_ + int(ly[i]) * STRIDE; ty = fy / 50.
    pz = np.concatenate([np.array(pcm[:fy * CF.HOP], np.int16), np.zeros(4000 * CF.HOP, np.int16)])
    inc = CF.CausalInputs(pz, wave, [x for x in w if CF.ended(x['end_time'], ty)], items[stem]['lang'], L + 2000, pst, vy.fm, vy.fs, None, Tpad=L + 2000, spk=False, feat_mode='inc')
    for k in range(K):
        tk = k * CHUNK + int(H[i])
        if tk <= int(ly[i]): continue
        Ta = s_ + tk * STRIDE; F_ = inc.feat(Ta)[s_:s_ + 256]
        lo = int(ly[i]) * STRIDE
        dmel = max(dmel, float(np.abs(F_[:lo, :80] - fk1[i, k, :lo, :80].numpy()).max()))
        dpro = max(dpro, float(np.abs(F_[:lo, 80:89] - fk1[i, k, :lo, 80:89].numpy()).max()))
        if tk * STRIDE > lo:
            dd = np.abs(F_[lo:tk * STRIDE] - fk1[i, k, lo:tk * STRIDE].numpy()).max(1)          # per frame after t_y
            bad_ = np.nonzero(dd > 1e-3)[0]; trans = max(trans, int(bad_.max()) + 1 if len(bad_) else 0)
            if len(dd) > 8: dsil = max(dsil, float(dd[8:].max()))
        n += 1
    spk, val, gr = causal_spk_select({k_: v[i:i + 1] for k_, v in b2.items()}, H[i:i + 1], K, ycf['cap'][i:i + 1])
    av = b2['cz_spkt'][i]; spk_ok &= all((int(g) < 0) or int(av[int(g)]) <= fy for g in gr[0])
ok('d training silence input == inference (mel / prosody < t_y; silence from t_y + 8 frames = the STFT spill)', n > 0 and dmel < 0.02 and dpro < 1e-5 and dsil < 1e-3 and trans <= 8,
   f'{n} chunks: max |mel| {dmel:.2e} |prosody| {dpro:.2e} |silence after 8| {dsil:.2e}; differing transition {trans} frames')
ok('d speaker grade capped at t_y', spk_ok)
print('ALL_OK' if not FAIL else f'SOME_FAIL {FAIL}'); sys.exit(1 if FAIL else 0)

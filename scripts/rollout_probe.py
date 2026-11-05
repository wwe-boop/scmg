"""Hold-attractor test (11-03, Claude): free-running vs ground-truth-history rollout.
    python scripts/rollout_probe.py <vae.pt> label=ckpt[@cfg_w] ...
For N dev clips (offline, generation seed 0) every model is rolled out twice:
  free    normal streaming: each chunk conditioned on its OWN previous chunks
  oracle  after every tick the committed chunk is replaced by the ground-truth latents, so every chunk is a one-step
          prediction from GT history (the generated latents are recorded before the replacement)
Reported per mode (frames within 2 of a chunk boundary excluded from the motion statistics):
  speed     mean arm speed / GT;  still  fraction of frames slower than 0.25 x GT median speed
  copy      mean ||gen chunk k - history chunk k-1|| / mean ||GT chunk k - GT chunk k-1|| in latent space
            (history = own chunk k-1 for free, GT chunk k-1 for oracle; << 1 = the model repeats what it was shown)
Speed / stillness use the ARM joints of the embodiment only (scmg.common.ARMS): EX002 channels 4..17 of 34; ECHO-G G1
channels 25..38 of 39 (left + right arm 7 each). G1 legs (10..21), waist (22..24) and root (0..9: 6D orientation, yaw
change, root velocity) are NOT part of speed / still. The copy ratio is in latent space (all channels).
Data (11-04): EX002 defaults (v36_1 val split, 43-clip scan). Other embodiments need explicit ROLLOUT_VIEW / ROLLOUT_STATS /
ROLLOUT_TEXT (view, stats, e5 cache; dev split 'val' -- never the test view); ROLLOUT_SCAN = clips scanned (default 43),
ROLLOUT_NCLIP = clips used (default 20). Engine mode (ROLLOUT_TEXT_MODE=x2) needs SCMG_TEXT_CAUSAL of the same rows."""
import argparse, json, sys
import numpy as np, torch
import os
from scmg.data import View, WindowDataset, clip_lm_capacity
from scmg.token_cache import checkpoint_config, resolve_token_cache
from scmg.evaluate import load, ARMS
from scmg.infer import Streamer
from scmg.landing import apply_cond
from scmg.common import CHUNK, CHUNK_FRAMES, U80, EMBODIMENT, ARMS as ARMS_EMB
from scmg.data import VIEW
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'; dev = 'cuda'
if ARMS != ARMS_EMB: raise RuntimeError('evaluate.ARMS differs from the embodiment arm joints')
_EX = EMBODIMENT == 'ex002'
_need = lambda k, d: os.environ.get(k) or (d if _EX else sys.exit(f'{k} is required for embodiment {EMBODIMENT} (no EX002 default)'))
R_VIEW = _need('ROLLOUT_VIEW', VIEW); R_STATS = _need('ROLLOUT_STATS', D + '/STATS_v36_1.json'); R_TEXT = _need('ROLLOUT_TEXT', D + '/text-e5-v36_1')
SCAN = int(os.environ.get('ROLLOUT_SCAN', '43'))
st = json.load(open(R_STATS)); qm, qs = np.array(st['q_mean'], np.float32), np.array(st['q_std'], np.float32)
ap = argparse.ArgumentParser(); ap.add_argument('vae'); ap.add_argument('models', nargs='+')
ap.add_argument('--tok-cache'); args = ap.parse_args()
VAE = args.vae; NCLIP = int(os.environ.get('ROLLOUT_NCLIP', '20'))
MAXCHUNK = int(os.environ.get('ROLLOUT_MAXCHUNK', '40'))   # cap rollout length per clip (quick curves use 20)
if os.environ.get('ROLLOUT_TEXT_CACHE') and not os.environ.get('ROLLOUT_TEXT'):   # exp/llm-text name for the same switch
    R_TEXT = os.environ['ROLLOUT_TEXT_CACHE']
WIN = os.environ.get('ROLLOUT_WINDOW_LM_TEXT') == '1'   # 11-03 review: windowed LM text (training protocol) for VoxCPM
TM = os.environ.get('ROLLOUT_TEXT_MODE', 'final')      # 11-03 user: deployment text streams word by word -> x2 (+ deploy timing, windowed LM)
X2_LAG = float(os.environ.get('ROLLOUT_X2_LAG', '0.15'))
_H = os.environ.get('ROLLOUT_H', 'inf')                 # 11-04 baseline protocol: audio look-ahead in 80 ms units ('inf' = offline)
SEED = int(os.environ.get('ROLLOUT_SEED', '0'))
FIX = os.environ.get('ROLLOUT_FIX', '1') == '1'
RF = os.environ.get('ROLLOUT_REFRESH'); RF = tuple(float(x) for x in RF.split(',')) if RF else None   # 'thr,t_low' still refresh              # trim_start + cfg_null_fix (agreed default protocol, 11-03 V4)
if TM != 'final': WIN = True


def clips(tok):
    v = View(R_VIEW, split='val', stats=st, mmap=True, text_cache=R_TEXT, tok_cache=tok); out = []
    for ci in range(min(SCAN, len(v.clips))):
        nch = min(v.clips[ci]['frames'] // CHUNK_FRAMES, MAXCHUNK)
        if nch < 8: continue
        s = WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=256, max_ids=clip_lm_capacity(v, ci) if WIN else 160).sample(ci, 0, np.random.default_rng(ci))
        if s['mask'].min() < 1: continue
        out.append((nch, s))
        if len(out) == NCLIP: break
    return out


def speed(q): return np.linalg.norm(np.diff(q[:, ARMS], axis=0), axis=1) * 50


@torch.no_grad()
def run(m, vae, data, w, oracle):
    sp, gsp, still, cp = [], [], [], []
    for nch, s in data:
        q = torch.from_numpy(s['q']).float().to(dev)[None]
        mu, _ = vae.encode(q); gl = ((mu - m.lat_mean) / m.lat_std)[0]                 # [N, z] GT latents (normalised)
        cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c') if k in s}
        cond = apply_cond(cond)                                                         # SCMG_COND (11-05 landing)
        S = Streamer(m, vae, s['feat'], cond, st, H=10 ** 9 if _H == 'inf' else int(_H) * U80, steps=10, seed=SEED, standby=st['q_mean'], ctx_t=m.ctx_t, cfg_w=w,
                     window_lm_text=WIN and m.cfg.planner == 'voxcpm', text_mode=TM, x2_rate=15.0, x2_lag=X2_LAG, deploy_timing=TM != 'final', trim_start=FIX, cfg_null_fix=FIX, still_refresh=RF)
        gen = []
        for k in range(nch):
            S.tick(None, None, None)
            gen.append(torch.stack(S.lat[-CHUNK:]))
            if oracle:
                for i in range(CHUNK): S.lat[-CHUNK + i] = gl[k * CHUNK + i].clone()
        G = torch.cat(gen)                                                             # [nch*CHUNK, z] generated
        y = vae.decode((G * m.lat_std + m.lat_mean)[None].float())[0].cpu().numpy() * qs + qm
        x = s['q'][:len(y)] * qs + qm
        keep = np.ones(len(y) - 1, bool); f = np.arange(len(keep)) + 1
        keep &= ~((f % CHUNK_FRAMES <= 2) | (f % CHUNK_FRAMES >= CHUNK_FRAMES - 2))
        sy, sx = speed(y)[keep], speed(x)[keep]
        sp.append(sy.mean()); gsp.append(sx.mean()); still.append((sy < 0.25 * np.median(sx)).mean())
        Gc = G.view(nch, CHUNK, -1); Tc = gl[:nch * CHUNK].view(nch, CHUNK, -1)
        prev = Tc[:-1] if oracle else Gc[:-1]
        cp.append(float((Gc[1:] - prev).flatten(1).norm(dim=1).mean() / (Tc[1:] - Tc[:-1]).flatten(1).norm(dim=1).mean()))
    return dict(speed=float(np.mean(sp) / np.mean(gsp)), still=float(np.mean(still)), copy=float(np.mean(cp)))


cache = {}
for spec in args.models:
    lab, ck = spec.split('=', 1); w = 2.0
    if '@' in ck: ck, w = ck.rsplit('@', 1); w = float(w)
    cfg = checkpoint_config(ck, 'rollout_probe.py')
    tok = resolve_token_cache(cfg, args.tok_cache)
    m, vae = load(ck, VAE, dev); m.eval()
    if tok not in cache: cache[tok] = clips(tok)
    pr = dict(text_mode=TM, x2_lag=X2_LAG, H=_H, seed=SEED, fix=FIX, window=WIN, refresh=RF)
    if not _EX: pr.update(embodiment=EMBODIMENT, view=R_VIEW, clips=len(cache[tok]), still_joints=f'arms {ARMS[0]}..{ARMS[-1]}')
    r = dict(protocol=pr, label=lab, free=run(m, vae, cache[tok], w, False), oracle=run(m, vae, cache[tok], w, True))
    print(json.dumps(r), flush=True)
    del m; torch.cuda.empty_cache()

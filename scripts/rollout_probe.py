"""Hold-attractor test (11-03, Claude): free-running vs ground-truth-history rollout.
    python scripts/rollout_probe.py <vae.pt> label=ckpt[@cfg_w] ...
For N dev clips (offline, generation seed 0) every model is rolled out twice:
  free    normal streaming: each chunk conditioned on its OWN previous chunks
  oracle  after every tick the committed chunk is replaced by the ground-truth latents, so every chunk is a one-step
          prediction from GT history (the generated latents are recorded before the replacement)
Reported per mode (frames within 2 of a chunk boundary excluded from the motion statistics):
  speed     mean arm speed / GT;  still  fraction of frames slower than 0.25 x GT median speed
  copy      mean ||gen chunk k - history chunk k-1|| / mean ||GT chunk k - GT chunk k-1|| in latent space
            (history = own chunk k-1 for free, GT chunk k-1 for oracle; << 1 = the model repeats what it was shown)"""
import json, sys
import numpy as np, torch
import os
from scmg.data import View, WindowDataset, clip_lm_capacity
from scmg.evaluate import load, ARMS
from scmg.infer import Streamer
from scmg.common import CHUNK, CHUNK_FRAMES
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'; dev = 'cuda'
st = json.load(open(D + '/STATS_v36_1.json')); qm, qs = np.array(st['q_mean'], np.float32), np.array(st['q_std'], np.float32)
VAE = sys.argv[1]; NCLIP = 20
WIN = os.environ.get('ROLLOUT_WINDOW_LM_TEXT') == '1'   # 11-03 review: windowed LM text (training protocol) for VoxCPM
TM = os.environ.get('ROLLOUT_TEXT_MODE', 'final')      # 11-03 user: deployment text streams word by word -> x2 (+ deploy timing, windowed LM)
X2_LAG = float(os.environ.get('ROLLOUT_X2_LAG', '0.15'))
if TM != 'final': WIN = True


def clips(tok):
    v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', tok_cache=tok); out = []
    for ci in range(43):
        nch = min(v.clips[ci]['frames'] // CHUNK_FRAMES, 40)
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
        S = Streamer(m, vae, s['feat'], cond, st, H=10 ** 9, steps=10, seed=0, standby=st['q_mean'], ctx_t=m.ctx_t, cfg_w=w,
                     window_lm_text=WIN and m.cfg.planner == 'voxcpm', text_mode=TM, x2_rate=15.0, x2_lag=X2_LAG, deploy_timing=TM != 'final')
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
for spec in sys.argv[2:]:
    lab, ck = spec.split('=', 1); w = 2.0
    if '@' in ck: ck, w = ck.rsplit('@', 1); w = float(w)
    m, vae = load(ck, VAE, dev)
    tok = D + '/tok-qwen25-v36_1'
    if 'qwen3' in (getattr(m.cfg, 'vox_lm', '') or '').lower(): tok = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/tok-qwen3-0.6b-v36_1'
    if 'voxcpm' in (getattr(m.cfg, 'vox_lm', '') or '').lower(): tok = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/tok-voxcpm15-v36_1'
    if tok not in cache: cache[tok] = clips(tok)
    r = dict(label=lab, free=run(m, vae, cache[tok], w, False), oracle=run(m, vae, cache[tok], w, True))
    print(json.dumps(r), flush=True)
    del m; torch.cuda.empty_cache()

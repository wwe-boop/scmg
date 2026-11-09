"""Long-horizon stability (Claude's protocol, all model families): 60 s free-running rollouts.
    python scripts/long_stability.py <kind: scmg|dots> <ckpt> <out.json> [dots code dir] [cfg_w] [n_long]
Long inputs: dev clips of ONE speaker concatenated back to back (audio features, text tokens with shifted times,
speaker/source of the first clip) until >= 60 s; n_long such sequences. Start = the first clip's first GT frames.
Per 10-s bin: arm speed (rad/s), joint-limit fraction, max |q|; plus the GT arm speed of the same audio."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from scmg.data import View, WindowDataset, VIEW, collate

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
KIND, CK, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
CFGW = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0; NL = int(sys.argv[6]) if len(sys.argv) > 6 else 6
SEC = 60; T = SEC * 50 // 16 * 16
meta = json.load(open(f'{VIEW}/META.json')); lo, hi = np.array(meta['lower']), np.array(meta['upper']); A = slice(4, 18)
dev = 'cuda'
if KIND == 'scmg':
    from scmg.evaluate import load
    from scmg.infer import generate
    model, vae = load(CK, '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/trained/vae-r3/final.pt', dev)
    st = json.load(open(D + '/STATS_v36_1.json'))
else:
    sys.path.insert(0, sys.argv[4])
    from dots_motion.model import MotionFlow, build_chunk
    ck = torch.load(CK, map_location='cpu', weights_only=False); st = ck['stats']
    model = MotionFlow(**ck['model_config']).to(dev); model.load_state_dict(ck['model']); model.eval()
v = View(VIEW, 'val', D + '/text-e5-v36_1', st, mmap=True, tok_cache=D + '/tok-qwen25-v36_1'); v.glitch = None
qm, qs = np.asarray(st['q_mean'], np.float32), np.asarray(st['q_std'], np.float32)
by = defaultdict(list)
for ci, c in enumerate(v.clips):
    if c['frames'] >= 160: by[c['speaker']].append(ci)
spk = sorted(by, key=lambda s_: -sum(v.clips[c]['frames'] for c in by[s_]))[:NL]


def long_sample(cis):
    parts, t0 = [], 0
    for ci in cis * 20:
        n = v.clips[ci]['frames'] // 16 * 16
        s = WindowDataset(v, n, ref_frames=16, max_tokens=64).sample(ci, 0, np.random.default_rng(ci))
        s['ttime'] = s['ttime'] + t0 / 50; parts.append(s); t0 += n
        if t0 >= T: break
    L = dict(parts[0]); L['q'] = np.concatenate([p['q'] for p in parts])[:T]; L['feat'] = np.concatenate([p['feat'] for p in parts])[:T]
    L['mask'] = np.ones(T, np.float32)
    tk = [(p['text'][p['tmask']], p['ttime'][p['tmask']]) for p in parts]
    text = np.concatenate([x[0] for x in tk]); tt = np.concatenate([x[1] for x in tk]); K = min(len(text), 256)
    L['text'] = np.zeros((256, text.shape[1]), np.float32); L['text'][:K] = text[:K]
    L['ttime'] = np.full(256, np.nan, np.float32); L['ttime'][:K] = tt[:K]; L['tmask'] = np.zeros(256, bool); L['tmask'][:K] = True
    return L


rows = []
for s_ in spk:
    L = long_sample(by[s_]); gt = L['q'] * qs + qm
    if KIND == 'scmg':
        cond = {k: L[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm')}
        y, _, _ = generate(model, vae, L['feat'], cond, st, T // 16, H=10 ** 6, steps=10, standby=gt[0],
                           ctx_t=getattr(model, 'ctx_t', 1.0), cfg_w=CFGW, cfg_text=model.cfg.planner != 'voxcpm')
    else:
        b = {k: x.to(dev) for k, x in collate([L]).items()}
        for k in list(b):
            if b[k].is_floating_point(): b[k] = b[k].float()
        g = torch.Generator(device=dev).manual_seed(0)
        with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
            for index in range(1, T - 1, 16):
                c, _, _ = build_chunk(b, index, model.history)
                pred = model.sample(c, steps=8, noise=torch.randn(1, 16, 34, device=dev, generator=g)).float()
                n = min(16, T - 1 - index); b['q'][:, index + 1:index + 1 + n] = pred[:, :n]
        y = b['q'][0].float().cpu().numpy() * qs + qm
    n = min(len(y), len(gt)); y, g_ = y[:n], gt[:n]
    bins = []
    for a in range(0, n - 499, 500):
        yy, gg = y[a:a + 500], g_[a:a + 500]
        bins.append(dict(speed=float(np.abs(np.diff(yy[:, A], axis=0)).mean() * 50), gt_speed=float(np.abs(np.diff(gg[:, A], axis=0)).mean() * 50),
                         limit=float(((yy < lo) | (yy > hi)).mean()), maxabs=float(np.abs(yy).max())))
    rows.append(dict(speaker=s_, bins=bins)); print(s_, [round(b_['speed'] / max(b_['gt_speed'], 1e-6), 2) for b_ in bins], flush=True)
agg = {k: [float(np.mean([r['bins'][i][k] for r in rows if i < len(r['bins'])])) for i in range(6)] for k in ('speed', 'gt_speed', 'limit', 'maxabs')}
agg['speed_ratio_per_10s'] = [s_ / max(g_, 1e-6) for s_, g_ in zip(agg['speed'], agg['gt_speed'])]
OUT.write_text(json.dumps(dict(kind=KIND, ckpt=CK, cfg_w=CFGW, aggregate=agg, rows=rows), indent=1)); print(json.dumps(agg))

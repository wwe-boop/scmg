"""Is the planner pulled toward the average motion? (11-03, Claude)  usage: python scripts/planner_probe.py <vae.pt> label=ckpt ...
On fixed dev windows (val view, 256 frames, offline, GT history = teacher forcing) for each checkpoint:
  plan shrinkage   p_aux(h) predicts each chunk's latents from the plan h: var(pred) / var(GT latents) over chunks
                   (1 = full spread, << 1 = regression to the mean) and R^2 of that prediction
  flow spread      one Local Flow sample per chunk with GT context: var(generated chunk latents) / var(GT) and their MSE
  modality use     ||h(no audio) - h|| / ||h|| and ||h(no text) - h|| / ||h|| (how much the plan depends on each input)
Prints one JSON line per model."""
import json, sys
import numpy as np, torch
sys.path.insert(0, __import__('os').path.dirname(__file__) + '/..')
from scmg.data import View, WindowDataset, collate
from scmg.evaluate import load
from scmg.common import CHUNK, STRIDE
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'; dev = 'cuda'
st = json.load(open(D + '/STATS_v36_1.json'))
VAE = sys.argv[1]; NW = 48


def windows(tok):
    v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', tok_cache=tok)
    ds = WindowDataset(v, 256, max_tokens=128)
    rng = np.random.default_rng(0); out = []
    for ci in range(min(len(v.clips), 380)):
        L = v.clips[ci]['frames']
        if L < 300: continue
        s = int(rng.integers(0, L - 256)) // 16 * 16
        out.append(ds.sample(ci, s, np.random.default_rng(ci)))
        if len(out) == NW: break
    return out


@torch.no_grad()
def probe(m, vae, samples):
    import scmg.losses as Lm
    rec = {}
    def hook(name):
        def f(mod, inp, outp): rec[name] = (inp[0].detach().float(), outp.detach().float())
        return f
    hd = m.p_aux.register_forward_hook(hook('aux'))
    fv = m.flow_velocity
    res = dict(pred=[], gt=[], h=[], h_na=[], h_nt=[])
    for i in range(0, len(samples), 8):
        b = {k: x.to(dev) for k, x in collate(samples[i:i + 8]).items()}
        B = b['q'].shape[0]
        mu, _ = vae.encode(b['q']); x1 = (mu - m.lat_mean) / m.lat_std; K = x1.shape[1] // CHUNK
        lat_k = x1.view(B, K, CHUNK, -1)
        hs = {}
        for tag, dt, da in (('h', 0., 0.), ('h_na', 0., 1.), ('h_nt', 1., 0.)):
            m.cfg.drop_text, m.cfg.drop_audio = dt, da
            torch.manual_seed(0)
            Lm.step_losses(m, vae, b, Lm.StageCfg(self_force_p=0.0, p_noctx=0.0), dict(qm=torch.tensor(st['q_mean'], device=dev), qs=torch.tensor(st['q_std'], device=dev)))
            hs[tag] = rec['aux']
        h, pred = hs['h']; res['h'].append(h.flatten(0, -2).cpu()); res['h_na'].append(hs['h_na'][0].flatten(0, -2).cpu())
        res['h_nt'].append(hs['h_nt'][0].flatten(0, -2).cpu())
        res['pred'].append(pred.flatten(0, -2).cpu()); res['gt'].append(lat_k.flatten(-2).flatten(0, 1).float().cpu())
    hd.remove()
    P = torch.cat(res['pred']); G = torch.cat(res['gt']); H = torch.cat(res['h'])
    shrink = float(P.var(0).sum() / G.var(0).sum()); r2 = float(1 - ((P - G) ** 2).sum() / ((G - G.mean(0)) ** 2).sum())
    rel = lambda a: float((torch.cat(res[a]) - H).norm(dim=-1).mean() / H.norm(dim=-1).mean())
    return dict(plan_shrink=shrink, plan_r2=r2, h_change_no_audio=rel('h_na'), h_change_no_text=rel('h_nt'))


samples_cache = {}
for spec in sys.argv[2:]:
    lab, ck = spec.split('=', 1)
    m, vae = load(ck, VAE, dev); m.eval()
    tok = D + '/tok-qwen25-v36_1'
    if getattr(m.cfg, 'vox_lm', ''):
        tok = {'qwen3': '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/tok-qwen3-0.6b-v36_1',
               'minicpm': '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/tok-minicpm4-v36_1',
               'voxcpm': '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/tok-voxcpm15-v36_1'}[
            next(k for k in ('qwen3', 'minicpm', 'voxcpm') if k in m.cfg.vox_lm.lower())]
    if tok not in samples_cache: samples_cache[tok] = windows(tok)
    r = probe(m, vae, samples_cache[tok]); r['label'] = lab
    print(json.dumps(r), flush=True)
    del m; torch.cuda.empty_cache()

"""Contact-aware DECODER fine-tuning of the G1 motion VAE (11-05).

The VAE alone doubles the official contact sliding (round trip 0.043 -> 0.086) and removes a quarter of the jerk. The encoder
is frozen, so the latent space -- and every generator trained on it -- stays valid: only the decoder is trained, with the
VAE's own reconstruction terms plus the G1 foot-contact consistency term (scmg/g1_contact.py, official sole geometry) on the
decoded physical frames. Latents are sampled from the frozen posterior (z = mu + sigma * eps) as in VAE training, so the
decoder stays robust to latent error. Latent statistics are copied unchanged (the encoder did not change).

    SCMG_J=39 python scripts/echog_g1/vae_decoder_ft.py --vae <G1 vae final.pt> --out <dir> [--w-foot 1.0 --steps 3000]
"""
import argparse, copy, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg.common import EMBODIMENT
from scmg.data import View, WindowDataset, collate
from scmg.vae import MotionVAE
from scmg.g1_contact import foot_contact_loss

ap = argparse.ArgumentParser()
ap.add_argument('--vae', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--view', default='/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/echog-g1-v1')
ap.add_argument('--stats', default='/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/STATS_echog_g1_v1.json')
ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--batch', type=int, default=64); ap.add_argument('--frames', type=int, default=256)
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--w-foot', type=float, default=1.0); ap.add_argument('--ema', type=float, default=0.999)
ap.add_argument('--workers', type=int, default=6); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--log-every', type=int, default=50)
a = ap.parse_args()
if EMBODIMENT != 'g1': raise SystemExit('run with SCMG_J=39')
torch.manual_seed(a.seed); dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
vck = torch.load(a.vae, map_location='cpu')
vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema'])
if getattr(vae, 'identity', False): raise SystemExit('identity codec has no decoder to fine-tune')
for p in vae.enc.parameters(): p.requires_grad_(False)
dec = [p for p in vae.dec.parameters()]
ema = copy.deepcopy(vae).eval().requires_grad_(False)
stats = json.loads(Path(a.stats).read_text())
qm = torch.tensor(stats['q_mean'], dtype=torch.float32, device=dev); qs = torch.tensor(stats['q_std'], dtype=torch.float32, device=dev)
view = View(a.view, 'train', None, stats, mmap=True)
ds = WindowDataset(view, a.frames, seed=a.seed * 1000, semsynth_share=0.)
dl = iter(torch.utils.data.DataLoader(ds, a.batch, num_workers=a.workers, collate_fn=collate, pin_memory=True))
opt = torch.optim.AdamW(dec, lr=a.lr, weight_decay=0.0)
(out / 'ARGS.json').write_text(json.dumps(vars(a), indent=1))
t0 = time.time(); agg = {}; vae.train(); vae.enc.eval()
for step in range(1, a.steps + 1):
    b = next(dl); x = b['q'].to(dev, non_blocking=True); m = b['mask'].to(dev, non_blocking=True)
    for g in opt.param_groups: g['lr'] = a.lr * min(1., step / 200)
    with torch.autocast('cuda', torch.bfloat16):
        y, l = vae(x, m, deriv=1.0)
    fc = foot_contact_loss(y.float() * qs + qm, x.float() * qs + qm, m.float(), 50.)
    loss = l['loss'] + a.w_foot * (fc['foot_slide'] + 10. * fc['foot_height'])
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(dec, 1.0); opt.step()
    with torch.no_grad():
        for pe, pm in zip(ema.dec.parameters(), vae.dec.parameters()): pe.lerp_(pm.detach(), 1 - a.ema)
    for k, v in {**l, **fc, 'loss_total': loss}.items(): agg[k] = agg.get(k, 0.) + float(v)
    if step % a.log_every == 0:
        rec = {k: round(v / a.log_every, 5) for k, v in agg.items()}; rec.update(step=step, min=round((time.time() - t0) / 60, 1))
        print(json.dumps(rec), flush=True); (out / 'train.jsonl').open('a').write(json.dumps(rec) + '\n'); agg = {}
# the encoder is untouched: its weights in the EMA copy are the original ones; latent statistics are copied
assert all(torch.equal(pe, pm) for pe, pm in zip(ema.enc.parameters(), vae.enc.parameters()))
d = dict(cfg=vck['cfg'], model=vae.state_dict(), ema=ema.state_dict(), step=a.steps, vae_cfg=vck.get('vae_cfg', vck['cfg']),
         lat_mean=vck['lat_mean'], lat_std=vck['lat_std'], decoder_ft=dict(source=a.vae, w_foot=a.w_foot, steps=a.steps, lr=a.lr))
tmp = out / 'final.pt.tmp'; torch.save(d, tmp); tmp.replace(out / 'final.pt'); print('wrote', out / 'final.pt')

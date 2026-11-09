"""EX002 decoder-only fine-tune r3d (11-08; criteria docs/criteria/EX002_CTL.md "Decoder-only fine-tune r3d", 600946f).
Encoder and latent statistics frozen (asserted at save) -> the latent space and every EX002 generator stay valid. Loss = r3 terms
(rec + w_vel vel + w_acc acc, normalised units) + relative physical-unit velocity / acceleration L1 on the moving upper body (joints 2-33).
Arm A: z = mu + sigma eps (frozen posterior); arm B: on --noise-p of the batches z = mu + --noise eps.
    SCMG_J=34 python scripts/ex002/vae_decoder_ft_ex002.py --vae vae-r3/final.pt --out DIR [--arm A|B --steps 2000 --init DIR/final.pt]"""
import argparse, copy, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg.common import EMBODIMENT
from scmg.data import View, WindowDataset, collate
from scmg.vae import MotionVAE, recon_losses

C = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'
ap = argparse.ArgumentParser()
ap.add_argument('--vae', required=True, help='r3 (source of the frozen encoder / latent stats)'); ap.add_argument('--out', required=True)
ap.add_argument('--init', default=None, help='continue the decoder from this r3d checkpoint (quick lane -> 10k)')
ap.add_argument('--view', default=C + '/train-v1-20261101/views/data-v36_1'); ap.add_argument('--stats', default=C + '/scmg/STATS_v36_1.json')
ap.add_argument('--arm', default='A', choices=['A', 'B']); ap.add_argument('--noise', type=float, default=0.3); ap.add_argument('--noise-p', type=float, default=0.5)
ap.add_argument('--steps', type=int, default=2000); ap.add_argument('--batch', type=int, default=64); ap.add_argument('--frames', type=int, default=256)
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--w-pv', type=float, default=1.0); ap.add_argument('--w-pa', type=float, default=1.0)
ap.add_argument('--ema', type=float, default=0.999); ap.add_argument('--workers', type=int, default=6); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--log-every', type=int, default=50)
a = ap.parse_args()
if EMBODIMENT != 'ex002': raise SystemExit('run with SCMG_J=34')
torch.manual_seed(a.seed); dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
vck = torch.load(a.vae, map_location='cpu'); vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema'])
enc0 = [p.detach().clone() for p in vae.enc.parameters()]
step0 = 0
if a.init:
    ick = torch.load(a.init, map_location='cpu'); vae.load_state_dict(ick['ema']); step0 = int(ick['step'])
    assert all(torch.equal(p.cpu(), q.cpu()) for p, q in zip(vae.enc.parameters(), enc0)), 'init checkpoint changed the encoder'
for p in vae.enc.parameters(): p.requires_grad_(False)
dec = list(vae.dec.parameters()); ema = copy.deepcopy(vae).eval().requires_grad_(False)
st = json.loads(Path(a.stats).read_text())
qm = torch.tensor(st['q_mean'], dtype=torch.float32, device=dev); qs = torch.tensor(st['q_std'], dtype=torch.float32, device=dev)
UB = list(range(2, 34))
view = View(a.view, 'train', None, st, mmap=True)
dl = iter(torch.utils.data.DataLoader(WindowDataset(view, a.frames, seed=a.seed * 1000 + step0, semsynth_share=0.), a.batch, num_workers=a.workers,
                                      collate_fn=collate, pin_memory=True))
opt = torch.optim.AdamW(dec, lr=a.lr, weight_decay=0.0); (out / 'ARGS.json').write_text(json.dumps(vars(a), indent=1))
g = torch.Generator(device='cpu').manual_seed(a.seed)


def rel(d_hat, d, m):
    m = m[..., None]; return ((d_hat - d).abs() * m).sum() / ((d.abs() * m).sum()).clamp(min=1e-6)


t0 = time.time(); agg = {}; vae.train(); vae.enc.eval()
for step in range(1, a.steps + 1):
    b = next(dl); x = b['q'].to(dev, non_blocking=True); m = b['mask'].to(dev, non_blocking=True).float()
    for pg in opt.param_groups: pg['lr'] = a.lr * min(1., step / 200)
    with torch.no_grad():
        mu, lv = vae.encode(x)
        noisy = a.arm == 'B' and float(torch.rand(1, generator=g)) < a.noise_p
        z = mu + torch.randn_like(mu) * (a.noise if noisy else (0.5 * lv).exp())
    with torch.autocast('cuda', torch.bfloat16):
        y = vae.decode(z)
    y = y.float(); l = recon_losses(y, x, m)
    base = l['rec'] + vae.w_vel * l['vel'] + vae.w_acc * l['acc'] + vae.w_accmag * l['accmag']
    yp, xp = (y * qs + qm)[..., UB], (x * qs + qm)[..., UB]
    mv = m[:, 1:] * m[:, :-1]; ma = mv[:, 1:] * mv[:, :-1]
    dv = lambda t: t[:, 1:] - t[:, :-1]
    pv = rel(dv(yp), dv(xp), mv); pa = rel(dv(dv(yp)), dv(dv(xp)), ma)
    loss = base + a.w_pv * pv + a.w_pa * pa
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(dec, 1.0); opt.step()
    with torch.no_grad():
        for pe, pm in zip(ema.dec.parameters(), vae.dec.parameters()): pe.lerp_(pm.detach(), 1 - a.ema)
    for k, v in dict(base=base, rec=l['rec'], pv=pv, pa=pa, loss=loss, noisy=float(noisy)).items(): agg[k] = agg.get(k, 0.) + float(v)
    if step % a.log_every == 0:
        rec = {k: round(v / a.log_every, 5) for k, v in agg.items()}; rec.update(step=step0 + step, min=round((time.time() - t0) / 60, 1))
        print(json.dumps(rec), flush=True); (out / 'train.jsonl').open('a').write(json.dumps(rec) + '\n'); agg = {}
assert all(torch.equal(p.cpu(), q.cpu()) for p, q in zip(ema.enc.parameters(), enc0)), 'encoder changed'
d = dict(cfg=vck['cfg'], model=vae.state_dict(), ema=ema.state_dict(), step=step0 + a.steps, lat_mean=vck['lat_mean'], lat_std=vck['lat_std'],
         decoder_ft=dict(source=a.vae, arm=a.arm, noise=a.noise, noise_p=a.noise_p, w_pv=a.w_pv, w_pa=a.w_pa, steps=step0 + a.steps, lr=a.lr))
tmp = out / 'final.pt.tmp'; torch.save(d, tmp); tmp.replace(out / 'final.pt'); (out / 'DONE.json').write_text(json.dumps(dict(step=step0 + a.steps, min=(time.time() - t0) / 60)))
print('R3D_DONE', out / 'final.pt', flush=True); import os; os._exit(0)   # skip DataLoader worker teardown noise

"""Train/inference consistency of VAE decoding / encoding.   python scripts/vae_consistency.py <vae.pt>
Decoder (frames of chunk k, ground-truth latents):
  full      : whole-window decode (reference = what the streamer emits with enough context + committed look-ahead)
  loss_r4   : the r1-r5 continuity-loss window  = [ctx 4 latents | target 4 latents], replicate padding both sides
  loss_fix  : [3 extra history chunks | ctx | target | next latent], frames of the target chunk only
  stream    : the streamer (4-chunk tail, emission waits for the next latent)
Error vs full on the target chunk frames (deg, arms) and chunk-boundary jerk ratio.
Encoder: latents of a 64-frame segment (state-adapter targets) vs whole-window encoding."""
import json, sys
import numpy as np, torch
from scmg.common import CHUNK, CHUNK_FRAMES, STRIDE
from scmg.data import View, WindowDataset, collate, fixed_windows
from scmg.vae import MotionVAE
from scmg.evaluate import boundary_ratio

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st = json.load(open(D + '/STATS_v36_1.json'))
ck = torch.load(sys.argv[1], map_location='cpu')
vae = MotionVAE(**ck['cfg']).cuda().eval(); vae.load_state_dict(ck['ema'])
v = View(split='val', stats=st, mmap=True)
b = collate([WindowDataset(v, 256).sample(ci, 0, np.random.default_rng(0)) for ci, _ in fixed_windows(v)[:64]])
q = b['q'].cuda(); qs = torch.tensor(st['q_std']).cuda(); A = slice(4, 18)
res = {}
with torch.no_grad():
    mu, _ = vae.encode(q); Nl = mu.shape[1]; K = Nl // CHUNK
    full = vae.decode(mu)
    def assemble(fn):
        out = []
        for k in range(1, K - 1): out.append(fn(k))
        return torch.cat(out, 1)                                     # frames of chunks 1..K-2
    ref = full[:, CHUNK_FRAMES:(K - 1) * CHUNK_FRAMES]
    def r4(k):
        y = vae.decode(mu[:, (k - 1) * CHUNK:(k + 1) * CHUNK]); return y[:, CHUNK_FRAMES:]
    def fix(k):
        a = max(0, (k - 4) * CHUNK); y = vae.decode(mu[:, a:(k + 1) * CHUNK + 1])
        o = (k * CHUNK - a) * STRIDE; return y[:, o:o + CHUNK_FRAMES]
    for name, fn in (('loss_r4', r4), ('loss_fix', fix)):
        y = assemble(fn)
        res[name + '_err_deg'] = float(((y - ref) * qs).abs()[..., A].mean() * 57.3)
        res[name + '_err_first4_deg'] = float(((y - ref) * qs).abs().view(y.shape[0], -1, CHUNK_FRAMES, y.shape[-1])[:, :, :4, A].mean() * 57.3)
        res[name + '_err_last4_deg'] = float(((y - ref) * qs).abs().view(y.shape[0], -1, CHUNK_FRAMES, y.shape[-1])[:, :, -4:, A].mean() * 57.3)
        res[name + '_boundary'] = float(np.mean([boundary_ratio((y[i] * qs).cpu().numpy()) for i in range(y.shape[0])]))
    res['full_boundary'] = float(np.mean([boundary_ratio((ref[i] * qs).cpu().numpy()) for i in range(ref.shape[0])]))
    # encoder: 64-frame segments ending at chunk boundaries vs whole-window latents (last CHUNK latents)
    errs = []
    for k in range(4, K):
        seg = q[:, (k - 3) * CHUNK_FRAMES:(k + 1) * CHUNK_FRAMES]
        ms, _ = vae.encode(seg); errs.append((ms[:, -CHUNK:] - mu[:, k * CHUNK:(k + 1) * CHUNK]).abs().mean() / mu.std())
    res['encoder_segment_vs_full_rel'] = float(torch.stack(errs).mean())
print(json.dumps({k: round(v_, 4) for k, v_ in res.items()}, indent=1))

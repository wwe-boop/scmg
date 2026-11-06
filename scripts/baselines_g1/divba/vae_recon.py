"""Is the smoothness already in the motion VAE? (11-06 Div/BA diagnosis, dev only, CPU ok)

D-spk dev GT (the 50 Hz view rows of split val) -> one of
  gt   : unchanged (pipeline floor: 50 Hz view -> 30 fps resampling only)
  vae  : VAE encode (posterior mean) -> decode with the given checkpoint (e.g. the decft-w3d0m05 decoder used in every run)
-> official-format .pt at 30 fps (same writer as scripts/echog_g1/vae_roundtrip.py), then diag_beats.py compares the
upper-body speed spectrum with the raw 30 fps GT. Latents are 12.5 Hz (stride 4 at 50 Hz): everything above 6.25 Hz must
come from the decoder.
    SCMG_J=39 PYTHONPATH=<scmg code> python vae_recon.py --vae V --stats S --view <view> --text-cache T --out D --mode gt|vae
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from scmg.common import CHUNK_FRAMES, EMBODIMENT
from scmg.data import View, WindowDataset
from scmg.vae import MotionVAE
from scmg import g1

ap = argparse.ArgumentParser()
ap.add_argument('--vae', required=True); ap.add_argument('--stats', required=True); ap.add_argument('--view', required=True)
ap.add_argument('--text-cache', required=True); ap.add_argument('--out', required=True); ap.add_argument('--mode', choices=('gt', 'vae'), required=True)
ap.add_argument('--split', default='val'); ap.add_argument('--shard', default='0/1'); ap.add_argument('--device', default='cpu')
a = ap.parse_args()
if EMBODIMENT != 'g1': raise SystemExit('run with SCMG_J=39')
torch.set_num_threads(2)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
stats = json.loads(Path(a.stats).read_text()); qm = np.asarray(stats['q_mean'], np.float64); qs = np.asarray(stats['q_std'], np.float64)
view = View(a.view, a.split, a.text_cache, stats)
vck = torch.load(a.vae, map_location='cpu', weights_only=False); vae = MotionVAE(**vck['cfg']).to(a.device); vae.load_state_dict(vck['ema']); vae.eval()
si, sn = (int(x) for x in a.shard.split('/'))
for ci in range(si, len(view.clips), sn):
    c = view.clips[ci]; it = c['items'][0]; stem = it['id']; o = out / f'{stem}.pt'
    if o.exists(): continue
    L = c['frames']; n30 = int(it['echog']['frames30']); T = -(-L // CHUNK_FRAMES) * CHUNK_FRAMES
    s = WindowDataset(view, T, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
    qn = s['q'][:T].astype(np.float32)
    if a.mode == 'vae':
        with torch.no_grad():
            mu, _ = vae.encode(torch.from_numpy(qn)[None].to(a.device)); qn = vae.decode(mu)[0].float().cpu().numpy()
    q50 = qn[:L].astype(np.float64) * qs + qm
    r30 = g1.to30(q50, n30).astype(np.float32)
    torch.save(dict(robot_repr=torch.from_numpy(r30), real_num_frames=n30, fps=30.0, representation_units='physical',
                    representation_schema='projecthermes-g1-39d-standard-v2', source=stem,
                    generator=dict(diagnostic=f'vae_recon mode={a.mode}', vae=str(a.vae), frames50=L)), o)
print('done', a.mode, len(list(out.glob('*.pt'))))

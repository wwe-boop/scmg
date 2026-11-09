"""11-04 diagnosis: how much of the public-metric gap (Div / jerk / foot sliding) is already in the motion VAE?
GT val_common motion (the 50 Hz view rows) -> [mode gt: unchanged | mode vae: VAE encode (mean) -> decode] -> 30 fps
-> official-format .pt, then score with bench.sh eval.  mode gt is the pipeline floor (30->50->30 resampling only).
    SCMG_J=39 python scripts/echog_g1/vae_roundtrip.py --vae <G1 vae> --stats <STATS> --view <view>/test --text-cache <e5 test>
        --out <dir> --mode gt|vae [--split test]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg.common import J, CHUNK_FRAMES, EMBODIMENT
from scmg.data import View, WindowDataset
from scmg.vae import MotionVAE
from scmg import g1

ap = argparse.ArgumentParser()
ap.add_argument('--vae', required=True); ap.add_argument('--stats', required=True); ap.add_argument('--view', required=True)
ap.add_argument('--text-cache', required=True); ap.add_argument('--out', required=True); ap.add_argument('--mode', choices=('gt', 'vae'), required=True)
ap.add_argument('--split', default='test'); ap.add_argument('--shard', default='0/1')
a = ap.parse_args()
if EMBODIMENT != 'g1': raise SystemExit('run with SCMG_J=39')
dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
stats = json.loads(Path(a.stats).read_text()); qm = np.asarray(stats['q_mean'], np.float64); qs = np.asarray(stats['q_std'], np.float64)
view = View(a.view, a.split, a.text_cache, stats)
vck = torch.load(a.vae, map_location='cpu'); vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema']); vae.eval()
si, sn = (int(x) for x in a.shard.split('/'))
for ci in range(si, len(view.clips), sn):
    c = view.clips[ci]; it = c['items'][0]; stem = it['id']; o = out / f'{stem}.pt'
    if o.exists(): continue
    L = c['frames']; n30 = int(it['echog']['frames30']); T = -(-L // CHUNK_FRAMES) * CHUNK_FRAMES
    s = WindowDataset(view, T, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
    qn = s['q'][:T].astype(np.float32)                                         # normalized, zero-padded past L
    if a.mode == 'vae':
        with torch.no_grad():
            mu, _ = vae.encode(torch.from_numpy(qn)[None].to(dev)); qn = vae.decode(mu)[0].float().cpu().numpy()
    q50 = (qn[:L].astype(np.float64) * qs + qm)
    r30 = g1.to30(q50, n30).astype(np.float32)
    torch.save(dict(robot_repr=torch.from_numpy(r30), real_num_frames=n30, fps=30.0, representation_units='physical',
                    representation_schema='projecthermes-g1-39d-standard-v2', source=stem,
                    generator=dict(diagnostic=f'vae_roundtrip mode={a.mode}', vae=str(a.vae), frames50=L)), o)
print('done', a.mode, len(list(out.glob('*.pt'))))

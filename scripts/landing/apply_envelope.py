"""Apply the inference arm envelope (scmg.landing.ArmEnvelope) to exported generations (Claude 11-05, landing L1).

    python scripts/landing/apply_envelope.py <src dir> <dst dir> [--knee 99] [--bound 99.9] [--extra 0.05] [--joints upper+wrist]

Every gen*.npy of <src> ([T, 34] rad) is written to <dst> after the envelope; gt*.npy and every other *.json /
*.jsonl file are copied unchanged (so FGD / semantic / arm scorers read <dst> exactly like <src>). The envelope is
stateless per frame, so this equals what the deployment stream would send with the envelope switched on.
<dst>/ENVELOPE.json records the configuration, the source dir and the per-file max |change|.
"""
import argparse, glob, json, os, shutil
import numpy as np
from scmg.landing import ArmEnvelope

ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('dst')
ap.add_argument('--envelope', default=os.environ.get('HUMAN_ENVELOPE', '/home/xr-25b7/bulk/landing-1105/HUMAN_ENVELOPE.json'))
ap.add_argument('--knee', type=float, default=99.0); ap.add_argument('--bound', type=float, default=99.9)
ap.add_argument('--extra', type=float, default=0.05); ap.add_argument('--joints', default='upper+wrist')
a = ap.parse_args()
if os.path.abspath(a.src) == os.path.abspath(a.dst): raise SystemExit('dst must differ from src')
env = ArmEnvelope(a.envelope, a.knee, a.bound, a.extra, a.joints)
os.makedirs(a.dst, exist_ok=True)
gens = sorted(glob.glob(a.src + '/gen*.npy'))
if not gens: raise SystemExit(f'no gen*.npy in {a.src}')
changed = {}
for p in gens:
    x = np.load(p); y = env(x)
    if y.shape != x.shape or not np.isfinite(y).all(): raise SystemExit(f'bad output for {p}')
    np.save(os.path.join(a.dst, os.path.basename(p)), y.astype(x.dtype)); changed[os.path.basename(p)] = float(np.abs(y - x).max())
for p in glob.glob(a.src + '/gt*.npy') + glob.glob(a.src + '/*.json') + glob.glob(a.src + '/*.jsonl'):
    if os.path.basename(p) != 'ENVELOPE.json': shutil.copy2(p, a.dst)
json.dump(dict(src=a.src, **env.cfg, files=len(gens), max_abs_change=max(changed.values()),
               frac_files_changed=round(float(np.mean([v > 0 for v in changed.values()])), 4), per_file=changed),
          open(os.path.join(a.dst, 'ENVELOPE.json'), 'w'), indent=1)
print(a.dst, len(gens), 'files; max |change| %.4f rad' % max(changed.values()))

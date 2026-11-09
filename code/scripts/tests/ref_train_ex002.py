"""11-08 reference-training regression check (see ref_train_ex002.sh). summarise <out> [--check REF.json] [--write REF.json]
Summary of one reference run:
  curves : train.jsonl flow / loss / gn / self_forced at every 25 steps (log-every 25); dev.jsonl dev_flow at 150 / 300
  motion : 13-point liveliness stats (lively_floor_ex002.py: still, wrist speed, wrist amplitude, held share, stutter) of the offline
           bench clips (shard 0/4 of the dev view) + the GT of the same clips
Check (exit 1 on failure): every quantity within its tolerance of the stored reference. Tolerances are stored in the reference
(calibrated from repeated runs of the same code; see TOL_* below for the defaults used when a reference is written):
  curve band   : mean over steps 100..300 of flow / gn, relative
  dev_flow     : final dev_flow, relative
  motion       : still (absolute, share of frames), wrist speed / amplitude (relative)
A change of training code that is MEANT to change the training (a fix, a new default) must re-write the reference in the same commit,
with the old -> new numbers in the commit message; a failure without such a commit is a drift and blocks lane submission."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ex002'))
TOL_CURVE = 0.005         # relative, mean flow / gn over steps 100..300 (11-08 calibration: run-to-run 4e-5 / 1e-4; pre-fix 5d8f154 -8 % / -17 %)
TOL_DEV = 0.002           # relative, final dev_flow (run-to-run 5e-6)
TOL_STILL = 0.003         # absolute share of frames (run-to-run 1.2e-4)
TOL_WRIST = 0.02          # relative, wrist speed / amplitude (run-to-run 3e-4)


def summarise(o):
    o = Path(o); tr = [json.loads(l) for l in (o / 'train' / 'train.jsonl').read_text().splitlines() if l.strip()]
    dv = [json.loads(l) for l in (o / 'train' / 'dev.jsonl').read_text().splitlines() if l.strip()]
    band = [r for r in tr if 100 <= r['step'] <= 300]
    s = dict(curve={k: [round(float(r[k]), 5) for r in tr] for k in ('flow', 'loss', 'gn', 'self_forced')}, steps=[r['step'] for r in tr],
             band=dict(flow=float(np.mean([r['flow'] for r in band])), gn=float(np.mean([r['gn'] for r in band]))),
             dev_flow=float(dv[-1]['dev_flow']), skipped=int(tr[-1].get('skipped', 0)))
    import lively_floor_ex002 as LF
    res = LF.guard([('GT', str(o / 'offline')), ('gen', str(o / 'offline' / 's0'))])
    s['motion'] = {k: res['gen'][k] for k in ('clips', 'still', 'wrist_speed', 'wrist_amp', 'held_share', 'stutter')}
    s['motion_gt'] = {k: res['GT'][k] for k in ('clips', 'still', 'wrist_speed', 'wrist_amp', 'held_share', 'stutter')}
    s['lively_gates'] = res['gen']['gates']
    args = json.loads((o / 'train' / 'ARGS.json').read_text())
    s['recipe'] = {k: args.get(k) for k in ('init', 'cfg', 'sc', 'self_force_p', 'bagger_share', 'semsynth_share', 'steps', 'seed', 'batch', 'lr')}
    return s


def check(s, ref):
    t = ref.get('tol', {}); bad = []
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-9)
    for k in ('flow', 'gn'):
        if rel(s['band'][k], ref['band'][k]) > t.get('curve', TOL_CURVE): bad.append(f"band {k} {s['band'][k]:.4f} vs ref {ref['band'][k]:.4f}")
    if rel(s['dev_flow'], ref['dev_flow']) > t.get('dev', TOL_DEV): bad.append(f"dev_flow {s['dev_flow']:.4f} vs ref {ref['dev_flow']:.4f}")
    if abs(s['motion']['still'] - ref['motion']['still']) > t.get('still', TOL_STILL):
        bad.append(f"still {s['motion']['still']:.4f} vs ref {ref['motion']['still']:.4f}")
    for k in ('wrist_speed', 'wrist_amp'):
        if rel(s['motion'][k], ref['motion'][k]) > t.get('wrist', TOL_WRIST): bad.append(f"{k} {s['motion'][k]:.4f} vs ref {ref['motion'][k]:.4f}")
    if s['recipe'] != ref['recipe']: bad.append('recipe differs from the reference recipe')
    return bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('summarise'); p.add_argument('out'); p.add_argument('--check', default=None); p.add_argument('--write', default=None)
    p.add_argument('--commit', default=None, help='code commit of the run (stored with --write)')
    a = ap.parse_args()
    s = summarise(a.out); (Path(a.out) / 'REF_SUMMARY.json').write_text(json.dumps(s, indent=1))
    print('REF_SUMMARY', json.dumps(dict(band=s['band'], dev_flow=s['dev_flow'], motion=s['motion'], gates=s['lively_gates'])), flush=True)
    if a.write:
        s['tol'] = dict(curve=TOL_CURVE, dev=TOL_DEV, still=TOL_STILL, wrist=TOL_WRIST); s['commit'] = a.commit
        Path(a.write).parent.mkdir(parents=True, exist_ok=True); Path(a.write).write_text(json.dumps(s, indent=1)); print('REF_WRITTEN', a.write)
    if a.check:
        bad = check(s, json.loads(Path(a.check).read_text()))
        print('REF_TRAIN_CHECK', 'OK' if not bad else 'FAIL', *bad, sep='\n  ' if bad else ' '); sys.exit(1 if bad else 0)

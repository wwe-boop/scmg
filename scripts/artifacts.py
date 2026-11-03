"""Validate artifacts and publish successful subprocess output atomically. Failed temporaries are retained."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

SCHEMAS = {'planner': ('label', 'plan_shrink', 'plan_r2', 'h_change_no_audio', 'h_change_no_text'),
           'rollout': ('label', 'free', 'oracle'), 'score': ('centre', 'n_neg_windows'),
           'eval': ('inf',), 'train': ('step',)}


def valid_json(path, kind):
    try:
        data = json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if not isinstance(data, dict) or not data or not all(k in data for k in SCHEMAS[kind]): return False
        if kind == 'score': return isinstance(data['centre'], dict) and all(k in data['centre'] for k in ('top1', 'top5', 'balanced_top1'))
        if kind == 'rollout': return all(isinstance(data[k], dict) and all(x in data[k] for x in ('speed', 'still', 'copy')) for k in ('free', 'oracle'))
        return True
    except (OSError, ValueError, TypeError): return False


def valid_semantic(manifest, folder, variants, n):
    import numpy as np
    entries = json.loads(Path(manifest).read_text())['entries'][:n]
    if not entries: return False
    for variant in variants.split(','):
        for e in entries:
            try:
                x = np.load(Path(folder) / variant / f"gen_{e['ci']}.npy", allow_pickle=False)
                if x.shape != (e['frames'], 34) or not np.isfinite(x).all(): return False
            except (OSError, ValueError): return False
    return True


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='action', required=True)
    v = sub.add_parser('valid'); v.add_argument('path'); v.add_argument('kind', choices=SCHEMAS)
    r = sub.add_parser('run'); r.add_argument('path'); r.add_argument('kind', choices=SCHEMAS)
    r.add_argument('mode', choices=('stdout', 'file', 'directory')); r.add_argument('command', nargs=argparse.REMAINDER)
    g = sub.add_parser('semantic'); g.add_argument('manifest'); g.add_argument('folder'); g.add_argument('variants'); g.add_argument('n', type=int)
    a = ap.parse_args()
    if a.action == 'valid': return 0 if valid_json(a.path, a.kind) else 1
    if a.action == 'semantic': return 0 if valid_semantic(a.manifest, a.folder, a.variants, a.n) else 1
    dest = Path(a.path); dest.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == 'directory':
        tmp = Path(tempfile.mkdtemp(prefix=dest.name + '.pending-', dir=dest.parent))
        result = subprocess.run(a.command + ['--out', str(tmp)])
        valid = valid_json(tmp / 'RESULTS.json', a.kind)
    else:
        fd, name = tempfile.mkstemp(prefix=dest.name + '.pending-', dir=dest.parent); os.close(fd); tmp = Path(name)
        if a.mode == 'stdout':
            with tmp.open('w') as f: result = subprocess.run(a.command, stdout=f)
        else: result = subprocess.run(a.command + ['--out', str(tmp)])
        valid = valid_json(tmp, a.kind)
    if result.returncode or not valid:
        print(f'artifact failed: status={result.returncode}, valid={valid}, retained={tmp}', file=sys.stderr)
        return result.returncode or 1
    if a.mode == 'directory' and dest.exists():
        old = Path(tempfile.mkdtemp(prefix=dest.name + '.previous-', dir=dest.parent))
        os.replace(dest, old / dest.name)
    os.replace(tmp, dest)
    return 0


if __name__ == '__main__': sys.exit(main())

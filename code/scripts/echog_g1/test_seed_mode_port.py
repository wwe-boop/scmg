"""Test of the --seed-mode port in infer_test.py (11-06; rule of exp/divba ebb03ae). Two dev clips, 2 flow steps, local mirror.
 A default == pre-port code (git HEAD export) bitwise      B --seed-mode run == default bitwise
 C per_clip differs from run                               D per_clip output of clip 2 is the same whether clip 1 is in the run
 E generator records seed_mode / clip_seed = seed + crc32(stem)
    SCMG_LOCAL_MIRROR=~/bulk/cpfs-mirror python scripts/echog_g1/test_seed_mode_port.py --ckpt C --vae V --view HV --stats S --text T --stems ST2
"""
import argparse, json, os, subprocess, sys, tempfile, zlib
from pathlib import Path
import torch

CODE = Path(__file__).resolve().parents[2]


def run(code, out, stems, extra=()):
    env = dict(os.environ, SCMG_J='39', SCMG_NO_GLITCH='1', PYTHONPATH=str(code))
    cmd = [sys.executable, str(Path(code) / 'scripts/echog_g1/infer_test.py'), '--ckpt', A.ckpt, '--vae', A.vae, '--stats', A.stats,
           '--view', A.view, '--text-cache', A.text, '--out', str(out), '--stems', str(stems), '--steps', '2', '--cfg-w', '2.5', '--split', 'val', *extra]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=code)
    if r.returncode: print(r.stderr[-2000:]); raise SystemExit('run failed')


def main():
    global A
    ap = argparse.ArgumentParser()
    for k in ('ckpt', 'vae', 'view', 'stats', 'text', 'stems'): ap.add_argument('--' + k, required=True)
    A = ap.parse_args(); tmp = Path(tempfile.mkdtemp(prefix='seedmode-', dir=os.environ.get('SCMG_TEST_TMP', '/tmp/claude-1000')))
    orig = tmp / 'orig'; orig.mkdir(); subprocess.run(f'git -C {CODE} archive HEAD scmg scripts | tar -x -C {orig}', shell=True, check=True)
    st = Path(A.stems).read_text().split()[:2]; s2 = tmp / 's2.txt'; s2.write_text('\n'.join(st) + '\n'); s1 = tmp / 's1.txt'; s1.write_text(st[1] + '\n')
    run(orig, tmp / 'old', s2); run(CODE, tmp / 'def', s2); run(CODE, tmp / 'run', s2, ('--seed-mode', 'run'))
    run(CODE, tmp / 'pc', s2, ('--seed-mode', 'per_clip')); run(CODE, tmp / 'pc1', s1, ('--seed-mode', 'per_clip'))
    L = lambda d, s: torch.load(tmp / d / f'{s}.pt', weights_only=False)
    ok = {}
    ok['A_default_eq_old'] = all(torch.equal(L('old', s)['robot_repr'], L('def', s)['robot_repr']) for s in st)
    ok['B_run_eq_default'] = all(torch.equal(L('run', s)['robot_repr'], L('def', s)['robot_repr']) for s in st)
    ok['C_per_clip_differs'] = not torch.equal(L('pc', st[0])['robot_repr'], L('def', st[0])['robot_repr'])
    ok['D_per_clip_order_free'] = bool(torch.equal(L('pc', st[1])['robot_repr'], L('pc1', st[1])['robot_repr']))
    g = L('pc', st[0])['generator']; ok['E_meta'] = g.get('seed_mode') == 'per_clip' and g.get('clip_seed') == (0 + zlib.crc32(st[0].encode())) % (2 ** 63)
    print(json.dumps(ok)); print('ALL_OK' if all(ok.values()) else 'SOME_FAIL')


if __name__ == '__main__':
    main()

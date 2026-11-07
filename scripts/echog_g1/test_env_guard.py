"""Test of the infer_test.py global-switch guard (11-06). One dev clip, 2 flow steps, local mirror data.
 A default environment: robot_repr bitwise equal to the pre-guard code (git HEAD export or --orig); generator.env all unset
 B SCMG_BF16=1 without --allow-env: refused (non-zero exit, no output)
 C SCMG_BF16=1 with --allow-env: runs, generator.env records SCMG_BF16 = '1'
Prints ALL_OK / SOME_FAIL.
    SCMG_LOCAL_MIRROR=~/bulk/cpfs-mirror python scripts/echog_g1/test_env_guard.py --ckpt C --vae V --view HV --stats S --text T --stems ST [--orig DIR]
"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
import torch

CODE = Path(__file__).resolve().parents[2]


def run(code, out, extra=(), env_add=None):
    env = {k: v for k, v in os.environ.items() if k not in ('SCMG_SEM_SCALE', 'SCMG_BF16', 'SCMG_SEED0', 'SCMG_STRIDE')}
    env.update(SCMG_J='39', SCMG_NO_GLITCH='1', PYTHONPATH=str(code), **(env_add or {}))
    cmd = [sys.executable, str(Path(code) / 'scripts/echog_g1/infer_test.py'), '--ckpt', A.ckpt, '--vae', A.vae, '--stats', A.stats,
           '--view', A.view, '--text-cache', A.text, '--out', str(out), '--stems', A.stems, '--n', '1', '--steps', '2', '--cfg-w', '2.5',
           '--split', 'val', *extra]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=code)


def main():
    global A
    ap = argparse.ArgumentParser()
    for k in ('ckpt', 'vae', 'view', 'stats', 'text', 'stems'): ap.add_argument('--' + k, required=True)
    ap.add_argument('--orig'); A = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix='envguard-', dir=os.environ.get('SCMG_TEST_TMP', '/tmp/claude-1000')))
    orig = A.orig
    if not orig:
        orig = tmp / 'orig'; orig.mkdir()
        subprocess.run(f'git -C {CODE} archive HEAD scmg scripts | tar -x -C {orig}', shell=True, check=True)
    ok = {}
    r0 = run(orig, tmp / 'old'); r1 = run(CODE, tmp / 'new')
    if r0.returncode or r1.returncode: print(r0.stderr[-2000:], r1.stderr[-2000:]); raise SystemExit('run failed')
    stem = Path(A.stems).read_text().split()[0]
    o, n = (torch.load(tmp / d / f'{stem}.pt', weights_only=False) for d in ('old', 'new'))
    ok['A_bitwise'] = bool(torch.equal(o['robot_repr'], n['robot_repr']))
    ok['A_env_recorded_unset'] = n['generator'].get('env') == {k: None for k in ('SCMG_SEM_SCALE', 'SCMG_BF16', 'SCMG_SEED0', 'SCMG_STRIDE')}
    rb = run(CODE, tmp / 'bf16', env_add={'SCMG_BF16': '1'})
    ok['B_refused'] = rb.returncode != 0 and not (tmp / 'bf16' / f'{stem}.pt').exists() and 'allow-env' in rb.stderr + rb.stdout
    rc = run(CODE, tmp / 'bf16ok', extra=('--allow-env',), env_add={'SCMG_BF16': '1'})
    ok['C_allowed_recorded'] = rc.returncode == 0 and torch.load(tmp / 'bf16ok' / f'{stem}.pt', weights_only=False)['generator']['env']['SCMG_BF16'] == '1'
    print(json.dumps(ok)); print('ALL_OK' if all(ok.values()) else 'SOME_FAIL')


if __name__ == '__main__':
    main()

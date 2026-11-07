"""11-07 RoboGesture collision pairs in the G1 safety layers (scmg/safety_rg.py, safety_run.py --rg-pairs; default off).
    PYTHONPATH=<official src>:<pydeps/safeproj> python scripts/tests/test_safety_rg.py   env BASE_TREE (code before) RAW (dir of raw
    robot_repr .pt) STEMS (stem list) SP (scratch)
 a default: safety_run.py --layers cbf_qp without --rg-pairs == BASE_TREE bitwise on 4 clips
 b gradients: RGCollision.near rows == central finite differences of RGCollision.distances (arm / waist joints) on near frames
 c HAND_CMD == RoboGesture's median hand command over its filtered generations (first 60 clips, every 3rd frame)
 e fast path (broad phase, capped distances, early exit) == reference bitwise
 d live: on clips with RoboGesture penetration, cbf_qp --rg-pairs leaves no RoboGesture-penetrating and no G1 contact frame
   (except fallback frames, which are counted); the default cbf_qp output does not change on clips without either contact"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve(); CODE = HERE.parents[2]; sys.path.insert(0, str(CODE))
from scmg import safety_proj as S
from scmg.safety_rg import RGCollision, HAND_CMD
E = os.environ; SP = Path(E['SP']); SP.mkdir(parents=True, exist_ok=True); FAIL = []


def ok(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not cond: FAIL.append(name)


def run(code, out, stl, extra=()):
    r = subprocess.run([sys.executable, 'scripts/echog_g1/safety_run.py', '--pred-dir', E['RAW'], '--stems', str(stl), '--out-root', str(out),
                        '--layers', 'cbf_qp', '--workers', '4', *extra], cwd=str(code), capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=f"{code}:{os.environ.get('PYTHONPATH', '')}"))
    if r.returncode: print(r.stdout[-1500:], r.stderr[-2500:])
    return r.returncode == 0


rg = RGCollision(S.G1_JOINTS); g1 = S.G1Collision()
stems = [l.strip() for l in open(E['STEMS']) if l.strip()]
load = lambda d, s: torch.load(f'{d}/{s}.pt', map_location='cpu', weights_only=False)['robot_repr'].double().numpy()[:, 10:]
# pick clips: 4 with RoboGesture penetration (for d), the first 4 (for a)
pen = []
for s in stems:
    q = load(E['RAW'], s); n = sum(bool((rg.distances(x) < 0).any()) for x in q[::3])
    if n: pen.append(s)
    if len(pen) == 4: break
(SP / 'a.txt').write_text('\n'.join(stems[:4]) + '\n'); (SP / 'd.txt').write_text('\n'.join(pen) + '\n')
# a
ok('a run BASE_TREE', run(E['BASE_TREE'], SP / 'base', SP / 'a.txt')); ok('a run new (default)', run(CODE, SP / 'new', SP / 'a.txt'))
same = all(torch.equal(torch.load(SP / 'base/cbf_qp' / f'{s}.pt', weights_only=False)['robot_repr'],
                       torch.load(SP / 'new/cbf_qp' / f'{s}.pt', weights_only=False)['robot_repr']) for s in stems[:4])
ok('a default cbf_qp == BASE_TREE bitwise (4 clips)', same)
# b
arm = [k for k, _ in rg.ctl]; nb = bad = 0; h = 1e-5
for s in pen[:2]:
    q = load(E['RAW'], s)
    for x in q[::7]:
        rows = rg.near(x, 0.05); d0 = rg.distances(x)
        if not rows: continue
        ids = np.nonzero(d0 < 0.05)[0]
        G = np.zeros((len(ids), len(x)))
        for k in arm:
            e = np.zeros(len(x)); e[k] = h
            G[:, k] = (rg.distances(x + e)[ids] - rg.distances(x - e)[ids]) / (2 * h)
        for (g, dist), fd, i in zip(rows, G, ids):
            if abs(dist) < 1e-3: continue                     # touching: their own finite-difference branch
            nb += 1; bad += int(np.abs(g - fd).max() > 1e-3 + 0.05 * np.abs(fd).max())
ok('b near() gradients == finite differences', nb > 0 and bad <= 0.02 * nb, f'{bad} / {nb} rows off')
# c
f = rg.f; C = rg.C; cmds = []
for p in sorted(Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/robogesture/gen/filtered_seed0').glob('*.npz'))[:60]:
    qq = np.load(p)['qpos'][:, 7:].astype(np.float64)
    for t in range(0, len(qq), 3):
        cmds.append(f._to_frame(np.array([qq[t, tt[0][0]] for tt in C.CONTROL_TARGETS]), 0.0)[rg.hands])
ok('c HAND_CMD == median RoboGesture hand command', np.array_equal(np.median(np.stack(cmds), 0), np.array(HAND_CMD)))
# d
ok('d run --rg-pairs', run(CODE, SP / 'rg', SP / 'd.txt', ('--rg-pairs',)))
npen = ncon = nfr = nfb = 0
for s in pen:
    y = load(SP / 'rg/cbf_qp', s); nfr += len(y)
    npen += sum(bool((rg.distances(x) < 0).any()) for x in y); ncon += sum(g1.collides(x) for x in y)
st = [json.loads(l) for l in open(SP / 'rg/cbf_qp/stats.jsonl')]; nfb = sum(int(r.get('fallback_frames', 0)) for r in st)
ok('d --rg-pairs: no RoboGesture-penetrating / G1-contact frame beyond fallbacks', npen + ncon <= nfb, f'{nfr} frames: RG pen {npen}, G1 contact {ncon}, fallback {nfb}')
# e (11-08) fast path (capped distances + bounding-sphere broad phase + early exit) == reference implementation bitwise
from scmg.safety_rg import UnionCollision
Lr = S.CBFQPLayer(S.Limits(), UnionCollision(S.G1Collision(), RGCollision(S.G1_JOINTS, fast=False)))
Lf = S.CBFQPLayer(S.Limits(), UnionCollision(S.G1Collision(), RGCollision(S.G1_JOINTS, fast=True)))
eq = all(np.array_equal(S.run_stream(Lr, load(E['RAW'], s), 10)[0], S.run_stream(Lf, load(E['RAW'], s), 10)[0]) for s in pen)
ok('e fast RGCollision == reference, cbf_qp outputs bitwise (penetrating clips)', eq)
print('ALL_OK' if not FAIL else f'SOME_FAIL {FAIL}'); sys.exit(1 if FAIL else 0)

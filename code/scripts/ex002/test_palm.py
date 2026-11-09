"""11-08 palm-sign test (present metric / IK fix): at the EX002 rest pose (template extended fingers) both palms face the torso midline
(left palm . -y > 0.9, right palm . +y > 0.9) with palm = PALM_SIGN[h] * cross(index_base - wrist, third_base - wrist), and curling the
fingers moves the index tip toward the palm side. A present IK solve with palm_fix gives a forward palm (angle to +x < 45 deg) on a
same-side target. Prints ALL_OK / SOME_FAIL."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from scmg import ex002_ctl as E
from ex002_present import geo_all, PALM_SIGN
from ex002_keyframe import IK
V = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1'
SH = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002/tokens/HAND_SHAPES.json'


def palm(g, h, f=0):
    w = g['wrist'][f, h]; n = np.cross(g['index_base'][f, h] - w, g['third_base'][f, h] - w); return PALM_SIGN[h] * n / np.linalg.norm(n)


def main():
    m = json.load(open(V + '/META.json'))['ex002']; rest = np.asarray(m['rest34'], np.float32); sh = json.load(open(SH))['shapes']
    qe, qc = rest.copy(), rest.copy()
    for h, (lo, hi) in ((0, (18, 26)), (1, (26, 34))): qe[lo:hi] = sh[f'{h}_3']; qc[lo:hi] = sh[f'{h}_0']
    g = geo_all(qe[None]); ok = {}
    ok['rest_midline'] = bool(palm(g, 0) @ np.array([0, -1, 0]) > 0.9 and palm(g, 1) @ np.array([0, 1, 0]) > 0.9)
    ge, gc = E.points(qe[None]), E.points(qc[None])
    ok['curl_toward_palm'] = all(float((gc['index_tip'][0, h] - ge['index_tip'][0, h]) @ palm(g, h)) > 0 for h in (0, 1))
    ik = IK()
    for h in (0, 1):
        q, info = ik.solve(qe, h, 'present', present_off=(0.30, 0.05, -0.12), pen_w=100.0, zmin=0.15, palm_fix=True)
        gg = geo_all(q[None]); a = float(np.degrees(np.arccos(np.clip(palm(gg, h) @ np.array([1., 0, 0]), -1, 1))))
        w, s = gg['wrist'][0, h], gg['shoulder'][0, h]; out = (w[1] - s[1]) * (1 if h == 0 else -1)
        print(f'hand {h}: palm angle to +x {a:.1f} deg, wrist outward offset {out:+.3f} m, pen {info["pen"]:.1e}')
        ok[f'ik_palm_forward_{h}'] = a < 45 and out > -0.03
    print(ok); print('ALL_OK' if all(ok.values()) else 'SOME_FAIL')


if __name__ == '__main__':
    main()

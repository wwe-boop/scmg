"""RoboGesture outputs -> ECHO-G official G1 39-d robot_repr (.pt per stem), plus an FK cross-check.

RoboGesture (official code + weights, reproduced in baseline-repro-1104/robogesture/gen/<variant>/<stem>.npz) writes
qpos[T,60] in its replay contract (docs/inference.md): 0:7 reserved root (unused, zeros), 7:60 = the 53 hinge joints of
its fixed-base MJCF robogesture/assets/data2/G1_brainco_hands_collision.xml in MJCF order (12 legs, 3 waist, left arm 7,
left BrainCo hand 12, right arm 7, right hand 12).  raw_* = official --raw (qpos[:,19:60] = raw_motion41, legs zero);
filtered_* = official default branch (smooth_model_blocks + CollisionFilter, qpos[:,7:60] = expand @ control, legs zero).

Mapping: each of the 29 ECHO-G G1_JOINT_NAMES is taken by name from the MJCF joint '<name>_joint' (hands dropped).
Root: RoboGesture is fixed-base / upper-body only (its legs are held by the Unitree standing controller), so the root is
the identity orientation (6-D [1,0,0,0,1,0]), zero yaw increment and zero local velocity; legs as output (zeros) or,
for a diagnostic only, the ECHO-G train-split mean leg pose (--legs trainmean, from STATS_echog_g1_v1.json q_mean).
Timing: RoboGesture is natively 30 fps (one 30-frame block per nominal second, last block padded); frames are trimmed to
(or edge-padded up to) the ECHO-G GT frame count real_num_frames; no resampling.

  rg_to_g1.py convert --src <gen dir> --out <dir> --stems <txt> --ref-dir <motion_39d_30fps> [--legs zero|trainmean]
  rg_to_g1.py fkcheck --src <gen dir> --conv <converted dir> --stems <txt> --out <json>
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

RG = Path('/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/robogesture/official/robogesture')
XML = RG / 'assets/data2/G1_brainco_hands_collision.xml'
OFF = Path('/home/xr-25b7/bulk/echog-official-ro/official/src/echo_g/evaluation')
STATS = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/STATS_echog_g1_v1.json')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


RR = _load('rr_official', OFF / 'robot_repr.py')            # G1_JOINT_NAMES (official canonical order)
G1_JOINT_NAMES = RR.G1_JOINT_NAMES


def mj_model():
    import mujoco
    root = ET.parse(XML).getroot(); root.find('compiler').set('meshdir', str(RG / 'assets/meshes'))
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    assert m.nq == 53 and m.njnt == 53
    return m


def joint_index():
    """qpos60 column for each official G1 joint (canonical order)."""
    import mujoco
    m = mj_model()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    cols = []
    for nm in G1_JOINT_NAMES:
        j = names.index(nm + '_joint')
        cols.append(7 + int(m.jnt_qposadr[j]))
    return np.array(cols), m, names


def convert(a):
    cols, m, _ = joint_index()
    lo, hi = m.jnt_range[cols - 7, 0], m.jnt_range[cols - 7, 1]
    leg_fill = None
    if a.legs == 'trainmean':
        leg_fill = np.asarray(json.load(open(STATS))['q_mean'], np.float32)[10:22]
    stems = [l.strip() for l in open(a.stems) if l.strip()]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = dict(n=0, frames=0, gen_frames=0, padded_frames=0, trimmed_frames=0, out_of_limit_frac=0.0)
    for s in stems:
        z = np.load(Path(a.src) / f'{s}.npz')
        q = np.asarray(z['qpos'], np.float64)
        assert q.shape[1] == 60 and float(z['fps']) == 30.0, (s, q.shape)
        assert np.all(q[:, :7] == 0)
        if 'raw_motion41' in z.files:                       # raw export: qpos[:,19:60] is raw_motion41 verbatim
            assert np.array_equal(q[:, 19:60], np.asarray(z['raw_motion41'], np.float64)) and np.all(q[:, 7:19] == 0)
        n = int(torch.load(Path(a.ref_dir) / f'{s}.pt', map_location='cpu', weights_only=False)['real_num_frames'])
        T = len(q)
        joints = q[:, cols]
        rep['out_of_limit_frac'] += float(((joints < lo - 1e-6) | (joints > hi + 1e-6)).any(1).sum())
        if T >= n:
            joints = joints[:n]; rep['trimmed_frames'] += T - n
        else:
            joints = np.concatenate([joints, np.repeat(joints[-1:], n - T, 0)]); rep['padded_frames'] += n - T
        if leg_fill is not None:
            assert np.all(joints[:, :12] == 0)
            joints[:, :12] = leg_fill
        rr = np.zeros((n, 39), np.float32)
        rr[:, 0] = 1.0; rr[:, 4] = 1.0                       # identity 6-D (first two rows of I)
        rr[:, 10:] = joints
        torch.save(dict(robot_repr=torch.from_numpy(rr), real_num_frames=n, fps=30.0, representation_units='physical',
                        representation_schema='projecthermes-g1-39d-standard-v2', source=s,
                        generator=f'RoboGesture official ({Path(a.src).name}) | rg_to_g1.py name-mapped 29 joints, '
                                  f'fixed root, legs={a.legs}'), out / f'{s}.pt')
        rep['n'] += 1; rep['frames'] += n; rep['gen_frames'] += T
    rep['out_of_limit_frac'] /= max(rep['gen_frames'], 1)
    rep.update(src=a.src, legs=a.legs, mapping={nm: int(c) for nm, c in zip(G1_JOINT_NAMES, cols)})
    json.dump(rep, open(out / 'CONVERT.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'mapping'}))


def fkcheck(a):
    """Official ECHO-G FK on the converted 39-d vs MuJoCo FK of RoboGesture's own MJCF on the source qpos."""
    import mujoco
    K = _load('g1k_official', OFF / 'g1_kinematics.py')
    cols, m, _ = joint_index()
    d = mujoco.MjData(m)
    bodies = ['left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_elbow_link', 'right_elbow_link',
              'left_wrist_yaw_link', 'right_wrist_yaw_link', 'torso_link']
    mj_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies]
    bfs = [K.G1_BFS_BODY_NAMES.index(b) for b in bodies]
    stems = [l.strip() for l in open(a.stems) if l.strip()][:a.limit]
    err = {b: [] for b in bodies}
    for s in stems:
        q = np.load(Path(a.src) / f'{s}.npz')['qpos']
        rr = torch.load(Path(a.conv) / f'{s}.pt', weights_only=False)['robot_repr'].double()
        n = min(len(q), len(rr))
        pos_mj = np.zeros((n, len(bodies), 3))
        for t in range(n):
            d.qpos[:] = q[t, 7:60]; mujoco.mj_kinematics(m, d)
            pos_mj[t] = d.xpos[mj_ids]
        rot = torch.eye(3, dtype=torch.float64).expand(1, n, 3, 3)
        P = K.g1_forward_kinematics(rr[None, :n, 10:], rot, torch.zeros(1, n, 3, dtype=torch.float64))[0].numpy()
        # MuJoCo world = RoboGesture pelvis frame (pelvis is the fixed world body); official pelvis at origin, identity.
        for i, b in enumerate(bodies):
            err[b].append(np.linalg.norm(P[:, bfs[i]] - pos_mj[:, i], axis=-1))
    res = {b: dict(mean_m=float(np.concatenate(v).mean()), max_m=float(np.concatenate(v).max())) for b, v in err.items()}
    res['clips'] = len(stems); res['note'] = 'official g1_forward_kinematics (identity root) vs mujoco mj_kinematics on RoboGesture MJCF'
    json.dump(res, open(a.out, 'w'), indent=1); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    c = sp.add_parser('convert'); c.add_argument('--src', required=True); c.add_argument('--out', required=True)
    c.add_argument('--stems', required=True); c.add_argument('--ref-dir', required=True)
    c.add_argument('--legs', default='zero', choices=['zero', 'trainmean'])
    f = sp.add_parser('fkcheck'); f.add_argument('--src', required=True); f.add_argument('--conv', required=True)
    f.add_argument('--stems', required=True); f.add_argument('--out', required=True); f.add_argument('--limit', type=int, default=30)
    a = ap.parse_args()
    convert(a) if a.cmd == 'convert' else fkcheck(a)

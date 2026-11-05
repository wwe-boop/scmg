"""Executable-gesture contract on G1 (11-05): how much of each method's RAW output can be sent to the robot as is?

Contract (hardware limits from the Unitree URDF g1_custom_collision_29dof.urdf, 29 revolute joints):
  C1 position  : every joint inside [lower, upper]
  C2 velocity  : |dq/dt| <= URDF velocity limit (central difference at the clip fps)
  C3 self-collision: no contact between robot links in MuJoCo (g1_mocap_29dof.xml with collisions switched ON for the
                 robot meshes; MuJoCo already filters parent-child pairs). Link pairs that touch in >= 50 % of the GT frames
                 are structural overlaps of the mesh hulls, not contacts -> excluded for every method (list printed).
Reported per method: share of frames violating each clause, share of frames violating any clause, share of clips with
zero violations (= executable without any safety intervention), and the mean |excess| (rad, rad/s) where violated.
Base pose fixed (identity orientation, origin): only joint-space executability is tested; root motion is ignored.

    python scripts/echog_g1/executability.py --ref-dir R --stems S [--limit N] --urdf U --mjcf M label=pred_dir ...
robot_repr .pt (physical, 30 fps): channels 10: = the 29 joints in echo_g G1_JOINT_NAMES order.
"""
import argparse, json
import xml.etree.ElementTree as ET
import numpy as np
import torch
import mujoco
from echo_g.evaluation.robot_repr import G1_JOINT_NAMES

ap = argparse.ArgumentParser()
ap.add_argument('--ref-dir', required=True); ap.add_argument('--stems', required=True); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--urdf', required=True); ap.add_argument('--mjcf', required=True); ap.add_argument('--fps', type=float, default=30.)
ap.add_argument('--collision-stride', type=int, default=3, help='check contacts every k-th frame (speed)')
ap.add_argument('--out', default=None); ap.add_argument('panels', nargs='+')
a = ap.parse_args()

lim = {j.get('name'): j.find('limit').attrib for j in ET.parse(a.urdf).getroot().findall('joint') if j.get('type') == 'revolute'}
names = [x if x.endswith('_joint') else x + '_joint' for x in G1_JOINT_NAMES]   # echo_g names lack the URDF/MJCF '_joint' suffix
missing = [n for n in names if n not in lim]
if missing: raise SystemExit(f'joints missing in URDF: {missing}')
lo = np.array([float(lim[n]['lower']) for n in names]); hi = np.array([float(lim[n]['upper']) for n in names])
vmax = np.array([float(lim[n]['velocity']) for n in names])

m = mujoco.MjModel.from_xml_path(a.mjcf)
floor = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
for g in range(m.ngeom):
    if g not in floor: m.geom_contype[g] = 1; m.geom_conaffinity[g] = 1
    else: m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0
d = mujoco.MjData(m)
qadr = []
for n in names:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    if jid < 0: raise SystemExit(f'joint {n} not in MJCF')
    qadr.append(m.jnt_qposadr[jid])
qadr = np.array(qadr)
free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]


def joints(path):
    x = torch.load(path, map_location='cpu', weights_only=False)
    rr = x['robot_repr'] if isinstance(x, dict) else x
    return np.asarray(rr, dtype=np.float64)[:, 10:]


def contacts(q):
    """set of (bodyA, bodyB) link pairs in contact for joint angles q [29]"""
    d.qpos[:] = 0
    for j in free: d.qpos[m.jnt_qposadr[j] + 3] = 1.0          # identity quaternion at the origin
    d.qpos[qadr] = q
    mujoco.mj_forward(m, d)
    out = set()
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist < 0:
            b1, b2 = sorted((m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])); out.add((b1, b2))
    return out


stems = [s.strip() for s in open(a.stems) if s.strip()][: a.limit or None]
cols = [('GT', a.ref_dir)] + [tuple(p.split('=', 1)) for p in a.panels]
data = {lab: [] for lab, _ in cols}
for s in stems:
    try: qs = [joints(f'{d_}/{s}.pt') for _, d_ in cols]
    except FileNotFoundError: continue
    n = min(q.shape[0] for q in qs)
    if n < 4: continue
    for (lab, _), q in zip(cols, qs): data[lab].append(q[:n])

# structural overlaps: link pairs in contact in >= 50 % of the sampled GT frames
cnt, tot = {}, 0
for q in data['GT']:
    for t in range(0, q.shape[0], a.collision_stride):
        tot += 1
        for p in contacts(q[t]): cnt[p] = cnt.get(p, 0) + 1
structural = {p for p, c in cnt.items() if c >= 0.5 * tot}
bn = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
print(f'{len(data["GT"])} clips; structural overlaps excluded: {sorted((bn(x), bn(y)) for x, y in structural)}')

res = {}
for lab, _ in cols:
    F = P = V = C = A = 0; clean = 0; pex = []; vex = []; pairs = {}
    for q in data[lab]:
        T = q.shape[0]
        pv = (q < lo) | (q > hi)
        v = np.gradient(q, 1.0 / a.fps, axis=0); vv = np.abs(v) > vmax
        cv = np.zeros(T, bool)
        for t in range(0, T, a.collision_stride):
            ps = contacts(q[t]) - structural
            if ps:
                cv[t: t + a.collision_stride] = True
                for p in ps: pairs[p] = pairs.get(p, 0) + 1
        anyv = pv.any(1) | vv.any(1) | cv
        F += T; P += pv.any(1).sum(); V += vv.any(1).sum(); C += cv.sum(); A += anyv.sum(); clean += not anyv.any()
        if pv.any(): pex.append(np.maximum(lo - q, q - hi)[pv].mean())
        if vv.any(): vex.append((np.abs(v) - vmax)[vv].mean())
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:3]
    res[lab] = dict(clips=len(data[lab]), frames=int(F), pos=P / F, vel=V / F, coll=C / F, any=A / F, clean_clips=clean / max(len(data[lab]), 1),
                    pos_excess=float(np.mean(pex)) if pex else 0., vel_excess=float(np.mean(vex)) if vex else 0.,
                    top_pairs=[(bn(x), bn(y), c) for (x, y), c in top])
    r = res[lab]
    print(f'{lab:18s} any {100*r["any"]:.2f}% | pos {100*r["pos"]:.2f}% vel {100*r["vel"]:.2f}% coll {100*r["coll"]:.2f}% | '
          f'clips fully executable {100*r["clean_clips"]:.1f}% | excess pos {r["pos_excess"]:.3f} rad vel {r["vel_excess"]:.2f} rad/s | {r["top_pairs"]}')
if a.out: json.dump(res, open(a.out, 'w'), indent=1)

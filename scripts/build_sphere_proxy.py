"""Differentiable self-collision proxy of EX002+RM003 for stage-1 physics losses (scmg/kinematics.py).
    python scripts/build_sphere_proxy.py <out.npz> [spheres_per_body=6]
From the SAME MuJoCo model the phase-2 simulator uses (EX002Sim: lift 0.32 held, contact excludes for structural
overlaps and hand-internal pairs) it exports
  * the kinematic tree (topological body order): parent, body pos/quat in parent, one joint per body at most
    (hinge/slide axis + anchor), which of the 34 generated joints drives it, constant q for the held joints;
  * collision spheres: every collision geom's surface points (mesh vertices / primitive samples) in body coordinates,
    k-means into `spheres_per_body` spheres per body, radius = max distance of the assigned points (conservative);
  * candidate sphere pairs = body pairs that MuJoCo may collide (not excluded, not parent-child, not the same body)
    and of which at least one body moves with an ARM/HAND joint.
Validation (printed, stored in the npz): torch-free numpy FK vs MuJoCo xpos over random poses; proxy penetration vs
MuJoCo contact penetration on random arm poses (frame-level recall / false alarm)."""
import json, sys
import numpy as np, mujoco
from scmg.physim import EX002Sim

out = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 6
sim = EX002Sim(); m, d = sim.m, sim.d
J34 = sim.joints; jid34 = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n): i for i, n in enumerate(J34)}
nb = m.nbody
parent = m.body_parentid.copy(); bpos = m.body_pos.copy(); bquat = m.body_quat.copy()
bj = np.full(nb, -1); jtype = np.zeros(nb, int); jaxis = np.zeros((nb, 3)); janchor = np.zeros((nb, 3)); jconst = np.zeros(nb)
for b in range(nb):
    n = m.body_jntnum[b]
    if n == 0: continue
    assert n == 1, f'body {b} has {n} joints'
    j = m.body_jntadr[b]; t = m.jnt_type[j]
    if t == mujoco.mjtJoint.mjJNT_FREE: raise ValueError('free joint in model')
    jtype[b] = 1 if t == mujoco.mjtJoint.mjJNT_SLIDE else 0
    jaxis[b] = m.jnt_axis[j]; janchor[b] = m.jnt_pos[j]
    if j in jid34: bj[b] = jid34[j]
    else:
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        jconst[b] = sim.held.get(name, 0.)            # lift / wheels held; mimic joints approximated by 0
mimic = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.body_jntadr[b]) for b in range(nb) if m.body_jntnum[b] and bj[b] < 0]


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def axis_rot(a, th):
    a = a / np.linalg.norm(a); K_ = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K_ + (1 - np.cos(th)) * K_ @ K_


def fk_np(q34):
    R = np.zeros((nb, 3, 3)); p = np.zeros((nb, 3)); R[0] = np.eye(3)
    for b in range(1, nb):
        Rb = R[parent[b]] @ quat_mat(bquat[b]); pb = p[parent[b]] + R[parent[b]] @ bpos[b]
        if m.body_jntnum[b]:
            qv = q34[bj[b]] if bj[b] >= 0 else jconst[b]
            if jtype[b] == 0:
                Rj = axis_rot(jaxis[b], qv); pb = pb + Rb @ (janchor[b] - Rj @ janchor[b]); Rb = Rb @ Rj
            else:
                pb = pb + Rb @ (jaxis[b] * qv)
        R[b], p[b] = Rb, pb
    return R, p


def set_q(q34):
    d.qpos[:] = 0
    for b in range(nb):
        if m.body_jntnum[b] and bj[b] < 0: d.qpos[m.jnt_qposadr[m.body_jntadr[b]]] = jconst[b]
    d.qpos[sim.qadr] = q34; mujoco.mj_forward(m, d)


rng = np.random.default_rng(0); lo, hi = sim.lo, sim.hi
err = []
for _ in range(50):
    q = rng.uniform(lo, hi); set_q(q); R, p = fk_np(q)
    err.append(max(np.abs(p - d.xpos).max(), np.abs(R - d.xmat.reshape(-1, 3, 3)).max()))
print('FK max error vs MuJoCo', max(err), flush=True)

# collision geoms -> body-frame surface points
pts = {}
for g in range(m.ngeom):
    if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0: continue
    b = m.geom_bodyid[g]; Rg = quat_mat(m.geom_quat[g]); pg = m.geom_pos[g]; t = m.geom_type[g]; s = m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        u = rng.normal(size=(200, 3)); v = s[0] * u / np.linalg.norm(u, axis=1, keepdims=True)
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        u = rng.normal(size=(400, 3)); u = s[0] * u / np.linalg.norm(u, axis=1, keepdims=True); u[:, 2] += rng.uniform(-s[1], s[1], 400); v = u
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        a = rng.uniform(0, 2 * np.pi, 400); v = np.c_[s[0] * np.cos(a), s[0] * np.sin(a), rng.uniform(-s[1], s[1], 400)]
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        v = rng.uniform(-1, 1, (400, 3)) * s[:3]; k_ = rng.integers(0, 3, 400); v[np.arange(400), k_] = np.sign(v[np.arange(400), k_]) * s[k_]
    else:
        continue
    pts.setdefault(b, []).append(np.asarray(v, float) @ Rg.T + pg)


def kmeans(x, k, it=30):
    c = x[rng.choice(len(x), min(k, len(x)), replace=False)]
    for _ in range(it):
        a = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), 1)
        c = np.array([x[a == i].mean(0) if (a == i).any() else c[i] for i in range(len(c))])
    a = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), 1)
    r = np.array([np.linalg.norm(x[a == i] - c[i], axis=1).max() if (a == i).any() else 0. for i in range(len(c))])
    return c, r


VOX = float(__import__('os').environ.get('PROXY_VOX', '0.04'))
sb, sc, sr = [], [], []
for b, P in sorted(pts.items()):                       # surface shell: one sphere per occupied VOX-sized voxel
    P = np.concatenate(P)
    key = np.floor(P / VOX).astype(int)
    _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.reshape(-1)
    for v_ in range(inv.max() + 1):
        Q = P[inv == v_]; c = Q.mean(0); r = np.linalg.norm(Q - c, axis=1).max()
        sb.append(b); sc.append(c); sr.append(max(r, 0.005))
sb, sc, sr = np.array(sb), np.array(sc), np.array(sr)

# movable-by-arm/hand bodies; excluded body pairs
names34 = J34
armhand = set(i for i, n in enumerate(names34) if 'arm_joint' in n or 'hand' in n or 'finger' in n or 'thumb' in n)
def drivers(b):
    s = set()
    while b > 0:
        if bj[b] >= 0: s.add(int(bj[b]))
        b = parent[b]
    return s
moves = {b: bool(drivers(b) & armhand) for b in range(nb)}
excl = set()
for e in range(m.nexclude):
    sig = int(m.exclude_signature[e]); excl.add((sig >> 16, sig & 0xFFFF)); excl.add((sig & 0xFFFF, sig >> 16))
pairs = []
for i in range(len(sb)):
    for j in range(i + 1, len(sb)):
        a, b = int(sb[i]), int(sb[j])
        if a == b or (a, b) in excl or parent[a] == b or parent[b] == a: continue
        if not (moves[a] or moves[b]): continue
        pairs.append((i, j))
pairs = np.array(pairs)


def tree_dist(a, b):
    anc = {}; x, k = a, 0
    while True:
        anc[x] = k
        if x == 0: break
        x = parent[x]; k += 1
    x, k = b, 0
    while x not in anc: x = parent[x]; k += 1
    return k + anc[x]


def proxy_dd(q):
    R, p = fk_np(q); w = np.einsum('sij,sj->si', R[sb], sc) + p[sb]
    return np.linalg.norm(w[pairs[:, 0]] - w[pairs[:, 1]], axis=1) - sr[pairs[:, 0]] - sr[pairs[:, 1]]


rest_q = np.clip(np.asarray(json.load(open('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/STATS_v36_1.json'))['q_mean']), lo + 1e-3, hi - 1e-3)
keep = (proxy_dd(rest_q) > 0.01) & np.array([tree_dist(int(sb[i]), int(sb[j])) > 3 for i, j in pairs])
pairs = pairs[keep]
# reachability: keep only sphere pairs that come within 3 cm of each other in some random arm/hand pose
near = np.zeros(len(pairs), bool)
for _ in range(1500):
    q = rest_q.copy()
    for i in armhand: q[i] = rng.uniform(lo[i], hi[i])
    near |= proxy_dd(q) < 0.03
keep = near
pairs = pairs[keep]
# calibration: body pairs that the proxy flags much more often than MuJoCo reports a contact for them are structural
# neighbours whose shells are inflated (e.g. shoulder link vs torso) -> dropped
import collections
pb = lambda i: (int(sb[pairs[i, 0]]), int(sb[pairs[i, 1]]))
body_of_pair = np.array([pb(i) for i in range(len(pairs))])
cnt_px, cnt_mj, NCAL = collections.Counter(), collections.Counter(), 800
for _ in range(NCAL):
    q = rest_q.copy()
    for i in armhand: q[i] = rng.uniform(lo[i], hi[i])
    set_q(q)
    for c_ in d.contact[:d.ncon]:
        if c_.dist < -1e-4:
            a_, b_ = int(m.geom_bodyid[c_.geom1]), int(m.geom_bodyid[c_.geom2]); cnt_mj[(a_, b_)] += 1; cnt_mj[(b_, a_)] += 1
    hit = proxy_dd(q) < 0
    for bp in set(map(tuple, body_of_pair[hit])): cnt_px[bp] += 1
drop = {bp for bp, c in cnt_px.items() if c / NCAL > 0.2 and cnt_mj[bp] / NCAL < 0.02}
print('calibration dropped body pairs:', sorted((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, a), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)) for a, b in drop), flush=True)
keep = np.array([tuple(bp) not in drop for bp in body_of_pair])
print('pairs before filter', len(pairs), 'kept', int(keep.sum()), flush=True)
pairs = pairs[keep]
print('bodies with geoms', len(pts), 'spheres', len(sb), 'pairs', len(pairs), 'mimic/held joints approximated', mimic, flush=True)

# validation vs MuJoCo contacts on random arm poses: proxy distances below 0 at the rest-pose structural pairs are
# expected (spheres are conservative); report frame-level agreement on random poses
import collections
FAP = collections.Counter(); rng = np.random.default_rng(1)   # fresh poses for validation
rec, fa, tot_c, tot_n = 0, 0, 0, 0
rest = rest_q
base_pen = None
for t_ in range(400):
    q = np.array(rest, float).copy()
    for i in armhand: q[i] = rng.uniform(lo[i], hi[i])
    set_q(q); mj = sum(-c_.dist for c_ in d.contact[:d.ncon] if c_.dist < 0) > 1e-4
    dd_ = proxy_dd(q); px = (dd_ < 0).any()
    if not mj:
        for i_ in np.where(dd_ < 0)[0]:
            FAP[(int(sb[pairs[i_, 0]]), int(sb[pairs[i_, 1]]))] += 1
    tot_c += mj; tot_n += (not mj); rec += (mj and px); fa += ((not mj) and px)
print(f'proxy vs MuJoCo on 400 random arm poses: contact frames {tot_c}, recall {rec / max(tot_c, 1):.2f}, '
      f'false alarm {fa / max(tot_n, 1):.2f}', flush=True)
bn = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
fb = collections.Counter()
for (a, b), c in FAP.items(): fb[tuple(sorted((bn(a), bn(b))))] += c
print('false-alarm body pairs (sphere-pair hits):', fb.most_common(15), flush=True)
np.savez(out, joints=np.array(J34), parent=parent, body_pos=bpos, body_quat=bquat, body_joint=bj, joint_type=jtype,
         joint_axis=jaxis, joint_anchor=janchor, joint_const=jconst, sphere_body=sb, sphere_center=sc, sphere_radius=sr,
         pairs=pairs, fk_err=max(err), recall=rec / max(tot_c, 1), false_alarm=fa / max(tot_n, 1))
print('saved', out)

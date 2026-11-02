"""Differentiable self-collision proxy of EX002+RM003 for the stage-1 physics loss (scmg/physloss.py).
    python scripts/build_collision_proxy.py <out.npz> [validation frames]
From the same compiled MuJoCo model the simulator uses (scmg.physim.EX002Sim: lift held at 0.32 m, contact excludes for
structural overlaps and hand-internal pairs):
  kinematic tree  every body: parent, local pos/quat, its joint (hinge/slide axis + anchor) -> index into the 34
                  generated joints, or a held constant (lift 0.32, wheels 0)
  spheres         collidable geoms' surface points (mesh vertices / primitive surface samples) in the body frame,
                  k-means per body, radius = q-quantile of point distances to the centre (q = RAD_Q)
  pairs           body pairs MuJoCo would test (contype/conaffinity, not parent-child, not excluded) with at least one
                  body in an arm/hand subtree -> sphere-pair list
Validation on ground-truth training frames: MuJoCo penetration (mj_forward contacts, dist < 0) vs proxy penetration
(max over sphere pairs of r_i + r_j - |c_i - c_j|): recall of real contacts and the proxy false-alarm rate."""
import json, sys
import numpy as np, mujoco
from scmg.physim import EX002Sim

RAD_Q = 0.9; SPH_PER_M = 1 / 0.05      # about one sphere per 5 cm of body extent
out = sys.argv[1]; NV = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
sim = EX002Sim(); m = sim.m; d = mujoco.MjData(m)
joints = sim.joints; jidx = {n: i for i, n in enumerate(joints)}
held = dict(sim.held)
rng = np.random.default_rng(0)

# ---- kinematic tree (bodies are ordered parent-before-child in MuJoCo)
nb = m.nbody
par = m.body_parentid.copy(); bpos = m.body_pos.copy(); bquat = m.body_quat.copy()
jtype = np.zeros(nb, np.int64); jaxis = np.zeros((nb, 3)); jpos = np.zeros((nb, 3)); jsrc = np.full(nb, -1); jconst = np.zeros(nb)
for b in range(nb):
    n_ = m.body_jntnum[b]
    if n_ == 0: continue
    assert n_ == 1, f'body {b} has {n_} joints'
    j = m.body_jntadr[b]; name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    t = m.jnt_type[j]
    if t in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL): raise ValueError('free/ball joint')
    jtype[b] = 1 if t == mujoco.mjtJoint.mjJNT_HINGE else 2
    jaxis[b] = m.jnt_axis[j]; jpos[b] = m.jnt_pos[j]
    if name in jidx: jsrc[b] = jidx[name]
    else: jconst[b] = held.get(name, 0.)
# mimic equalities (joint couplings) -> treat the follower as a function of its leader: q_f = c0 + c1 q_l
mimic = []
for e in range(m.neq):
    if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and m.eq_active0[e]:
        jf, jl = m.eq_obj1id[e], m.eq_obj2id[e]; c = m.eq_data[e][:2]
        nf = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jf); nl = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jl)
        bf = m.jnt_bodyid[jf]
        if nl in jidx and nf not in jidx: mimic.append((int(bf), jidx[nl], float(c[0]), float(c[1])))

# ---- collidable geoms -> body-frame points -> spheres
def quat2mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def geom_points(g, n=400):
    t = m.geom_type[g]; s = m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]; a = m.mesh_vertadr[mid]; k = m.mesh_vertnum[mid]
        p = m.mesh_vert[a:a + k].astype(np.float64)
        if len(p) > 3000: p = p[rng.choice(len(p), 3000, replace=False)]
    else:
        u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        if t == mujoco.mjtGeom.mjGEOM_SPHERE: p = u * s[0]
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            z = rng.uniform(-s[1], s[1], n); p = u * s[0]; p[:, 2] += z
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            ang = rng.uniform(0, 2 * np.pi, n); p = np.c_[s[0] * np.cos(ang), s[0] * np.sin(ang), rng.uniform(-s[1], s[1], n)]
        elif t == mujoco.mjtGeom.mjGEOM_BOX: p = rng.uniform(-1, 1, (n, 3)) * s[:3]
        else: return np.zeros((0, 3))
    return p @ quat2mat(m.geom_quat[g]).T + m.geom_pos[g]


def kmeans(p, k, it=30):
    c = p[rng.choice(len(p), k, replace=False)]
    for _ in range(it):
        a = np.argmin(((p[:, None] - c[None]) ** 2).sum(-1), 1)
        c = np.stack([p[a == i].mean(0) if (a == i).any() else c[i] for i in range(k)])
    return c, a


coll = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
body_pts = {}
for g in coll:
    body_pts.setdefault(int(m.geom_bodyid[g]), []).append(geom_points(g))
sph_body, sph_c, sph_r = [], [], []
for b, ps in sorted(body_pts.items()):
    p = np.concatenate(ps)
    if len(p) < 4: continue
    ext = np.linalg.norm(p.max(0) - p.min(0)); k = int(np.clip(np.ceil(ext * SPH_PER_M), 1, 12))
    c, a = kmeans(p, min(k, len(p)))
    for i in range(len(c)):
        q = p[a == i]
        if not len(q): continue
        sph_body.append(b); sph_c.append(c[i]); sph_r.append(np.quantile(np.linalg.norm(q - c[i], axis=1), RAD_Q))
sph_body = np.array(sph_body); sph_c = np.array(sph_c); sph_r = np.array(sph_r)

# ---- body pairs MuJoCo would test + arm/hand relevance
excl = set(int(x) for x in m.exclude_signature[:m.nexclude]) if m.nexclude else set()
arm = set()
for side in ('left', 'right'):
    arm |= {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in EX002Sim._subtree(m, f'{side}_arm_link1')}
bodies = sorted(set(sph_body.tolist()))
gmask = {b: [(m.geom_contype[g], m.geom_conaffinity[g]) for g in coll if m.geom_bodyid[g] == b] for b in bodies}
pairs_b = []
for i, b1 in enumerate(bodies):
    for b2 in bodies[i + 1:]:
        if not (b1 in arm or b2 in arm): continue
        if par[b1] == b2 or par[b2] == b1: continue
        if ((b1 + 1) << 16) + b2 + 1 in excl or ((b2 + 1) << 16) + b1 + 1 in excl: continue
        if not any((t1 & a2) or (t2 & a1) for t1, a1 in gmask[b1] for t2, a2 in gmask[b2]): continue
        pairs_b.append((b1, b2))
pi, pj = [], []
for b1, b2 in pairs_b:
    for i in np.where(sph_body == b1)[0]:
        for j in np.where(sph_body == b2)[0]: pi.append(i); pj.append(j)
pi, pj = np.array(pi), np.array(pj)
print(f'bodies {nb}, collidable bodies {len(bodies)}, spheres {len(sph_r)}, body pairs {len(pairs_b)}, sphere pairs {len(pi)}', flush=True)

# ---- numpy FK (same math as scmg/physloss.py) for validation
def fk(q):
    R = np.zeros((nb, 3, 3)); P = np.zeros((nb, 3)); R[0] = np.eye(3)
    for b in range(1, nb):
        Rl = quat2mat(bquat[b]); pl = bpos[b].copy()
        if jtype[b]:
            v = q[jsrc[b]] if jsrc[b] >= 0 else jconst[b]
            for (bf, src, c0, c1) in mimic:
                if bf == b: v = c0 + c1 * q[src]
            if jtype[b] == 1:
                ax = jaxis[b]; K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                Rj = np.eye(3) + np.sin(v) * K + (1 - np.cos(v)) * K @ K
                pl = pl + Rl @ (jpos[b] - Rj @ jpos[b]); Rl = Rl @ Rj
            else:
                pl = pl + Rl @ (jaxis[b] * v)
        R[b] = R[par[b]] @ Rl; P[b] = R[par[b]] @ pl + P[par[b]]
    return R, P


def proxy_pen(q):
    R, P = fk(q); c = np.einsum('sij,sj->si', R[sph_body], sph_c) + P[sph_body]
    dd = np.linalg.norm(c[pi] - c[pj], axis=1)
    return float(np.max(sph_r[pi] + sph_r[pj] - dd)), (R, P)


# ---- frames: ground truth (train) + generated trajectories that DO collide (for recall)
from pathlib import Path
from scmg.data import View, VIEW
st = json.load(open('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/STATS_v36_1.json'))
v = View(VIEW, 'train', None, st, mmap=True)
Qgt = np.asarray(v.q[np.sort(rng.choice(len(v.q), NV, replace=False))], np.float64)
gen = []
for dname in ('/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/robogesture/eval-dev/raw', '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/dots-eval/v2-long',
              '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/dots-eval/v1-long'):
    for f in sorted(Path(dname).glob('gen_Hinf_*.npy'))[:8]: gen.append(np.load(f)[::5])
Qgen = np.concatenate(gen).astype(np.float64)
qadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joints}
mim_q = [(m.jnt_qposadr[m.body_jntadr[bf]], src, c0, c1) for bf, src, c0, c1 in mimic]


def mj_state(q):
    d.qpos[:] = m.qpos0
    for n, val in held.items(): d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = val
    for n, i in jidx.items(): d.qpos[qadr[n]] = q[i]
    for a, src, c0, c1 in mim_q: d.qpos[a] = c0 + c1 * q[src]
    mujoco.mj_forward(m, d)
    return max([-c_.dist for c_ in d.contact[:d.ncon] if c_.dist < 0] + [0.])


def pair_pen(q):
    R, P = fk(q); c = np.einsum('sij,sj->si', R[sph_body], sph_c) + P[sph_body]
    return sph_r[pi] + sph_r[pj] - np.linalg.norm(c[pi] - c[pj], axis=1), P


# structural proxy overlaps: sphere pairs penetrating in > 2 % of ground-truth frames are dropped
Pg = []; fk_err = []
for q in Qgt:
    q = np.clip(q, sim.lo, sim.hi); mj_state(q); pp, P = pair_pen(q); Pg.append(pp > 0); fk_err.append(float(np.abs(P - d.xpos).max()))
keep = np.mean(Pg, 0) <= 0.02
pi, pj = pi[keep], pj[keep]
print('dropped structural sphere pairs', int((~keep).sum()), 'kept', int(keep.sum()), 'fk max err', max(fk_err), flush=True)


def proxy(q): return float(np.max(pair_pen(q)[0])) if len(pi) else -1.


rep = dict(fk_max_err_m=float(np.max(fk_err)), rad_q=RAD_Q, spheres=int(len(sph_r)), sphere_pairs=int(len(pi)))
for tag, Q in (('gt', Qgt), ('gen', Qgen)):
    mj, px = [], []
    for q in Q:
        q = np.clip(q, sim.lo, sim.hi); mj.append(mj_state(q)); px.append(proxy(q))
    mj, px = np.array(mj), np.array(px); real = mj > 1e-3
    rep[tag + '_sweep'] = {f'{dl * 1000:.0f}mm': dict(pos=float((px > dl).mean()), recall=float((px[real] > dl).mean()) if real.any() else None)
                           for dl in (0., .005, .01, .015, .02, .03)}
    np.save(out.replace('.npz', f'_{tag}_mj.npy'), mj); np.save(out.replace('.npz', f'_{tag}_px.npy'), px)
    rep[tag] = dict(frames=len(Q), mujoco_contact_frac=float(real.mean()), proxy_pos_frac=float((px > 0).mean()),
                    recall=float((px[real] > 0).mean()) if real.any() else None, false_alarm=float((px[~real] > 0).mean()),
                    corr=float(np.corrcoef(mj, np.maximum(px, 0))[0, 1]) if real.any() else None)
print(json.dumps(rep), flush=True)
np.savez(out, parent=par, pos=bpos, quat=bquat, jtype=jtype, jaxis=jaxis, jpos=jpos, jsrc=jsrc, jconst=jconst,
         mimic=np.array(mimic, np.float64).reshape(-1, 4), sph_body=sph_body, sph_c=sph_c, sph_r=sph_r, pi=pi, pj=pj,
         report=json.dumps(rep), joints=np.array(joints))

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

Revised checker (design v4 §4.2, 11-05; the defaults above keep the original behaviour bit for bit):
  --collision-stride 1           contacts on every frame (default 3 = every 3rd frame, smeared over the next 2)
  --exclusion-from D [--exclusion-stems L] --exclusion-file X.json
                                 build the structural-overlap table ONCE from training-set GT (all *.pt in D, or the
                                 stems of list L in D; or D = a list file of .pt paths): link pairs in contact in >= 50 %
                                 of ALL those frames (every --exclusion-stride-th frame, default 1). Written to X.json;
                                 without --ref-dir / panels the script only builds the table and exits.
  --exclusion-file X.json        (alone) read that table instead of deriving it from the GT of the evaluated set, so the
                                 exclusions no longer depend on which clips are evaluated.
Always reported now: every violating link pair with its number of checked frames (GT printed in full, json pairs_all),
and per-joint position / velocity violation frame counts (json joint_pos_frames / joint_vel_frames).
robot_repr .pt (physical, 30 fps): channels 10: = the 29 joints in echo_g G1_JOINT_NAMES order.
"""
import argparse, json
import xml.etree.ElementTree as ET
import numpy as np
import torch
import mujoco
from echo_g.evaluation.robot_repr import G1_JOINT_NAMES

ap = argparse.ArgumentParser()
ap.add_argument('--ref-dir', default=None); ap.add_argument('--stems', default=None); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--urdf', required=True); ap.add_argument('--mjcf', required=True); ap.add_argument('--fps', type=float, default=30.)
ap.add_argument('--collision-stride', type=int, default=3, help='check contacts every k-th frame (speed)')
ap.add_argument('--exclusion-from', default=None, help='training-set GT dir (or list file of .pt paths) -> build the exclusion table')
ap.add_argument('--exclusion-stems', default=None, help='with --exclusion-from DIR: only these stems (e.g. official train_drop.txt)')
ap.add_argument('--exclusion-stride', type=int, default=1, help='frame stride when building the table (1 = every frame)')
ap.add_argument('--exclusion-workers', type=int, default=1)
ap.add_argument('--exclusion-file', default=None, help='json table: written by --exclusion-from, read otherwise')
ap.add_argument('--out', default=None); ap.add_argument('panels', nargs='*')
a = ap.parse_args()
build_only = a.exclusion_from is not None and a.ref_dir is None and not a.panels
if not build_only and (a.ref_dir is None or a.stems is None or not a.panels):
    ap.error('--ref-dir, --stems and at least one label=pred_dir panel are required (unless only building --exclusion-from)')
if a.exclusion_from is not None and a.exclusion_file is None: ap.error('--exclusion-from needs --exclusion-file (output json)')

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


bn = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)


def _sha(path):
    import hashlib
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def _excl_count(paths):
    cnt, tot = {}, 0
    for pth in paths:
        q = joints(pth)
        for t in range(0, q.shape[0], a.exclusion_stride):
            tot += 1
            for p in contacts(q[t]): cnt[p] = cnt.get(p, 0) + 1
    return cnt, tot


def build_exclusion():
    """fixed rule: link pairs in contact in >= 50 % of the training-set GT frames (sampled every exclusion_stride)"""
    import os, hashlib
    src = a.exclusion_from
    if os.path.isdir(src):
        ids = ([s.strip() for s in open(a.exclusion_stems) if s.strip()] if a.exclusion_stems
               else sorted(f[:-3] for f in os.listdir(src) if f.endswith('.pt')))
        paths = [os.path.join(src, f'{s}.pt') for s in ids]
    else:
        paths = [s.strip() for s in open(src) if s.strip()]
    miss = [p for p in paths if not os.path.isfile(p)]
    if miss: raise SystemExit(f'{len(miss)} exclusion source files missing, e.g. {miss[:3]}')
    if a.exclusion_workers > 1:
        from multiprocessing import Pool
        chunks = [paths[i::a.exclusion_workers] for i in range(a.exclusion_workers)]
        with Pool(a.exclusion_workers) as pool: parts = pool.map(_excl_count, chunks)
    else: parts = [_excl_count(paths)]
    cnt, tot = {}, 0
    for c, t in parts:
        tot += t
        for p, v in c.items(): cnt[p] = cnt.get(p, 0) + v
    frac = sorted(((bn(x), bn(y), v / tot) for (x, y), v in cnt.items()), key=lambda r: -r[2])
    excl = [[x, y, f] for x, y, f in frac if f >= 0.5]
    rec = dict(rule='link pairs in contact (MuJoCo dist < 0, collisions on for all robot geoms, identity base) in >= 50 % '
                    'of the training-set GT frames; fixed before looking at any method', threshold=0.5,
               source=os.path.abspath(src), source_stems=os.path.abspath(a.exclusion_stems) if a.exclusion_stems else None,
               source_stems_sha256=_sha(a.exclusion_stems) if a.exclusion_stems else None, n_clips=len(paths),
               clip_list_sha256=hashlib.sha256('\n'.join(paths).encode()).hexdigest(), n_frames=tot,
               frame_stride=a.exclusion_stride, mjcf=os.path.abspath(a.mjcf), mjcf_sha256=_sha(a.mjcf),
               excluded=excl, contact_fraction_top=[[x, y, f] for x, y, f in frac[:40]])
    json.dump(rec, open(a.exclusion_file, 'w'), indent=1)
    print(f'exclusion table from {len(paths)} clips / {tot} frames -> {a.exclusion_file}: {[(x, y) for x, y, _ in excl]}')
    return rec


def load_exclusion(path):
    rec = json.load(open(path)); out = set()
    for x, y, _ in rec['excluded']:
        ix, iy = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in (x, y))
        if ix < 0 or iy < 0: raise SystemExit(f'exclusion pair {x}-{y} not in {a.mjcf}')
        out.add(tuple(sorted((ix, iy))))
    return out, rec


if a.exclusion_from is not None: build_exclusion()
if build_only: raise SystemExit(0)

stems = [s.strip() for s in open(a.stems) if s.strip()][: a.limit or None]
cols = [('GT', a.ref_dir)] + [tuple(p.split('=', 1)) for p in a.panels]
data = {lab: [] for lab, _ in cols}
for s in stems:
    try: qs = [joints(f'{d_}/{s}.pt') for _, d_ in cols]
    except FileNotFoundError: continue
    n = min(q.shape[0] for q in qs)
    if n < 4: continue
    for (lab, _), q in zip(cols, qs): data[lab].append(q[:n])

if a.exclusion_file is not None:
    # fixed table (training-set GT, built once) -> independent of the evaluated subset
    structural, excl_rec = load_exclusion(a.exclusion_file)
    excl_src = f'file {a.exclusion_file} ({excl_rec["n_clips"]} training clips, {excl_rec["n_frames"]} frames)'
else:
    # legacy: structural overlaps = link pairs in contact in >= 50 % of the sampled GT frames of THIS evaluation set
    cnt, tot = {}, 0
    for q in data['GT']:
        for t in range(0, q.shape[0], a.collision_stride):
            tot += 1
            for p in contacts(q[t]): cnt[p] = cnt.get(p, 0) + 1
    structural = {p for p, c in cnt.items() if c >= 0.5 * tot}
    excl_src = 'dynamic (GT of the evaluated set)'
print(f'{len(data["GT"])} clips; structural overlaps excluded: {sorted((bn(x), bn(y)) for x, y in structural)}')

res = {}; extra = {}
for lab, _ in cols:
    F = P = V = C = A = 0; clean = 0; pex = []; vex = []; pairs = {}; jp = np.zeros(29, int); jv = np.zeros(29, int)
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
        jp += pv.sum(0); jv += vv.sum(0)
        anyv = pv.any(1) | vv.any(1) | cv
        F += T; P += pv.any(1).sum(); V += vv.any(1).sum(); C += cv.sum(); A += anyv.sum(); clean += not anyv.any()
        if pv.any(): pex.append(np.maximum(lo - q, q - hi)[pv].mean())
        if vv.any(): vex.append((np.abs(v) - vmax)[vv].mean())
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:3]
    res[lab] = dict(clips=len(data[lab]), frames=int(F), pos=P / F, vel=V / F, coll=C / F, any=A / F, clean_clips=clean / max(len(data[lab]), 1),
                    pos_excess=float(np.mean(pex)) if pex else 0., vel_excess=float(np.mean(vex)) if vex else 0.,
                    top_pairs=[(bn(x), bn(y), c) for (x, y), c in top])
    r = res[lab]
    r_extra = dict(pairs_all=[(bn(x), bn(y), c) for (x, y), c in sorted(pairs.items(), key=lambda kv: -kv[1])],
                   pairs_unit=f'checked frames (collision stride {a.collision_stride})',
                   joint_pos_frames={G1_JOINT_NAMES[j]: int(jp[j]) for j in range(29) if jp[j]},
                   joint_vel_frames={G1_JOINT_NAMES[j]: int(jv[j]) for j in range(29) if jv[j]})
    print(f'{lab:18s} any {100*r["any"]:.2f}% | pos {100*r["pos"]:.2f}% vel {100*r["vel"]:.2f}% coll {100*r["coll"]:.2f}% | '
          f'clips fully executable {100*r["clean_clips"]:.1f}% | excess pos {r["pos_excess"]:.3f} rad vel {r["vel_excess"]:.2f} rad/s | {r["top_pairs"]}')
    extra[lab] = r_extra
g = extra['GT']
print(f'GT violations by link pair ({g["pairs_unit"]}): {g["pairs_all"]}')
print(f'GT joint-limit frames: pos {g["joint_pos_frames"]} vel {g["joint_vel_frames"]}')
if a.out:
    for lab in res: res[lab].update(extra[lab])
    res['GT']['checker'] = dict(collision_stride=a.collision_stride, exclusion=excl_src,      # label set unchanged
                                excluded_pairs=sorted([bn(x), bn(y)] for x, y in structural))
    json.dump(res, open(a.out, 'w'), indent=1)

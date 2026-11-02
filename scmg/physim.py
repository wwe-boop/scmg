"""Phase 2, P0: physics simulation of EX002 PVT1 + RM003 v4 (the training embodiment) with a joint tracking controller.

Model: dataset/zehan/gesture/ex002-rm003-v4-20261020/model-r1/scene.xml (fixed base, 46 joints, torque ranges from the
URDF, 3 mimic equalities). Added here with MjSpec: position servos on the 34 generated joints (force-limited to the
joint's actfrcrange), stiff holds on the lift (0.32 m, the training embodiment) and the wheels. 50 Hz commands,
500 Hz integration (10 substeps). Servo bandwidth is calibrated so the step response lag matches the real robot
(69g logs: gain ~1.00, pure lag 50-80 ms; calibrate() reports the 63 % rise time).
Rollout returns executed joints, velocities, actuator torques, torque saturation, self-contact penetration and
joint-limit violations; physics_score() turns them into the per-chunk executability signal used by phase 2.
"""
import json
import numpy as np
import mujoco

MODEL = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/ex002-rm003-v4-20261020/model-r1/scene.xml'
META = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1/META.json'
SUB = 10                     # 0.002 s x 10 = 20 ms per command frame
GROUP_KP = dict(waist=3000., head=200., shoulder=900., elbow=600., wrist=150., finger=8.)   # N m / rad (x kp_scale)
ARMATURE = dict(waist=0.5, head=0.02, shoulder=0.1, elbow=0.08, wrist=0.03, finger=0.002)  # kg m^2 (rotor inertia;
# without it the fingers diverged at 2 ms steps)
DELAY = 3                    # nominal pure latency (frames): real robot gain ~1, pure lag 50-80 ms
import os as _os
CACHE = _os.environ.get('SCMG_SIM_CACHE', '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/cache/scmg-ex002-servo.mjb')


def group(name):
    if name.startswith('waist'): return 'waist'
    if name.startswith('head'): return 'head'
    if 'arm_joint' in name:
        k = int(name[-1]); return 'shoulder' if k <= 3 else ('elbow' if k == 4 else 'wrist')
    return 'finger'


class EX002Sim:
    def __init__(self, model_path=MODEL, joints=None, kp_scale=1.0, dampratio=1.0, lift=0.32, seed=0, cache=CACHE):
        self.joints = joints or json.load(open(META))['joints']
        import os, hashlib
        key = hashlib.sha256(open(model_path, 'rb').read() + repr((kp_scale, dampratio, lift, GROUP_KP, ARMATURE)).encode()).hexdigest()[:16]
        if cache and os.path.exists(cache) and os.path.exists(cache + '.key') and open(cache + '.key').read() == key:
            self.m = mujoco.MjModel.from_binary_path(cache)          # seconds instead of ~2 min of mesh compilation
        else:
            self.m = self._build(model_path, kp_scale, dampratio, lift)
            if cache:
                os.makedirs(os.path.dirname(cache), exist_ok=True); mujoco.mj_saveModel(self.m, cache, None)
                open(cache + '.key', 'w').write(key)
        m = self.m
        self.held = {}
        for i in range(m.nu):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if n.startswith('hold_'):
                jn = n[5:]; self.held[jn] = lift if jn.startswith('lift_joint') else 0.
        self.d = mujoco.MjData(self.m)
        self.kp = {n: GROUP_KP[group(n)] * kp_scale for n in self.joints}
        self.qadr = np.array([self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.joints])
        self.dadr = np.array([self.m.jnt_dofadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.joints])
        self.act = np.array([mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'srv_' + n) for n in self.joints])
        self.hold_act = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hold_' + n) for n in self.held}
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joints]
        self.lo, self.hi = self.m.jnt_range[jid, 0].copy(), self.m.jnt_range[jid, 1].copy()
        self.frange = self.m.actuator_forcerange[self.act].copy()
        self.rng = np.random.default_rng(seed)
        self.nominal = dict(mass=self.m.body_mass.copy(), damping=self.m.dof_damping.copy(),
                            gain=self.m.actuator_gainprm[:, 0].copy(), bias=self.m.actuator_biasprm[:, :3].copy())
        self.delay = DELAY

    # ------------------------------------------------------------------ domain randomisation
    def randomize(self, mass=0.1, damping=(0., 2.), gain=0.2, delay_frames=(2, 4)):
        r = self.rng; m = self.m; nom = self.nominal
        m.body_mass[:] = nom['mass'] * (1 + mass * (2 * r.random(len(nom['mass'])) - 1))
        m.dof_damping[:] = nom['damping']; m.dof_damping[self.dadr] = r.uniform(*damping, len(self.dadr))
        s = 1 + gain * (2 * r.random(len(self.act)) - 1)
        m.actuator_gainprm[self.act, 0] = nom['gain'][self.act] * s
        m.actuator_biasprm[self.act, 1] = nom['bias'][self.act, 1] * s
        self.delay = int(r.integers(delay_frames[0], delay_frames[1] + 1))

    def nominalize(self):
        nom = self.nominal; m = self.m
        m.body_mass[:] = nom['mass']; m.dof_damping[:] = nom['damping']
        m.actuator_gainprm[:, 0] = nom['gain']; m.actuator_biasprm[:, :3] = nom['bias']; self.delay = DELAY

    # ------------------------------------------------------------------ rollout
    def reset(self, q, dq=None):
        mujoco.mj_resetData(self.m, self.d)
        for n, v in self.held.items():
            self.d.qpos[self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = v
            self.d.ctrl[self.hold_act[n]] = v
        self.d.qpos[self.qadr] = q
        if dq is not None: self.d.qvel[self.dadr] = dq
        self.d.ctrl[self.act] = q
        self._queue = None                       # latency queue persists across rollouts (chunk after chunk)
        mujoco.mj_forward(self.m, self.d)

    def _build(self, model_path, kp_scale, dampratio, lift):
        """MjSpec: servos + holds + contact excludes (structural rest-pose overlaps, hand-internal pairs)."""
        spec = mujoco.MjSpec.from_file(model_path)
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        base = mujoco.MjModel.from_xml_path(model_path)
        held = {}
        for i in range(base.njnt):
            n = mujoco.mj_id2name(base, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n in self.joints: continue
            if n.startswith('lift_joint'): held[n] = lift
            elif 'wheel' in n: held[n] = 0.
        self.kp = {}
        for n in self.joints:
            jid = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_JOINT, n)
            fr = base.jnt_actfrcrange[jid]
            kp = GROUP_KP[group(n)] * kp_scale; self.kp[n] = kp
            a = spec.add_actuator(); a.name = 'srv_' + n; a.target = n; a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            spec.joint(n).armature = ARMATURE[group(n)]
            a.set_to_position(kp=kp, dampratio=dampratio)
            if fr[1] > fr[0] and fr[1] >= 2.0:          # head range in the URDF is +-0.5 N m (cannot hold the head
                                                       # against gravity: implausible) -> left unlimited
                a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE; a.forcerange = [float(fr[0]), float(fr[1])]
        for n, v in held.items():
            a = spec.add_actuator(); a.name = 'hold_' + n; a.target = n; a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.set_to_position(kp=2e5, dampratio=1.0)
        # contacts: exclude (a) pairs that already overlap at the rest pose (structural mesh overlaps: chassis/wheels/
        # lift, wrist link5/link7, waist/head) and (b) pairs inside one hand (no grasp simulated). What remains is the
        # meaningful self-collision set: arm/hand vs torso/head/other arm.
        m0 = spec.compile(); d0 = mujoco.MjData(m0)
        rest = self._rest_from(m0)
        for jn, v in list(zip(self.joints, rest)) + list(held.items()):
            d0.qpos[m0.jnt_qposadr[mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = v
        mujoco.mj_forward(m0, d0)
        bname = lambda g: mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_BODY, m0.geom_bodyid[g])
        pairs = {tuple(sorted((bname(c.geom1), bname(c.geom2)))) for c in d0.contact[:d0.ncon] if c.dist < 0.002}
        for side in ('left', 'right'):
            bl = sorted(self._subtree(m0, f'{side}_arm_link6'))
            for a_ in range(len(bl)):
                for b_ in range(a_ + 1, len(bl)): pairs.add((bl[a_], bl[b_]))
        for b1, b2 in sorted(pairs):
            e = spec.add_exclude(); e.bodyname1 = b1; e.bodyname2 = b2
        return spec.compile()

    @staticmethod
    def _subtree(m, root):
        rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root); out = set()
        for b in range(m.nbody):
            p = b
            while p > 0 and p != rid: p = m.body_parentid[p]
            if p == rid: out.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b))
        return out

    def _rest_from(self, m):
        st = json.load(open('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/STATS_v36_1.json'))
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joints]
        return np.clip(np.asarray(st['q_mean']), m.jnt_range[jid, 0] + 1e-3, m.jnt_range[jid, 1] - 1e-3)

    def rest(self):
        """Mean training pose (stats file) clipped into the joint range: a collision-free standby posture."""
        st = json.load(open('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/STATS_v36_1.json'))
        return np.clip(np.asarray(st['q_mean']), self.lo + 1e-3, self.hi - 1e-3)

    def state(self):
        return self.d.qpos[self.qadr].copy(), self.d.qvel[self.dadr].copy()

    def rollout(self, cmd, push=None):
        """cmd [T, 34] rad at 50 Hz (applied with self.delay frames of extra latency). push: optional dict(frame, joints,
        torque, frames) external joint torque. Returns per-frame arrays."""
        T = len(cmd); q = np.zeros((T, len(self.joints))); dq = np.zeros_like(q); tau = np.zeros_like(q)
        pen = np.zeros(T); sat = np.zeros(T)
        queue = getattr(self, '_queue', None) or [cmd[0]] * self.delay
        for t in range(T):
            queue.append(cmd[t]); c = queue.pop(0)
            self.d.ctrl[self.act] = c
            self.d.qfrc_applied[:] = 0
            if push is not None and push['frame'] <= t < push['frame'] + push['frames']:
                self.d.qfrc_applied[self.dadr[push['joints']]] = push['torque']
            for _ in range(SUB): mujoco.mj_step(self.m, self.d)
            q[t], dq[t] = self.state(); tau[t] = self.d.actuator_force[self.act]
            lim_ = np.abs(self.frange).max(1); has = lim_ > 0               # head servos are unlimited
            sat[t] = np.mean(np.abs(tau[t][has]) >= 0.98 * lim_[has])
            pen[t] = sum(-c_.dist for c_ in self.d.contact[:self.d.ncon] if c_.dist < 0)
        self._queue = queue
        return dict(q=q, dq=dq, tau=tau, sat=sat, pen=pen)

    # ------------------------------------------------------------------ calibration
    def calibrate(self, step=0.2, frames=40):
        """63 % rise time (ms) of each servo group for a small step from the zero pose, nominal dynamics."""
        out = {}
        for gname in GROUP_KP:
            idx = [i for i, n in enumerate(self.joints) if group(n) == gname]
            if not idx: continue
            q0 = self.rest(); self.reset(q0)
            q_settle = self.rollout(np.tile(q0, (25, 1)))['q'][-1]             # gravity sag settles first
            self.reset(q_settle); tgt = q_settle.copy(); tgt[idx] += np.where(q_settle[idx] + step < self.hi[idx], step, -step)
            r = self.rollout(np.tile(tgt, (frames, 1)))['q']
            frac = (r[:, idx] - q_settle[idx]) / (tgt[idx] - q_settle[idx])
            t63 = [(np.argmax(frac[:, k] >= 0.63) + 1) * 20 if (frac[:, k] >= 0.63).any() else None for k in range(len(idx))]
            ok = [x for x in t63 if x is not None]
            out[gname] = dict(t63_ms_median=float(np.median(ok)) if ok else None, reached=len(ok), joints=len(idx),
                              final_frac=float(np.median(frac[-1])))
        return out


def physics_score(r, cmd, lim=None):
    """Per-rollout executability terms (lower = better) and a weighted total."""
    err = np.abs(r['q'] - cmd)
    acc = np.diff(r['dq'], axis=0) * 50
    terms = dict(track_rms=float(np.sqrt((err ** 2).mean())), track_max=float(err.max()),
                 sat=float(r['sat'].mean()), pen_mm=float(r['pen'].mean() * 1000),
                 energy=float(np.abs(r['tau'] * r['dq']).sum(1).mean()),
                 acc_p99=float(np.percentile(np.abs(acc), 99)) if len(acc) else 0.)
    if lim is not None:
        terms['limit_viol'] = float(((r['q'] < lim['lo']) | (r['q'] > lim['hi'])).any(1).mean())
    w = dict(track_rms=10., sat=5., pen_mm=1., energy=0.002, acc_p99=0.01, limit_viol=5.)
    terms['total'] = float(sum(w[k] * terms[k] for k in w if k in terms))
    return terms

"""Phase 2, P1+P2: candidate chunks scored by physics simulation, closed loop.
    python scripts/p2_rerank.py <ckpt> <vae> <out.json> [n_clips] [n_cand] [cfg_w]
Every tick: N candidate chunks (different noise seeds) are generated from the same streaming state; each is executed
in the EX002+RM003 MuJoCo simulation from the CURRENT simulator state (copied), passed through the CBF-QP safety
projection first; physics score = tracking error, torque saturation, self-penetration, acceleration p99, joint-limit
violation + CBF modification. Energy guard: candidates slower than 0.8 x the median candidate speed are not eligible
(physics must not be bought with smaller gestures). The best eligible candidate is committed and the simulator
advances with it. Baseline policy: always candidate 0 (= plain sampling). Candidates + scores are saved for P3."""
import json, sys, copy
import numpy as np, torch, mujoco
from scmg.common import CHUNK_FRAMES
from scmg.data import View, WindowDataset
from scmg.evaluate import load, beat_align
from scmg.common import PROSODY_COLS
from scmg.infer import Streamer
from scmg.model import INF
from scmg.physim import EX002Sim, physics_score
from scmg.safety import limits34, CBFFilter

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
st = json.load(open(D + '/STATS_v36_1.json'))
meta = json.load(open('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1/META.json'))
v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1')
m, vae = load(sys.argv[1], sys.argv[2], 'cuda')
OUT = sys.argv[3]; NC = int(sys.argv[4]) if len(sys.argv) > 4 else 20; NCAND = int(sys.argv[5]) if len(sys.argv) > 5 else 6
CFGW = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
qm, qs = np.asarray(st['q_mean'], np.float32), np.asarray(st['q_std'], np.float32)
lim = limits34(meta['lower'], meta['upper'])
sim = EX002Sim(); A = slice(4, 18)
COND = ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm')


def snap(S): return (len(S.lat), len(S.frames), S.n_emitted, S.k)
def restore(S, s): del S.lat[s[0]:]; del S.frames[s[1]:]; S.n_emitted, S.k = s[2], s[3]


def score_candidate(fr, d_now, cbf_now):
    """Execute candidate frames from the current simulator state (copied) through a copy of the CBF filter."""
    d_save = mujoco.MjData(sim.m); mujoco.mj_copyData(d_save, sim.m, sim.d); q_save = list(sim._queue or [])
    f = copy.deepcopy(cbf_now); cmd = np.array([f.step(x) for x in fr])
    mod = float(np.abs(cmd - fr).mean())
    r = sim.rollout(cmd); sc = physics_score(r, cmd, lim)
    mujoco.mj_copyData(sim.d, sim.m, d_save); sim._queue = q_save or None   # roll the simulator back
    sc['cbf_mod'] = mod; sc['total'] += 20 * mod
    return sc


def run(policy, s, nch, seed0):
    cond = {k: s[k] for k in COND}
    S = Streamer(m, vae, s['feat'], cond, st, H=INF, steps=10, standby=qm, ctx_t=getattr(m, 'ctx_t', 1.0), cfg_w=CFGW)
    q0 = sim.rest(); sim.nominalize(); sim.reset(q0)
    cbf = CBFFilter(lim, q0)
    executed, commanded, log = [], [], []
    for k in range(nch):
        base = snap(S); cands = []
        for c in range(NCAND if policy == 'rerank' else 1):
            restore(S, base); S.g.manual_seed(seed0 * 1000 + k * 37 + c)
            fr = S.tick().cpu().numpy().astype(np.float64)
            lat = torch.stack(S.lat[-4:]).cpu().numpy()
            cands.append((c, fr, lat))
        if policy == 'rerank':
            scores = [score_candidate(fr, None, cbf) for _, fr, _ in cands]
            spd = np.array([np.abs(np.diff(fr[:, A], axis=0)).mean() if len(fr) > 1 else 0. for _, fr, _ in cands])
            ok = spd >= 0.8 * np.median(spd)
            tot = np.array([sc['total'] for sc in scores]); tot[~ok] = np.inf
            best = int(np.argmin(tot))
            log.append(dict(k=k, totals=[sc['total'] for sc in scores], speed=spd.tolist(), chosen=best,
                            lat=[c[2].tolist() for c in cands]))
        else:
            best = 0
        restore(S, base); S.g.manual_seed(seed0 * 1000 + k * 37 + cands[best][0])
        fr = S.tick().cpu().numpy().astype(np.float64)            # deterministic re-generation of the chosen one
        if k == nch - 1: fr = np.concatenate([fr, S.flush().cpu().numpy()], 0)
        cmd = np.array([cbf.step(x) for x in fr])
        r = sim.rollout(cmd); executed.append(r); commanded.append(cmd)
    cmd = np.concatenate(commanded)
    r = {key: np.concatenate([e[key] for e in executed]) for key in executed[0]}
    return cmd, physics_score(r, cmd, lim), log


res = {'baseline': [], 'rerank': []}; logs = []
clips = [i for i, c in enumerate(v.clips) if c['frames'] >= 320][:NC]
for n, ci in enumerate(clips):
    nch = v.clips[ci]['frames'] // CHUNK_FRAMES
    s = WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(0))
    gt = s['q'] * qs + qm
    beat = np.clip(s['feat'][:, PROSODY_COLS['onset']] + s['feat'][:, PROSODY_COLS['beat']], 0, None)
    for pol in res:
        cmd, ph, log = run(pol, s, nch, seed0=n)
        L = min(len(cmd), len(gt))
        ph['speed_ratio'] = float(np.abs(np.diff(cmd[:L, A], axis=0)).mean() / np.abs(np.diff(gt[:L, A], axis=0)).mean())
        ph['beat_align'] = beat_align(cmd[:L], beat[:L])
        res[pol].append(ph)
        if pol == 'rerank': logs.append(dict(clip=v.clips[ci]['clip'], chunks=log))
    print(n, {p: round(res[p][-1]['total'], 3) for p in res}, flush=True)
summary = {p: {k: float(np.mean([x[k] for x in rows])) for k in rows[0]} for p, rows in res.items()}
json.dump(dict(summary=summary, n_clips=len(clips), n_cand=NCAND, cfg_w=CFGW), open(OUT, 'w'), indent=1)
json.dump(logs, open(OUT.replace('.json', '_candidates.json'), 'w'))
print(json.dumps(summary, indent=1))

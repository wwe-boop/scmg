"""Phase 2 (physics-feedback fine-tuning), data collection on TRAIN windows.
    python scripts/p3_collect.py <ckpt> <vae> <out_dir> <shard> <n_shards> <n_windows> [n_cand] [cfg_w] [cfg_text]
Each window (5.12 s = 16 chunks, random train clip/offset, seeded by window id) is generated in closed loop: every
tick N candidate chunks are sampled from the same streaming state, each executed in the EX002+RM003 MuJoCo
simulation (through the CBF-QP projection) from the current simulator state, scored (physics_score + CBF
modification); the best ELIGIBLE candidate (speed >= 0.8 x median) is committed and becomes the history of the next
chunk (on-policy history). Saved per window: clip/start, committed latents (normalised), candidate latents, scores,
speeds, eligibility -> scripts/p3_finetune.py.
v2 (11-02, user: phase 2 targets only where phase 1 is not SOTA + physics-specific metrics): every candidate's individual
terms are saved (TERMS below + speed vs ground-truth speed of the same chunk), and the committed candidate is chosen by
the TARGET score P3_TARGET (JSON weights over TERMS + 'speed_gap' = |log(speed / gt speed)|); fine-tuning can re-weight
the saved terms without recollecting."""
import json, sys, copy, time
from pathlib import Path
import numpy as np, torch, mujoco
from scmg.common import CHUNK, CHUNK_FRAMES
from scmg.data import View, WindowDataset
from scmg.evaluate import load
from scmg.infer import Streamer
from scmg.model import INF
from scmg.physim import EX002Sim, physics_score
from scmg.safety import limits34, CBFFilter

D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'
ck, vae_p, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
SH, NSH, NW = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
NC = int(sys.argv[7]) if len(sys.argv) > 7 else 6
CFGW = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
CFGT = (sys.argv[9] != 'audio') if len(sys.argv) > 9 else True
out.mkdir(parents=True, exist_ok=True)
st = json.load(open(D + '/STATS_v36_1.json'))
meta = json.load(open('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1/META.json'))
v = View(split='train', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', tok_cache=D + '/tok-qwen25-v36_1')
ds = WindowDataset(v, 256, max_tokens=128)
m, vae = load(ck, vae_p, 'cuda')
qm, qs = np.asarray(st['q_mean'], np.float32), np.asarray(st['q_std'], np.float32)
lim = limits34(meta['lower'], meta['upper']); sim = EX002Sim(); A = slice(4, 18)
COND = ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm')
w = ds.w


def snap(S): return (len(S.lat), len(S.frames), S.n_emitted, S.k)
def restore(S, s): del S.lat[s[0]:]; del S.frames[s[1]:]; S.n_emitted, S.k = s[2], s[3]


TERMS = ('collision_rate', 'pen_mm', 'acc_p99', 'limit_viol', 'track_rms', 'sat', 'cbf_mod', 'total')
TARGET = json.loads(__import__('os').environ.get('P3_TARGET', '{"collision_rate": 50, "pen_mm": 5, "acc_p99": 0.01, "limit_viol": 5, "speed_gap": 2}'))


def score(fr, cbf):
    d_save = mujoco.MjData(sim.m); mujoco.mj_copyData(d_save, sim.m, sim.d); q_save = list(sim._queue or [])
    f = copy.deepcopy(cbf); cmd = np.array([f.step(x) for x in fr]); mod = float(np.abs(cmd - fr).mean())
    sc = physics_score(sim.rollout(cmd), cmd, lim); sc['total'] += 20 * mod; sc['cbf_mod'] = mod
    mujoco.mj_copyData(sim.d, sim.m, d_save); sim._queue = q_save or None
    return np.array([sc.get(t, 0.) for t in TERMS], np.float32)


def target(terms, spd, gspd):
    gap = np.abs(np.log(np.maximum(spd, 1e-4) / max(gspd, 1e-4)))
    return sum(w_ * (gap if k == 'speed_gap' else terms[..., TERMS.index(k)]) for k, w_ in TARGET.items())


t0 = time.time(); done = 0
for wid in range(SH, NW, NSH):
    f = out / f'w{wid:06d}.npz'
    if f.exists(): continue
    rng = np.random.default_rng([20261102, wid])
    while True:
        ci = int(rng.choice(len(w), p=w)); L = v.clips[ci]['frames']
        if L >= 256: break
    s0 = int(rng.integers(0, L - 256 + 1)) // 4 * 4
    s = ds.sample(ci, s0, rng); gt = s['q'] * qs + qm; nch = 16
    cond = {k: s[k] for k in COND}
    S = Streamer(m, vae, s['feat'], cond, st, H=INF, steps=10, standby=gt[0], ctx_t=getattr(m, 'ctx_t', 1.0), cfg_w=CFGW, cfg_text=CFGT)
    sim.nominalize(); sim.reset(np.clip(gt[0], sim.lo + 1e-3, sim.hi - 1e-3)); cbf = CBFFilter(lim, gt[0])
    cand_lat = np.zeros((nch, NC, CHUNK, m.cfg.z), np.float32); tot = np.zeros((nch, NC), np.float32); spd = np.zeros((nch, NC), np.float32)
    terms = np.zeros((nch, NC, len(TERMS)), np.float32); gspd = np.zeros(nch, np.float32)
    for k in range(nch):
        base = snap(S); frs = []
        for c in range(NC):
            restore(S, base); S.g.manual_seed(wid * 100003 + k * 37 + c)
            fr = S.tick().cpu().numpy().astype(np.float64)
            cand_lat[k, c] = torch.stack(S.lat[-CHUNK:]).cpu().numpy(); frs.append(fr)
            terms[k, c] = score(fr, cbf); spd[k, c] = np.abs(np.diff(fr[:, A], axis=0)).mean() if len(fr) > 1 else 0.
        e1 = S.n_emitted; e0 = e1 - len(frs[-1]); g_ = gt[max(e0 - 1, 0):e1, A]
        gspd[k] = np.abs(np.diff(g_, axis=0)).mean() if len(g_) > 1 else spd[k].mean()
        tot[k] = target(terms[k], spd[k], gspd[k])
        ok = spd[k] >= 0.8 * np.median(spd[k]); t_ = np.where(ok, tot[k], np.inf); best = int(np.argmin(t_))
        restore(S, base); S.g.manual_seed(wid * 100003 + k * 37 + best)
        fr = S.tick().cpu().numpy().astype(np.float64)
        cmd = np.array([cbf.step(x) for x in fr]); sim.rollout(cmd)
    committed = torch.stack(S.lat).cpu().numpy().astype(np.float32)
    elig = spd >= 0.8 * np.median(spd, 1, keepdims=True)
    np.savez(f, ci=ci, start=s0, committed=committed, cand_lat=cand_lat, total=tot, speed=spd, eligible=elig,
             terms=terms, term_names=np.array(TERMS), gt_speed=gspd, target=json.dumps(TARGET))
    done += 1
    print(f'shard {SH} window {wid} done ({done}) {time.time() - t0:.0f}s best-mean {np.mean(np.min(np.where(elig, tot, np.inf), 1)):.3f} '
          f'cand-mean {tot.mean():.3f}', flush=True)

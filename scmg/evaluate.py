"""Evaluation on the fixed dev set (380 BEAT2 dev utterances, never trained on).

Open loop, for each audio horizon H (INF offline, 8 / 4 / 2 latents look-ahead, 0 strictly causal):
  FKD          Frechet distance of 1-s kinematic features (pose mean, pose std, mean |velocity|) vs ground truth
  beat_align   mean normalised arm acceleration on audio onsets/beats (generated / ground truth ratio)
  boundary     jerk at chunk-boundary frames / jerk elsewhere (1.0 = seamless; ground truth reported too)
  speed_ratio  mean arm joint speed generated / ground truth
  diversity    mean |difference| between two seeds (arm joints, rad)
  tick_ms      p50 / p95 time per 320 ms chunk tick (real-time factor = p95 / 320 ms)
Closed loop (--closed-loop): a first-order tracking robot (tau 60 ms) receives every committed chunk; at a random
time a person HOLDS four arm joints displaced by 0.3 rad for 0.4-1.0 s, then releases. With / without state feedback:
  fight_rms       RMS(command - executed) on the held joints while held (how hard the robot fights the person)
  snap_acc_p99    executed arm acceleration p99 in the 0.5 s after release (snap-back violence), rad/s^2
  recover_s       time after release until |command - executed| < 0.05 rad on every arm joint
  track_rms_after RMS(command - executed) after release
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from scipy import linalg

from .common import CHUNK_FRAMES, PROSODY_COLS, J, U80
from .data import View, WindowDataset, VIEW
from .infer import generate
from .model import INF
from .sim import ARMS
from .train import build_model
from .vae import MotionVAE


def kin_feats(q, win=50):
    out = []
    for s in range(0, len(q) - win + 1, win // 2):
        w = q[s:s + win]
        out.append(np.concatenate([w.mean(0), w.std(0), np.abs(np.diff(w, axis=0)).mean(0) * 50]))
    return out


def frechet(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m1, m2 = a.mean(0), b.mean(0); s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    try:
        cs, _ = linalg.sqrtm(s1 @ s2, disp=False)
    except TypeError:                                   # scipy >= 1.14 dropped disp
        cs = linalg.sqrtm(s1 @ s2)
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs.real))


def beat_align(q, beat):
    acc = np.linalg.norm(q[2:, ARMS] - 2 * q[1:-1, ARMS] + q[:-2, ARMS], axis=-1)
    acc = acc / max(acc.mean(), 1e-6); b = beat[1:-1]
    return float((acc * b).sum() / max(b.sum(), 1e-6))


def boundary_ratio(q):
    j = np.linalg.norm(np.diff(q[:, ARMS], 3, axis=0), axis=-1)
    f = np.arange(len(j)) + 1
    at = (f % CHUNK_FRAMES <= 1) | (f % CHUNK_FRAMES >= CHUNK_FRAMES - 1)
    return float(j[at].mean() / max(j[~at].mean(), 1e-9))


class Robot:
    """First-order tracking (q += alpha (cmd - q)) with a HOLD disturbance: from hold_start for hold_len frames a person
    holds 4 arm joints, displaced by amp (ramped in over 0.1 s); then releases them and tracking resumes."""

    def __init__(self, q0, tau=0.06, hold_start=None, hold_len=40, amp=0.3, seed=0):
        self.q = np.asarray(q0, np.float64).copy(); self.dq = np.zeros_like(self.q)
        self.a = (1 / 50) / (tau + 1 / 50); self.f = 0
        r = np.random.default_rng(seed)
        self.h0, self.hl = hold_start, hold_len
        self.idx = r.choice(ARMS, 4, replace=False); self.off = amp * r.choice([-1, 1], 4)
        self.held = None

    def step(self, frames):
        frames = frames.detach().cpu().numpy() if torch.is_tensor(frames) else frames
        ex = []
        for c in frames:
            prev = self.q.copy()
            self.q = self.q + self.a * (c - self.q)
            if self.h0 is not None and self.h0 <= self.f < self.h0 + self.hl:
                if self.held is None: self.held = prev[self.idx].copy()
                ramp = min(1., (self.f - self.h0 + 1) / 5)
                self.q[self.idx] = self.held + ramp * self.off
            self.dq = (self.q - prev) * 50; self.f += 1; ex.append(self.q.copy())
        return self.q.astype(np.float32), self.dq.astype(np.float32), np.array(ex)


def load(ckpt, vae_path, dev):
    vck = torch.load(vae_path, map_location='cpu')
    vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema']); vae.eval()
    ck = torch.load(ckpt, map_location='cpu')
    m, _ = build_model(ck['base_cfg'], {}, ck['extra_cfg'])
    m.load_state_dict(ck['ema']); m.to(dev).eval()
    m.hist_measured = bool((ck.get('sc') or {}).get('state_hist_measured', False))
    return m, vae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--vae', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--stats', required=True); ap.add_argument('--text-cache', required=True); ap.add_argument('--view', default=VIEW)
    ap.add_argument('--n', type=int, default=380); ap.add_argument('--horizons', default='inf,8,4,0')
    ap.add_argument('--steps', type=int, default=10); ap.add_argument('--closed-loop', action='store_true')
    ap.add_argument('--tok-cache', default=None)
    ap.add_argument('--n-future', type=int, default=0, help='open loop: provisional chunks (closed loop always uses 1)')
    ap.add_argument('--ndiv', type=int, default=20, help='clips that get a second seed for the diversity metric'); ap.add_argument('--mmap', action='store_true')
    a = ap.parse_args()
    dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stats = json.loads(Path(a.stats).read_text())
    view = View(a.view, 'val', a.text_cache, stats, mmap=a.mmap, tok_cache=a.tok_cache)
    m, vae = load(a.ckpt, a.vae, dev)
    qm, qs = np.asarray(stats['q_mean'], np.float32), np.asarray(stats['q_std'], np.float32)
    standby = qm
    res = {}; per = []
    clips = list(range(min(a.n, len(view.clips))))
    for hname in a.horizons.split(','):
        H = INF if hname == 'inf' else int(hname) * U80          # CLI horizons are in 80 ms units
        F_gen, F_gt, ba_g, ba_t, br_g, br_t, sp, dv, ticks = [], [], [], [], [], [], [], [], []
        for ci in clips:
            L = view.clips[ci]['frames']; nch = L // CHUNK_FRAMES
            if nch < 3: continue
            ds = WindowDataset(view, nch * CHUNK_FRAMES, max_tokens=256)
            s = ds.sample(ci, 0, np.random.default_rng(ci))
            cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm')}
            gt = s['q'] * qs + qm
            beat = np.clip(s['feat'][:, PROSODY_COLS['onset']] + s['feat'][:, PROSODY_COLS['beat']], 0, None)
            gens = []
            for seed in ((0, 1) if len(dv) < a.ndiv else (0,)):
                cmd, _, tm = generate(m, vae, s['feat'], cond, stats, nch, H=H, steps=a.steps, seed=seed,
                                      n_future=a.n_future, standby=standby)
                gens.append(cmd); ticks += list(tm[1:])
            g0 = gens[0]; n = min(len(g0), len(gt))
            F_gen += kin_feats(g0[:n]); F_gt += kin_feats(gt[:n])
            ba_g.append(beat_align(g0[:n], beat[:n])); ba_t.append(beat_align(gt[:n], beat[:n]))
            br_g.append(boundary_ratio(g0[:n])); br_t.append(boundary_ratio(gt[:n]))
            sp.append(np.abs(np.diff(g0[:n, ARMS], axis=0)).mean() / max(np.abs(np.diff(gt[:n, ARMS], axis=0)).mean(), 1e-6))
            if len(gens) > 1: dv.append(float(np.abs(gens[0][:n, ARMS] - gens[1][:n, ARMS]).mean()))
            per.append(dict(H=hname, clip=view.clips[ci]['clip'], ba=ba_g[-1], br=br_g[-1], sp=sp[-1], dv=dv[-1] if len(gens) > 1 else None))
            if ci < 8: np.save(out / f'gen_H{hname}_{ci}.npy', g0); np.save(out / f'gt_{ci}.npy', gt)
        tk = np.array(ticks) * 1000
        res[hname] = dict(clips=len(ba_g), FKD=frechet(F_gen, F_gt) if len(F_gen) > 110 else None,
                          beat_align=float(np.mean(ba_g)), beat_align_gt=float(np.mean(ba_t)),
                          boundary=float(np.mean(br_g)), boundary_gt=float(np.mean(br_t)), speed_ratio=float(np.mean(sp)),
                          diversity=float(np.mean(dv)), tick_ms_p50=float(np.percentile(tk, 50)), tick_ms_p95=float(np.percentile(tk, 95)))
        print(hname, json.dumps(res[hname]), flush=True)
    if a.closed_loop:
        cl = {}
        for use_state in ((True, False) if m.cfg.state else (False,)):
            hm = m.hist_measured and use_state
            tr, fight, snap, rec = [], [], [], []
            for ci in clips:
                L = view.clips[ci]['frames']; nch = L // CHUNK_FRAMES
                if nch < 6: continue
                ds = WindowDataset(view, nch * CHUNK_FRAMES, max_tokens=256)
                s = ds.sample(ci, 0, np.random.default_rng(ci))
                cond = {k: s[k] for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm')}
                rg = np.random.default_rng(ci)
                hl = int(rg.integers(20, 51)); h0 = int(rg.integers(2 * CHUNK_FRAMES, max(2 * CHUNK_FRAMES + 1, (nch - 2) * CHUNK_FRAMES - hl)))
                robot = Robot(standby, hold_start=h0, hold_len=hl, seed=ci)
                if not use_state:
                    class NoFeedback:
                        def step(self, fr):
                            _, _, ex = robot.step(fr); return None, None, ex
                    rb = NoFeedback()
                else:
                    rb = robot
                cmd, ex, _ = generate(m, vae, s['feat'], cond, stats, nch, robot=rb, H=4 * U80, steps=a.steps, seed=0,
                                      n_future=1, standby=standby, hist_measured=hm)
                n = min(len(cmd), len(ex)); rel = h0 + hl
                if rel + 25 > n: continue
                err = cmd[:n] - ex[:n]
                fight.append(float(np.sqrt((err[h0:rel][:, robot.idx] ** 2).mean())))
                acc = np.linalg.norm(np.diff(ex[rel - 2:rel + 25][:, ARMS], 2, axis=0), axis=-1) * 2500
                snap.append(float(np.percentile(acc, 99)))
                ok = np.all(np.abs(err[rel:, ARMS]) < 0.05, axis=1)
                rec.append(float(np.argmax(ok) / 50) if ok.any() else float(len(ok) / 50))
                tr.append(float(np.sqrt((err[rel:][:, ARMS] ** 2).mean())))
            cl['state' if use_state else 'no_state'] = dict(clips=len(tr), fight_rms=float(np.mean(fight)),
                                                           snap_acc_p99=float(np.mean(snap)), recover_s=float(np.mean(rec)),
                                                           track_rms_after=float(np.mean(tr)))
            print('closed-loop', 'state' if use_state else 'no_state', json.dumps(cl), flush=True)
        res['closed_loop'] = cl
    (out / 'RESULTS.json').write_text(json.dumps(res, indent=1))
    with open(out / 'per_clip.jsonl', 'w') as f:
        for p in per: f.write(json.dumps({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in p.items()}) + '\n')


if __name__ == '__main__':
    main()

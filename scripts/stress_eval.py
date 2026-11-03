"""Stress evaluation of our models on the perturbed-audio packs of stress_feats.py (11-04 Claude).
    stress_eval.py <out.jsonl> <packs dir> label=ckpt[@cfg_w] ...
Env: STRESS_CONDS (comma, default all in INDEX), STRESS_TEXT own|notext|swap (default own), STRESS_TM final|x2,
     STRESS_H inf|<80 ms units>, STRESS_SEED (0), STRESS_LAG (x2 lag s, .15).  Inference fixes are default-on.
Every row: model, clip, cond, text, tm, H, seed and
  speed      mean arm joint speed / clean-GT speed;   still  frames slower than .25 x clean-GT median speed
  still_bins stillness per 10 s;   post_sil  stillness in the 3 s after the inserted silence ends (silence3 only), pre_sil
  hf         share of arm motion energy above 4 Hz (jitter; catches 'moving' by tremor)
  jerk       mean |d3 q| arms x 50^3 / clean GT;   lim_pos / lim_vel  fraction of frames outside 3 % position margin /
             above the joint velocity limit (safety.limits34);  tick_p50/p95 ms
  beat_gap   visible beat: 4 Hz low-passed arm acceleration on prosodic-prominence peaks of THIS condition's audio,
             real minus circularly shifted (+-0.6..2.4 s), the validated gesture-scale form of the beat metric"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from scipy.signal import butter, filtfilt, find_peaks
from scmg.data import View, WindowDataset, clip_lm_capacity
from scmg.evaluate import load
from scmg.infer import generate
from scmg.common import CHUNK_FRAMES, U80, PROSODY_COLS
from scmg.safety import limits34
D = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg'; dev = 'cuda'; FPS = 50
ARMS = list(range(4, 18)); BA, BB = butter(4, 4.0 / (FPS / 2)); HB, HA = butter(4, 4.0 / (FPS / 2), 'high')
st = json.load(open(D + '/STATS_v36_1.json')); qm, qs = np.array(st['q_mean'], np.float32), np.array(st['q_std'], np.float32)
fm, fs_ = np.array(st['f_mean'], np.float32), np.array(st['f_std'], np.float32)
OUT, PK = sys.argv[1], Path(sys.argv[2]); index = json.load(open(PK / 'INDEX.json'))
CONDS = os.environ.get('STRESS_CONDS', ''); CONDS = set(CONDS.split(',')) if CONDS else None
TXT = os.environ.get('STRESS_TEXT', 'own'); TM = os.environ.get('STRESS_TM', 'final'); HS = os.environ.get('STRESS_H', 'inf')
SEED = int(os.environ.get('STRESS_SEED', '0')); LAG = float(os.environ.get('STRESS_LAG', '0.15'))
H = 10 ** 9 if HS == 'inf' else int(HS) * U80
meta = json.load(open('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1/META.json'))
LIM = limits34(meta['lower'], meta['upper']) if 'lower' in meta else None
TIMES = ('ttime', 'titime', 'tseg', 'tiseg', 'tiend')


def arm_speed(q): return np.linalg.norm(np.diff(q[:, ARMS], axis=0), axis=1) * FPS


def beat_gap(q, prom):
    if len(q) < 60: return None
    ql = filtfilt(BA, BB, q[:, ARMS], axis=0); acc = np.linalg.norm(ql[2:] - 2 * ql[1:-1] + ql[:-2], axis=-1); acc /= max(acc.mean(), 1e-9)
    b = np.clip(prom[1:-1][:len(acc)], 0, None); acc = acc[:len(b)]
    sc = lambda w: float((acc * w).sum() / max(w.sum(), 1e-9))
    return sc(b) - np.mean([sc(np.roll(b, int(s * FPS))) for s in (-2.4, -1.6, -1.0, -0.6, 0.6, 1.0, 1.6, 2.4)])


def main():
    tok = D + '/tok-qwen25-v36_1'
    v = View(split='val', stats=st, mmap=True, text_cache=D + '/text-e5-v36_1', tok_cache=tok)
    rows = [r for r in index if CONDS is None or r['cond'] in CONDS]
    clip_ids = sorted({r['ci'] for r in rows}); samples = {}
    for ci in clip_ids:
        nch = v.clips[ci]['frames'] // CHUNK_FRAMES
        samples[ci] = WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=256, max_ids=clip_lm_capacity(v, ci)).sample(ci, 0, np.random.default_rng(ci))
    f = open(OUT, 'a')
    for spec in sys.argv[3:]:
        lab, ck = spec.split('=', 1); w = 2.0
        pre = ck.startswith('dir:')                # external baseline: precomputed <clip>__<cond>.npy (EX002 34-d rad 50 Hz)
        if '@' in ck and not pre: ck, w = ck.rsplit('@', 1); w = float(w)
        if not pre:
            m, vae = load(ck, '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/trained/vae-r3/final.pt', dev)
            vox = m.cfg.planner == 'voxcpm'
        for r in rows:
            ci = r['ci']; s = samples[ci]; z = np.load(PK / r['npz'])
            feat = (z['feat'] - fm) / fs_; T = int(z['frames']); nch = T // CHUNK_FRAMES
            src = s if TXT != 'swap' else samples[clip_ids[(clip_ids.index(ci) + 1) % len(clip_ids)]]
            cond = {k: np.array(src[k]) for k in ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c') if k in src}
            for k in ('spk', 'spk_valid', 'source', 'ref', 'ref_valid'): cond[k] = np.array(s[k])
            for k in TIMES:
                if k in cond:
                    x = cond[k]; ok = np.isfinite(x); x[ok] = np.interp(x[ok], z['knots_old'], z['knots_new']); cond[k] = x
            if TXT == 'notext':
                for k in ('tmask', 'tidm'):
                    if k in cond: cond[k] = np.zeros_like(cond[k])
            gt = s['q'] * qs + qm; gsp = arm_speed(gt); thr = 0.25 * np.median(gsp)
            if pre:
                fp = Path(ck[4:]) / r['npz'].replace('.npz', '.npy')
                if not fp.exists(): continue
                cmd, tm = np.load(fp), [0.]
            else:
                cmd, _, tm = generate(m, vae, feat, cond, st, nch, H=H, steps=10, seed=SEED, standby=st['q_mean'], ctx_t=m.ctx_t, cfg_w=w,
                                      text_mode=TM, x2_rate=15.0, x2_lag=LAG, deploy_timing=TM != 'final', window_lm_text=vox)
            y = cmd[:T]; sp = arm_speed(y); still = sp < thr
            bins = [float(still[i:i + 500].mean()) for i in range(0, len(still), 500)]
            row = dict(model=lab, ci=ci, clip=r['clip'], lang=r['lang'], cond=r['cond'], text=TXT, tm=TM, H=HS, seed=SEED, cfg_w=w,
                       seconds=T / FPS, speed=float(sp.mean() / gsp.mean()), still=float(still.mean()), still_bins=bins)
            if r['cond'] == 'silence3':
                k0 = np.flatnonzero(np.diff(z['knots_new']) > 0)
                t1 = float(z['knots_new'][2]) if len(z['knots_new']) == 4 else None
                if t1 is not None:
                    a = int(t1 * FPS); row['post_sil'] = float(still[a:a + 150].mean()); row['in_sil'] = float(still[a - 150:a].mean()); row['pre_sil'] = float(still[max(0, a - 300):a - 150].mean())
            hp = filtfilt(HB, HA, y[:, ARMS], axis=0); row['hf'] = float((hp ** 2).sum() / max(((y[:, ARMS] - y[:, ARMS].mean(0)) ** 2).sum(), 1e-9))
            j = np.abs(np.diff(y[:, ARMS], 3, axis=0)).mean(); jg = np.abs(np.diff(gt[:, ARMS], 3, axis=0)).mean(); row['jerk'] = float(j / max(jg, 1e-9))
            if LIM is not None:
                J = [0, 1] + ARMS                    # waist + arms; HARD limits (the 3 % margin flags upright waist pitch, range 0.23 rad)
                lo_h, hi_h = np.asarray(meta['lower'])[J], np.asarray(meta['upper'])[J]
                row['lim_pos'] = float(((y[:, J] < lo_h) | (y[:, J] > hi_h)).any(1).mean())
                row['waist_pitch'] = float(y[:, 1].mean()); row['waist_pitch_gt'] = float(gt[:, 1].mean())
                row['lim_vel'] = float((np.abs(np.diff(y[:, J], axis=0)) * FPS > LIM['v'][J]).any(1).mean())
                row['lim_pos_gt'] = float(((gt[:, J] < lo_h) | (gt[:, J] > hi_h)).any(1).mean())
            row['beat_gap'] = beat_gap(y, z['feat'][:T, PROSODY_COLS['prominence']])
            tk = np.array(tm[1:]) * 1000 if len(tm) > 1 else np.array([0.]); row['tick_p50'] = float(np.percentile(tk, 50)); row['tick_p95'] = float(np.percentile(tk, 95))
            f.write(json.dumps(row) + '\n'); f.flush(); print(lab, r['clip'], r['cond'], round(row['still'], 3), round(row['speed'], 2), flush=True)
        if not pre: del m; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

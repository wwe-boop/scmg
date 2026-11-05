"""Landing metrics on EX002 dev eval exports (Claude 11-05; user 11-04 L1 / L2).

    python scripts/landing/landing_metrics.py <out.jsonl> [--envelope HUMAN_ENVELOPE.json] label=<dir>[+<dir seed 1>] ...
<dir> = scmg.evaluate output (gen_H*_<ci>.npy + gt_<ci>.npy, ci = val-view clip index, frame 0 = clip start), e.g.
<lane>/eval/final/engine/eval-s0. Several dirs joined by '+' (inference seeds) are pooled.

Per label (gen and, where meaningful, the GT of the same clips):
  wrist_out     frames with any wrist joint 5-7 (either arm) outside the clean-human [p0.1, p99.9] interval
                (HUMAN_ENVELOPE.json pooled; joint-space proxy of "wrist / hand-arm junction everted")
  upper_out     same for the upper-arm joints 1-3
  still         frames with arm speed < 0.25 x the clip's human median speed (rollout_probe definition, all frames)
  still_gap     |still_gen - still_gt| (user 11-04: stillness should match the human, not be minimal)
  freeze        share of speaking frames inside >= 1.5 s still segments that lie entirely in speech;
                freeze_per_min = such segments per minute of speech
  overspeed     frames with arm speed above the clip's human p95 speed (human = 5 % by construction; >> 5 % = flailing)
  speed         mean over clips of mean gen arm speed / mean GT arm speed
  amp_upper / amp_wrist   median over clips of mean_joint std_gen / std_gt (1 = like the human; << 1 = rigid)
Speech = raw 'voiced' prosody column > 0.5 with gaps <= 0.3 s closed, from the val view (same clips as the export).
"""
import argparse, glob, json, os
import numpy as np
from scmg.common import P, CPFS, PROSODY_COLS, FPS
from scmg.landing import (UPPER, WRIST, load_envelope, arm_speed, speech_mask, stillness, freeze_segments,
                          out_of_envelope, ARMS14)

VIEW = CPFS + '/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1'


def val_voiced():
    """ci -> raw voiced column of the val-view clip (the View class builds clips from SPLIT val_ids in train.json order)."""
    from scmg.data import View
    st = json.load(open(P(CPFS + '/dataset/zehan/gesture/scmg/STATS_v36_1.json')))
    v = View(VIEW, 'val', None, st, mmap=True)
    col = PROSODY_COLS['voiced']
    return v, lambda ci, n: np.asarray(v.f[v.clips[ci]['start']:v.clips[ci]['start'] + n, col], np.float64)


def files(d):
    g = sorted(glob.glob(d + '/gen_H*_*.npy'), key=lambda p: int(p.rsplit('_', 1)[1][:-4]))
    return [(int(p.rsplit('_', 1)[1][:-4]), p, f'{d}/gt_{p.rsplit("_", 1)[1]}') for p in g]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('specs', nargs='+')
    ap.add_argument('--envelope', default=os.environ.get('HUMAN_ENVELOPE', '/home/xr-25b7/bulk/landing-1105/HUMAN_ENVELOPE.json'))
    a = ap.parse_args()
    e, qs, q = load_envelope(a.envelope)
    col = lambda p: q[:, qs.index(p)]
    lo, hi = col(0.1), col(99.9)
    view, voiced = val_voiced()
    fo = open(a.out, 'a')
    for spec in a.specs:
        lab, ds = spec.split('=', 1)
        acc = {w: dict(frames=0, wrist=0, upper=0, still=0, over=0, spk=0, frz=0, nseg=0) for w in ('gen', 'gt')}
        sp_ratio, amp_u, amp_w, n_clips = [], [], [], 0
        for d in ds.split('+'):
            fl = files(d)
            if not fl: raise SystemExit(f'{lab}: no gen_H*_*.npy in {d}')
            for ci, pg, pt in fl:
                g = np.load(pg).astype(np.float64); t = np.load(pt).astype(np.float64)
                n = min(len(g), len(t)); g, t = g[:n], t[:n]
                if view.clips[ci]['frames'] < n: raise SystemExit(f'{lab}: clip {ci} export longer than the view clip')
                speech = speech_mask(voiced(ci, n) > 0.5)
                st = arm_speed(t); med, p95 = float(np.median(st)), float(np.percentile(st, 95))
                for w, x in (('gen', g), ('gt', t)):
                    A = acc[w]; s = arm_speed(x)
                    A['frames'] += n
                    A['wrist'] += int(out_of_envelope(x, lo[WRIST], hi[WRIST], WRIST).sum())
                    A['upper'] += int(out_of_envelope(x, lo[UPPER], hi[UPPER], UPPER).sum())
                    A['still'] += int(stillness(x, med).sum()); A['over'] += int((s > p95).sum())
                    seg, fz, spk = freeze_segments(x, speech, med); A['frz'] += fz; A['spk'] += spk; A['nseg'] += len(seg)
                sp_ratio.append(float(arm_speed(g).mean() / max(st.mean(), 1e-6)))
                sd = lambda x, c: x[:, c].std(0)
                amp_u.append(float(np.mean(sd(g, UPPER) / np.maximum(sd(t, UPPER), 1e-3))))
                amp_w.append(float(np.mean(sd(g, WRIST) / np.maximum(sd(t, WRIST), 1e-3))))
                n_clips += 1
        r = dict(label=lab, dirs=ds, clips=n_clips, envelope=a.envelope)
        for w, A in acc.items():
            nf = max(1, A['frames']); nv = max(1, A['frames'] - n_clips)        # speed-based counts have T-1 frames per clip
            r[w] = dict(wrist_out=round(A['wrist'] / nf, 5), upper_out=round(A['upper'] / nf, 5), still=round(A['still'] / nv, 4),
                        overspeed=round(A['over'] / nv, 4), freeze=round(A['frz'] / max(1, A['spk']), 4),
                        freeze_per_min=round(A['nseg'] / max(1e-9, A['spk'] / FPS / 60), 3))
        r['still_gap'] = round(abs(r['gen']['still'] - r['gt']['still']), 4)
        r['speed'] = round(float(np.mean(sp_ratio)), 3)
        r['amp_upper'] = round(float(np.median(amp_u)), 3); r['amp_wrist'] = round(float(np.median(amp_w)), 3)
        print(json.dumps(r), flush=True); fo.write(json.dumps(r) + '\n'); fo.flush()


if __name__ == '__main__':
    main()

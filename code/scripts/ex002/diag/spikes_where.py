"""Read-only diagnosis (11-08, coordinator): where do acceleration spikes of generated streams sit?  numbers only.
Statistic (as the 9b diagnosis): per frame a = max over arms of || 2nd difference of the arm's 7 joint angles || * fps^2 (rad/s^2);
random 0.2 s windows (stride 0.1 s) -> peak; share of frames > 100 rad/s^2 and > the GT p99 of the same embodiment.
Location: spike frames (a > GT p99) by phase of the 320 ms chunk grid (16 frames at 50 Hz; frame f phase f mod 16; latent = 4 frames;
emission boundary of the streamer at phase 12), by joint (argmax |a_j| at the spike, arms / fingers / head-waist), speech vs silence."""
import json, glob, sys, os
from collections import OrderedDict
import numpy as np
from scipy.interpolate import CubicSpline
X = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; DS = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'
VIEW = DS + '/train-v1-20261101/views/data-v36_1'; GV = DS + '/echog-g1-v1'
E = X + '/ex002-sfscope-1108/enc-s0'; SIB = X + '/control-token-1105/ex002/m78b-sibag'
ARM_E = (list(range(4, 11)), list(range(11, 18))); FING_E = list(range(18, 34)); HW_E = list(range(0, 4))
ARM_G = (list(range(25, 32)), list(range(32, 39)))
R = {}


def speech_masks():
    val = set(json.load(open(VIEW + '/SPLIT.json'))['val_ids']); cl = OrderedDict()
    for it in json.load(open(VIEW + '/train.json')):
        if it['id'] in val: cl.setdefault(it['clip'], []).append(it)
    out = []
    for c, xs in cl.items():
        xs.sort(key=lambda x: x['offset']); st = xs[0]['offset']; L = sum(x['frames'] for x in xs); m = np.zeros(L, bool)
        for it in xs:
            if not it.get('speech'): continue
            i0 = it['offset'] - st; s0, s1 = (int(round(float(x) * 50)) for x in it['speech']); m[max(0, i0 + s0):min(L, i0 + s1)] = True
        out.append(m)
    return out


def acc(y, jj, fps):
    return np.linalg.norm(np.diff(y[:, jj], n=2, axis=0), axis=1) * fps ** 2      # index i = frame i + 1


def per_joint(y, fps): return np.abs(np.diff(y, n=2, axis=0)) * fps ** 2


def to30(y):
    t50 = np.arange(len(y)) / 50.; n = int(np.floor(t50[-1] * 30 + 1e-9)) + 1
    return CubicSpline(t50, y, axis=0)(np.arange(n) / 30.)


def stat(clips, fps, arms, thr=None):
    a_all = []; pk = []
    w = int(round(0.2 * fps)) + 1; st = max(1, int(round(0.1 * fps)))
    for y in clips:
        a = np.max([acc(y, jj, fps) for jj in arms], 0); a_all.append(a)
        pk += [a[i:i + w].max() for i in range(0, len(a) - w + 1, st)]
    a = np.concatenate(a_all); pk = np.array(pk)
    d = dict(frames=int(len(a)), share_gt100=float((a > 100).mean()), p99=float(np.percentile(a, 99)), max=float(a.max()),
             win_median=float(np.median(pk)), win_p95=float(np.percentile(pk, 95)), win_max=float(pk.max()))
    if thr is not None: d['share_gt_GTp99'] = float((a > thr).mean())
    return d, a_all


def where(clips, a_all, thr, sm=None, fps=50):
    ph16 = np.zeros(16); ph16_all = np.zeros(16); jcount = OrderedDict((k, 0) for k in ('L arm j1-7', 'R arm j1-7'))
    jj_hist = np.zeros(34); sp = dict(speech_frames=0, speech_spikes=0, sil_frames=0, sil_spikes=0); rel = []
    for ci, (y, a) in enumerate(zip(clips, a_all)):
        f = np.arange(1, len(a) + 1); s = a > thr
        ph16 += np.bincount(f[s] % 16, minlength=16); ph16_all += np.bincount(f % 16, minlength=16)
        pj = per_joint(y, fps)                           # [T-2, J], row i = frame i + 1
        for i in np.nonzero(s)[0]: jj_hist[np.argmax(pj[i, 4:18]) + 4] += 1
        if sm is not None:
            m = sm[ci][1:len(a) + 1] if len(sm[ci]) >= len(a) + 1 else np.pad(sm[ci], (0, len(a) + 1 - len(sm[ci])))[1:len(a) + 1]
            sp['speech_frames'] += int(m.sum()); sp['speech_spikes'] += int((s & m).sum()); sp['sil_frames'] += int((~m).sum()); sp['sil_spikes'] += int((s & ~m).sum())
    rate = ph16 / np.maximum(ph16_all, 1); rate = rate / rate.mean()
    out = dict(n_spikes=int(ph16.sum()), phase16_rel_rate=[round(float(x), 2) for x in rate],
               phase4_rel_rate=[round(float(x), 2) for x in (lambda r4: r4 / r4.mean())(np.array([ph16[k::4].sum() / ph16_all[k::4].sum() for k in range(4)]))],
               joint_share={f'{"L" if j < 11 else "R"} j{(j - 4) % 7 + 1}': round(float(jj_hist[j] / max(1, jj_hist.sum())), 3) for j in range(4, 18)})
    if sm is not None:
        out['spike_rate_speech'] = sp['speech_spikes'] / max(1, sp['speech_frames']); out['spike_rate_silence'] = sp['sil_spikes'] / max(1, sp['sil_frames'])
        out['speech_frame_share'] = sp['speech_frames'] / max(1, sp['speech_frames'] + sp['sil_frames'])
    return out


def load_npys(d): return [np.load(f) for f in sorted((f for f in glob.glob(d + '/*.npy') if os.path.basename(f)[:-4].isdigit()), key=lambda p: int(os.path.basename(p)[:-4]))]


def main():
    sm = speech_masks()
    gt = [np.load(f'{E}/offline/gt_{i}.npy') for i in range(len(glob.glob(E + '/offline/gt_*.npy')))]
    g, ga = stat(gt, 50, ARM_E); thr = g['p99']; R['EX002 GT dev (50 Hz)'] = g
    R['EX002 GT where'] = where(gt, ga, thr, sm)
    # fingers / head-waist separately on GT and enc-s0 none offline s0
    for lab, src in [('enc-s0 offline', [f'{E}/offline/s{k}' for k in range(3)]), ('enc-s0 psync', [f'{E}/psync/s{k}' for k in range(3)]),
                     ('sibag-s2 offline', [f'{SIB}/offline/none_s{k}' for k in range(3)]), ('sibag-s2 psync', [f'{SIB}/psync/none_s{k}' for k in range(3)])]:
        cl = []; masks = []
        for d in src:
            c = load_npys(d); cl += c; masks += sm[:len(c)]
        d_, aa = stat(cl, 50, ARM_E, thr); R[f'EX002 {lab} none (50 Hz)'] = d_
        R[f'EX002 {lab} where'] = where(cl, aa, thr, masks)
        fg = np.concatenate([np.abs(np.diff(y[:, FING_E], n=2, axis=0)).max(1) * 2500 for y in cl]); hw = np.concatenate([np.abs(np.diff(y[:, HW_E], n=2, axis=0)).max(1) * 2500 for y in cl])
        R[f'EX002 {lab} fingers/head-waist max-joint acc p99'] = [float(np.percentile(fg, 99)), float(np.percentile(hw, 99))]
        R[f'EX002 {lab} none resampled to 30 fps'] = stat([to30(y) for y in cl], 30, ARM_E)[0]
    fg = np.concatenate([np.abs(np.diff(y[:, FING_E], n=2, axis=0)).max(1) * 2500 for y in gt]); hw = np.concatenate([np.abs(np.diff(y[:, HW_E], n=2, axis=0)).max(1) * 2500 for y in gt])
    R['EX002 GT fingers/head-waist max-joint acc p99'] = [float(np.percentile(fg, 99)), float(np.percentile(hw, 99))]
    R['EX002 GT resampled to 30 fps'] = stat([to30(y) for y in gt], 30, ARM_E)[0]
    # G1: dev GT (view train_q, 50 Hz cubic from 30 fps mocap) -> 30 fps; preds at 30 fps (robot_repr, joints 25:39 arms)
    import torch
    sp = json.load(open(X + '/views/echog-g1-v1-hos/SPLIT.json')); val = set(sp['val_ids'])
    q = np.load(GV + '/train_q.npy', mmap_mode='r'); items = [it for it in json.load(open(X + '/views/echog-g1-v1-hos/train.json')) if it['id'] in val]
    g1gt = []
    for it in items:
        y = np.asarray(q[it['offset']:it['offset'] + it['frames']], np.float64); n30 = it['echog']['frames30']
        t50 = np.arange(len(y)) / 50.; n30 = min(n30, int(np.floor(t50[-1] * 30 + 1e-9)) + 1)
        g1gt.append(CubicSpline(t50, y[:, 10:39], axis=0)(np.arange(n30) / 30.))
    shift = lambda arms: tuple([j - 10 for j in jj] for jj in arms)
    G, ga = stat(g1gt, 30, shift(ARM_G)); R['G1 GT dev (30 fps)'] = G
    G50 = stat([np.asarray(q[it['offset']:it['offset'] + it['frames']], np.float64)[:, 10:39] for it in items], 50, shift(ARM_G))[0]; R['G1 GT dev (50 Hz view frames)'] = G50

    def g1(d, lim=None):
        fs = sorted(glob.glob(d + '/*.pt'))[:lim]; out = []
        for f in fs:
            z = torch.load(f, map_location='cpu', weights_only=False); rr = z['robot_repr'].double().numpy()[:int(z.get('real_num_frames', len(z['robot_repr'])))]
            out.append(rr[:, 10:39])
        return out
    C6 = X + '/causal-train-1105/czm6kws-dev-1107/dev/czm6kws/bench'
    for pr in ('offline', 'psync'):
        cl = g1(f'{C6}/{pr}/seed_000'); R[f'G1 czm6kws dev {pr} (30 fps, {len(cl)} clips)'] = stat(cl, 30, shift(ARM_G), G['p99'])[0]
    F3 = X + '/causal-train-1105/final3-1106/test'
    for pr in ('offline', 'psync'):
        cl = []
        for s in range(3): cl += g1(f'{F3}/czmws-s{s}/final-test/{pr}/seed_000', lim=int(sys.argv[1]) if len(sys.argv) > 1 else None)
        R[f'G1 czmws test {pr} (30 fps, {len(cl)} clips, 3 train seeds)'] = stat(cl, 30, shift(ARM_G), G['p99'])[0]
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()

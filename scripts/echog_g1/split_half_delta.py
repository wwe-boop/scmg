"""Non-inferiority margins delta by ground-truth split halves (design v4 §0, 11-05).

The ground-truth clips are split into two halves, stratified by speaker (within each speaker: random permutation, first
floor(n/2) to half A, the rest to B; an odd extra clip goes to A or B at random), R = 1000 times. For each split the
metrics are computed on the two halves with the OFFICIAL evaluator code (echo_g.evaluation.g1_motion_cls) and the 95th
percentile over the splits is the margin delta:
  FGD        FGD(half A latents, half B latents)       - official map2latent + FGD.frechet_distance (official encoder)
  FGD_leggt  the same after the leg-GT replacement (scripts/echog_g1/fgd_leggt.py). For ground truth against ground truth
             the replaced dims are each clip's own GT, so this equals FGD exactly; it is computed, not assumed.
  Div        |Div(A) - Div(B)|                          - official Div (EMAGE L1div, fk_positions_ba, frame-weighted)
  BA         |BA(A) - BA(B)|                            - official BeatAlign (audio_to_motion, frame-weighted mean)
  JerkW      |JerkW(A) - JerkW(B)|                      - official Jerk._jerk on fk_positions_world, weight T-3
Per-clip quantities (latents, Div sums, BA scores, jerk values) are computed once; each split only re-aggregates them
exactly as the official classes do (Div: sum / frames; BA: sum(BA_i T_i) / sum(T_i) over non-NaN clips; JerkW: weighted
mean), so every split value equals running the official evaluator on that half (checked by --verify-official).
Speakers: --speaker-map json {stem: speaker} / two-column tsv, default = first three '_' fields of the stem
(english_1_wayne_...). Output: json with the per-split values, delta (95th pct), signed 2.5 / 97.5 pct, the seed.

    PYTHONPATH=$OFF/src python scripts/echog_g1/split_half_delta.py --ref-dir R --stems S --wav-dir W --mmae M \
        --fgd-ckpt F [--speaker-map X] [--splits 1000] [--seed 20261105] --out delta.json
"""
import argparse, json, os, sys, time
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from echo_g.evaluation import g1_motion_cls as GM
import fgd_leggt as L

METRICS = ('FGD', 'FGD_leggt', 'Div', 'BA', 'JerkW')


def speakers_of(stems, path):
    if path is None: return {s: '_'.join(s.split('_')[:3]) for s in stems}
    if path.endswith('.json'): mp = json.load(open(path))
    else: mp = dict(l.split()[:2] for l in open(path) if l.strip())
    miss = [s for s in stems if s not in mp]
    if miss: raise SystemExit(f'{len(miss)} stems without speaker, e.g. {miss[:3]}')
    return {s: mp[s] for s in stems}


def per_clip(pairs, net, cfg, dev, wav_dir, fps):
    """official per-clip pieces of GT-vs-GT evaluation (pred = gt)."""
    beat = GM.BeatAlign(fps, direction='audio_to_motion'); jerk = GM.Jerk(fps)
    rec = []
    for s, pr, gr in pairs:
        pr, gr = GM.crop_to_common_length(pr, gr)
        lat = GM.map2latent_feature(net, cfg, pr, dev)
        prl, _ = L.leg_gt_replace(pr, gr)
        lat_lg = GM.map2latent_feature(net, cfg, prl, dev)
        pj = GM.fk_positions_ba(pr); x = pj.reshape(pj.shape[0], -1)
        div_sum = float(np.sum(np.abs(x - np.mean(x, axis=0)), axis=None))
        ba = beat._beat_consistency(pj, GM.wav_of(wav_dir, s))
        jw = jerk._jerk(GM.fk_positions_world(pr, fps), fps)
        rec.append(dict(stem=s, T=int(pr.shape[0]), lat=lat, lat_lg=lat_lg, div_sum=div_sum, ba=ba, jerk=jw))
    return rec


def aggregate(rec, idx):
    r = [rec[i] for i in idx]
    T = np.array([x['T'] for x in r], float)
    div = sum(x['div_sum'] for x in r) / T.sum()
    ba_ok = [(x['ba'], x['T']) for x in r if x['ba'] == x['ba']]
    ba = sum(v * w for v, w in ba_ok) / sum(w for _, w in ba_ok)
    jw = [(x['jerk'], x['T'] - 3) for x in r if x['jerk'] == x['jerk'] and x['T'] - 3 > 0]
    jerk = float(np.average([v for v, _ in jw], weights=[w for _, w in jw]))
    lat = [x['lat'] for x in r if x['lat'] is not None]; lat_lg = [x['lat_lg'] for x in r if x['lat_lg'] is not None]
    return dict(Div=div, BA=ba, JerkW=jerk, lat=np.concatenate(lat), lat_lg=np.concatenate(lat_lg))


def split(groups, rng):
    A, B = [], []
    for g in groups:
        p = rng.permutation(g); h = len(p) // 2
        if len(p) % 2 and rng.random() < 0.5: h += 1
        A += p[:h].tolist(); B += p[h:].tolist()
    return sorted(A), sorted(B)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ref-dir', required=True); ap.add_argument('--stems', required=True); ap.add_argument('--wav-dir', required=True)
    ap.add_argument('--mmae', required=True, help='official eval_assets/mmae/g1_mmae_30body_30fps.npy (BA)')
    ap.add_argument('--fgd-ckpt', required=True); ap.add_argument('--speaker-map', default=None)
    ap.add_argument('--splits', type=int, default=1000); ap.add_argument('--seed', type=int, default=20261105)
    ap.add_argument('--fps', type=float, default=30.0); ap.add_argument('--quantile', type=float, default=0.95)
    ap.add_argument('--verify-official', type=int, default=0, help='re-check the first k splits with the official classes')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    GM._G1_MEAN_VEL_PATH = a.mmae; GM._G1_MEAN_VEL = None; GM._g1_mean_vel()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net, cfg = GM.load_g1_ae(a.fgd_ckpt, dev, a.fps)
    stems = L.read_stems(a.stems)
    pairs = L.load_pairs(a.ref_dir, a.ref_dir, stems, a.fps)
    spk = speakers_of([s for s, _, _ in pairs], a.speaker_map)
    t0 = time.time(); rec = per_clip(pairs, net, cfg, dev, a.wav_dir, a.fps)
    print(f'{len(rec)} clips, per-clip pieces in {time.time() - t0:.0f} s', flush=True)
    by = defaultdict(list)
    for i, x in enumerate(rec): by[spk[x['stem']]].append(i)
    groups = [np.array(by[k]) for k in sorted(by)]
    rng = np.random.default_rng(a.seed)
    vals = {k: [] for k in METRICS}; fd = GM.FGD.frechet_distance
    for it in range(a.splits):
        iA, iB = split(groups, rng)
        mA, mB = aggregate(rec, iA), aggregate(rec, iB)
        vals['FGD'].append(fd(mA['lat'], mB['lat'])); vals['FGD_leggt'].append(fd(mA['lat_lg'], mB['lat_lg']))
        for k in ('Div', 'BA', 'JerkW'): vals[k].append(mA[k] - mB[k])
        if it < a.verify_official: verify(pairs, rec, iA, iB, mA, mB, vals, it, net, cfg, dev, a)
        if (it + 1) % 100 == 0: print(f'split {it + 1}/{a.splits}', flush=True)
    res = dict(rule=f'speaker-stratified random halves of the GT, {a.splits} splits, delta = {a.quantile:.0%} quantile; '
                    'FGD / FGD_leggt = FGD(half A, half B); Div / BA / JerkW = |metric(A) - metric(B)|',
               ref_dir=a.ref_dir, stems=a.stems, n_clips=len(rec), speakers={k: len(v) for k, v in sorted(by.items())},
               seed=a.seed, splits=a.splits, fgd_ckpt=a.fgd_ckpt, encoder_input_mode=cfg['input_mode'], delta={}, values=vals)
    for k, v in vals.items():
        v = np.asarray(v, float); absv = np.abs(v)
        res['delta'][k] = dict(delta=float(np.quantile(absv, a.quantile)), mean_abs=float(absv.mean()),
                               signed_q025=float(np.quantile(v, 0.025)), signed_q975=float(np.quantile(v, 0.975)))
    res['FGD_leggt_equals_FGD'] = bool(np.allclose(vals['FGD'], vals['FGD_leggt'], rtol=0, atol=1e-9))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1)
    for k in METRICS: print(f"delta {k:10s} = {res['delta'][k]['delta']:.4f}  (mean |diff| {res['delta'][k]['mean_abs']:.4f})")
    print(f'wrote {a.out}')


def verify(pairs, rec, iA, iB, mA, mB, vals, it, net, cfg, dev, a):
    """recompute one split with the official metric classes, half by half."""
    for half, idx, m in (('A', iA, mA), ('B', iB, mB)):
        div = GM.Div(); beat = GM.BeatAlign(a.fps, direction='audio_to_motion'); jerk = GM.Jerk(a.fps)
        for i in idx:
            s, pr, gr = pairs[i]; pr, gr = GM.crop_to_common_length(pr, gr)
            pj = GM.fk_positions_ba(pr); div.compute(pj.reshape(pj.shape[0], -1))
            beat.update(pj, GM.wav_of(a.wav_dir, s)); w = GM.fk_positions_world(pr, a.fps); jerk.update(w, w)
        o = dict(Div=div.avg(), BA=beat.summary()['mean'], JerkW=jerk.summary_length_weighted()[0]['mean'])
        bad = {k: (o[k], m[k]) for k in o if abs(o[k] - m[k]) > 1e-9 * max(1.0, abs(o[k]))}
        print(f'verify split {it} half {half}: ' + ('OK ' if not bad else f'MISMATCH {bad} ') + str({k: round(v, 6) for k, v in o.items()}), flush=True)
    # FGD: official class with half A as "pred" and half B as "gt" latents
    f = GM.FGD(net, cfg, dev); f.pred_lat = [rec[i]['lat'] for i in iA if rec[i]['lat'] is not None]
    f.gt_lat = [rec[i]['lat'] for i in iB if rec[i]['lat'] is not None]
    v, _, _ = f.compute(); print(f'verify split {it} FGD: official {v:.6f} vs {vals["FGD"][-1]:.6f}', flush=True)


if __name__ == '__main__':
    main()

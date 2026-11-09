"""Development-set report for the C3 safety layers (design v6 C3): one row per safety layer for one generator.

    python scripts/echog_g1/safety_report.py --run-root O --raw-dir P --ref-dir GT --stems S --fgd-ckpt F --out J [--boot 1000]

O = safety_run.py output root (O/<layer>/<stem>.pt + stats.jsonl). Columns:
  independent checker (scripts/echog_g1/executability.py --collision-stride 1 --exclusion-file g1_exclusion_train.json, run
    as a subprocess): share of frames violating position / velocity / self-collision / any, clips fully executable
  fallback rate  (frames executed under a fallback / all frames; per layer's own definition: braking-tail or hold)
  acc / jerk     share of frames over the acc / jerk limits (g1_accjerk_limits.json; supplementary, not in the checker)
  FGD            official ECHO-G FGD (unmodified official class) and leg-neutral FGD (fgd_leggt.py leg_neutral: the 12 leg
                 joints = 0 rad in prediction and GT, prediction root = GT root)
  non-encoder quality (different source from the FGD encoder, against Goodhart):
                 |dJerk| = |mean_clip Jerk(pred) - mean_clip Jerk(GT)|, Jerk = official Jerk._jerk (mean third finite
                 difference of the 30 body positions * fps^3) on pose-only FK of the leg-neutral pair;
                 acc spike % = share of frames where any upper-body joint (waist + arms) has |a| above that joint's
                 training-GT 99.9th percentile (g1_accjerk_limits.json acc_p99_9)
  correction RMSE (rad, 29 joints; and upper body) vs the raw generator output; per-chunk solve time (stats.jsonl)
Paired bootstrap over RECORDINGS (stem minus '_utt_NNNN'): differences A - B of official FGD, leg-neutral FGD, |dJerk|,
acc spike % for the listed pairs, 95 % percentile interval and two-sided bootstrap p.
"""
import argparse, json, os, subprocess, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import fgd_leggt as FL
from scmg import safety_proj as S
from echo_g.evaluation import g1_motion_cls as GM

UPPER = list(range(12, 29))


def frechet(mu_p, cov_p, mu_g, cov_g, eps=1e-6):
    from scipy.linalg import sqrtm
    off = np.eye(cov_p.shape[0]) * eps
    cm, _ = sqrtm((cov_p + off) @ (cov_g + off), disp=False)
    if np.iscomplexobj(cm): cm = cm.real
    d = mu_p - mu_g
    return float(d @ d + np.trace(cov_p) + np.trace(cov_g) - 2 * np.trace(cm))


class LatStats:
    """per-clip sufficient statistics of latents -> FGD for any multiset of clips (bootstrap weights)."""

    def __init__(self, lat):                  # lat: {stem: (pred [n, D], gt [n, D])}
        self.stems = list(lat)
        self.n = np.array([lat[s][0].shape[0] for s in self.stems], float)
        self.sp = np.stack([lat[s][0].sum(0) for s in self.stems]).astype(np.float64)
        self.sg = np.stack([lat[s][1].sum(0) for s in self.stems]).astype(np.float64)
        self.qp = np.stack([lat[s][0].astype(np.float64).T @ lat[s][0].astype(np.float64) for s in self.stems])
        self.qg = np.stack([lat[s][1].astype(np.float64).T @ lat[s][1].astype(np.float64) for s in self.stems])

    def fgd(self, w=None):
        w = np.ones(len(self.stems)) if w is None else w
        n = (w * self.n).sum()
        out = []
        for s1, q2 in ((self.sp, self.qp), (self.sg, self.qg)):
            mu = (w[:, None] * s1).sum(0) / n
            cov = (np.tensordot(w, q2, 1) - n * np.outer(mu, mu)) / (n - 1)
            out += [mu, cov]
        return frechet(*out)


def load_q(d, s):
    x = torch.load(f'{d}/{s}.pt', map_location='cpu', weights_only=False)
    return (x['robot_repr'] if isinstance(x, dict) else x).double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-root', required=True); ap.add_argument('--raw-dir', required=True)
    ap.add_argument('--ref-dir', required=True); ap.add_argument('--stems', required=True)
    ap.add_argument('--fgd-ckpt', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--layers', default=','.join(S.LAYERS)); ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--pairs', default='proj_perc-proj_l2,proj_perc-cbf_qp,proj_perc-clip_coll,proj_l2-cbf_qp,proj_perc-clip')
    ap.add_argument('--tag', default=''); ap.add_argument('--skip-checker', action='store_true')
    a = ap.parse_args()
    layers = a.layers.split(','); stems = FL.read_stems(a.stems)
    lim = S.Limits(); aj = json.load(open(S.ACCJERK)); spike_thr = np.array(aj['acc_p99_9'])[UPPER]
    rep = dict(tag=a.tag, run_root=os.path.abspath(a.run_root), raw_dir=os.path.abspath(a.raw_dir), ref_dir=os.path.abspath(a.ref_dir),
               stems=os.path.abspath(a.stems), n_stems=len(stems), layers={})
    # ---------------- independent checker
    if not a.skip_checker:
        cj = os.path.join(a.run_root, 'executability.json')
        cmd = [sys.executable, os.path.join(HERE, 'executability.py'), '--ref-dir', a.ref_dir, '--stems', a.stems,
               '--urdf', S.URDF, '--mjcf', S.MJCF, '--collision-stride', '1', '--exclusion-file', S.EXCLUSION, '--out', cj] + \
              [f'{n}={a.run_root}/{n}' for n in layers]
        r = subprocess.run(cmd, capture_output=True, text=True)
        open(os.path.join(a.run_root, 'executability.log'), 'w').write(r.stdout + r.stderr)
        if r.returncode: raise SystemExit(r.stderr)
        chk = json.load(open(cj)); rep['checker_gt'] = {k: chk['GT'][k] for k in ('pos', 'vel', 'coll', 'any', 'clean_clips', 'frames')}
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net, cfg = GM.load_g1_ae(a.fgd_ckpt, dev, 30.0)
    rec_of = {s: s.rsplit('_utt_', 1)[0] for s in stems}
    recs = sorted(set(rec_of.values())); rid = {r: i for i, r in enumerate(recs)}
    stem_rec = np.array([rid[rec_of[s]] for s in stems])
    per = {}
    for n in layers:
        d = f'{a.run_root}/{n}'
        st = [json.loads(l) for l in open(f'{d}/stats.jsonl')]
        st = {x['stem']: x for x in st}
        pairs = FL.load_pairs(d, a.ref_dir, stems)
        lat_o = FL.latents_pairs(net, cfg, dev, pairs, 'official'); lat_n = FL.latents_pairs(net, cfg, dev, pairs, 'legneutral')
        Lo, Ln = LatStats(lat_o), LatStats(lat_n)
        assert Lo.stems == stems and Ln.stems == stems, 'every stem must give latents'
        jg, jp, spk, nfr, acc_v, jerk_v, se, se_u, nfr29 = [], [], [], [], [], [], [], [], []
        for s, pr, gr in pairs:
            po, go = FL.leg_neutral(pr, gr)
            jp.append(GM.Jerk._jerk(GM.fk_positions(po.float()), 30.0)); jg.append(GM.Jerk._jerk(GM.fk_positions(go.float()), 30.0))
            q = load_q(d, s)[:, 10:].numpy(); raw = load_q(a.raw_dir, s)[:, 10:].numpy()
            acc = np.abs(np.diff(q, 2, axis=0)) * 900; jerk = np.abs(np.diff(q, 3, axis=0)) * 27000
            spk.append(int((acc[:, UPPER] > spike_thr).any(1).sum())); nfr.append(len(q))
            acc_v.append(int((acc > lim.amax + 1e-6).any(1).sum())); jerk_v.append(int((jerk > lim.jmax + 1e-6).any(1).sum()))
            se.append(float(((q - raw) ** 2).sum())); se_u.append(float(((q[:, UPPER] - raw[:, UPPER]) ** 2).sum()))
        jp, jg, spk, nfr = map(np.array, (jp, jg, spk, nfr)); se, se_u = np.array(se), np.array(se_u)
        times = np.concatenate([np.array(st[s]['times']) for s in stems]) * 1e3
        fbf = sum(st[s]['fallback_frames'] for s in stems); fbc = sum(st[s]['fallback_chunks'] for s in stems)
        chunks = sum(st[s]['chunks'] for s in stems); frames = sum(st[s]['frames'] for s in stems)
        row = dict(FGD_official=Lo.fgd(), FGD_legneutral=Ln.fgd(),
                   FGD_official_classcheck=FL.fgd_pairs(net, cfg, dev, pairs, 'official')['FGD'],
                   dJerk=float(abs(jp.mean() - jg.mean())), Jerk=float(jp.mean()), Jerk_gt=float(jg.mean()),
                   spike_pct=100 * spk.sum() / nfr.sum(), acc_viol_pct=100 * sum(acc_v) / nfr.sum(), jerk_viol_pct=100 * sum(jerk_v) / nfr.sum(),
                   rmse=float(np.sqrt(se.sum() / (nfr.sum() * 29))), rmse_upper=float(np.sqrt(se_u.sum() / (nfr.sum() * len(UPPER)))),
                   fallback_frames_pct=100 * fbf / frames, fallback_chunks_pct=100 * fbc / chunks, chunks=chunks, frames=frames,
                   t_mean=float(times.mean()), t_p50=float(np.percentile(times, 50)), t_p95=float(np.percentile(times, 95)),
                   t_p99=float(np.percentile(times, 99)), t_max=float(times.max()))
        extra = {}
        for k in ('qp_fail', 'verify_fail', 'coll_iters', 'perc_reject', 'relax_acc', 'relax_jerk', 'errors'):
            if k in st[stems[0]]: extra[k] = int(sum(st[s][k] for s in stems))
        row['counters'] = extra
        if not a.skip_checker:
            c = chk[n]; row.update(chk_pos=100 * c['pos'], chk_vel=100 * c['vel'], chk_coll=100 * c['coll'], chk_any=100 * c['any'],
                                   chk_clean_clips=100 * c['clean_clips'], chk_top_pairs=c['top_pairs'])
        rep['layers'][n] = row
        per[n] = dict(Lo=Lo, Ln=Ln, jp=jp, jg=jg, spk=spk, nfr=nfr)
        print(f"[{a.tag}] {n:10s} " + ' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()
                                               if k not in ('chk_top_pairs', 'counters')), row['counters'], flush=True)
    # ---------------- paired bootstrap over recordings
    rng = np.random.default_rng(20261105)
    R = len(recs); boot = {}
    pairs_ab = [p.split('-', 1) for p in a.pairs.split(',') if all(x in per for x in p.split('-', 1))]
    metr = lambda P, w: dict(FGD_official=P['Lo'].fgd(w), FGD_legneutral=P['Ln'].fgd(w),
                             dJerk=abs((w * P['jp']).sum() / w.sum() - (w * P['jg']).sum() / w.sum()),
                             spike_pct=100 * (w * P['spk']).sum() / (w * P['nfr']).sum())
    full = {n: metr(per[n], np.ones(len(stems))) for n in per}
    draws = {n: [] for n in per if any(n in ab for ab in pairs_ab)}
    for b in range(a.boot):
        cnt = np.bincount(rng.integers(0, R, R), minlength=R).astype(float)
        w = cnt[stem_rec]
        for n in draws: draws[n].append(metr(per[n], w))
    for A, B in pairs_ab:
        res = {}
        for k in full[A]:
            dlt = np.array([x[k] - y[k] for x, y in zip(draws[A], draws[B])])
            res[k] = dict(diff=full[A][k] - full[B][k], lo=float(np.percentile(dlt, 2.5)), hi=float(np.percentile(dlt, 97.5)),
                          p=float(min(1.0, 2 * min((dlt <= 0).mean(), (dlt >= 0).mean()))))
        boot[f'{A}-{B}'] = res
        print(f'[{a.tag}] bootstrap {A} - {B}: ' + '; '.join(f"{k} {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] p={v['p']:.3f}"
                                                         for k, v in res.items()), flush=True)
    rep['bootstrap'] = dict(unit='recording', n_recordings=R, n_boot=a.boot, pairs=boot)
    json.dump(rep, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

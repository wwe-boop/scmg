"""11-06 G1 main table: recording-level paired bootstrap of the official FGD between two METHODS, each given as several runs
(training x inference seeds): per bootstrap draw the val_common recordings (stem minus '_utt_NNNN') are resampled with
replacement, every run's FGD is recomputed on the same multiset (per-clip latent sufficient statistics, LatStats of
safety_report.py = the official FGD for the full set) and averaged over the method's runs; reported: mean-FGD difference A - B,
95 % percentile interval, two-sided bootstrap p, and the per-run full-set FGD.
    SAFEPROJ=<exp/safe-proj tree> python scripts/echog_g1/boot_main_g1.py --ref-dir GT --stems S --fgd-ckpt F --out J --boot 2000 A=dir1,dir2,... B=dir1,...
"""
import argparse, json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
SAFE = os.environ.get('SAFEPROJ', os.path.expanduser('~/projects/gesture/scmg-wt-safeproj'))   # exp/safe-proj (fgd_leggt, LatStats)
sys.path.insert(0, os.path.join(SAFE, 'scripts', 'echog_g1')); sys.path.insert(0, SAFE)
import fgd_leggt as FL
from safety_report import LatStats
from echo_g.evaluation import g1_motion_cls as GM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-dir', required=True); ap.add_argument('--stems', required=True); ap.add_argument('--fgd-ckpt', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--boot', type=int, default=2000); ap.add_argument('groups', nargs='+')
    a = ap.parse_args()
    stems = FL.read_stems(a.stems); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net, cfg = GM.load_g1_ae(a.fgd_ckpt, dev, 30.0)
    rec_of = [s.rsplit('_utt_', 1)[0] for s in stems]; recs = sorted(set(rec_of)); rid = {r: i for i, r in enumerate(recs)}
    srec = np.array([rid[r] for r in rec_of]); R = len(recs)
    G = {}
    for g in a.groups:
        name, dirs = g.split('=', 1); L = []
        for d in dirs.split(','):
            pairs = FL.load_pairs(d, a.ref_dir, stems)
            Ls = LatStats(FL.latents_pairs(net, cfg, dev, pairs, 'official')); assert Ls.stems == stems
            L.append(Ls); print(name, d, 'FGD', round(Ls.fgd(), 4), flush=True)
        G[name] = L
    rng = np.random.default_rng(20261106); names = list(G)
    full = {n: [L.fgd() for L in G[n]] for n in names}
    draws = {n: [] for n in names}
    for b in range(a.boot):
        w = np.bincount(rng.integers(0, R, R), minlength=R).astype(float)[srec]
        for n in names: draws[n].append(np.mean([L.fgd(w) for L in G[n]]))
    out = dict(stems=os.path.abspath(a.stems), n_recordings=R, n_boot=a.boot, unit='recording', full=full, pairs={})
    for i, A in enumerate(names):
        for B in names[i + 1:]:
            d = np.array(draws[A]) - np.array(draws[B])
            res = dict(diff=float(np.mean(full[A]) - np.mean(full[B])), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                       p=float(min(1., 2 * min((d <= 0).mean(), (d >= 0).mean()))))
            out['pairs'][f'{A} - {B}'] = res
            print(f'{A} - {B}: mean FGD diff {res["diff"]:+.4f} [{res["lo"]:+.4f}, {res["hi"]:+.4f}] p={res["p"]:.4f}', flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

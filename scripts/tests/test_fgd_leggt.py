"""Tests for scripts/echog_g1/fgd_leggt.py (11-05).
    PYTHONPATH=$OFF/src python scripts/tests/test_fgd_leggt.py <gt_dir> <stems> <fgd_ckpt> [<pred_dir> <benchmark.json>]
1 GT vs GT: leg-GT FGD and official FGD ~ 0.
2 GT with legs-only perturbation: leg-GT FGD identical to (1) (bit-exact), official FGD moves.
3 (pred given) the same on a real prediction; official FGD of the unmodified pred reproduces the official evaluator's json
  (run the evaluator on the SAME host/GPU type: A800 vs RTX 5090 differ by ~2e-4 in FGD for identical inputs).
4 upper-body perturbation does move leg-GT FGD (the metric is not blind).
"""
import json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'echog_g1'))
import fgd_leggt as L
from echo_g.evaluation import g1_motion_cls as GM

ok = True
def check(name, cond, info=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True); ok &= bool(cond)

gt_dir, stems_f, ckpt = sys.argv[1:4]
pred_dir = sys.argv[4] if len(sys.argv) > 4 else None; bench = sys.argv[5] if len(sys.argv) > 5 else None
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
net, cfg = GM.load_g1_ae(ckpt, dev, 30.0)
stems = L.read_stems(stems_f)
gtp = L.load_pairs(gt_dir, gt_dir, stems)
print(f'{len(gtp)} clips, encoder mode {cfg["input_mode"]}, replaced dims {L.REPLACE_DIMS}')
check('leg dims are G1_JOINT_NAMES[0:12] (+10 offset)', L.LEG_DIMS == list(range(10, 22)),
      str([L.G1_JOINT_NAMES[i - 10] for i in L.LEG_DIMS]))


def perturb(pairs, dims, seed=0):
    g = torch.Generator().manual_seed(seed); out = []
    for s, pr, gr in pairs:
        p = pr.clone()
        p[:, dims] = p[:, dims] + 0.3 * torch.sin(torch.arange(p.shape[0])[:, None] / 7.0) \
                     + 0.1 * torch.randn(p.shape[0], len(dims), generator=g)
        p[:, dims] = torch.roll(p[:, dims], shifts=11, dims=0)
        out.append((s, p, gr))
    return out


f = lambda pairs, lg: L.fgd_pairs(net, cfg, dev, pairs, lg)['FGD']
g_lg, g_of = f(gtp, True), f(gtp, False)
check('GT self: leg-GT FGD ~ 0', abs(g_lg) < 1e-3, f'{g_lg:.3e}')
check('GT self: official FGD ~ 0', abs(g_of) < 1e-3, f'{g_of:.3e}')
legs_root = list(range(0, 22))
pp = perturb(gtp, legs_root)
p_lg, p_of = f(pp, True), f(pp, False)
check('GT legs+root perturbed: leg-GT FGD unchanged (bit-exact)', p_lg == g_lg, f'{p_lg:.6e} vs {g_lg:.6e}')
check('GT legs+root perturbed: official FGD changes', p_of > g_of + 0.05, f'official {g_of:.4f} -> {p_of:.4f}')
up = perturb(gtp, list(range(22, 39)))
u_lg = f(up, True)
check('GT upper-body perturbed: leg-GT FGD moves', u_lg > g_lg + 0.05, f'{g_lg:.4f} -> {u_lg:.4f}')

if pred_dir:
    pr = L.load_pairs(pred_dir, gt_dir, stems)
    r_lg, r_of = f(pr, True), f(pr, False)
    rp = perturb(pr, legs_root, seed=1)
    rp_lg, rp_of = f(rp, True), f(rp, False)
    check('pred legs+root perturbed: leg-GT FGD unchanged (bit-exact)', rp_lg == r_lg, f'{rp_lg:.6e} vs {r_lg:.6e}')
    check('pred legs+root perturbed: official FGD changes', abs(rp_of - r_of) > 0.05, f'official {r_of:.4f} -> {rp_of:.4f}')
    print(f'pred: official FGD {r_of:.4f}, leg-GT FGD {r_lg:.4f}')
    if bench:
        b = json.load(open(bench))['metrics']['FGD']
        check('pred official FGD == official evaluator json', abs(r_of - b) < 1e-6, f'{r_of:.6f} vs json {b:.6f}')
print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)

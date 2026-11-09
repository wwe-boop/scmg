"""r3d gates 1-2 (criteria "Decoder-only fine-tune r3d", 600946f): mu-decode of the 43 dev GT clips with r3 and the candidate.
    SCMG_J=34 python r3d_gates.py --cand r3d/final.pt --out G12.json [--r3 vae-r3/final.pt]"""
import argparse, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from spikes_where import stat, where, ARM_E
B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; ST = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/STATS_v36_1.json'
E = B + '/ex002-sfscope-1108/enc-s0'; UB = list(range(2, 34))


def recon(path, gt, dev='cuda'):
    from scmg.vae import MotionVAE
    st = json.load(open(ST)); qm = torch.tensor(st['q_mean'], device=dev).float(); qs = torch.tensor(st['q_std'], device=dev).float()
    ck = torch.load(path, map_location='cpu'); v = MotionVAE(**ck['cfg']).to(dev); v.load_state_dict(ck['ema']); v.eval(); out = []; l1 = []
    for y in gt:
        T = len(y); Tp = -(-T // 4) * 4; x = np.concatenate([y, np.repeat(y[-1:], Tp - T, 0)]) if Tp > T else y
        with torch.no_grad():
            xn = (torch.as_tensor(x, device=dev).float() - qm) / qs; mu, _ = v.encode(xn[None]); r = v.decode(mu)[0]
        l1.append(float((r[:T] - xn[:T]).abs().mean())); out.append((r[:T] * qs + qm).cpu().numpy().astype(np.float64))
    return out, float(np.mean(l1)), [p.detach().cpu() for p in v.enc.parameters()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cand', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--r3', default=B + '/trained/vae-r3/final.pt'); a = ap.parse_args()
    gt = [np.load(f'{E}/offline/gt_{i}.npy').astype(np.float64) for i in range(len(glob.glob(E + '/offline/gt_*.npy')))]
    g, _ = stat(gt, 50, ARM_E); thr = g['p99']; R = dict(GT=g)
    r3, l3, e3 = recon(a.r3, gt); rc, lc, ec = recon(a.cand, gt)
    R['encoder_identical'] = all(torch.equal(p, q) for p, q in zip(e3, ec))
    for lab, rr, l in (('r3', r3, l3), ('cand', rc, lc)):
        d, da = stat(rr, 50, ARM_E, thr); w = where(rr, da, thr)
        R[lab] = dict(l1_norm=l, rad_upper=float(np.mean([np.abs(r[:, UB] - y[:, UB]).mean() for r, y in zip(rr, gt)])), **d,
                      phase4=w['phase4_rel_rate'], joint_share=w['joint_share'])
    c, b = R['cand'], R['r3']
    R['gate1'] = bool(c['l1_norm'] <= 1.02 * b['l1_norm'] and c['rad_upper'] <= 1.02 * b['rad_upper'])
    R['gate2'] = bool(c['win_p95'] <= 1.2 * g['win_p95'] and c['share_gt_GTp99'] <= 0.015 and all(0.9 <= x <= 1.1 for x in c['phase4']))
    R['gate12'] = R['gate1'] and R['gate2'] and R['encoder_identical']
    open(a.out, 'w').write(json.dumps(R, indent=1))
    print('R3D_G12', 'PASS' if R['gate12'] else 'FAIL', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k != 'joint_share'}), 'r3', json.dumps({k: round(b[k], 4) for k in ('l1_norm', 'rad_upper', 'win_p95', 'share_gt_GTp99')}))


if __name__ == '__main__':
    main()

"""Minimal-change safety projection of the POST-INTERRUPTION motion (11-05, G1 interruption benchmark).

Only frames >= f_i (the motion the interruption handling produced) are changed; everything before t_i is copied bit for
bit, so the uninterrupted output and the pre-interruption motion are untouched.
For each <stem>@<frac>.pt (30 fps robot_repr) the joint correction d [n, 29] on frames f_i.. minimises
    sum |d|^2 + B * sum |second difference of d|^2 (d = 0 on the two frames before f_i: no jump, no kink)
    + LAM * sum_t (safe_lim + safe_lr + safe_torso)(q + d)        (scmg/g1_safety.py contract terms, margins below)
with Adam, then joints are clamped into [lower + 1e-3, upper - 1e-3] (URDF). Collision proxies use enlarged distances
(D_LR 0.12 m, D_T 0.16 m instead of the calibrated 0.10 / 0.14) so the projected motion clears the MuJoCo mesh contacts,
plus a wrist-hip proxy (--d-hip, the contacts the retraction to the arms-down rest pose produces).
    python scripts/interrupt/safety_project.py --in-dir D --out-dir D_safe --interrupts I.json [--every n]
Run with PYTHONPATH=<ECHO-G official>/src (torch FK); GPU if available.
"""
import argparse, json, os, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import g1_safety as S

ap = argparse.ArgumentParser()
ap.add_argument('--in-dir', required=True); ap.add_argument('--out-dir', required=True); ap.add_argument('--interrupts', required=True)
ap.add_argument('--every', type=int, default=1); ap.add_argument('--iters', type=int, default=200); ap.add_argument('--lam', type=float, default=200.)
ap.add_argument('--shard', default='0/1');
ap.add_argument('--d-hip', type=float, default=0.13, help='wrist-hip proxy distance (m), 0 = off')
ap.add_argument('--smooth', type=float, default=30.); ap.add_argument('--d-lr', type=float, default=0.12); ap.add_argument('--d-t', type=float, default=0.16)
a = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
rows = json.loads(Path(a.interrupts).read_text())['interrupts']
keep = set(sorted({r['stem'] for r in rows})[::a.every])
_si, _sn = (int(v) for v in a.shard.split('/')); keep = set(sorted(keep)[_si::_sn])
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
S.D_LR, S.D_T = a.d_lr, a.d_t                                # module constants read inside safety_terms
c = S._consts(dev, torch.float32); lo, hi = c['lo'] + 1e-3, c['hi'] - 1e-3
G1K = S._mods()['G1K']; BN = list(G1K.G1_BFS_BODY_NAMES)
WR = {s: [BN.index(f'{s}_wrist_{p}_link') for p in ('pitch', 'yaw')] for s in ('left', 'right')}
HP = {s: [BN.index(f'{s}_hip_{p}_link') for p in ('pitch', 'roll')] for s in ('left', 'right')}


def hip_term(q):
    """wrist-hip proxy: (relu(D_H - d) / D_H)^2, d = min same-side distance between the wrist pitch / yaw and the hip pitch /
    roll body origins (MuJoCo contacts of the post-interruption motion are wrist-thigh: d <= 0.107 m at 95 % of the
    colliding windows, >= 0.126 m at 95 % of the clean ones; dev diagnosis 11-05)"""
    if a.d_hip <= 0: return torch.zeros(q.shape[0], device=q.device)
    B = q.shape[0]; eye = torch.eye(3, device=q.device).expand(1, B, 3, 3)
    pos, _ = G1K.g1_forward_kinematics_with_rotations(q[None, :, 10:], eye, torch.zeros(1, B, 3, device=q.device))
    d = torch.stack([(pos[0][:, WR[s]][:, :, None] - pos[0][:, HP[s]][:, None]).norm(dim=-1).flatten(1).min(1).values for s in WR], 1).min(1).values
    return (torch.relu(a.d_hip - d) / a.d_hip) ** 2
n_done = n_changed = 0; tot_rmse = 0.
for r in rows:
    if r['stem'] not in keep: continue
    src = Path(a.in_dir) / f"{r['stem']}@{r['frac']:.1f}.pt"; dst = out / src.name
    if not src.exists() or dst.exists(): continue
    x = torch.load(src, map_location='cpu', weights_only=False)
    rr = x['robot_repr'].float().to(dev); f = r['f_i']
    post = rr[f:].clone(); n = len(post)
    st = S.safety_terms(post[None]); bad = bool(((st['lim'] > 0) | (st['lr'] > 0) | (st['torso'] > 0)).any() or (hip_term(post) > 0).any())
    newpost = post.clone()
    if bad:
        d = torch.zeros(n, 29, device=dev, requires_grad=True); opt = torch.optim.Adam([d], lr=0.01)
        for it in range(a.iters):
            q = post.clone(); q[:, 10:] = post[:, 10:] + d
            t = S.safety_terms(q[None]); dd = torch.cat([torch.zeros(2, 29, device=dev), d], 0)
            acc = dd[2:] - 2 * dd[1:-1] + dd[:-2]
            loss = (d ** 2).sum() + a.smooth * (acc ** 2).sum() + a.lam * (t['lim'] + t['lr'] + t['torso']).sum() + a.lam * hip_term(q).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        newpost[:, 10:] = post[:, 10:] + d.detach()
    newpost[:, 10:] = torch.maximum(torch.minimum(newpost[:, 10:], hi), lo)
    ch = float((newpost - post).pow(2).mean().sqrt()); tot_rmse += ch; n_changed += ch > 0
    y = rr.clone(); y[f:] = newpost
    x['robot_repr'] = y.cpu(); x.setdefault('generator', {})['safety_projection'] = dict(lam=a.lam, smooth=a.smooth, iters=a.iters,
        d_lr=a.d_lr, d_t=a.d_t, d_hip=a.d_hip, rmse=ch, from_frame=f)
    torch.save(x, dst.with_suffix('.pt.tmp')); os.replace(dst.with_suffix('.pt.tmp'), dst); n_done += 1
print('PROJECT_DONE', n_done, 'changed', n_changed, 'mean rmse %.4f' % (tot_rmse / max(n_done, 1)))

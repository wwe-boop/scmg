"""P2 sem_cf (11-09; docs/criteria/P2_SEMCF_1109.md on exp/text-path): the 16-D kinematic descriptor of a gesture window, one
torch function used on GT, on generations and in the training loss (differentiable in the joint angles).
Window W = 61 frames (t - 0.6 .. t + 0.6 s at 50 Hz); exact EX002 FK (scmg.ex002_ctl.points_t): wrist + shoulder of both hands.
h = commanded hand, o = the other, r_t = wrist_h - shoulder_h (base frame: x forward, y left, z up).
  0 amp_h  1 amp_o  2 disp  3 vpeak_h  4 vmean_h  5 vmean_o  6 tpeak  7 z_end  8 z_mean  9 x_mean  10 y_abs  11 iw  12 path
  13-15 dir = unit(mean_t (r_t - r_0))
D = [(d[:13] - mu) / sd, dir]; E(a, b) = |D(a) - D(b)|^2 / 16."""
import numpy as np
import torch
import torch.nn.functional as F

FPS, HALF = 50, 0.6
NW = int(round(2 * HALF * FPS)) + 1          # 61
NS, ND = 13, 16
NAMES = ['amp_h', 'amp_o', 'disp', 'vpeak_h', 'vmean_h', 'vmean_o', 'tpeak', 'z_end', 'z_mean', 'x_mean', 'y_abs', 'iw', 'path',
         'dir_x', 'dir_y', 'dir_z']


def _speed(p):                                   # p [N, T, 3] -> [N, T - 1] m/s
    return (p[:, 1:] - p[:, :-1]).norm(dim=-1) * FPS


def descriptor_pts(wrist, shoulder, hand):
    """wrist / shoulder [N, T, 2, 3] (torch), hand [N] long (0 L, 1 R) -> raw descriptor [N, 16] (dir unit)."""
    N, T = wrist.shape[:2]; ar = torch.arange(N, device=wrist.device)
    wh, wo = wrist[ar, :, hand], wrist[ar, :, 1 - hand]                                   # [N, T, 3]
    r = wh - shoulder[ar, :, hand]; ro = wo - shoulder[ar, :, 1 - hand]
    amp = lambda x: 2 * ((x - x.mean(1, keepdim=True)).square().sum(-1).mean(1) + 1e-8).sqrt()
    dm = (r - r[:, :1]).mean(1)                                                         # [N, 3]
    v = _speed(r); vo = _speed(ro); tau = 0.05
    vpeak = tau * (torch.logsumexp(v / tau, 1) - np.log(v.shape[1]))
    tt = torch.linspace(-1, 1, v.shape[1], device=v.device)
    tpeak = (torch.softmax(v / tau, 1) * tt).sum(1)
    cols = [amp(r), amp(ro), (dm.square().sum(-1) + 1e-8).sqrt(), vpeak, v.mean(1), vo.mean(1), tpeak, r[:, -5:, 2].mean(1),
            r[:, :, 2].mean(1), r[:, :, 0].mean(1), (r[:, :, 1].square() + 1e-4).sqrt().mean(1),
            (wrist[:, :, 0] - wrist[:, :, 1]).norm(dim=-1).mean(1), (r[:, 1:] - r[:, :-1]).norm(dim=-1).sum(1)]
    return torch.cat([torch.stack(cols, 1), F.normalize(dm, dim=-1, eps=1e-6)], 1)


def descriptor_q(q, hand):
    """q [N, T, 34] physical rad (torch, differentiable) -> raw descriptor [N, 16]."""
    from .ex002_ctl import points_t
    N, T = q.shape[:2]; g = points_t(q.reshape(N * T, -1))
    return descriptor_pts(g['wrist'].view(N, T, 2, 3), g['shoulder'].view(N, T, 2, 3), hand)


def dominant_hand(q):
    """numpy [T, 34] -> 0 / 1: the hand with the larger wrist-path diameter (APEX.json rule)."""
    from .ex002_ctl import points
    w = points(np.asarray(q, np.float32))['wrist']
    diam = [float(np.linalg.norm(w[:, None, h] - w[None, :, h], axis=-1).max()) for h in (0, 1)]
    return int(np.argmax(diam))


def window(q, t, fps=FPS):
    """numpy [T, 34] -> the NW-frame window centred on t (edge-held)."""
    c = int(round(t * fps)); idx = np.clip(np.arange(c - NW // 2, c + NW // 2 + 1), 0, len(q) - 1)
    return np.asarray(q, np.float32)[idx]


def standardise(d, mu, sd):
    """raw [..., 16] -> D [..., 16]: z-scored scalars, unit direction."""
    mu = torch.as_tensor(mu, dtype=d.dtype, device=d.device); sd = torch.as_tensor(sd, dtype=d.dtype, device=d.device)
    return torch.cat([(d[..., :NS] - mu) / sd, d[..., NS:]], -1)


def error(Da, Db):
    """standardised [..., 16] x2 -> E [...] = |Da - Db|^2 / 16."""
    return (Da - Db).square().sum(-1) / ND


def np_desc(qwin, hand):
    """numpy convenience: [NW, 34] window(s) -> raw [16] (or [N, 16] for [N, NW, 34])."""
    q = torch.as_tensor(np.asarray(qwin, np.float32)); one = q.dim() == 2
    if one: q = q[None]
    h = torch.as_tensor(np.atleast_1d(hand), dtype=torch.long).expand(q.shape[0])
    with torch.no_grad(): d = descriptor_q(q, h).numpy()
    return d[0] if one else d

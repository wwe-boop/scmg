"""Motion VAE: normalised joint frames [B, T, J] (50 Hz) <-> latents [B, T/4, Z] (12.5 Hz).

Encoder is strictly causal (latent i sees frames < 4(i+1)). Decoder sees latents <= i+1 for frames of latent i
(one-latent = 80 ms lookahead), so in streaming a chunk's last latent is finalised when the next chunk's first latent
is committed. Pretrained once (stage 0) and frozen afterwards ("pretrained Motion Decoder").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import J, STRIDE


class CConv(nn.Conv1d):
    """Conv1d on [B, C, T] with left padding (causal) or `future` frames of right context."""

    def __init__(self, cin, cout, k, stride=1, future=0):
        super().__init__(cin, cout, k, stride)
        self.lp, self.rp = k - 1 - future, future

    def forward(self, x):
        return super().forward(F.pad(x, (self.lp, self.rp), mode='replicate'))


class ChanNorm(nn.Module):
    """LayerNorm over channels of each frame. (GroupNorm would also normalise over TIME and erase the window's
    absolute pose -- the first VAE run could not reconstruct positions because of it.)"""

    def __init__(self, c):
        super().__init__()
        self.n = nn.LayerNorm(c)

    def forward(self, x):
        return self.n(x.transpose(1, 2)).transpose(1, 2)


class Res(nn.Module):
    def __init__(self, c, k=3, future=0):
        super().__init__()
        self.c1 = CConv(c, c, k, future=future); self.c2 = CConv(c, c, 1)
        self.n = ChanNorm(c)

    def forward(self, x):
        return x + self.c2(F.silu(self.c1(F.silu(self.n(x)))))


class MotionVAE(nn.Module):
    """Plain causal conv VAE (r3). The first design (residual stacks + stride-2 pairs) got stuck outputting the global
    mean pose even when over-fitting one batch (debug_vae*.py); this flat design fits it (L1 0.11 in 300 steps)."""

    def __init__(self, z=16, c=384, stride=STRIDE):
        super().__init__()
        if stride != STRIDE: raise ValueError(f'VAE built for SCMG_STRIDE={stride}, environment has {STRIDE}')
        self.z, self.stride = z, stride
        self.enc = nn.Sequential(CConv(J, c, 3), nn.SiLU(), CConv(c, c, 3), nn.SiLU(),
                                 nn.Conv1d(c, c, STRIDE, stride=STRIDE),          # latent i <- frames 4i..4i+3
                                 nn.SiLU(), CConv(c, c, 3), nn.SiLU(), CConv(c, c, 3), nn.SiLU(), nn.Conv1d(c, 2 * z, 1))
        with torch.no_grad(): self.enc[-1].bias[z:].fill_(-6.)       # small initial posterior noise (std ~0.05)
        self.dec = nn.Sequential(CConv(z, c, 3, future=1), nn.SiLU(), CConv(c, c, 3), nn.SiLU(), CConv(c, c, 3), nn.SiLU(),
                                 nn.ConvTranspose1d(c, c, STRIDE, stride=STRIDE),  # latent i -> frames 4i..4i+3
                                 nn.SiLU(), CConv(c, c, 3), nn.SiLU(), CConv(c, c, 3), nn.SiLU(), CConv(c, J, 3))

    def encode(self, x):
        """x [B, T, J] (T multiple of 4) -> mu, logvar [B, T/4, Z]."""
        mu, lv = self.enc(x.transpose(1, 2)).chunk(2, 1)
        return mu.transpose(1, 2), lv.clamp(-12, 6).transpose(1, 2)

    def decode(self, z):
        """z [B, N, Z] -> x [B, 4N, J]."""
        return self.dec(z.transpose(1, 2)).transpose(1, 2)

    def forward(self, x, mask=None, beta=1e-3, w=None, deriv=1.0):
        """deriv in [0, 1]: weight ramp of the velocity/acceleration terms. At full weight from step 0 they dominate the
        noisy initial output and the net escapes them by emitting a constant pose (stuck at the mean, KL collapses)."""
        mu, lv = self.encode(x)
        z = mu + deriv * torch.randn_like(mu) * (0.5 * lv).exp()      # sampling noise ramps in with the same schedule
        y = self.decode(z)
        losses = recon_losses(y, x, mask, w)
        kl = 0.5 * (mu.square() + lv.exp() - 1 - lv).sum(-1).mean()
        losses['kl'] = kl
        losses['loss'] = losses['rec'] + deriv * (0.3 * losses['vel'] + 0.1 * losses['acc'] + beta * kl)
        return y, losses


def recon_losses(y, x, mask=None, w=None):
    """L1 on positions / velocities / accelerations, optionally masked [B, T] and joint-weighted [J]."""
    w = torch.ones(x.shape[-1], device=x.device) if w is None else w
    m = torch.ones(x.shape[:2], device=x.device) if mask is None else mask.float()

    def l1(a, b, mm):
        return ((a - b).abs() * w).mean(-1).mul(mm).sum() / mm.sum().clamp(min=1)
    dv = lambda t: t[:, 1:] - t[:, :-1]
    mv = m[:, 1:] * m[:, :-1]; ma = mv[:, 1:] * mv[:, :-1]
    return dict(rec=l1(y, x, m), vel=l1(dv(y), dv(x), mv) * 10., acc=l1(dv(dv(y)), dv(dv(x)), ma) * 30.)

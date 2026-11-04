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

    def __init__(self, z=16, c=384, stride=STRIDE, beta=1e-3, w_vel=0.3, w_acc=0.1, w_accmag=0.0, dim_mask=(), j=None, ch_w=(),
                 identity=False):
        super().__init__()
        # 11-04 'identity' codec (ECHO-G generates in the raw robot space; our VAE alone doubles contact sliding and removes a
        # quarter of the jerk): a latent is the STRIDE consecutive normalised frames concatenated, z = STRIDE * J, lossless and
        # strictly causal (latent i = frames S*i .. S*i+S-1; the conv VAE decoder looks one latent ahead). No parameters.
        self.identity = bool(identity)
        if self.identity:
            if j is not None and j != J: raise ValueError(f'codec built for {j} motion channels, environment has SCMG_J={J}')
            if stride != STRIDE: raise ValueError(f'codec built for SCMG_STRIDE={stride}, environment has {STRIDE}')
            if z != STRIDE * J: raise ValueError(f'identity codec needs z = STRIDE * J = {STRIDE * J}, got {z}')
            if dim_mask or ch_w: raise ValueError('identity codec: dim_mask / ch_w do not apply')
            self.z, self.stride, self.c, self.dim_mask, self.ch_w = z, stride, c, (), ()
            return
        if j is not None and j != J: raise ValueError(f'VAE built for {j} motion channels, environment has SCMG_J={J}')
        if stride != STRIDE: raise ValueError(f'VAE built for SCMG_STRIDE={stride}, environment has {STRIDE}')
        self.z, self.stride, self.c = z, stride, c
        # 11-04 ECHO-G G1 'leg masking' lane (default off = bit-identical): these channels are zeroed (= the training mean in
        # normalised units) at the encoder input, carry no reconstruction loss, and are pinned to 0 at the decoder output,
        # so the latents never encode them and every generated frame has the mean pose there (saved in the checkpoint cfg)
        self.dim_mask = tuple(int(i) for i in dim_mask)
        if self.dim_mask:
            if min(self.dim_mask) < 0 or max(self.dim_mask) >= J: raise ValueError(f'dim_mask outside 0..{J - 1}')
            keep = torch.ones(J); keep[list(self.dim_mask)] = 0.
            self.register_buffer('dim_keep', keep, persistent=False)
        self.beta, self.w_vel, self.w_acc, self.w_accmag = beta, w_vel, w_acc, w_accmag     # training-loss weights (11-03 variants)
        # 11-04 G1 contact sliding: per-channel loss weights [[channel, weight], ...] (default none = bit-identical), e.g.
        # legs 10..21 and root velocity 7..9 up-weighted -- the VAE alone doubles contact sliding (0.043 -> 0.086)
        self.ch_w = tuple((int(c_), float(w_)) for c_, w_ in ch_w)
        if self.ch_w:
            cw = torch.ones(J)
            for c_, w_ in self.ch_w:
                if not 0 <= c_ < J or w_ < 0: raise ValueError(f'ch_w entry {(c_, w_)}')
                cw[c_] = w_
            self.register_buffer('ch_weight', cw / cw.mean(), persistent=False)   # mean 1: the total loss scale is unchanged
        self.enc = nn.Sequential(CConv(J, c, 3), nn.SiLU(), CConv(c, c, 3), nn.SiLU(),
                                 nn.Conv1d(c, c, STRIDE, stride=STRIDE),          # latent i <- frames 4i..4i+3
                                 nn.SiLU(), CConv(c, c, 3), nn.SiLU(), CConv(c, c, 3), nn.SiLU(), nn.Conv1d(c, 2 * z, 1))
        with torch.no_grad(): self.enc[-1].bias[z:].fill_(-6.)       # small initial posterior noise (std ~0.05)
        self.dec = nn.Sequential(CConv(z, c, 3, future=1), nn.SiLU(), CConv(c, c, 3), nn.SiLU(), CConv(c, c, 3), nn.SiLU(),
                                 nn.ConvTranspose1d(c, c, STRIDE, stride=STRIDE),  # latent i -> frames 4i..4i+3
                                 nn.SiLU(), CConv(c, c, 3), nn.SiLU(), CConv(c, c, 3), nn.SiLU(), CConv(c, J, 3))

    def encode(self, x):
        """x [B, T, J] (T multiple of 4) -> mu, logvar [B, T/4, Z]."""
        if self.identity:
            B, T, _ = x.shape
            if T % STRIDE: raise ValueError(f'identity codec: {T} frames not a multiple of {STRIDE}')
            mu = x.reshape(B, T // STRIDE, STRIDE * J)
            return mu, torch.full_like(mu, -12.)
        if self.dim_mask: x = x * self.dim_keep
        mu, lv = self.enc(x.transpose(1, 2)).chunk(2, 1)
        return mu.transpose(1, 2), lv.clamp(-12, 6).transpose(1, 2)

    def decode(self, z):
        """z [B, N, Z] -> x [B, 4N, J]."""
        if self.identity: return z.reshape(z.shape[0], z.shape[1] * STRIDE, J)
        y = self.dec(z.transpose(1, 2)).transpose(1, 2)
        return y * self.dim_keep if self.dim_mask else y

    def forward(self, x, mask=None, beta=None, w=None, deriv=1.0):
        """deriv in [0, 1]: weight ramp of the velocity/acceleration terms. At full weight from step 0 they dominate the
        noisy initial output and the net escapes them by emitting a constant pose (stuck at the mean, KL collapses)."""
        if self.identity: raise RuntimeError('the identity codec has nothing to train')
        mu, lv = self.encode(x)
        z = mu + deriv * torch.randn_like(mu) * (0.5 * lv).exp()      # sampling noise ramps in with the same schedule
        y = self.decode(z)
        if self.dim_mask and w is None: w = self.dim_keep
        if self.ch_w: w = self.ch_weight if w is None else w * self.ch_weight
        losses = recon_losses(y, x, mask, w)
        kl = 0.5 * (mu.square() + lv.exp() - 1 - lv).sum(-1).mean()
        losses['kl'] = kl
        beta = self.beta if beta is None else beta
        losses['loss'] = losses['rec'] + deriv * (self.w_vel * losses['vel'] + self.w_acc * losses['acc'] + beta * kl
                                                  + self.w_accmag * losses['accmag'])
        return y, losses


def recon_losses(y, x, mask=None, w=None):
    """L1 on positions / velocities / accelerations, optionally masked [B, T] and joint-weighted [J]."""
    w = torch.ones(x.shape[-1], device=x.device) if w is None else w
    m = torch.ones(x.shape[:2], device=x.device) if mask is None else mask.float()

    def l1(a, b, mm):
        return ((a - b).abs() * w).mean(-1).mul(mm).sum() / mm.sum().clamp(min=1)
    dv = lambda t: t[:, 1:] - t[:, :-1]
    mv = m[:, 1:] * m[:, :-1]; ma = mv[:, 1:] * mv[:, :-1]
    ay, ax = dv(dv(y)), dv(dv(x))
    out = dict(rec=l1(y, x, m), vel=l1(dv(y), dv(x), mv) * 10., acc=l1(ay, ax, ma) * 30.)
    # 11-03: per-frame acceleration MAGNITUDE (arms+hands energy, sign-free): signed L1 lets the decoder smooth short beat
    # strokes to ~0 acceleration; matching |acc| keeps their energy at the right frames (beat_align depends on it)
    nm = lambda t: (t * w).norm(dim=-1)
    out['accmag'] = ((nm(ay) - nm(ax)).abs() * ma).sum() / ma.sum().clamp(min=1) * 30.
    return out

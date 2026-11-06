"""11-06 Div/BA line: training terms on the decoded [ctx | target] frames of the local flow (one-step estimate x1hat).

Diagnosis (scripts/baselines_g1/divba): generated motion is too smooth -- speed power above 2 Hz 0.25-0.5x GT while the VAE
round trip keeps 0.9-1.5x -- so the latents the generator produces are too smooth. Both terms below use only the training
batch (decoded GT frames, the batch's own prosody onset channel); no evaluator, recogniser or dev/test data.

spec_loss  (a') two-sided spectrum matching: Hann-windowed rFFT of the upper-body (waist + arms) joint velocity over the
           32-frame [ctx | target] window (50 Hz, 1.56 Hz bins); band powers 1.5-3 / 3-6 / 6-12 Hz aggregated over the valid
           instances whose noise level t >= t_min (low-t one-step estimates are conditional means and SHOULD be smooth);
           loss = sum_b (log P_pred_b - log P_gt_b)^2 on the band SHARES of the 0.5-12 Hz power (shape, scale-free) plus the
           same on the total power with weight level_w (level kept close, two-sided: neither smoother nor busier than GT).
onset_loss (b') onset-conditioned motion beats, density-normalised: dip(f) = relu(local mean speed (+-k frames) - speed(f)) /
           mean speed of the instance (scale-free); R = mean dip at onset frames (weights = onset channel, clamped >= 0) minus
           mean dip over all valid frames, i.e. how much more the motion pauses ON onsets than anywhere else (adding pauses
           everywhere does not raise R). loss = relu(R_gt - R_pred), batch-aggregated (one-sided: up to the GT level).
"""
import torch
import torch.nn.functional as F

UPPER = list(range(22, 39))          # G1 robot_repr channels: waist (22-24) + arms (25-38)
BANDS = ((1.5, 3.0), (3.0, 6.0), (6.0, 12.0))


def _vel(y, fps):                     # [M, T, C] rad -> [M, T-1, C] rad/s
    return (y[:, 1:] - y[:, :-1]) * fps


def band_powers(y, w, fps=50.0, bands=BANDS, lo=0.5, hi=12.0):
    """y [M, T, C] (rad), w [M] instance weights -> (band powers [nb], total power in [lo, hi)) aggregated over instances."""
    v = _vel(y, fps); v = v - v.mean(1, keepdim=True)
    n = v.shape[1]; win = torch.hann_window(n, periodic=False, device=v.device, dtype=v.dtype)
    P = torch.fft.rfft(v * win[None, :, None], dim=1).abs().square().sum(-1)        # [M, n//2+1]
    f = torch.fft.rfftfreq(n, 1.0 / fps).to(v.device)
    Pw = (P * w[:, None]).sum(0) / w.sum().clamp(min=1e-6)
    bp = torch.stack([Pw[(f >= a) & (f < b)].sum() for a, b in bands])
    tot = Pw[(f >= lo) & (f < hi)].sum()
    return bp, tot


def spec_loss(y, gt, inst_w, level_w=0.5, eps=1e-8):
    """y, gt [M, T, J] in radians (decoded prediction / decoded-or-raw GT frames of the same windows); inst_w [M] (0 = skip)."""
    yu, gu = y[..., UPPER].float(), gt[..., UPPER].float()
    bp, tp = band_powers(yu, inst_w); bg, tg = band_powers(gu, inst_w)
    shape = ((bp / (tp + eps) + eps).log() - (bg / (tg + eps) + eps).log()).square().sum()
    level = ((tp + eps).log() - (tg + eps).log()).square()
    return dict(spec=shape + level_w * level, spec_shape=shape.detach(), spec_level=level.detach(),
                spec_hf_ratio=(bp[1:].sum() / (bg[1:].sum() + eps)).detach())


def _dip_ratio(y, onset, fw, k=4, eps=1e-6):
    """y [M, T, C] rad; onset [M, T] >= 0; fw [M, T] frame validity -> (sum dip*onset, sum onset, sum dip*fw, sum fw)."""
    s = _vel(y, 50.0).norm(dim=-1)                                                   # [M, T-1] speed
    s = torch.cat([s[:, :1], s], 1)                                                  # align to frames
    loc = F.avg_pool1d(s[:, None], 2 * k + 1, stride=1, padding=k, count_include_pad=False)[:, 0]
    ms = (s * fw).sum(1, keepdim=True) / fw.sum(1, keepdim=True).clamp(min=1)
    dip = F.relu(loc - s) / (ms + eps)
    ow = onset * fw
    return (dip * ow).sum(), ow.sum(), (dip * fw).sum(), fw.sum()


def onset_loss(y, gt, onset, fw, k=4):
    """y, gt [M, T, J] rad; onset [M, T] onset strength (>= 0, training audio); fw [M, T] frame weights (0 = skip)."""
    yu, gu = y[..., UPPER].float(), gt[..., UPPER].float()
    a, b, c, d = _dip_ratio(yu, onset, fw, k); ag, bg, cg, dg = _dip_ratio(gu, onset, fw, k)
    Rp = a / b.clamp(min=1e-6) - c / d.clamp(min=1e-6)
    Rg = ag / bg.clamp(min=1e-6) - cg / dg.clamp(min=1e-6)
    return dict(onset=F.relu(Rg.detach() - Rp), onset_R=Rp.detach(), onset_R_gt=Rg.detach())

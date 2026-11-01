"""Shared building blocks: attention with explicit q/k/v Linear layers (so LoRA can wrap them), AdaLN transformer
blocks with optional cross-attention, LoRA adapters, sinusoidal embeddings."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos(t, dim, max_period=10000.):
    """t [...] float -> [..., dim] sinusoidal embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[..., None] * freqs
    return torch.cat([a.cos(), a.sin()], -1)


class LoRALinear(nn.Module):
    """y = W x + (alpha/r) B A x ; base W frozen by apply_lora, B zero-initialised (exact identity at start)."""

    def __init__(self, base: nn.Linear, r=16, alpha=16.):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def apply_lora(module, names=('q', 'k', 'v', 'o', 'fc1', 'fc2'), r=16, alpha=16.):
    """Replace every nn.Linear child whose attribute name is in `names` (recursively) by a LoRALinear."""
    n = 0
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if name in names and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, r, alpha)); n += 1
    return n


class Attention(nn.Module):
    def __init__(self, d, heads, kv_dim=None):
        super().__init__()
        kv_dim = kv_dim or d
        self.h = heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(kv_dim, d); self.v = nn.Linear(kv_dim, d); self.o = nn.Linear(d, d)

    def forward(self, x, kv=None, mask=None):
        """mask: bool [B or 1, Lq, Lk] or [B, 1, Lk]; True = attend. Rows with no valid key output 0."""
        kv = x if kv is None else kv
        B, Lq, D = x.shape
        q = self.q(x).view(B, Lq, self.h, -1).transpose(1, 2)
        k = self.k(kv).view(B, kv.shape[1], self.h, -1).transpose(1, 2)
        v = self.v(kv).view(B, kv.shape[1], self.h, -1).transpose(1, 2)
        m = None
        if mask is not None:
            m = mask[:, None]                                             # [B,1,Lq,Lk]
            empty = ~m.any(-1, keepdim=True)
            m = m | empty                                                 # avoid NaN rows; zeroed below
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        y = y.transpose(1, 2).reshape(B, Lq, D)
        if mask is not None:
            y = y.masked_fill(empty[:, 0].expand(B, Lq, 1), 0.)
        return self.o(y)


class MLP(nn.Module):
    def __init__(self, d, mult=4, out=None):
        super().__init__()
        self.fc1 = nn.Linear(d, d * mult); self.fc2 = nn.Linear(d * mult, out or d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


DROPOUT = 0.1          # residual-branch dropout (r4: the r1 base overfit after ~6k steps of a 1024-window batch)


class Block(nn.Module):
    """Pre-norm block: self-attn -> (cross-attn) -> MLP, each modulated by AdaLN(c) (shift, scale, gate).
    c: [B, d] global or [B, L, d] per token. Gates start at 0 for the AdaLN projection (identity block at init)."""

    def __init__(self, d, heads, cross=False, mem_dim=None):
        super().__init__()
        self.drop = nn.Dropout(DROPOUT)
        self.n1 = nn.LayerNorm(d, elementwise_affine=False); self.sa = Attention(d, heads)
        self.cross = cross
        if cross:
            self.n2 = nn.LayerNorm(d, elementwise_affine=False); self.ca = Attention(d, heads, mem_dim or d)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False); self.ff = MLP(d)
        self.ada = nn.Linear(d, 9 * d)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, x, c, self_mask=None, mem=None, mem_mask=None):
        a = self.ada(F.silu(c))
        if a.ndim == 2: a = a[:, None]
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = a.chunk(9, -1)
        x = x + self.drop((1 + g1) * self.sa(self.n1(x) * (1 + s1) + b1, mask=self_mask))
        if self.cross and mem is not None:
            x = x + self.drop((1 + g2) * self.ca(self.n2(x) * (1 + s2) + b2, kv=mem, mask=mem_mask))
        return x + self.drop((1 + g3) * self.ff(self.n3(x) * (1 + s3) + b3))


class Encoder(nn.Module):
    """Plain transformer stack (no AdaLN condition: c = 0)."""

    def __init__(self, d, heads, layers, cross=False):
        super().__init__()
        self.d = d
        self.blocks = nn.ModuleList([Block(d, heads, cross) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask=None, mem=None, mem_mask=None, c=None):
        c = x.new_zeros(x.shape[0], self.d) if c is None else c
        for b in self.blocks: x = b(x, c, mask, mem, mem_mask)
        return self.norm(x)

"""SemREPA (11-09; docs/criteria/SEMREPA_1109.md on exp/text-path): REPA-style alignment of an early planner block to a frozen
semantic motion encoder (the teacher) applied to the GT motion around each target chunk. Training-time only: inference never
uses the teacher or the projector.

Teacher: 1.6 s windows (80 frames, 50 Hz) of the G1 view channels 22..38 (waist + both arms, physical), window mean removed,
divided by the train std, + first differences x 5; 4-frame patches -> Linear(136 -> 192) + learned positions -> 4 pre-norm
Transformer layers (4 heads, FFN 384, dropout 0.1). Heads: per-patch 5-class logits (none / beat / deictic / iconic / metaphoric),
window embedding e = LayerNorm(mean of the tokens) (the REPA target, L2-normalised) and Linear(192 -> 768) for the e5 cosine.
Trained by scripts/semantic/semrepa_teacher.py (G1 hos TRAIN ids only)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

CH = list(range(22, 39))
WIN, PATCH, NCLS, D, E5 = 80, 4, 5, 192, 768
CLASSES = ['none', 'beat', 'deictic', 'iconic', 'metaphoric']
_CACHE = {}


class Teacher(nn.Module):
    def __init__(self, std, win=WIN, d=D, layers=4, heads=4, p=0.1):
        super().__init__()
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32).clone())
        self.win, self.n = win, win // PATCH
        self.patch = nn.Linear(2 * len(CH) * PATCH, d)
        self.pos = nn.Parameter(torch.randn(1, self.n, d) * 0.02)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=p, batch_first=True, norm_first=True),
                                        layers, enable_nested_tensor=False)
        self.cls = nn.Linear(d, NCLS); self.norm = nn.LayerNorm(d); self.e5 = nn.Linear(d, E5)

    def forward(self, x):
        """x [B, win, 17] physical -> dict(logits [B, n, 5], emb [B, d] (not normalised), e5 [B, 768])."""
        x = x.float()
        z = (x - x.mean(1, keepdim=True)) / self.std
        v = torch.diff(z, dim=1, prepend=z[:, :1]) * 5.0
        f = torch.cat([z, v], -1)
        B = f.shape[0]
        h = self.tf(self.patch(f.reshape(B, self.n, PATCH * f.shape[-1])) + self.pos)
        e = self.norm(h.mean(1))
        return dict(logits=self.cls(h), emb=e, e5=self.e5(e))


def load_teacher(path, device):
    key = (str(path), str(device))
    if key not in _CACHE:
        s = torch.load(path, map_location='cpu', weights_only=False)
        t = Teacher(s['std'], **s.get('arch', {})); t.load_state_dict(s['model']); t.eval().requires_grad_(False)
        _CACHE[key] = t.to(device)
    return _CACHE[key]


def chunk_windows(qphys, K, chunk_frames=16, win=WIN):
    """qphys [B, T, J] physical -> [B, K, win, 17]: windows centred on chunk k's centre (16 k + 8), edge-held inside [0, T)."""
    T = qphys.shape[1]
    c = torch.arange(K, device=qphys.device) * chunk_frames + chunk_frames // 2
    idx = (c[:, None] - win // 2 + torch.arange(win, device=qphys.device)[None]).clamp(0, T - 1)       # [K, win]
    return qphys[:, idx][..., CH]


@torch.no_grad()
def teacher_targets(teacher, qphys, K, chunk_frames=16):
    """-> [B, K, d] L2-normalised teacher embeddings of the GT windows (fp32, autocast off)."""
    B = qphys.shape[0]
    with torch.autocast('cuda', enabled=False):
        w = chunk_windows(qphys.float(), K, chunk_frames, teacher.win)
        e = teacher(w.flatten(0, 1))['emb']
    return F.normalize(e.float(), dim=-1).view(B, K, -1)


def make_projector(d, out=D, seed=1109):
    """REPA projector d -> 2d -> 2d -> out (SiLU), default nn.Linear init drawn under a forked CPU RNG seeded with `seed`, so
    the global RNG stream (data order, noise, dropout) is exactly the control's."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, 2 * d), nn.SiLU(), nn.Linear(2 * d, out))

"""Landing requirements of the EX002 line (user 11-04 L1-L3; Claude 11-05). Inference-side only: nothing here changes
training or the default behaviour of any script (every switch is off unless set explicitly).

1. Condition override (SCMG_COND, read by scmg/evaluate.py, scripts/rollout_probe.py, scripts/gen_semantic.py)
   The dev evaluation conditions every clip on the TRUE speaker embedding and a reference motion of the same speaker
   (WindowDataset.sample), while deployment (scmg/deploy_stream.py) sends spk_valid = 0 and ref_valid = 0 (null style)
   with source = 5. SCMG_COND makes the evaluation run under deployment / persona conditions:
       SCMG_COND="spk=null,ref=null,source=5"         deployment conditions
       SCMG_COND="spk=null,ref=<persona.npy>,source=5" persona: one fixed reference motion for every sentence
   keys: spk=null|<[192] .npy>, ref=null|<[R, 34] .npy, NORMALISED with the training stats>, source=<int>,
   spk_alpha=<a in (0,1)> (after spk=<file>: blend a * speaker + (1 - a) * null in global_cond; inference only).
   NOTE: ref only matters for checkpoints trained with cfg.style=True; sibag / sibag-coll have style=False (ref ignored).
   Unset / empty = conditions untouched (bit-identical).

2. ArmEnvelope (L1 "upper arm within a human range, no wrist eversion")
   Stateless per-joint soft limit for the upper-arm joints 1-3 and the wrist joints 5-7 of both arms: identity inside
   the human knee interval [p_lo, p_hi] (clean BEAT2 train frames, scripts/landing/human_envelope.py), smooth tanh
   saturation towards the human bound beyond it. Properties (tests/test_landing.py): C1, non-decreasing, slope in [0, 1] (it can
   only slow a joint down, never add speed), output strictly inside the bound, zero latency (per frame, no state), so
   applying it to an exported gen_*.npy is exactly what the deployment stream would send.

3. Metrics (scripts/landing/landing_metrics.py)
   wrist eversion rate, envelope-exceedance rate, freeze segments while speaking (>= 1.5 s), stillness gap to the
   human, over-speed share, style descriptors for persona consistency.
"""
import json, os
import numpy as np

from .common import J, FPS, require_ex002

# ----------------------------------------------------------------------------------------------- 1. condition override


def cond_spec():
    return os.environ.get('SCMG_COND', '').strip()


def apply_cond(cond, spec=None):
    """Return a copy of `cond` with the SCMG_COND overrides applied (numpy or torch values; dtypes / shapes kept)."""
    spec = cond_spec() if spec is None else spec.strip()
    if not spec:
        return cond
    import torch
    c = dict(cond)

    def like(ref, arr):                       # same container type / dtype / device as the original entry
        if isinstance(ref, torch.Tensor):
            return torch.as_tensor(np.asarray(arr), dtype=ref.dtype, device=ref.device)
        return np.asarray(arr, dtype=np.asarray(ref).dtype)

    for kv in filter(None, (x.strip() for x in spec.split(','))):
        if '=' not in kv:
            raise ValueError(f'SCMG_COND item {kv!r}: expected key=value')
        k, v = (x.strip() for x in kv.split('=', 1))
        if k in ('spk', 'ref'):
            if k not in c or f'{k}_valid' not in c:
                raise KeyError(f'SCMG_COND {k}=...: cond has no {k} / {k}_valid')
            shape = tuple(np.shape(c[k]))
            if v == 'null':
                val, ok = np.zeros(shape, np.float32), 0.
            else:
                val = np.load(v).astype(np.float32); ok = 1.
                if val.shape != shape:
                    raise ValueError(f'SCMG_COND {k}={v}: shape {val.shape} != expected {shape}')
                if not np.isfinite(val).all():
                    raise ValueError(f'SCMG_COND {k}={v}: non-finite values')
            c[k] = like(c[k], val); c[f'{k}_valid'] = like(c[f'{k}_valid'], ok)
        elif k == 'spk_alpha':                  # blend a * speaker + (1 - a) * null speaker (scmg/model.py global_cond)
            al = float(v)
            if not 0 < al < 1: raise ValueError(f'SCMG_COND spk_alpha={v}: need 0 < a < 1 (0 / 1 = spk=null / spk=<file>)')
            c['spk_valid'] = like(c['spk_valid'], al)
        elif k == 'source':
            if 'source' not in c:
                raise KeyError('SCMG_COND source=...: cond has no source')
            c['source'] = like(c['source'], int(v))
        else:
            raise ValueError(f'SCMG_COND key {k!r}: supported are spk, ref, source, spk_alpha')
    return c


# ----------------------------------------------------------------------------------------------- 2. arm envelope

ARM_BASE = {'left': 4, 'right': 11}           # EX002 34-joint layout: arm joint k of side s = ARM_BASE[s] + k - 1


def arm_index(side, k):
    return ARM_BASE[side] + k - 1


UPPER = [arm_index(s, k) for s in ('left', 'right') for k in (1, 2, 3)]
WRIST = [arm_index(s, k) for s in ('left', 'right') for k in (5, 6, 7)]


def load_envelope(path):
    e = json.load(open(path))
    qs = e['quantiles']; q = np.asarray(e['pooled']['q'], np.float64)           # [18 joints, len(qs)]
    return e, qs, q


def _qcol(qs, q, p):
    if p not in qs:
        raise ValueError(f'quantile {p} not in the envelope file (has {qs})')
    return q[:, qs.index(p)]


def soft_limit(x, lo_k, hi_k, lo_b, hi_b):
    """Identity on [lo_k, hi_k]; beyond, k + (b - k) tanh((x - k) / (b - k)) -> strictly inside (lo_b, hi_b).
    C1 at the knee, non-decreasing, slope in [0, 1]."""
    x = np.asarray(x, np.float64)
    y = x.copy()
    w_hi = hi_b - hi_k; w_lo = lo_k - lo_b
    up = x > hi_k; dn = x < lo_k
    y = np.where(up, hi_k + w_hi * np.tanh((x - hi_k) / w_hi), y)
    y = np.where(dn, lo_k - w_lo * np.tanh((lo_k - x) / w_lo), y)
    return y


class ArmEnvelope:
    """Per-joint soft limit of the upper-arm (1-3) and wrist (5-7) joints, knee = human [100-knee, knee] quantile
    interval, bound = human [100-bound, bound] interval widened by `extra` rad on each side (> 0 so the knee width is
    never zero)."""

    def __init__(self, envelope_json, knee=99.0, bound=99.9, extra=0.05, joints='upper+wrist'):
        require_ex002('ArmEnvelope (EX002 arm joints)')
        if J != 34:
            raise RuntimeError('ArmEnvelope needs the 34-joint EX002 layout')
        if not (50 < knee < bound <= 100 and extra > 0):
            raise ValueError(f'ArmEnvelope: need 50 < knee < bound <= 100 and extra > 0 (got {knee}, {bound}, {extra})')
        e, qs, q = load_envelope(envelope_json)
        if e['joints'][4] != 'left_arm_joint1' or e['joints'][17] != 'right_arm_joint7':
            raise ValueError('envelope joint order is not the EX002 layout')
        sel = {'upper': UPPER, 'wrist': WRIST, 'upper+wrist': UPPER + WRIST}[joints]
        self.idx = np.asarray(sorted(sel))
        r = lambda p: round(100 - p, 4)
        self.lo_k = _qcol(qs, q, r(knee))[self.idx]; self.hi_k = _qcol(qs, q, knee)[self.idx]
        self.lo_b = _qcol(qs, q, r(bound))[self.idx] - extra; self.hi_b = _qcol(qs, q, bound)[self.idx] + extra
        if not ((self.lo_b < self.lo_k) & (self.lo_k < self.hi_k) & (self.hi_k < self.hi_b)).all():
            raise ValueError('envelope quantiles are not ordered')
        self.cfg = dict(envelope=str(envelope_json), knee=knee, bound=bound, extra=extra, joints=joints)

    def __call__(self, q):
        q = np.asarray(q)
        out = q.astype(np.float64, copy=True)
        out[..., self.idx] = soft_limit(out[..., self.idx], self.lo_k, self.hi_k, self.lo_b, self.hi_b)
        return out.astype(q.dtype if q.dtype.kind == 'f' else np.float64)


# ----------------------------------------------------------------------------------------------- 3. metrics

ARMS14 = list(range(4, 18))


def arm_speed(q):
    """[T-1] mean-over-arm-joints |dq/dt| norm (rad/s), the rollout_probe definition (norm over the 14 arm joints)."""
    return np.linalg.norm(np.diff(np.asarray(q, np.float64)[:, ARMS14], axis=0), axis=1) * FPS


def runs(mask):
    """[(start, end)) runs of True."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[0::2], d[1::2]))


def speech_mask(voiced, close_s=0.3):
    """Voiced frames with gaps shorter than close_s filled (syllable gaps inside speech are still speech)."""
    v = np.asarray(voiced, bool).copy(); g = int(round(close_s * FPS))
    for a, b in runs(~v):
        if a > 0 and b < len(v) and b - a <= g:
            v[a:b] = True
    return v


def stillness(q, ref_speed_median, thr=0.25):
    """Still frames: arm speed < thr x the clip's human median speed (rollout_probe 'still')."""
    return arm_speed(q) < thr * ref_speed_median


def freeze_segments(q, speech, ref_speed_median, min_s=1.5, thr=0.25):
    """Segments of >= min_s s in which the arms are still (stillness()) while the speaker is speaking (every frame of
    the segment inside `speech`). Returns (list of (a, b) frames, frozen speaking frames, speaking frames)."""
    st = stillness(q, ref_speed_median, thr); n = min(len(st), len(speech) - 1)
    both = st[:n] & np.asarray(speech[:n], bool)
    seg = [(a, b) for a, b in runs(both) if b - a >= int(round(min_s * FPS))]
    return seg, int(sum(b - a for a, b in seg)), int(np.asarray(speech[:n], bool).sum())


def out_of_envelope(q, lo, hi, idx):
    """[T] bool: any joint of idx outside [lo, hi] (arrays aligned with idx)."""
    x = np.asarray(q, np.float64)[:, idx]
    return ((x < lo) | (x > hi)).any(1)


def style_descriptor(q):
    """Per-clip arm style: [14 joint std, 14 joint mean, mean arm speed] (rad, rad, rad/s)."""
    x = np.asarray(q, np.float64)[:, ARMS14]
    return np.r_[x.std(0), x.mean(0), arm_speed(q).mean()]

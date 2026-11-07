"""Explicit control tokens for G1 (11-05 user: pointing + amplitude only; labels from the ground-truth motion, no manual labels).

Pointing follows the spatially grounded gesture benchmark (Deichler et al., arXiv 2610.03105): the control is a 3D TARGET in the
PELVIS frame (+ the dominant hand); the apex frame is NOT an input, the generator decides when to point.
Annotation tokens (group 0) are attached to one BEAT2 human-annotated gesture segment [a, b]:
  deictic (POINT):          hand (L / R), target centre (x, y, z) m, pelvis frame, amplitude bin
  iconic / metaphoric / beat: amplitude bin only
Lexical tokens (group 1): g1_lexicon category of a word, at its onset, no values (see TYPE_NAMES below).
Token time t (where it sits in the text stream, i.e. the word it is attached to): onset of the annotated keyword's aligned
occurrence nearest to the segment (within [a - 1 s, b + 1 s]); no such word -> onset of the first word starting >= a - 0.2 s,
else a. The apex is never used for t.
Geometry: official ECHO-G G1 FK (echo_g.evaluation.g1_kinematics), identity base orientation and zero translation, so every
point is in the pelvis frame (x forward, y left, z up). Per hand: wrist = wrist_yaw_link origin, elbow = elbow_link origin.
  hand  = the hand with the larger wrist-path diameter inside [a, b]
  apex  = frame of that hand's largest wrist distance from its zero-pose (rest) position, inside [a, b]
  ray   = from the wrist along elbow -> wrist at the apex (the benchmark's pointing ray)
  target (training label, BEAT2 has no scene targets) = wrist + d * ray direction, d ~ U[1.5, 3] m (seeded per token);
         target box = axis-aligned cube of side BOX = 0.20 m centred on it
  amplitude = wrist-path diameter (m) of the hand (GESTURE: max over the hands), binned by the TRAIN tertiles.
Value vector (5): [x / 3, y / 3, z / 3, hand (-1 left, +1 right), amp bin - 1]; value mask (3): [target, hand, amp].
Scores (benchmark, per POINT token, seconds instead of 30 fps frames): e_t = |a_hat - a_ref|, d_t = max(0, e_t - TAU),
  Score_a = exp(-d_t^2 / (2 SIG_T^2)), TAU = 10 / 30 s, SIG_T = 15 / 30 s; at a_hat: ray from the wrist along elbow -> wrist,
  d_miss = distance of the ray to the target AABB (0 if it hits), d_centroid = |wrist - centre|, theta = atan(d_miss / d_centroid),
  Score_b = exp(-theta^2 / (2 (20 deg)^2)); STG = Score_a * Score_b (per token, then averaged). a_hat = a frozen kinematic
  apex detector (logistic regression, +-3-frame positives, train split GT only; scripts/semantic/g1_ctl_apex.py).
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

# Token types (11-05 user: pointing / amplitude AND semantic intent in ONE token mechanism). Group 0 = from the BEAT2 human
# annotation (a segment [a, b], values measured on the GT motion); group 1 = from the TEXT only (g1_lexicon word category at the
# word onset, no values) -- the kind a lexicon / LLM inserts automatically at inference.
SEM_CLASSES = ['deictic', 'iconic', 'metaphoric', 'beat']
LEX_CLASSES = ['we', 'self', 'you', 'third', 'neg', 'number', 'big', 'small', 'up', 'down', 'this', 'there', 'quest', 'mental']
TYPE_NAMES = ['pad'] + SEM_CLASSES + ['lex_' + c for c in LEX_CLASSES]
NTYPES_V1 = len(TYPE_NAMES)    # 11-07 exp/duplex: types appended after this keep every existing id and encoding unchanged
TYPE_NAMES = TYPE_NAMES + ['yield']   # 11-07 duplex mode: the interrupt event (the robot must yield the floor), group 0, no values
TYPE_ID = {n: i for i, n in enumerate(TYPE_NAMES)}
NTYPES = len(TYPE_NAMES)
POINT = TYPE_ID['deictic']
YIELD = TYPE_ID['yield']
GESTURE = TYPE_ID['iconic']   # legacy name (amplitude-only types: iconic / metaphoric / beat)
TYPES = {c: TYPE_ID[c] for c in SEM_CLASSES}
NV, NVM = 5, 3                 # value channels / value-mask channels
KC = 16                        # control tokens per training window (padding)
# Rule map lexicon category -> BEAT2 semantic class for AUTOMATIC insertion of semantic tokens from text (decided before any
# result; deixis -> deictic, size / direction / number -> iconic, cognition / negation / question -> metaphoric).
LEX2SEM = dict(we='deictic', self='deictic', you='deictic', third='deictic', this='deictic', there='deictic',
               number='iconic', big='iconic', small='iconic', up='iconic', down='iconic',
               neg='metaphoric', quest='metaphoric', mental='metaphoric')
POS_SCALE = 3.0                # m: target coordinates / POS_SCALE enter the Fourier features
BOX = 0.20                     # m: side of the target cube
DIST = (1.5, 3.0)              # m: label target distance along the ray
TAU, SIG_T, SIG_B = 10 / 30, 15 / 30, math.radians(20.)
WIN_PAST, WIN_FUTURE = 2.0, 3.0   # flow route: a TIMED token reaches positions within [t - 2 s, t + 3 s]
BFS = dict(elbow=(22, 23), wrist=(28, 29), shoulder=(12, 13))


def fourier(x, n):
    """x [...] about [-1, 1] -> [..., 2n]: sin / cos (2^f pi x / 2), f = 0..n-1."""
    f = (2.0 ** torch.arange(n, device=x.device, dtype=torch.float32)) * (math.pi / 2)
    a = x.float()[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1)


def value_features(val, vmask, n):
    """val [..., 5], vmask [..., 3] -> [..., 5 * 2n + 3] (masked fields exactly 0; the mask itself is appended)."""
    m5 = torch.stack([vmask[..., 0]] * 3 + [vmask[..., 1], vmask[..., 2]], -1).float()
    return torch.cat([(fourier(val, n) * m5[..., None]).flatten(-2), vmask.float()], -1)


# ---------------------------------------------------------------- geometry (official FK, pelvis frame)
_REST = {}


def arm_points(joints):
    """joints [B, T, 29] physical rad (robot_repr[..., 10:]) -> dict(wrist, elbow, shoulder) [B, T, 2, 3], pelvis frame (L, R)."""
    from .g1_contact import _mods
    G1K = _mods()['G1K']
    q = torch.as_tensor(joints).float(); B, T = q.shape[:2]
    eye = torch.eye(3, device=q.device).expand(B, T, 3, 3)
    pos, _ = G1K.g1_forward_kinematics_with_rotations(q, eye, torch.zeros(B, T, 3, device=q.device))
    return {k: pos[:, :, list(v)] for k, v in BFS.items()}


def rest_wrist():
    if 'w' not in _REST: _REST['w'] = arm_points(torch.zeros(1, 1, 29))['wrist'][0, 0].numpy()
    return _REST['w']


def geometry_np(joints29, chunk=4096):
    """numpy [T, 29] -> dict of numpy [T, 2, 3] (CPU FK in chunks)."""
    out = {k: [] for k in BFS}
    for i in range(0, len(joints29), chunk):
        g = arm_points(torch.as_tensor(np.asarray(joints29[i:i + chunk], np.float32))[None])
        for k in BFS: out[k].append(g[k][0].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def diameter(p):
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    return float(d.max()) if len(p) > 1 else 0.


def measure(geo, fa, fb, hand=None):
    """Segment measurement (numpy geometry of geometry_np), frames [fa, fb). hand None -> larger wrist-path diameter.
    Returns dict(hand 0 L / 1 R, apex frame, wrist, ray (unit elbow -> wrist) at the apex, amp_hand, amp_max)."""
    W = geo['wrist']; fa, fb = max(0, int(fa)), min(len(W), int(fb))
    if fb - fa < 2: return None
    seg = W[fa:fb]; rest = rest_wrist()
    diam = [diameter(seg[:, h]) for h in (0, 1)]
    h = int(np.argmax(diam)) if hand is None else int(hand)
    apex = fa + int(np.argmax(np.linalg.norm(seg[:, h] - rest[h], axis=-1)))
    return dict(hand=h, apex=apex, wrist=W[apex, h].tolist(), ray=ray_at(geo, apex, h).tolist(), amp_hand=diam[h],
                amp_max=max(diam))


def ray_at(geo, f, h):
    r = geo['wrist'][f, h] - geo['elbow'][f, h]
    return r / max(np.linalg.norm(r), 1e-9)


def amp_bin(x, edges):
    return int(np.searchsorted(np.asarray(edges), x, side='right'))


def ray_box_distance(o, d, centre, side=BOX):
    """Distance from the ray o + s d (s >= 0) to the axis-aligned box (0 if it hits): numeric minimisation over s
    (point-to-box distance is convex in s)."""
    lo, hi = np.asarray(centre) - side / 2, np.asarray(centre) + side / 2
    o, d = np.asarray(o, np.float64), np.asarray(d, np.float64)
    def dist(s):
        p = o + s * d
        return float(np.linalg.norm(np.maximum(0, np.maximum(lo - p, p - hi))))
    a, b = 0.0, 20.0
    for _ in range(100):                                                # ternary search (convex)
        m1, m2 = a + (b - a) / 3, b - (b - a) / 3
        if dist(m1) <= dist(m2): b = m2
        else: a = m1
    return min(dist(0.0), dist(0.5 * (a + b)))


def score_b(wrist, ray, centre, side=BOX):
    dm = ray_box_distance(wrist, ray, centre, side); dc = float(np.linalg.norm(np.asarray(centre) - np.asarray(wrist)))
    th = math.atan2(dm, max(dc, 1e-9))
    return math.exp(-th ** 2 / (2 * SIG_B ** 2)), math.degrees(th)


def score_a(a_hat_s, a_ref_s):
    d = max(0., abs(a_hat_s - a_ref_s) - TAU)
    return math.exp(-d ** 2 / (2 * SIG_T ** 2))


def target_from_dir(az, el, hand, dist=2.25):
    """az / el (rad, pelvis frame) -> a target point dist m along that direction from the zero-pose shoulder of the hand."""
    sh = arm_points(torch.zeros(1, 1, 29))['shoulder'][0, 0, int(hand)].numpy()
    return sh + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def token_value(tok):
    """CTL.json token -> (type, val [5], vmask [3]). A POINT token may carry 'target' [3] or ('az', 'el')."""
    ty = int(tok['type']); v = np.zeros(NV, np.float32); m = np.zeros(NVM, np.float32)
    if ty == POINT:
        tgt = tok.get('target')
        if tgt is None and tok.get('az') is not None: tgt = target_from_dir(tok['az'], tok['el'], tok.get('hand', 1))
        if tgt is not None: v[:3] = np.asarray(tgt, np.float32) / POS_SCALE; m[0] = 1
        if tok.get('hand') is not None: v[3] = -1. if int(tok['hand']) == 0 else 1.; m[1] = 1
    if tok.get('amp_bin') is not None: v[4] = float(tok['amp_bin']) - 1.; m[2] = 1
    if ty != POINT and tok.get('count') is not None:          # 11-07 count control (non-POINT annotation tokens): slot 0
        v[0] = count_value(tok['count']); m[0] = 1
    return ty, v, m


# ---------------------------------------------------------------- counts (11-07 §7.55 item 3: "tap / raise N times")
# count of a span = strokes of the span's dominant hand (larger wrist bounding-box diagonal inside the span): local maxima of the
# smoothed wrist speed (moving average COUNT_SMOOTH frames at 50 Hz) that exceed every neighbour within +-COUNT_HALF frames and the
# action speed threshold (APEX.json, 25th pct of train peak speeds), capped at COUNT_MAX ('4 or more'). The hard count (labels,
# evaluation) and the soft count (training loss) are the same expression with indicators vs sigmoids.
COUNT_MAX, COUNT_SMOOTH, COUNT_HALF, COUNT_SPAN = 4, 4, 5, (0.1, 4.0)
COUNT_BETA_H, COUNT_BETA_N = 20.0, 100.0      # 1 / (m/s) sharpness of the soft height / neighbour sigmoids


def count_value(n):
    return (min(int(n), COUNT_MAX) - COUNT_MAX / 2) / (COUNT_MAX / 2)


def count_of_value(v):
    return int(round(float(v) * (COUNT_MAX / 2) + COUNT_MAX / 2))


def count_speed(wrist):
    """wrist [..., T, 2, 3] (m, 50 Hz) -> smoothed speed [..., T, 2] (m/s)."""
    w = torch.as_tensor(wrist).float()
    v = torch.linalg.norm(w[..., 1:, :, :] - w[..., :-1, :, :], dim=-1) * 50.
    v = torch.cat([v[..., :1, :], v], -2)
    sh = v.shape; x = v.movedim(-2, -1).reshape(-1, 1, sh[-2])
    x = F.pad(x, (COUNT_SMOOTH // 2, COUNT_SMOOTH - 1 - COUNT_SMOOTH // 2), mode='replicate')
    x = F.avg_pool1d(x, COUNT_SMOOTH, 1)
    return x.reshape(*sh[:-2], sh[-1], sh[-2]).movedim(-1, -2)


def peak_score(sp, thr, soft=False):
    """sp [..., T, H] -> per-frame peak indicator (hard) or soft score in [0, 1]."""
    T = sp.shape[-2]; pad = torch.full_like(sp[..., :COUNT_HALF, :], -1.)
    x = torch.cat([pad, sp, pad], -2)
    f = (lambda d, b: torch.sigmoid(b * d)) if soft else (lambda d, b: (d > 0).float())
    out = f(sp - thr, COUNT_BETA_H)
    for j in range(1, COUNT_HALF + 1):
        out = out * f(sp - x[..., COUNT_HALF - j:COUNT_HALF - j + T, :], COUNT_BETA_N) \
                  * f(sp - x[..., COUNT_HALF + j:COUNT_HALF + j + T, :], COUNT_BETA_N)
    return out


def span_hand(wrist_span):
    """wrist_span [L, 2, 3] -> dominant hand (larger bounding-box diagonal)."""
    w = torch.as_tensor(wrist_span).float()
    return int(torch.argmax(torch.linalg.norm(w.amax(0) - w.amin(0), dim=-1)))


def count_np(wrist, fa, fb, thr, sp=None):
    """hard count of frames [fa, fb) of one clip; wrist [T, 2, 3] numpy / torch. sp = precomputed count_speed(wrist)."""
    w = torch.as_tensor(np.asarray(wrist)).float(); fa, fb = max(0, int(fa)), min(len(w), int(fb))
    if fb - fa < 2: return None
    sp = count_speed(w) if sp is None else sp
    h = span_hand(w[fa:fb])
    return min(int(peak_score(sp, thr)[fa:fb, h].sum()), COUNT_MAX)


def group_of(ty):
    return 1 if TYPE_NAMES[int(ty)].startswith('lex_') else 0


def priority(ty):
    """truncation order when a window has more than KC tokens: annotated non-beat, lexical, beat."""
    n = TYPE_NAMES[int(ty)]
    return 2 if n == 'beat' else 1 if n.startswith('lex_') else 0


def window_arrays(toks, kc=KC):
    """list of (window-relative time s, segment a, b, apex (s), type, val, vmask) -> padded arrays (type 0 = padding);
    more than kc tokens: keep by priority() then time, stored in time order."""
    t = np.full(kc, np.nan, np.float32); a = np.full(kc, np.nan, np.float32); b = np.full(kc, np.nan, np.float32)
    ap = np.full(kc, np.nan, np.float32); ty = np.zeros(kc, np.int64); gr = np.zeros(kc, np.int64)
    v = np.zeros((kc, NV), np.float32); m = np.zeros((kc, NVM), np.float32)
    keep = sorted(sorted(toks, key=lambda x: (priority(x[4]), x[0]))[:kc], key=lambda x: x[0])
    for i, (t_, a_, b_, ap_, ty_, v_, m_) in enumerate(keep):
        t[i], a[i], b[i], ap[i], ty[i], v[i], m[i], gr[i] = t_, a_, b_, ap_, ty_, v_, m_, group_of(ty_)
    return dict(ctl_t=t, ctl_a=a, ctl_b=b, ctl_apex=ap, ctl_type=ty, ctl_g=gr, ctl_v=v, ctl_vm=m)


# ---------------------------------------------------------------- arrival / visibility (torch)
def anchor_arrival(arr, ttime, tvalid, ct):
    """Arrival of each control token = arrival of the last valid text token whose time is <= the control time (the word it
    is attached to); none -> the control time itself. arr / ttime / tvalid [B, Kt], ct [B, Kc] (s, same origin)."""
    tv = tvalid & torch.isfinite(ttime)
    cand = tv[:, None, :] & (torch.nan_to_num(ttime, 1e9)[:, None, :] <= torch.nan_to_num(ct, -1e9)[:, :, None] + 1e-3)
    score = torch.where(cand, torch.nan_to_num(ttime, 0.)[:, None, :].float(), torch.full(cand.shape, -1e9, device=ct.device))
    got = torch.gather(torch.nan_to_num(arr, 0.).float(), 1, score.argmax(-1))
    return torch.where(cand.any(-1), got, torch.nan_to_num(ct, 0.).float())


def visibility(cmask, ct, arr_c, final, now, heard, stream):
    """-> keep [B, K, Kc] (token arrived by chunk k), timed [B, K, Kc] (its time is known: voiced by now + H, or offline).
    cmask [B, Kc] bool, ct / arr_c [B, Kc] s, final / stream [B] bool, now [K] s, heard [B, K] s."""
    keep = ((arr_c[:, None, :] <= now[None, :, None]) | final[:, None, None]) & cmask[:, None, :]
    voiced = (torch.nan_to_num(ct, 1e9)[:, None, :] <= heard[:, :, None]) | ~stream[:, None, None]
    return keep, keep & voiced

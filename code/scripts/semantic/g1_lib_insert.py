"""Fixed-action library insertion plans (11-07 item 5; library = g1_gesture_lib.py, train speakers of one half only).
plan(lib, what, mode, stems, ctl_dir, pred) -> {stem: dict(given=[token dicts scored by g1_ctl_eval.py], inserts=[dict(t0, t1, kind,
frames [n, 39] physical 50 Hz)])}; inpaint(inserts, model, vae, qm, qs, n_lat) -> Streamer inpaint dict (normalised latents on the
global latent grid + soft weights with RAMP s cosine ramps at both ends).
  what point (mode gt) : dev POINT tokens with target + hand; the same-hand library entry whose elbow -> wrist ray is closest in
                         angle to (target - its own apex wrist); its keyword frame is aligned to the token time.
  what count (mode gt) : dev count tokens (non-POINT, 'count'); N strokes of one library stroke (N = 0: its first frame held over
                         the span), a stroke that fits N times into [a + 50 ms, b - 50 ms] (else time-compressed to fit), centred
                         in the span; the rest of [a, b] and 2 RAMP s outside it are held poses (the soft ramps sit on the outer half);
                         given the VAE, the first stroke (seeded order) whose composite counts exactly N after the VAE round
                         trip is used (the insertion lives in its latent space).
  what class           : deictic / iconic / metaphoric exemplar keyed by (class, keyword lexicon category), else class; at
                         mode gt = dev annotation tokens (oracle class + time), lex = lexicon words mapped by scmg.ctl.LEX2SEM,
                         llm = the --ctl-pred tokens (g1_llm_intent.py PRED.json). Keyword frame aligned to the token time.
Greedy in time, no overlaps (0.2 s gap). Choice among candidates seeded by crc32(stem:index) (independent of the noise seed).
"""
import json, zlib
from pathlib import Path
import numpy as np
import torch

FPS = 50; GAP = 0.2; RAMP = 0.3


def _load(lib):
    meta = json.loads((Path(lib) / 'LIB.json').read_text()); arr = np.load(Path(lib) / 'LIB.npz')
    return meta, arr


def _roundtrip_count(vae, f, qm, qs, a_, b_, thr):
    from scmg import ctl
    T = len(f); Tp = -(-T // 4) * 4
    ff = np.concatenate([f, np.repeat(f[-1:], Tp - T, 0)]) if Tp > T else f
    with torch.no_grad():
        mu, _ = vae.encode(((torch.as_tensor(ff, dtype=torch.float32, device=qm.device) - qm) / qs)[None])
        y = (vae.decode(mu)[0] * qs + qm)[:T].cpu().numpy().astype(np.float64)
    return ctl.count_np(ctl.geometry_np(y[:, 10:39])['wrist'], max(a_, 0), b_, thr)


def plan(lib, what, mode, stems, ctl_dir, pred=None, vae=None, qm=None, qs=None):
    from scmg import ctl
    from g1_lexicon import EN_MAP
    meta, arr = _load(lib); E = meta['entries']; thr = meta['action_speed']
    raw = json.loads((Path(ctl_dir) / 'CTL.json').read_text())
    P = json.loads(Path(pred).read_text()) if (mode == 'llm' and pred) else {}
    pts = [e for e in E if e['kind'] == 'point']; strokes = sorted([e for e in E if e['kind'] == 'stroke'], key=lambda e: e['frames'])
    bank = {}
    for e in E:
        if e['kind'] == 'class': bank.setdefault((e['cls'], e['cat']), []).append(e); bank.setdefault((e['cls'], '*'), []).append(e)
    fr = lambda e: np.asarray(arr[f"frames_{e['id']}"], np.float64)
    out = {}
    for s in stems:
        ts = raw.get(s, []); rng = lambda i: np.random.default_rng(zlib.crc32(f'{s}:{i}'.encode()))
        cand = []                                    # (t0, frames, kind, given token)
        if what == 'point':
            for i, t in enumerate(ts):
                if t['type'] != ctl.POINT or t.get('target') is None or t.get('hand') is None: continue
                tg = np.asarray(t['target']); best, bd = None, 9.
                for e in pts:
                    if e['hand'] != int(t['hand']): continue
                    d = tg - np.asarray(e['wrist']); d = d / max(np.linalg.norm(d), 1e-9)
                    ang = float(np.arccos(np.clip(np.dot(d, np.asarray(e['ray'])), -1, 1)))
                    if ang < bd: best, bd = e, ang
                if best is not None: cand.append((t['t'] - best['tref'], fr(best), 'point', dict(t, lib_id=best['id'], lib_angle_deg=float(np.degrees(bd)))))
        elif what == 'count':
            for i, t in enumerate(ts):
                if t.get('count') is None or t['type'] == ctl.POINT or not strokes: continue
                n, span = int(t['count']), t['b'] - t['a']
                core_max = max(span - 0.1, 0.2)                  # the strokes sit inside [a, b] with 50 ms margins
                fit = [e for e in strokes if max(n, 1) * e['frames'] / FPS <= core_max] or strokes[:1]
                order = [fit[j] for j in rng(i).permutation(len(fit))]
                pick = None
                for e in order:                                    # with the VAE: the first stroke whose composite survives the
                    f = fr(e); pad = int(round(2 * RAMP * FPS))    # VAE round trip with exactly N counted strokes in [a, b]
                    if n == 0: core = np.repeat(f[:1], int(round(span * FPS)), 0)
                    else:
                        core = np.concatenate([f] * n)
                        if len(core) / FPS > core_max:             # time-compress (linear interpolation) so all N strokes fit
                            m_ = max(int(core_max * FPS), 2 * n); x = np.linspace(0, len(core) - 1, m_)
                            core = np.stack([np.interp(x, np.arange(len(core)), core[:, k]) for k in range(core.shape[1])], -1)
                    # the whole span [a, b] is covered (held poses before / after the centred strokes: no free motion inside
                    # the counted span) and RAMP s of held pose outside it carry the soft ramps
                    pre = pad + int(round((span * FPS - len(core)) / 2)); post = pad + int(round(span * FPS)) - len(core) - (pre - pad)
                    comp = np.concatenate([np.repeat(core[:1], max(pre, pad), 0), core, np.repeat(core[-1:], max(post, pad), 0)])
                    t0 = t['a'] - pad / FPS
                    if pick is None: pick = (t0, comp, e)
                    if vae is None: break
                    a_, b_ = int(round((t['a'] - t0) * FPS)), int(round((t['b'] - t0) * FPS))
                    if _roundtrip_count(vae, comp, qm, qs, a_, b_, thr) == n: pick = (t0, comp, e); break
                t0, comp, e = pick
                cand.append((t0, comp, 'count', dict(t, lib_id=e['id'])))
        elif what == 'class':
            sem = {ctl.TYPE_ID[c]: c for c in ('deictic', 'iconic', 'metaphoric')}
            if mode == 'gt':
                qs = [(t['t'], sem[t['type']], EN_MAP.get((t.get('kw') or '').strip("'"), 'none'), t) for t in ts if t['type'] in sem]
            elif mode == 'lex':
                qs = [(t['t'], ctl.LEX2SEM[t['cls'][4:]], t['cls'][4:], dict(t, type=ctl.TYPE_ID[ctl.LEX2SEM[t['cls'][4:]]]))
                      for t in ts if t['cls'].startswith('lex_') and t['cls'][4:] in ctl.LEX2SEM]
            else:
                qs = [(t['t'], t['cls'], EN_MAP.get((t.get('word') or '').lower(), 'none'), t) for t in P.get(s, []) if t.get('cls') in sem.values()]
            for i, (tq, cls, cat, tok) in enumerate(sorted(qs, key=lambda x: x[0])):
                c = bank.get((cls, cat)) or bank.get((cls, '*'))
                if not c: continue
                e = c[rng(i).integers(len(c))]
                cand.append((tq - e['tref'], fr(e), 'class', dict(tok, lib_id=e['id'])))
        else: raise ValueError(what)
        ins, given, last = [], [], -1e9
        for t0, f, kind, tok in sorted(cand, key=lambda x: x[0]):
            if t0 < last + GAP: continue
            t0 = max(0., t0); ins.append(dict(t0=float(t0), t1=float(t0 + len(f) / FPS), kind=kind, frames=f)); given.append(tok)
            last = t0 + len(f) / FPS
        out[s] = dict(given=given, inserts=ins)
    return out


@torch.no_grad()
def inpaint(inserts, model, vae, qm, qs, n_lat, seed):
    """-> dict(lat [n_lat, z] normalised, w [n_lat], seed) or None. qm / qs: torch stats of the VAE input normalisation."""
    if not inserts: return None
    dev = qm.device; z = model.lat_mean.shape[-1]
    lat = torch.zeros(n_lat, z, device=dev); w = torch.zeros(n_lat, device=dev); r = max(1, int(round(RAMP * FPS / 4)))
    for it in inserts:
        f = np.asarray(it['frames'], np.float32); T = len(f); Tp = -(-T // 4) * 4
        if Tp > T: f = np.concatenate([f, np.repeat(f[-1:], Tp - T, 0)])
        mu, _ = vae.encode(((torch.as_tensor(f, device=dev) - qm) / qs)[None])
        l = ((mu[0] - model.lat_mean.reshape(-1)) / model.lat_std.reshape(-1)).float()
        i0 = int(round(it['t0'] * FPS / 4)); n = len(l)
        ww = torch.ones(n, device=dev)
        k = min(r, n // 2)
        if k > 0:
            ramp = 0.5 - 0.5 * torch.cos(torch.pi * (torch.arange(k, device=dev) + 0.5) / k)
            ww[:k] = ramp; ww[n - k:] = ramp.flip(0)
        lo, hi = max(0, i0), min(n_lat, i0 + n)
        if hi <= lo: continue
        lat[lo:hi] = l[lo - i0:hi - i0]; w[lo:hi] = torch.maximum(w[lo:hi], ww[lo - i0:hi - i0])
    return dict(lat=lat, w=w, seed=seed)

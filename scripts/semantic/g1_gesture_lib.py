"""Fixed-action library for G1 (11-07 item 5, "单独的固定动作，类似动作库"). Built ONLY from train-split speakers of ONE half
(SPLIT_RK.json of g1_speaker_split.py, default K) -- never dev / test speakers -- so the semantic recognisers of the other half (R)
score insertions speaker-disjointly. Ground-truth G1 motion of the hos view (50 Hz, physical, 39 channels).
Entries (all store the 39-channel frames, 50 Hz, and their decft-VAE latents):
  point  : every train POINT token of the half: snippet [t - 0.6 s, b + 0.3 s] (t = its keyword time, kept as 'tref'), hand,
           apex offset, wrist at the apex and the elbow -> wrist ray (scmg/ctl.py geometry). A query (target, hand) takes the
           same-hand entry whose ray is closest in angle to (target - its own apex wrist) = finest direction binning.
  stroke : one tap / raise stroke cut from spans whose count_np count is 1: from the speed minimum before the peak to the one
           after (within 0.4 s), closed (|wrist end - start| <= 4 cm), counting exactly 1 between held poses (peak >= 1.3x threshold) and exactly 2
           when doubled (count_np), and exactly N for N = 1..4 after the decft VAE round trip (insertion is in its latents). A count-N query
           repeats a stroke N times (count 0 = its first frame held), so the count is exact by construction of the frames.
  class  : deictic / iconic / metaphoric annotation tokens (as the paste bank), keyed by (class, lexicon category of the keyword).
    python scripts/semantic/g1_gesture_lib.py --ctl <ctl-tokens-hos-count> --view <hos view> --split SPLIT_RK.json --half K \
        --vae <decft VAE> --stats <STATS json> --out <lib dir>
Writes LIB.json (entries without arrays) + LIB.npz (frames_<i>, lat_<i> = raw VAE means).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from scmg import ctl
from g1_lexicon import EN_MAP

FPS = 50; PRE, POST = 0.6, 0.3
SEM = {ctl.TYPE_ID[c]: c for c in ('deictic', 'iconic', 'metaphoric')}


def spk_of(stem): return '_'.join(stem.split('_')[1:3])


def encode(vae, frames, qm, qs, dev):
    """frames [T, 39] physical -> VAE posterior means [ceil(T / 4), z] (raw VAE units; the generator normalises them with its own
    lat_mean / lat_std at insertion). Frames padded by edge replication to a multiple of 4."""
    T = len(frames); Tp = -(-T // 4) * 4
    f = np.concatenate([frames, np.repeat(frames[-1:], Tp - T, 0)]) if Tp > T else frames
    x = (torch.as_tensor(f, dtype=torch.float32, device=dev) - qm) / qs
    with torch.no_grad():
        mu, _ = vae.encode(x[None])
    return mu[0].cpu().numpy().astype(np.float32)


def roundtrip(vae, frames, qm, qs, dev):
    T = len(frames); Tp = -(-T // 4) * 4
    f = np.concatenate([frames, np.repeat(frames[-1:], Tp - T, 0)]) if Tp > T else frames
    x = (torch.as_tensor(f, dtype=torch.float32, device=dev) - qm) / qs
    with torch.no_grad():
        mu, _ = vae.encode(x[None]); y = vae.decode(mu)[0] * qs + qm
    return y[:T].cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    for k in ('ctl', 'view', 'split', 'vae', 'stats', 'out'): ap.add_argument('--' + k, required=True)
    ap.add_argument('--half', default='K'); ap.add_argument('--device', default='cuda'); a = ap.parse_args()
    dev = torch.device(a.device)
    keep = set(json.loads(Path(a.split).read_text())[a.half])
    raw = json.loads((Path(a.ctl) / 'CTL.json').read_text()); thr = json.loads((Path(a.ctl) / 'APEX.json').read_text())['action_speed']
    items = {it['id']: it for it in json.loads((Path(a.view) / 'train.json').read_text())}
    q = np.load(Path(a.view) / 'train_q.npy', mmap_mode='r')
    from scmg.vae import MotionVAE
    vck = torch.load(a.vae, map_location='cpu'); vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema']); vae.eval()
    st = json.loads(Path(a.stats).read_text())
    qm, qs = torch.tensor(st['q_mean'], device=dev).float(), torch.tensor(st['q_std'], device=dev).float()
    entries, arrs = [], {}

    def add(e, fr):
        i = len(entries); e['id'] = i; e['frames'] = int(len(fr)); entries.append(e); arrs[f'frames_{i}'] = fr.astype(np.float32)
        arrs[f'lat_{i}'] = encode(vae, fr, qm, qs, dev)

    n_seen = 0
    for s, ts in raw.items():
        if not ts or ts[0]['split'] != 'train' or spk_of(s) not in keep or s not in items: continue
        it = items[s]; clip = np.asarray(q[it['offset']:it['offset'] + it['frames']], np.float64); n_seen += 1
        geo = None
        for t in ts:
            if ctl.group_of(t['type']) != 0: continue
            f0 = max(0, int(round((t['t'] - PRE) * FPS))); f1 = min(len(clip), int(round((t['b'] + POST) * FPS)))
            if f1 - f0 < 10: continue
            base = dict(stem=s, speaker=spk_of(s), rec=it['echog']['recording'], tref=float(t['t'] - f0 / FPS))
            if t['type'] in SEM:
                add(dict(base, kind='class', cls=SEM[t['type']], cat=EN_MAP.get((t.get('kw') or '').strip("'"), 'none')), clip[f0:f1])
            if t['type'] == ctl.POINT and t.get('ray') is not None and t.get('hand') is not None and t.get('apex') is not None:
                add(dict(base, kind='point', hand=int(t['hand']), ray=t['ray'], wrist=t['wrist'], apex=float(t['apex'] - f0 / FPS)), clip[f0:f1])
            if t.get('count'):
                if geo is None: geo = ctl.geometry_np(clip[:, 10:39]); sp = ctl.count_speed(geo['wrist'])
                fa, fb = int(round(t['a'] * FPS)), int(round(t['b'] * FPS))
                h = ctl.span_hand(torch.as_tensor(geo['wrist'][fa:fb])); pk = ctl.peak_score(sp, thr)[:, h].numpy()
                peaks = [f for f in range(fa, fb) if pk[f] > 0]
                for p in peaks:
                    v = sp[:, h].numpy()
                    lo = max(0, p - 20); hi = min(len(v), p + 21)
                    s0 = lo + int(np.argmin(v[lo:p])) if p > lo else p; s1 = p + int(np.argmin(v[p:hi])) if hi > p else p
                    if s1 - s0 < 8: continue
                    W = geo['wrist'][:, h]
                    if np.linalg.norm(W[s1] - W[s0]) > 0.06: continue
                    fr = clip[s0:s1 + 1]
                    # the stroke must count exactly 1 BETWEEN HELD POSES (as it is used: repeated, with held padding) and peak
                    # clearly above the threshold (1.3x) so that VAE smoothing does not lose it
                    hp = 15; comp = np.concatenate([np.repeat(fr[:1], hp, 0), fr, np.repeat(fr[-1:], hp, 0)])
                    wc = ctl.geometry_np(comp[:, 10:39])['wrist']
                    if ctl.count_np(wc, hp, hp + len(fr), thr) != 1: continue
                    spc = ctl.count_speed(wc)[hp:hp + len(fr), ctl.span_hand(torch.as_tensor(wc[hp:hp + len(fr)]))]
                    if float(spc.max()) < 1.3 * thr: continue
                    two = np.concatenate([np.repeat(fr[:1], hp, 0), fr, fr, np.repeat(fr[-1:], hp, 0)])
                    if ctl.count_np(ctl.geometry_np(two[:, 10:39])['wrist'], hp, hp + 2 * len(fr), thr) != 2: continue
                    # and the same must survive the decft VAE round trip (insertion happens in its latent space) for N = 1..4
                    rt_ok = True
                    for n_ in (1, 2, 3, 4):
                        cn = np.concatenate([np.repeat(fr[:1], hp, 0)] + [fr] * n_ + [np.repeat(fr[-1:], hp, 0)])
                        y = roundtrip(vae, cn, qm, qs, dev)
                        if ctl.count_np(ctl.geometry_np(y[:, 10:39])['wrist'], hp, hp + n_ * len(fr), thr) != n_: rt_ok = False; break
                    if not rt_ok: continue
                    add(dict(base, kind='stroke', hand=h, tref=0.0, peak=float((p - s0) / FPS)), fr)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'LIB.npz', **arrs)
    kinds = {k: sum(e['kind'] == k for e in entries) for k in ('point', 'stroke', 'class')}
    meta = dict(half=a.half, speakers=sorted(keep), clips_seen=n_seen, action_speed=thr, vae=a.vae, counts=kinds,
                latents='raw VAE posterior means (decft VAE), 4 frames per latent',
                entries=entries)
    (out / 'LIB.json').write_text(json.dumps(meta)); print(json.dumps(dict(clips=n_seen, **kinds)))


if __name__ == '__main__':
    main()

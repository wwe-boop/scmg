"""Text vs audio contribution inside semantic-gesture windows (11-06 user criterion 1 of the semantic line).
Windows = dev BEAT2 annotated deictic / iconic / metaphoric segments [a - 0.5 s, b + 0.5 s] (labels only pick the windows to
score; inference never sees them). Same distance as scripts/echog_g1/text_sensitivity.py (30 fps robot_repr, per frame L2 norm
over a channel group, upper = waist + arms 22..38), but averaged over window frames only:
  r_text  = D(own, no-text) / D(own, own seed 1)        r_audio = D(own, no-audio) / D(own, own seed 1)
(ratio of the window-frame means over all clips). Recording-clustered bootstrap (2000 draws, seed 0) of r_text - r_audio;
criterion: r_text > r_audio with the 95 % interval above 0.
    python scripts/semantic/sem_window_sens.py --ctl <CTL dir> --out res.json name=<own>,<seed1>,<notext>,<noaudio> ...
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

G = dict(upper=list(range(22, 39)), joints=list(range(10, 39))); FPS = 30; PAD = 0.5


def load(d, stem):
    return torch.load(Path(d) / f'{stem}.pt', map_location='cpu', weights_only=False)['robot_repr'].numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ctl', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('specs', nargs='+'); a = ap.parse_args()
    raw = json.loads((Path(a.ctl) / 'CTL.json').read_text())
    win = {s: [(t['a'] - PAD, t['b'] + PAD) for t in ts if t['type'] in (1, 2, 3)] for s, ts in raw.items() if ts and ts[0]['split'] == 'dev'}
    win = {s: w for s, w in win.items() if w}
    res = {}
    for spec in a.specs:
        name, dirs = spec.split('=', 1); own, s1, nt, na = dirs.split(',')
        per = defaultdict(lambda: dict(seed=0., text=0., audio=0., n=0))          # recording -> summed distances
        clips = 0
        for s, ws in sorted(win.items()):
            if not all((Path(d) / f'{s}.pt').exists() for d in (own, s1, nt, na)): continue
            X = {k: load(d, s) for k, d in (('own', own), ('s1', s1), ('nt', nt), ('na', na))}
            T = min(len(x) for x in X.values()); m = np.zeros(T, bool); tt = np.arange(T) / FPS
            for lo, hi in ws: m |= (tt >= lo) & (tt <= hi)
            if not m.any(): continue
            rec = re.sub(r'_utt_\d+$', '', s); clips += 1
            dist = lambda y: np.linalg.norm(X['own'][:T][m][:, G['upper']] - y[:T][m][:, G['upper']], axis=-1).sum()
            p = per[rec]; p['seed'] += dist(X['s1']); p['text'] += dist(X['nt']); p['audio'] += dist(X['na']); p['n'] += int(m.sum())
        recs = sorted(per); arr = np.array([[per[r]['seed'], per[r]['text'], per[r]['audio']] for r in recs])
        def stat(A):
            s = A.sum(0); return s[1] / s[0], s[2] / s[0]
        rt, ra = stat(arr); rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            x, y = stat(arr[rng.integers(0, len(arr), len(arr))]); bs.append(x - y)
        n = sum(per[r]['n'] for r in recs)
        res[name] = dict(clips=clips, recordings=len(recs), window_frames=n, r_text=float(rt), r_audio=float(ra), diff=float(rt - ra),
                         ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                         d_seed=float(arr[:, 0].sum() / n), d_notext=float(arr[:, 1].sum() / n), d_noaudio=float(arr[:, 2].sum() / n),
                         text_dominant=bool(np.quantile(bs, .025) > 0))
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name].items()}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

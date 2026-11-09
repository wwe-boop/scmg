"""Text sensitivity of G1 predictions (11-05 input-modality ablation): how much the motion changes when ONLY the text is
replaced (same audio, same speaker, same seed; infer_test.py --text-swap), relative to how much it changes between two
inference seeds with everything kept (the sampling floor).
    python scripts/echog_g1/text_sensitivity.py --out res.json  name=<own dir>,<swap dir>,<own seed-1 dir> ...
Per clip (30 fps robot_repr, common length): mean over frames of the L2 norm over channel group G of the difference (rad);
groups: upper = waist + arms (channels 22..38, the semantic recogniser's channels), joints = all 29 joints (10..38).
Reported per name and group: d_swap (own vs text-swapped), d_seed (own seed 0 vs own seed 1), ratio = d_swap / d_seed
(ratio of the clip means; 1 = swapping the text changes the motion no more than resampling does), the median per-clip
ratio and the share of clips whose swap distance exceeds their seed distance."""
import argparse, json
from pathlib import Path
import numpy as np, torch

G = dict(upper=list(range(22, 39)), joints=list(range(10, 39)))


def load(d, stem):
    return torch.load(Path(d) / f'{stem}.pt', map_location='cpu', weights_only=False)['robot_repr'].numpy().astype(np.float64)


def dist(a, b, ch):
    n = min(len(a), len(b)); return float(np.linalg.norm(a[:n, ch] - b[:n, ch], axis=-1).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('specs', nargs='+')
    a = ap.parse_args(); res = {}
    for spec in a.specs:
        name, dirs = spec.split('=', 1); own, swap, s1 = dirs.split(',')
        stems = sorted(p.stem for p in Path(own).glob('*.pt'))
        miss = [s for s in stems if not (Path(swap) / f'{s}.pt').exists() or not (Path(s1) / f'{s}.pt').exists()]
        if miss: raise SystemExit(f'{name}: {len(miss)} stems missing in swap / seed-1 dirs, e.g. {miss[:3]}')
        r = dict(n_clips=len(stems))
        for g, ch in G.items():
            ds, dn = [], []
            for s in stems:
                y0 = load(own, s); ds.append(dist(y0, load(swap, s), ch)); dn.append(dist(y0, load(s1, s), ch))
            ds, dn = np.asarray(ds), np.asarray(dn)
            r[g] = dict(d_swap=float(ds.mean()), d_seed=float(dn.mean()), ratio_of_means=float(ds.mean() / dn.mean()),
                        clip_ratio_median=float(np.median(ds / np.maximum(dn, 1e-9))), frac_clips_swap_gt_seed=float((ds > dn).mean()))
        res[name] = r; print(name, json.dumps(r), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

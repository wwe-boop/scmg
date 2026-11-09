"""Quick jitter check of G1 predictions vs ground truth (11-05): mean |2nd difference| per channel group, prediction / GT,
on the clips present in <pred dir> (works on a partial benchmark directory).  1 = GT-like; >>1 = residual sampling noise.
    python scripts/echog_g1/jitter_probe.py <pred dir> <motion_39d_30fps dir> [n]"""
import sys, glob, numpy as np, torch
P, R = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
G = dict(rot6d=range(0, 6), yaw=[6], vel=[7, 8, 9], legs=range(10, 22), waist=range(22, 25), arms=range(25, 39))
hf = lambda x: np.abs(np.diff(x, 2, axis=0)).mean(0)
fs = sorted(glob.glob(P + '/*.pt'))[:n]; a, g = [], []
for f in fs:
    p = torch.load(f, weights_only=False)['robot_repr'].numpy(); x = torch.load(f"{R}/{f.split('/')[-1]}", weights_only=False)
    x = np.asarray(x['robot_repr'] if isinstance(x, dict) else x)[:len(p)]; a.append(hf(p)); g.append(hf(x))
a, g = np.mean(a, 0), np.mean(g, 0)
print(f'{len(fs)} clips |', ' '.join(f'{k} {a[list(v)].mean() / g[list(v)].mean():.2f}' for k, v in G.items()))

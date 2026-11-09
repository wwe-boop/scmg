"""11-08 EX002 design D2 (docs/criteria/EX002_DECEQ_1108.md): causal polyphase equaliser on the decoded motion (radians).
y[t, j] = sum_{k=0..K} W[phi(t), k, j] x[t-k, j] + b[phi(t), j], phi(t) = t mod P (P = 4 = the decoder stride; frame 0 = the stream start
on the latent grid), frames before the start = frame 0. mode 'poly' (per phase), 'agnostic' (W, b shared across phases: the control).
Fit: per (phase, joint) ridge least squares toward the identity tap (W[., 0, .] = 1, others 0, b = 0) on round-trip pairs.
Default off everywhere: nothing imports this unless --dec-eq is given."""
from __future__ import annotations
import json
import numpy as np

P = 4


def lagged(x, K):
    """x [T, J] -> [T, K + 1, J]: column k = x[t - k] (clamped at frame 0)."""
    T = len(x); idx = np.clip(np.arange(T)[:, None] - np.arange(K + 1)[None], 0, None)
    return x[idx]


def fit(pairs, K=7, lam=1e-2, mode='poly'):
    """pairs: list of (x_hat [T, J], target [T, J]) on the stream grid. Returns dict(W [P, K+1, J], b [P, J], K, mode)."""
    J = pairs[0][0].shape[1]; nph = P if mode == 'poly' else 1
    A = np.zeros((nph, J, K + 2, K + 2)); r = np.zeros((nph, J, K + 2)); n = np.zeros(nph)
    for xh, tg in pairs:
        L = lagged(np.asarray(xh, np.float64), K); ph = (np.arange(len(xh)) % P) if mode == 'poly' else np.zeros(len(xh), int)
        F = np.concatenate([L, np.ones((len(xh), 1, J))], 1)            # [T, K + 2, J]
        for p in range(nph):
            m = ph == p; Fm = F[m]; tm = np.asarray(tg, np.float64)[m]
            A[p] += np.einsum('tkj,tlj->jkl', Fm, Fm); r[p] += np.einsum('tkj,tj->jk', Fm, tm); n[p] += m.sum()
    prior = np.zeros(K + 2); prior[0] = 1.0
    W = np.zeros((nph, K + 1, J)); b = np.zeros((nph, J))
    for p in range(nph):
        s = lam * n[p]                                                   # ridge scaled by the sample count (lambda per frame)
        for j in range(J):
            M = A[p, j] + s * np.eye(K + 2); v = r[p, j] + s * prior
            w = np.linalg.solve(M, v); W[p, :, j] = w[:K + 1]; b[p, j] = w[K + 1]
    if mode != 'poly': W = np.repeat(W, P, 0); b = np.repeat(b, P, 0)
    return dict(W=W, b=b, K=K, mode=mode, lam=lam)


def apply(eq, x):
    """x [T, J] decoded motion (frame 0 on the latent grid) -> equalised [T, J] (causal)."""
    x = np.asarray(x, np.float64); L = lagged(x, eq['K']); ph = np.arange(len(x)) % P
    return np.einsum('tkj,tkj->tj', L, np.asarray(eq['W'])[ph]) + np.asarray(eq['b'])[ph]


def save(eq, path):
    np.savez(path, W=eq['W'], b=eq['b'], K=eq['K'], mode=eq['mode'], lam=eq['lam'])


def load(path):
    z = np.load(path); return dict(W=z['W'], b=z['b'], K=int(z['K']), mode=str(z['mode']), lam=float(z['lam']))

"""11-06 Div/BA line: the stochastic flow sampler (scmg.infer.sde_step, Streamer(sde_eta=gamma)).
1. 1-D Gaussian data x1 ~ N(m, s^2) with the EXACT rectified-flow velocity: Euler ODE and the SDE (gamma 0.5 / 1 / 2) both
   reproduce mean m and std s (many steps); gamma = 0 equals the Euler step exactly.
2. The SDE keeps the per-step marginal: starting from exact samples of x_t, one step lands on the x_{t+dt} marginal.
3. Streamer default (sde_eta = 0) never calls sde_step (source check: the default branch is the original Euler line).
    python scripts/tests/test_sde_sampler.py      (CPU, seconds)
"""
import inspect, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import infer
from scmg.infer import sde_step

torch.manual_seed(0)
m, s = 1.5, 0.4
ok = True


def vel(x, t):   # exact E[x1 - n | x_t = x] for x1 ~ N(m, s^2), n ~ N(0, 1)
    var = t * t * s * s + (1 - t) ** 2
    ex1 = m + t * s * s * (x - t * m) / var; en = (1 - t) * (x - t * m) / var
    return ex1 - en


def sample(gamma, steps, N=200000):
    g = torch.Generator().manual_seed(1); x = torch.randn(N, dtype=torch.float64, generator=g)
    for k in range(steps):
        t = k / steps; v = vel(x, t)
        x = sde_step(x, v, t, 1 / steps, gamma, g) if (gamma > 0 and k < steps - 1) else x + v / steps
    return x


for gamma in (0.0, 0.5, 1.0, 2.0):
    x = sample(gamma, 400); mu, sd = float(x.mean()), float(x.std())
    good = abs(mu - m) < 0.01 and abs(sd - s) < 0.01
    ok &= good; print(f'gamma {gamma}: mean {mu:.4f} (want {m}) std {sd:.4f} (want {s}) {"OK" if good else "FAIL"}')
x = torch.randn(1000, dtype=torch.float64); v = vel(x, 0.3)
e0 = (sde_step(x, v, 0.3, 0.1, 0.0, torch.Generator().manual_seed(0)) - (x + v * 0.1)).abs().max().item()
ok &= e0 == 0.0; print(f'gamma 0 == Euler: max diff {e0}')
g = torch.Generator().manual_seed(2); N = 400000; t, dt = 0.4, 0.05
x1 = m + s * torch.randn(N, dtype=torch.float64, generator=g); n = torch.randn(N, dtype=torch.float64, generator=g)
xt = t * x1 + (1 - t) * n; y = sde_step(xt, vel(xt, t), t, dt, 1.0, g)
t2 = t + dt; want_m, want_s = t2 * m, (t2 * t2 * s * s + (1 - t2) ** 2) ** 0.5
good = abs(float(y.mean()) - want_m) < 0.01 and abs(float(y.std()) - want_s) < 0.01
ok &= good; print(f'one-step marginal: mean {float(y.mean()):.4f}/{want_m:.4f} std {float(y.std()):.4f}/{want_s:.4f} {"OK" if good else "FAIL"}')
src = inspect.getsource(infer.Streamer)
good = 'if self.sde_eta > 0 and s < self.steps - 1 and tval < self.sde_tmax:' in src and 'x = torch.where(known[..., None], x, x + v / self.steps)' in src
ok &= good; print('default branch = original Euler line:', 'OK' if good else 'FAIL')
print('ALL_OK' if ok else 'FAILED'); sys.exit(0 if ok else 1)

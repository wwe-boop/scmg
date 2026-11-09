"""Execution-deviation simulator (torch, vectorised) for the Robot State Adapter.

Given planned trajectories around a chunk boundary, produce what the robot ACTUALLY did up to the boundary and the
recovery target the generator should output after it:
  lag    first-order tracking lag (tau 30-120 ms) + gain error           (servo / controller tracking error)
  push   critically damped external displacement starting inside the history (human touch, collision, handshake)
  pose   large static offset (resume from a non-gesture pose: after pointing, handover, safety stop)
  clamp  velocity saturation of the controller (vmax 30-70 % of the plan's peak)
  none   actual == plan (the adapter must learn to follow the plan when nothing went wrong)
Recovery target after the boundary: plan + quintic-Hermite blend that starts at the actual position AND velocity
and reaches the plan with zero velocity/acceleration error after R frames (R 16-48 frames = 0.32-0.96 s).
All quantities in radians; frames at 50 Hz.
"""
import torch

from .common import ARMS                      # arm joints of the embodiment (EX002: 4..17 of 34; G1: 25..38 of 39)
KINDS = ('none', 'lag', 'push', 'pose', 'clamp')
DT = 1 / 50.


def hermite(s):
    """Quintic blend: b(0)=1, b'(0)=0, b''(0)=0, b(1)=0, b'(1)=0, b''(1)=0 ; c: c(0)=0, c'(0)=1 (per unit s)."""
    s = s.clamp(0, 1)
    b = 1 - 10 * s ** 3 + 15 * s ** 4 - 6 * s ** 5
    c = s - 6 * s ** 3 + 8 * s ** 4 - 3 * s ** 5
    return b, c


def simulate(plan, boundary, probs=(0.2, 0.3, 0.2, 0.15, 0.15), generator=None):
    """plan [M, F, J] radians; frames < boundary are history (executed), >= boundary future.
    Returns actual history [M, boundary, J], state (q, dq) at boundary-1, recovery target [M, F, J], kind [M]."""
    M, F, Jn = plan.shape; dev = plan.device
    g = generator
    u = lambda *s: torch.rand(*s, device=dev, generator=g)
    kind = torch.multinomial(torch.tensor(probs, device=dev), M, replacement=True, generator=g)
    hist = plan[:, :boundary]
    act = hist.clone()
    arm = torch.zeros(Jn, device=dev, dtype=torch.bool); arm[ARMS] = True
    # lag: first-order filter with gain error
    lag = kind == 1
    if lag.any():
        tau = 0.03 + 0.09 * u(int(lag.sum()), 1, 1)
        a = DT / (tau + DT); gain = 1 + 0.1 * (u(int(lag.sum()), 1, Jn) - .5)
        h = hist[lag]; y = torch.empty_like(h); y[:, 0] = h[:, 0]
        for f in range(1, boundary): y[:, f] = y[:, f - 1] + a[:, 0] * (h[:, f] * gain[:, 0] + h[:, 0] * (1 - gain[:, 0]) - y[:, f - 1])
        act[lag] = y
    # push: critically damped displacement on a random subset of arm joints
    push = kind == 2
    if push.any():
        n = int(push.sum())
        t0 = (boundary * (0.3 + 0.65 * u(n, 1, 1))).floor()
        amp = (0.05 + 0.35 * u(n, 1, Jn)) * torch.where(u(n, 1, Jn) < .5, -1., 1.) * (u(n, 1, Jn) < .5) * arm
        td = 0.08 + 0.25 * u(n, 1, 1)
        f = torch.arange(boundary, device=dev)[None, :, None].float()
        s = ((f - t0) * DT / td).clamp(min=0)
        act[push] = hist[push] + amp * (s * torch.exp(1 - s)) * (f >= t0)          # rises then decays
    # pose: static offset of the arms (resume from another pose)
    pose = kind == 3
    if pose.any():
        n = int(pose.sum())
        off = 0.3 * torch.randn(n, 1, Jn, device=dev, generator=g) * arm
        act[pose] = hist[pose] + off
    # clamp: velocity saturation
    cl = kind == 4
    if cl.any():
        h = hist[cl]; v = (h[:, 1:] - h[:, :-1]) / DT
        vmax = (0.3 + 0.4 * u(int(cl.sum()), 1, 1)) * v.abs().amax(1, keepdim=True).clamp(min=1e-3)
        y = torch.empty_like(h); y[:, 0] = h[:, 0]
        for f in range(1, boundary):
            y[:, f] = y[:, f - 1] + (h[:, f] - y[:, f - 1]).clamp(-vmax[:, 0] * DT, vmax[:, 0] * DT)
        act[cl] = y
    q = act[:, -1]; dq = (act[:, -1] - act[:, -2]) / DT
    dq_plan = (hist[:, -1] - hist[:, -2]) / DT
    e, de = q - hist[:, -1], dq - dq_plan
    R = (16 + 32 * u(M, 1, 1)).floor()
    tf = torch.arange(F - boundary, device=dev)[None, :, None].float() + 1
    b, c = hermite(tf / R)
    target = plan.clone()
    target[:, boundary:] = plan[:, boundary:] + e[:, None] * b + de[:, None] * c * R * DT
    target[:, :boundary] = act
    return act, q, dq, target, kind


def state_vector(q, dq, q_plan, qm, qs, contact=None):
    """Robot state -> adapter input [M, 3J+2] (normalised position, scaled velocity, deviation, contact flags)."""
    c = torch.zeros(q.shape[0], 2, device=q.device) if contact is None else contact
    return torch.cat([(q - qm) / qs, dq * 0.2, (q - q_plan) / qs, c], -1)

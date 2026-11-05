"""Tests of scmg/landing.py (Claude 11-05).   PYTHONPATH=. python scripts/tests/test_landing.py [HUMAN_ENVELOPE.json]"""
import json, os, sys, tempfile
import numpy as np
import torch
from scmg import landing as L

ENV = sys.argv[1] if len(sys.argv) > 1 else '/home/xr-25b7/bulk/landing-1105/HUMAN_ENVELOPE.json'
ok = lambda c, m: print(('PASS ' if c else 'FAIL ') + m) or c
res = []

# ---- soft limit: identity inside, C1, monotone, slope <= 1, strictly inside the bound
x = np.linspace(-6, 6, 200001)
y = L.soft_limit(x, -1.0, 0.8, -1.3, 1.1)
inside = (x >= -1) & (x <= 0.8)
res.append(ok(np.array_equal(y[inside], x[inside]), 'identity on the knee interval'))
dy = np.diff(y) / np.diff(x)
res.append(ok((dy >= 0).all() and dy.max() <= 1 + 1e-9, f'non-decreasing, slope in [0, 1] (max {dy.max():.6f}; tanh saturates to 0 far out)'))
res.append(ok(y.max() < 1.1 and y.min() > -1.3, 'strictly inside the bound'))
k = np.searchsorted(x, 0.8); res.append(ok(abs(dy[k] - 1) < 1e-3 and abs(dy[k - 2] - 1) < 1e-9, 'C1 at the knee'))

# ---- ArmEnvelope: only the selected joints change; speed can only shrink
env = L.ArmEnvelope(ENV)
rng = np.random.default_rng(0)
q = np.cumsum(rng.normal(0, 0.05, (5000, 34)), 0).astype(np.float32)
o = env(q)
other = np.setdiff1d(np.arange(34), env.idx)
res.append(ok(o.dtype == q.dtype and o.shape == q.shape, 'dtype / shape kept'))
res.append(ok(np.array_equal(o[:, other], q[:, other]), 'non-envelope joints untouched (incl. elbow 4, fingers, waist, head)'))
res.append(ok(sorted(env.idx.tolist()) == sorted(L.UPPER + L.WRIST) and 7 not in env.idx and 14 not in env.idx, 'joints = arm 1-3, 5-7'))
v_in = np.abs(np.diff(q[:, env.idx].astype(np.float64), axis=0)); v_out = np.abs(np.diff(o[:, env.idx].astype(np.float64), axis=0))
res.append(ok((v_out <= v_in + 1e-5).all(), 'per-joint frame-to-frame change never grows'))
res.append(ok(((o[:, env.idx] > env.lo_b - 1e-6) & (o[:, env.idx] < env.hi_b + 1e-6)).all(), 'output inside the bound'))
o1 = np.stack([env(f) for f in q[:50]])
res.append(ok(np.allclose(o1, o[:50]), 'per-frame call == batch call (stateless, streaming-safe)'))
e = json.load(open(ENV)); med = np.asarray(e['pooled']['median'], np.float32)
qm = np.tile(np.r_[med, np.zeros(16, np.float32)], (10, 1))
res.append(ok(np.array_equal(env(qm), qm), 'human median pose unchanged'))
try: L.ArmEnvelope(ENV, knee=99.9, bound=99.0); res.append(ok(False, 'bad quantile order rejected'))
except ValueError: res.append(ok(True, 'bad quantile order rejected'))

# ---- apply_cond
c = dict(spk=np.ones(192, np.float32), spk_valid=np.float32(1), ref=np.ones((128, 34), np.float32), ref_valid=np.float32(1),
         source=np.int64(2), text=np.zeros((4, 768), np.float32))
os.environ.pop('SCMG_COND', None)
res.append(ok(L.apply_cond(c) is c, 'unset SCMG_COND = the same object (bit-identical)'))
d = L.apply_cond(c, 'spk=null,ref=null,source=5')
res.append(ok(float(d['spk_valid']) == 0 and float(d['ref_valid']) == 0 and not d['spk'].any() and not d['ref'].any()
              and int(d['source']) == 5 and d['source'].dtype == np.int64 and d['ref'].shape == (128, 34), 'deployment conditions'))
res.append(ok(float(c['spk_valid']) == 1 and c['spk'].all(), 'input not modified'))
with tempfile.TemporaryDirectory() as t:
    r = rng.normal(size=(128, 34)).astype(np.float32); np.save(t + '/p.npy', r)
    d = L.apply_cond(c, f'ref={t}/p.npy')
    res.append(ok(np.array_equal(d['ref'], r) and float(d['ref_valid']) == 1 and float(d['spk_valid']) == 1, 'persona reference loaded'))
    np.save(t + '/bad.npy', r[:64])
    try: L.apply_cond(c, f'ref={t}/bad.npy'); res.append(ok(False, 'wrong ref shape rejected'))
    except ValueError: res.append(ok(True, 'wrong ref shape rejected'))
ct = {k: torch.as_tensor(v) for k, v in c.items()}
d = L.apply_cond(ct, 'spk=null,source=5')
res.append(ok(isinstance(d['spk'], torch.Tensor) and d['source'].dtype == torch.int64 and float(d['spk_valid']) == 0
              and float(d['ref_valid']) == 1, 'torch conditions'))
for bad in ('foo=1', 'spk', 'ref=null,bar=2'):
    try: L.apply_cond(c, bad); res.append(ok(False, f'bad spec {bad!r} rejected'))
    except (ValueError, KeyError): res.append(ok(True, f'bad spec {bad!r} rejected'))

# ---- metrics
t = np.zeros((500, 34)); t[:, 4:18] = np.sin(np.arange(500) / 10.)[:, None]            # moving
g = t.copy(); g[100:250] = g[100]                                                        # 3 s frozen
sp = np.zeros(500, bool); sp[50:400] = True
med = float(np.median(L.arm_speed(t)))
seg, fz, spk = L.freeze_segments(g, sp, med)
res.append(ok(len(seg) == 1 and 145 <= fz <= 150 and spk == 350, f'freeze segment found ({seg}, {fz}/{spk})'))
seg, fz, _ = L.freeze_segments(t, sp, med)
res.append(ok(len(seg) == 0 and fz == 0, 'no freeze on moving motion'))
sp2 = np.zeros(100, bool); sp2[10:20] = sp2[30:40] = sp2[80:] = True
m2 = L.speech_mask(sp2, 0.3)
res.append(ok(m2[10:40].all() and not m2[45:75].any() and m2[80:].all(), 'speech gaps <= 0.3 s closed, longer kept'))
res.append(ok(L.runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)], 'runs'))

# ---- fractional speaker condition (model.global_cond): 0 / 1 unchanged, a in (0, 1) blends
from types import SimpleNamespace
from scmg.model import SCMG
torch.manual_seed(0)
stub = SimpleNamespace(spk_in=torch.nn.Linear(192, 8), spk_null=torch.nn.Parameter(torch.randn(8)), source=torch.nn.Embedding(8, 8))
spk_t = torch.randn(4, 192); src = torch.tensor([5, 5, 1, 5]); v = torch.tensor([0., 1., 1., 0.])
with torch.no_grad():
    old = torch.where(v[:, None] > .5, stub.spk_in(spk_t), stub.spk_null) + stub.source(src)
    res.append(ok(torch.equal(SCMG.global_cond(stub, spk_t, v, src), old), 'global_cond: 0 / 1 spk_valid bit-identical to the pre-change formula'))
    v2 = torch.tensor([0., .3, 1., .5]); out = SCMG.global_cond(stub, spk_t, v2, src); ref = stub.spk_in(spk_t)
    exp1 = .3 * ref[1] + .7 * stub.spk_null + stub.source(src)[1]
    res.append(ok(torch.equal(out[0], old[0]) and torch.equal(out[2], old[2]) and float((out[1] - exp1).abs().max()) < 1e-6, 'global_cond: fractional rows blend, others unchanged'))
d = L.apply_cond(c, 'spk_alpha=0.3')
res.append(ok(abs(float(d['spk_valid']) - .3) < 1e-6, 'spk_alpha sets a fractional spk_valid'))
try: L.apply_cond(c, 'spk_alpha=1.0'); res.append(ok(False, 'spk_alpha outside (0, 1) rejected'))
except ValueError: res.append(ok(True, 'spk_alpha outside (0, 1) rejected'))

# ---- still overlap
gs = np.zeros(200, bool); ts = np.zeros(200, bool); ts[50:80] = True; gs[55:85] = True
o = L.still_overlap(gs, ts, 0); res.append(ok(o == (25, 30, 25, 30), f'still_overlap exact {o}'))
o = L.still_overlap(gs, ts, 12); res.append(ok(o == (30, 30, 30, 30), f'still_overlap with tolerance {o}'))
o = L.still_overlap(np.roll(ts, 100), ts, 12); res.append(ok(o[0] == 0 and o[2] == 0, 'disjoint pauses -> 0'))
res.append(ok(np.array_equal(L.dilate(np.array([0, 0, 1, 0, 0, 0], bool), 1), np.array([0, 1, 1, 1, 0, 0], bool)), 'dilate'))

print('ALL PASS' if all(res) else f'{res.count(False)} FAILED'); sys.exit(0 if all(res) else 1)

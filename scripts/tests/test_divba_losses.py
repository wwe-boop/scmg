"""11-06 Div/BA line: StageCfg w_spec (a') / w_onset (b') (scmg/divba_losses.py, hooked into scmg/losses.py step_losses).
    SCMG_J=39 python scripts/tests/test_divba_losses.py              part A only (CPU, seconds)
    SCMG_J=39 CUDA_VISIBLE_DEVICES=k PARITY=1 CK=.. VAE=.. STATS=.. TRV=<view> TEXT=.. BASE_TREE=<exported pre-change code>
        python scripts/tests/test_divba_losses.py                  + part B (real model / batch, scripts/tests/_causal_loss_probe.py)
A  functions: spec_loss(gt, gt) = 0; a smoothed copy is penalised and a gradient step on it raises its 3-12 Hz power; the
   shape term is scale-free; spec_loss is two-sided (an over-busy copy is penalised as well); onset_loss: pauses on the onsets
   give R > 0 and beat pauses elsewhere; adding pauses everywhere does not raise R (density-normalised); R is scale-free;
   loss = 0 once R_pred >= R_gt.
B  real training batch (train split of the D-spk view, never dev/test): (1) the final recipe with w_spec = w_onset = 0 ->
   loss terms + every parameter gradient bitwise equal to BASE_TREE (the code before this change); (2) w_spec / w_onset on ->
   every pre-existing loss term bitwise unchanged (no RNG shift), the new terms finite, gradients reach the local flow.
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve(); CODE = HERE.parents[2]
sys.path.insert(0, str(CODE))
from scmg.divba_losses import spec_loss, onset_loss, band_powers, UPPER

FAIL = []


def ok(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not cond: FAIL.append(name)


# ------------------------------------------------------------------ A: functions
g = torch.Generator().manual_seed(0)
M, T, Jn = 64, 32, 39
tt = torch.arange(T, dtype=torch.float64) / 50.0
gt = torch.zeros(M, T, Jn, dtype=torch.float64)
for f, a in ((1.0, 0.3), (2.5, 0.12), (4.5, 0.06), (8.0, 0.02)):
    ph = 2 * np.pi * torch.rand(M, 1, Jn, generator=g, dtype=torch.float64)
    gt = gt + a * torch.sin(2 * np.pi * f * tt[None, :, None] + ph)
w = torch.ones(M, dtype=torch.float64)
l0 = spec_loss(gt, gt, w)['spec'].item(); ok('A spec_loss(gt, gt) == 0', abs(l0) < 1e-10, f'{l0:.2e}')
k = torch.tensor([1, 2, 3, 2, 1], dtype=torch.float64) / 9
sm = torch.nn.functional.conv1d(gt.permute(0, 2, 1).reshape(-1, 1, T), k[None, None], padding=2).reshape(M, Jn, T).permute(0, 2, 1)
y = sm.clone().requires_grad_(True); L = spec_loss(y, gt, w)
ok('A smoothed copy penalised', L['spec'].item() > 0.05, f"{L['spec'].item():.4f} hf ratio {L['spec_hf_ratio'].item():.3f}")
L['spec'].backward()
y2 = (sm - 0.5 * y.grad).detach(); hf = lambda z: band_powers(z[..., UPPER].float(), w.float())[0][1:].sum().item()
ok('A gradient step raises 3-12 Hz power', hf(y2) > hf(sm), f'{hf(sm):.4g} -> {hf(y2):.4g}')
s1 = spec_loss(sm, gt, w)['spec_shape'].item(); s2 = spec_loss(3.0 * sm, gt, w)['spec_shape'].item()
ok('A shape term scale-free', abs(s1 - s2) < 1e-6 * max(1, s1), f'{s1:.6f} vs {s2:.6f}')
busy = gt + 0.05 * torch.sin(2 * np.pi * 10 * tt[None, :, None] + torch.rand(M, 1, Jn, generator=g, dtype=torch.float64))
ok('A two-sided: over-busy copy penalised', spec_loss(busy, gt, w)['spec'].item() > 0.05, f"{spec_loss(busy, gt, w)['spec'].item():.4f}")

T2 = 200; fw = torch.ones(8, T2, dtype=torch.float64)
on = torch.zeros(8, T2, dtype=torch.float64); on_idx = torch.arange(10, T2 - 10, 25); on[:, on_idx] = 1.0
pos = torch.cumsum(torch.ones(8, T2, dtype=torch.float64), 1) / 50


def traj(pause_at):        # 1-D motion along every upper joint; speed dips (pauses) at the given frames
    sp = torch.ones(T2, dtype=torch.float64)
    for p in pause_at:
        for d in range(-3, 4):
            if 0 <= p + d < T2: sp[p + d] = min(float(sp[p + d]), abs(d) / 4)
    x = torch.cumsum(sp, 0) / 50
    return x[None, :, None].expand(8, T2, Jn).clone()


R = lambda y_: onset_loss(y_, y_, on, fw)['onset_R'].item()
r_on, r_off = R(traj(on_idx.tolist())), R(traj((on_idx + 12).tolist()))
ok('A pauses on onsets: R > 0, pauses off onsets: R < R_on', r_on > 0.1 and r_off < 0.5 * r_on, f'{r_on:.3f} / {r_off:.3f}')
r_all = R(traj(list(range(5, T2, 6))))
ok('A pauses everywhere do not raise R (density-normalised)', r_all < 0.5 * r_on, f'{r_all:.3f}')
ok('A R scale-free', abs(R(3.0 * traj(on_idx.tolist())) - r_on) < 1e-5 * abs(r_on), f'{R(3.0 * traj(on_idx.tolist())):.8f} vs {r_on:.8f}')
lo = onset_loss(traj(on_idx.tolist()), traj((on_idx + 12).tolist()), on, fw)['onset'].item()
ok('A loss 0 when R_pred >= R_gt', lo == 0.0)
lo2 = onset_loss(traj((on_idx + 12).tolist()), traj(on_idx.tolist()), on, fw)['onset'].item()
ok('A loss > 0 when R_pred < R_gt', lo2 > 0.1, f'{lo2:.3f}')

# ------------------------------------------------------------------ B: real batch parity (GPU)
if os.environ.get('PARITY') == '1':
    E = os.environ; SP = Path(E.get('SP', tempfile.mkdtemp(prefix='divba_')))
    CK, VAE, STATS, TRV, TEXT, BT = E['CK'], E['VAE'], E['STATS'], E['TRV'], E['TEXT'], E['BASE_TREE']
    from scmg.data import View, WindowDataset, collate
    stats = json.loads(Path(STATS).read_text())
    vt = View(TRV, 'train', TEXT, stats); ds = WindowDataset(vt, 256, max_tokens=256)
    rng = np.random.default_rng(7); cis = rng.choice(len(vt.clips), 8, replace=False)
    smp = [ds.sample(int(c), int(rng.integers(0, max(1, vt.clips[int(c)]['frames'] - 256))), np.random.default_rng(int(c))) for c in cis]
    torch.save(collate(smp), SP / 'batch.pt')
    SC = dict(p_stream=0.67, p_async=1.0, p_async_seg=0.0, self_force_p=0, w_safe=300)

    def probe(code, sc, tag):
        out = SP / f'probe_{tag}.pt'
        r = subprocess.run([sys.executable, str(HERE.parent / '_causal_loss_probe.py'), str(SP / 'batch.pt'), str(out), CK, VAE, STATS,
                            '{"adsc": true}', json.dumps(sc)], env=dict(os.environ, PYTHONPATH=str(code)), cwd=str(code), capture_output=True, text=True)
        if r.returncode: print(r.stdout[-2000:], r.stderr[-3000:])
        return torch.load(out, weights_only=False) if r.returncode == 0 else None

    pb, pn = probe(BT, SC, 'base'), probe(CODE, SC, 'new')
    same = pb is not None and pn is not None and pb['loss'].keys() == pn['loss'].keys() and all(torch.equal(pb['loss'][k], pn['loss'][k]) for k in pb['loss']) \
        and pb['grad'].keys() == pn['grad'].keys() and all(torch.equal(pb['grad'][k], pn['grad'][k]) for k in pb['grad'])
    ok('B switches off: loss + grads bitwise == BASE_TREE', same, f"loss {({k: round(float(v), 5) for k, v in pn['loss'].items()}) if pn else None}")
    for tag, extra, key in (('spec', dict(w_spec=0.3), 'spec'), ('onset', dict(w_onset=1.0), 'onset')):
        pt = probe(CODE, {**SC, **extra}, tag)
        if pt is None: ok(f'B {tag} probe', False); continue
        pre = [k for k in pn['loss'] if k != 'loss']
        ok(f'B {tag} on: pre-existing terms bitwise unchanged', all(torch.equal(pn['loss'][k], pt['loss'][k]) for k in pre))
        ok(f'B {tag} on: term finite', bool(torch.isfinite(pt['loss'][key])), f"{key} {float(pt['loss'][key]):.5f}")
        dg = sum(float((pt['grad'][k] - pn['grad'][k]).abs().sum()) for k in pt['grad'] if 'flow' in k or 'local' in k)
        ok(f'B {tag} on: gradient reaches the local flow', dg > 0, f'|dgrad| {dg:.3e}')
print('ALL_OK' if not FAIL else f'FAILED {FAIL}'); sys.exit(1 if FAIL else 0)

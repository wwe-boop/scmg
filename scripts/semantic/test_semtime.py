"""sem_time (semantic-first temporal stream, 11-07) acceptance tests; helpers from test_ctl.py (same env).
 A hos-base + adsc == + sem_time modules (zero-init), offline and engine, bitwise (with the full text present)
 B causality (engine streaming, route perturbed): masking every word token from time T0 on cannot change frames before the
   first masked token's x2 arrival (minus the decoder's 1-latent look-ahead)
 C training: finite; gradients reach semt / semt_flow / semt_plan after one step; training default path is covered by test_ctl A
Prints ALL_OK / SOME_FAIL."""
import json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ctl as TC


def main():
    ok = {}
    st, v, vae = TC.setup(); dv = TC.setup(split='val')[1]
    base = TC.model({'adsc': True}); full = TC.model({'adsc': True, 'sem_time': True})
    for p in ('offline', 'engine'):
        ok[f'A_zero_init_{p}'] = all(np.array_equal(TC.gen(base, vae, dv, st, ci, p), TC.gen(full, vae, dv, st, ci, p)) for ci in (0, 1))
    with torch.no_grad():
        full.semt_flow.weight.normal_(0, 0.05); full.semt_plan.weight.normal_(0, 0.05)
    ci = max(range(len(dv.clips)), key=lambda i: dv.clips[i]['frames'] if dv.clips[i]['frames'] < 1200 else 0)
    nch = dv.clips[ci]['frames'] // 16
    from scmg.data import WindowDataset
    from scmg.model import x2_arrival
    s = WindowDataset(dv, nch * 16, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
    T0 = nch * 0.32 * 0.6
    cut = np.isfinite(s['ttime']) & (s['ttime'] >= T0) & s['tmask']
    tm2 = s['tmask'] & ~cut
    y0 = TC.gen(full, vae, dv, st, ci, 'engine', nch=nch); y1 = TC.gen(full, vae, dv, st, ci, 'engine', nch=nch, ctl=dict(tmask=tm2))
    arr = x2_arrival(torch.tensor(s['ttime'])[None], torch.tensor(s['tmask'])[None], torch.tensor([15.]), torch.tensor([0.15]))[0].numpy()
    first_arr = float(arr[cut].min())
    allowed = int(np.ceil(first_arr / 0.32 - 1e-6)) * 16 - 4
    d = np.abs(y0 - y1).max(-1); first = int(np.argmax(d > 0)) if (d > 0).any() else len(d)
    ok['B_causal'] = first >= allowed
    print('B', dict(first_changed=first, allowed_from=allowed, first_arrival_s=round(first_arr, 3), changed=bool((d > 0).any())))
    b = TC.batch_of(v, n=8, seed=3)
    l, g = TC.grads_losses(full, vae, b, st, TC.SC)
    with torch.no_grad():
        for n_, p_ in full.named_parameters():
            if n_.startswith('semt') and p_.grad is not None: p_ -= 1e-2 * p_.grad.sign()
    l, g = TC.grads_losses(full, vae, b, st, TC.SC)
    reach = {k: any(n_.startswith(k) and float(x.abs().sum()) > 0 for n_, x in g.items()) for k in ('semt.', 'semt_flow', 'semt_plan')}
    ok['C_train'] = all(np.isfinite(list(l.values()))) and all(reach.values())
    print('C', reach)
    print(json.dumps(ok)); print('ALL_OK' if all(ok.values()) else 'SOME_FAIL')


if __name__ == '__main__':
    main()

"""mask-conditioned Local Flow (cfg mask_cond + StageCfg p_mask, 11-07 item 5b) acceptance tests; helpers from test_ctl.py.
 A with p_mask 0 the mask_cond model's losses equal the base model's bitwise (f_mask unused); default training path vs HEAD = test_ctl A
 B with p_mask 1: finite losses, inserted positions excluded from the loss, gradient reaches f_mask
 C inference without insertion: mask_cond model (f_mask randomised) == base model bitwise, offline and engine
Prints ALL_OK / SOME_FAIL."""
import json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ctl as TC


def main():
    ok = {}
    st, v, vae = TC.setup(); dv = TC.setup(split='val')[1]
    base = TC.model({'adsc': True}); mc = TC.model({'adsc': True, 'mask_cond': True})
    b = TC.batch_of(v, n=8, seed=3)
    l0, _ = TC.grads_losses(base, vae, b, st, TC.SC); l1, _ = TC.grads_losses(mc, vae, b, st, TC.SC)
    ok['A_p_mask0_bitwise'] = l0 == l1
    l2, g2 = TC.grads_losses(mc, vae, b, st, dict(TC.SC, p_mask=1.0))
    ok['B_finite'] = all(np.isfinite(list(l2.values())))
    ok['B_grad_f_mask'] = 'f_mask' in g2 and float(g2['f_mask'].abs().sum()) > 0
    ok['B_loss_changes'] = l2['flow'] != l1['flow']
    with torch.no_grad(): mc.f_mask.normal_(0, 0.1)
    for p in ('offline', 'engine'):
        ok[f'C_no_insert_{p}'] = all(np.array_equal(TC.gen(base, vae, dv, st, ci, p), TC.gen(mc, vae, dv, st, ci, p)) for ci in (0, 1))
    print(dict(flow_p0=l1['flow'], flow_p1=l2['flow']))
    print(json.dumps(ok)); print('ALL_OK' if all(ok.values()) else 'SOME_FAIL')


if __name__ == '__main__':
    main()

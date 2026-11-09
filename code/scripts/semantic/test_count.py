"""count control (11-07 §7.55 item 3) acceptance tests; helpers from test_ctl.py (same env).
    python scripts/semantic/test_count.py --ctl <ctl-tokens-hos-count>
 A token_value: no 'count' field (or None) -> the 11-05 values exactly; a count token sets slot 0 only; POINT tokens ignore counts
 B labels: re-measuring the dev GT with count_np reproduces every stored dev count (label == evaluation rule); the soft count with
   very sharp sigmoids equals the hard count
 C training with count tokens: ctl_count present, finite, > 0 tokens, gradients reach the ctl modules (after one step);
   counterfactual counts change the loss target
Prints ALL_OK / SOME_FAIL."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ctl as TC


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ctl', required=True); a = ap.parse_args()
    from scmg import ctl
    from g1_ctl_apex import gt_geo
    ok = {}
    t0 = dict(type=2, t=1.0, a=0.8, b=1.9, amp_bin=1)
    v0, v1 = ctl.token_value(t0), ctl.token_value(dict(t0, count=None))
    v2 = ctl.token_value(dict(t0, count=3)); vp = ctl.token_value(dict(type=1, t=1., a=.8, b=1.9, target=[1, 0, .3], hand=1, amp_bin=0, count=2))
    vp0 = ctl.token_value(dict(type=1, t=1., a=.8, b=1.9, target=[1, 0, .3], hand=1, amp_bin=0))
    ok['A_token_value'] = (all(np.array_equal(x, y) for x, y in zip(v0[1:], v1[1:])) and v2[1][0] == ctl.count_value(3) and v2[2][0] == 1
                           and np.array_equal(v2[1][1:], v0[1][1:]) and all(np.array_equal(x, y) for x, y in zip(vp[1:], vp0[1:])))
    raw = json.loads((Path(a.ctl) / 'CTL.json').read_text()); thr = json.loads((Path(a.ctl) / 'APEX.json').read_text())['action_speed']
    dev = [s for s, ts in raw.items() if ts and ts[0]['split'] == 'dev' and any(t.get('count') is not None for t in ts)][:40]
    from scmg.common import P
    G = gt_geo(P(TC.VIEW), dev)
    same = []; soft_ok = []
    for s in dev:
        sp = ctl.count_speed(G[s]['wrist'])
        for t in raw[s]:
            if t.get('count') is None: continue
            fa, fb = int(round(t['a'] * 50)), int(round(t['b'] * 50))
            same.append(ctl.count_np(G[s]['wrist'], fa, fb, thr, sp) == t['count'])
            h = ctl.span_hand(torch.as_tensor(G[s]['wrist'][fa:fb]))
            bh, bn = ctl.COUNT_BETA_H, ctl.COUNT_BETA_N; ctl.COUNT_BETA_H, ctl.COUNT_BETA_N = 1e6, 1e6
            soft = float(ctl.peak_score(sp, thr, soft=True)[fa:fb, h].sum()); ctl.COUNT_BETA_H, ctl.COUNT_BETA_N = bh, bn
            soft_ok.append(abs(min(soft, ctl.COUNT_MAX) - min(float(ctl.peak_score(sp, thr)[fa:fb, h].sum()), ctl.COUNT_MAX)) < 1e-3)
    ok['B_labels_reproduced'] = bool(same) and all(same); ok['B_soft_limit_is_hard'] = all(soft_ok)
    print('B', dict(n=len(same), reproduced=float(np.mean(same)), soft_ok=float(np.mean(soft_ok))))
    st, vc, vae = TC.setup(a.ctl)
    full = TC.model({'adsc': True, 'ctl': True, 'ctl_flow': True})
    b = TC.batch_of(vc, n=8, seed=3, ctl_focus=0.5)
    sc = dict(TC.SC, ctl_w=1.0, w_ctl_aux=1.0, w_ctl_count=1.0, ctl_drop=0.0, ctl_field_drop=0.0, ctl_count_thr=thr)
    l, g = TC.grads_losses(full, vae, b, st, sc)
    with torch.no_grad():
        for n_, p_ in full.named_parameters():
            if n_.startswith('ctl') and p_.grad is not None: p_ -= 1e-2 * p_.grad.sign()
    l, g = TC.grads_losses(full, vae, b, st, sc)
    reach = {k: any(n_.startswith(k) and float(x.abs().sum()) > 0 for n_, x in g.items()) for k in ('ctl_type', 'ctl_val', 'ctl_dt', 'ctl_fmlp')}
    ok['C_train'] = 'ctl_count' in l and np.isfinite(l['ctl_count']) and l['ctl_count_n'] > 0 and all(reach.values())
    lc, _ = TC.grads_losses(full, vae, b, st, dict(sc, ctl_cf_count_p=1.0))
    ok['C_counterfactual_changes_target'] = lc['ctl_count'] != l['ctl_count']
    print('C', dict(ctl_count=l.get('ctl_count'), n=l.get('ctl_count_n'), abs=l.get('ctl_count_abs'), cf=lc.get('ctl_count'), reach=reach))
    print(json.dumps(ok)); print('ALL_OK' if all(ok.values()) else 'SOME_FAIL')


if __name__ == '__main__':
    main()

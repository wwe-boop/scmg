"""control-token look-ahead (StageCfg / Streamer ctl_lookahead, 11-07) acceptance tests; helpers from test_ctl.py.
    python scripts/semantic/test_lookahead.py --ctl <ctl-tokens-hos>
 A training: ctl_lookahead 0 == the field's default bitwise; 0.5 changes the loss on a batch with annotation tokens
 B inference offline (final text: every token visible anyway): look-ahead changes nothing (bitwise)
 C inference engine (streaming): look-ahead changes the motion, but never before t - 0.5 s - the 3 s route window (causal bound)
Prints ALL_OK / SOME_FAIL."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ctl as TC


def gen(m, vae, v, st, ci, proto, nch, ctl, la):
    from scmg.common import U80, CHUNK_FRAMES
    from scmg.data import WindowDataset
    from scmg.infer import generate
    from scmg.model import INF
    s = WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
    keys = ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c')
    cond = {k: s[k] for k in keys if k in s}; cond.update(ctl)
    kw = dict(H=INF) if proto == 'offline' else dict(H=2 * U80, text_mode='x2', x2_rate=15.0, x2_lag=0.15, deploy_timing=True)
    if la > 0: kw['ctl_lookahead'] = la
    with torch.no_grad():
        return np.asarray(generate(m, vae, s['feat'], cond, st, nch, steps=4, seed=0, ctx_t=0.9, cfg_w=2.5, **kw)[0], np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ctl', required=True); a = ap.parse_args()
    from scmg.ctl import token_value
    ok = {}
    stc, vc, vae = TC.setup(a.ctl); full = TC.model({'adsc': True, 'ctl': True, 'ctl_flow': True})
    b = TC.batch_of(vc, n=8, seed=3, ctl_focus=0.5)
    sc = dict(TC.SC, ctl_w=1.0, ctl_drop=0.0, ctl_field_drop=0.0)
    with torch.no_grad():
        full.ctl_fmlp[-1].weight.normal_(0, 0.05); full.ctl_text.weight.normal_(0, 0.05)
    l0, _ = TC.grads_losses(full, vae, b, stc, sc); lz, _ = TC.grads_losses(full, vae, b, stc, dict(sc, ctl_lookahead=0.0))
    l5, _ = TC.grads_losses(full, vae, b, stc, dict(sc, ctl_lookahead=0.5))
    ok['A_default_bitwise'] = l0 == lz; ok['A_lookahead_changes_loss'] = l5['loss'] != l0['loss']
    dv = TC.setup(split='val')[1]; full.eval()
    ci = max(range(len(dv.clips)), key=lambda i: dv.clips[i]['frames'] if dv.clips[i]['frames'] < 1200 else 0)
    nch = dv.clips[ci]['frames'] // 16
    tok = dict(type=1, t=nch * 0.32 * 0.6, target=[1.5, 0.3, 0.4], hand=1, amp_bin=2); tv = token_value(tok)
    cl = dict(ctl_t=np.array([tok['t']], np.float32), ctl_type=np.array([tv[0]]), ctl_v=tv[1][None], ctl_vm=tv[2][None])
    ok['B_offline_unchanged'] = np.array_equal(gen(full, vae, dv, stc, ci, 'offline', nch, cl, 0.0), gen(full, vae, dv, stc, ci, 'offline', nch, cl, 0.5))
    y_none = gen(full, vae, dv, stc, ci, 'engine', nch, {}, 0.0)
    y0 = gen(full, vae, dv, stc, ci, 'engine', nch, cl, 0.0); y5 = gen(full, vae, dv, stc, ci, 'engine', nch, cl, 0.5)
    f = lambda y: (lambda d: int(np.argmax(d > 0)) if (d > 0).any() else len(d))(np.abs(y - y_none).max(-1))
    f0, f5 = f(y0), f(y5)
    d05 = np.abs(y0 - y5).max(-1); fd = int(np.argmax(d05 > 0)) if (d05 > 0).any() else len(d05)
    # with x2 text streaming the (untimed) token arrives with its word long before it is spoken in both cases; look-ahead only
    # makes its TIME known 0.5 s early -> the two runs may differ only from (t - 0.5 s - the 3 s route window) on
    lo = int((tok['t'] - 0.5 - 3.0) * 50) - 4
    ok['C_engine_differs_only_after_lookahead_bound'] = fd < len(d05) and fd >= lo
    print(dict(first_change_vs_notoken_no_la=f0, first_change_vs_notoken_la=f5, first_diff_la_vs_no_la=fd, lower_bound=lo,
               token_frame=int(tok['t'] * 50)))
    print(json.dumps(ok)); print('ALL_OK' if all(ok.values()) else 'SOME_FAIL')


if __name__ == '__main__':
    main()

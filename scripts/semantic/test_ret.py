"""Retrieval-route acceptance tests (11-06). Uses scripts/semantic/test_ctl.py helpers (same view / checkpoints / env).
    SCMG_J=39 SCMG_NO_GLITCH=1 SCMG_LOCAL_MIRROR=~/bulk/cpfs-mirror python scripts/semantic/test_ret.py --ret <bank dir> [--ctl <CTL dir>]
 A default path: hos-base + adsc == hos-base + adsc + ret modules (zero-init) without exemplars, offline and engine (bitwise)
 B zero-init: WITH exemplars at initialisation the output is still bitwise unchanged (the route's last layer is zero)
 C causality (engine streaming): after perturbing the route, an exemplar attached to a late word changes nothing before the word
   arrives AND is voiced (its time must be known to align the snippet): first changed frame >= first such chunk - 1 latent
 D training: finite losses; gradients reach ret_in / ret_out after one step; batches carry ret_* only with View(ret=...)
 E data: sampler with ret loaded draws the same windows as without (no extra RNG)
Prints ALL_OK / SOME_FAIL."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ctl as TC


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ret', required=True); a = ap.parse_args(); ok = {}
    st, v, vae = TC.setup(); dv = TC.setup(split='val')[1]
    base = TC.model({'adsc': True}); full = TC.model({'adsc': True, 'ret': True})
    bank = np.load(Path(a.ret) / 'BANK.npz')['lat'].astype(np.float32)
    for p in ('offline', 'engine'):
        ya = [TC.gen(base, vae, dv, st, ci, p) for ci in (0, 1)]; yb = [TC.gen(full, vae, dv, st, ci, p) for ci in (0, 1)]
        ok[f'A_no_exemplar_{p}'] = all(np.array_equal(x, y) for x, y in zip(ya, yb))
    rc = dict(ret_t=np.array([0.6, 1.4], np.float32), ret_lat=bank[:2])
    ok['B_zero_init_with_exemplar'] = bool(np.array_equal(TC.gen(base, vae, dv, st, 0, 'offline'), TC.gen(full, vae, dv, st, 0, 'offline', ctl=rc)))
    with torch.no_grad(): full.ret_out[-1].weight.normal_(0, 0.05)
    ci = max(range(len(dv.clips)), key=lambda i: dv.clips[i]['frames'] if dv.clips[i]['frames'] < 1200 else 0)
    nch = dv.clips[ci]['frames'] // 16; tl = nch * 0.32 - 0.6
    late = dict(ret_t=np.array([tl], np.float32), ret_lat=bank[:1])
    y0 = TC.gen(full, vae, dv, st, ci, 'engine', nch=nch); y1 = TC.gen(full, vae, dv, st, ci, 'engine', nch=nch, ctl=late)
    diff = np.abs(y0 - y1).max(-1); first = int(np.argmax(diff > 0)) if (diff > 0).any() else len(diff)
    # earliest admissible chunk: arrived (word arrival) AND voiced (t <= chunk start + H = 0.16 s); snippet starts ret_pre before t
    allowed_chunk = int(np.ceil((tl - 0.16) / 0.32 - 1e-6))
    ok['C_causal'] = first >= allowed_chunk * 16 - 4
    print('C', dict(first_changed=first, allowed_from=allowed_chunk * 16 - 4, changed=bool((diff > 0).any())))
    from scmg.data import View
    stc = TC.setup()[0]
    from scmg.common import P
    vr = View(TC.VIEW, 'train', TC.TEXT, stc, mmap=True, ret=a.ret + ':RET_train.json')
    b = TC.batch_of(vr, n=16, seed=4)
    ok['D_batch_has_ret'] = 'ret_t' in b and bool(b['ret_m'].any())
    l, g = TC.grads_losses(full, vae, b, stc, dict(TC.SC, ret_drop=0.0))
    with torch.no_grad():
        for n_, p_ in full.named_parameters():
            if n_.startswith('ret_') and p_.grad is not None: p_ -= 1e-2 * p_.grad.sign()
    l, g = TC.grads_losses(full, vae, b, stc, dict(TC.SC, ret_drop=0.0))
    ok['D_train'] = all(np.isfinite(list(l.values()))) and all(any(n_.startswith(k) and float(x.abs().sum()) > 0 for n_, x in g.items()) for k in ('ret_in', 'ret_out'))
    b0 = TC.batch_of(v, n=6, seed=11); b1 = TC.batch_of(vr, n=6, seed=11)
    eq = lambda x, y: x.shape == y.shape and bool(torch.equal(torch.nan_to_num(x.float(), 12345.), torch.nan_to_num(y.float(), 12345.)))
    ok['E_sampler_default'] = all(eq(b0[k], b1[k]) for k in b0)
    print(json.dumps(ok)); print('ALL_OK' if all(ok.values()) else 'SOME_FAIL')


if __name__ == '__main__':
    main()

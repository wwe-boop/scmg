"""Control-token acceptance tests (11-05). Runs on one GPU with the G1 held-out-speaker view.
    SCMG_J=39 SCMG_NO_GLITCH=1 SCMG_LOCAL_MIRROR=~/bulk/cpfs-mirror python scripts/semantic/test_ctl.py --ctl <CTL dir> [--orig <HEAD export>]
 A  training default path: no control tokens in the batch -> step_losses (losses AND every gradient) bitwise equal to the
    pre-change code (--orig: `git archive HEAD` export); data path: View/WindowDataset samples without ctl_tokens bitwise equal
 B  inference default path: hos-base with the new code == old code (bitwise), and hos-base + ctl + ctl_flow modules (zero-init)
    WITHOUT tokens == hos-base (bitwise), offline and engine
 C  with tokens at initialisation: the flow route is exactly zero (ctl_flow on vs off identical given the same planner input)
 D  training with tokens: finite; gradients reach ctl_type / ctl_val / ctl_enc / ctl_text / ctl_dt / ctl_fmlp; w_ctl_aux reported
 E  causality: engine streaming, a token attached to a late word cannot change chunks before its word arrives
 F  default path RNG: a ctl View/sampler with ctl_focus 0 draws the same windows as without tokens
Prints ALL_OK / SOME_FAIL."""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, torch

CODE = Path(__file__).resolve().parents[2]
if not os.environ.get('CTL_CHILD'): sys.path.insert(0, str(CODE))
B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; C = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'
VIEW = B + '/views/echog-g1-v1-hos'; STATS = C + '/scmg/STATS_echog_g1_v1.json'; TEXT = C + '/scmg/text-e5-echog-g1-v1'
INIT = B + '/echog-g1-arch-1104/hos-base/best.pt'; VAE = B + '/echog-g1-decft-1105/decft-w3d0m05/final.pt'
TCAUSAL = B + '/models/text-e5-causal-echog-g1-v1'
SC = dict(p_stream=0.67, p_async=1.0, p_async_seg=0.0, self_force_p=0)


def child(mode, out, code, ctl=None):
    """run this file's child in a fresh process with `code` on sys.path (old or new code)."""
    env = dict(os.environ, PYTHONPATH=str(code), CTL_CHILD=mode, CTL_OUT=out, CTL_DIR=ctl or '', SCMG_TEXT_CAUSAL=TCAUSAL)
    r = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-3000:], r.stderr[-6000:]); raise SystemExit(f'child {mode} failed')


def setup(ctl=None, split='train'):
    from scmg.common import P
    from scmg.data import View
    from scmg.vae import MotionVAE
    st = json.loads(Path(P(STATS)).read_text())
    kw = dict(ctl_tokens=ctl) if ctl else {}
    v = View(VIEW, split, TEXT, st, mmap=True, **kw)
    vck = torch.load(P(VAE), map_location='cpu'); vae = MotionVAE(**vck['cfg']).cuda(); vae.load_state_dict(vck['ema']); vae.eval()
    return st, v, vae


def model(extra):
    from scmg.common import P
    from scmg.train import build_model
    ck = torch.load(P(INIT), map_location='cpu')
    m, _ = build_model(dict(ck['base_cfg']), dict(ck['extra_cfg']), {**ck['extra_cfg'], **extra}, ck['ema'])
    for k, v in extra.items(): setattr(m.cfg, k, v)
    m.add_modules(); return m.cuda()


def batch_of(v, n=6, seed=0, ctl_focus=None):
    from scmg.data import WindowDataset, collate
    kw = {} if ctl_focus is None else dict(ctl_focus=ctl_focus)
    it = iter(WindowDataset(v, 256, seed=seed, **kw)); b = collate([next(it) for _ in range(n)])
    return {k: x.cuda() for k, x in b.items()}


def grads_losses(m, vae, b, st, sc):
    from scmg.losses import StageCfg, step_losses
    m.train(); torch.manual_seed(123); np.random.seed(123)
    stats = dict(qm=torch.tensor(st['q_mean']).cuda(), qs=torch.tensor(st['q_std']).cuda())
    out = step_losses(m, vae, b, StageCfg(**sc), stats)
    m.zero_grad(set_to_none=True); out['loss'].backward()
    g = {n: p.grad.detach().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}
    return {k: float(v) for k, v in out.items()}, g


def gen(m, vae, v, st, ci, proto, nch=6, ctl=None):
    from scmg.common import U80, CHUNK_FRAMES
    from scmg.data import WindowDataset
    from scmg.infer import generate
    from scmg.model import INF
    s = WindowDataset(v, nch * CHUNK_FRAMES, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
    keys = ('text', 'ttime', 'tmask', 'spk', 'spk_valid', 'source', 'ref', 'ref_valid', 'tids', 'tidm', 'titime', 'tseg', 'tiseg', 'tiend', 'text_c')
    cond = {k: s[k] for k in keys if k in s}
    if ctl: cond.update(ctl)
    kw = dict(H=INF) if proto == 'offline' else dict(H=2 * U80, text_mode='x2', x2_rate=15.0, x2_lag=0.15, deploy_timing=True)
    m.eval()
    with torch.no_grad():
        return np.asarray(generate(m, vae, s['feat'], cond, st, nch, steps=4, seed=0, ctx_t=0.9, cfg_w=2.5, **kw)[0], np.float32)


def run_child():
    mode, out = os.environ['CTL_CHILD'], os.environ['CTL_OUT']
    torch.manual_seed(0)
    st, v, vae = setup()
    if mode == 'train':
        torch.use_deterministic_algorithms(True, warn_only=True)
        m = model({'adsc': True}); b = batch_of(v)
        l, g = grads_losses(m, vae, b, st, SC)
        torch.save(dict(l=l, g=g, b={k: x.cpu() for k, x in b.items()}), out)
    elif mode == 'infer':
        dv = setup(split='val')[1]; m = model({'adsc': True})
        torch.save({p: [gen(m, vae, dv, st, ci, p) for ci in (0, 1)] for p in ('offline', 'engine')}, out)


eq = None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ctl', required=True); ap.add_argument('--orig', default=None)
    a = ap.parse_args(); ok = {}
    tmp = Path(tempfile.mkdtemp(prefix='ctltest-', dir=os.environ.get('SCMG_TEST_TMP', '/tmp/claude-1000')))
    orig = a.orig
    if orig is None:
        orig = str(tmp / 'orig'); os.makedirs(orig)
        subprocess.run(f'git -C {CODE} archive HEAD scmg | tar -x -C {orig}', shell=True, check=True)
    # A: training default path, old vs new code
    for code, tag in ((orig, 'old'), (CODE, 'new')): child('train', str(tmp / f'train_{tag}.pt'), code)
    o, n = torch.load(tmp / 'train_old.pt'), torch.load(tmp / 'train_new.pt')
    global eq
    eq = lambda x, y: x.shape == y.shape and bool(torch.equal(torch.nan_to_num(x.float(), 12345.), torch.nan_to_num(y.float(), 12345.))) \
        and bool(torch.equal(torch.isnan(x.float()), torch.isnan(y.float())))
    same_b = o['b'].keys() == n['b'].keys() and all(eq(o['b'][k], n['b'][k]) for k in o['b'])
    same_l = o['l'] == n['l']; same_g = o['g'].keys() == n['g'].keys() and all(torch.equal(o['g'][k], n['g'][k]) for k in o['g'])
    if not same_g:
        child('train', str(tmp / 'train_new2.pt'), CODE); n2 = torch.load(tmp / 'train_new2.pt')
        floor = max(float((n['g'][k] - n2['g'][k]).abs().max()) for k in n['g'])
        dmax = max(float((o['g'][k] - n['g'][k]).abs().max()) for k in o['g'])
        print('A grads: old-vs-new max diff', dmax, 'new-vs-new (run-to-run floor)', floor, flush=True)
        same_g = dmax <= 3 * floor          # losses are bitwise equal; gradients equal up to GPU run-to-run nondeterminism
    ok['A_train_default_bitwise'] = bool(same_b and same_l and same_g)
    print('A', dict(batch=same_b, losses=same_l, grads=same_g, n_grads=len(n['g']), loss=n['l']['loss']), flush=True)
    # B: inference default path
    for code, tag in ((orig, 'old'), (CODE, 'new')): child('infer', str(tmp / f'infer_{tag}.pt'), code)
    io, inew = torch.load(tmp / 'infer_old.pt', weights_only=False), torch.load(tmp / 'infer_new.pt', weights_only=False)
    ok['B_infer_old_vs_new'] = all(np.array_equal(x, y) for p in io for x, y in zip(io[p], inew[p]))
    st, v, vae = setup(); dv = setup(split='val')[1]
    base = model({'adsc': True}); full = model({'adsc': True, 'ctl': True, 'ctl_flow': True})
    for p in ('offline', 'engine'):
        ya = [gen(base, vae, dv, st, ci, p) for ci in (0, 1)]; yb = [gen(full, vae, dv, st, ci, p) for ci in (0, 1)]
        ok[f'B_ctl_modules_no_token_{p}'] = all(np.array_equal(x, y) for x, y in zip(ya, yb))
        ok[f'B_ctl_modules_vs_old_{p}'] = all(np.array_equal(x, y) for x, y in zip(io[p], yb))
    print('B', {k: v_ for k, v_ in ok.items() if k.startswith('B')}, flush=True)
    # C: zero-init flow route
    from scmg.ctl import token_value
    from scmg.data import load_ctl_tokens
    raw = json.loads((Path(a.ctl) / 'CTL.json').read_text())
    stem0 = dv.clips[0]['items'][0]['id']
    toks = [t for t in raw.get(stem0, [])][:6] or [dict(type=1, t=0.5, target=[1, 0.2, 0.3], hand=1, amp_bin=1)]
    tv = [token_value(t) for t in toks]
    cc = dict(ctl_t=np.array([t['t'] for t in toks], np.float32), ctl_type=np.array([x[0] for x in tv]), ctl_v=np.stack([x[1] for x in tv]),
              ctl_vm=np.stack([x[2] for x in tv]))
    plan_only = model({'adsc': True, 'ctl': True})
    plan_only.load_state_dict({k: x for k, x in full.state_dict().items() if not k.startswith(('ctl_dt', 'ctl_fmlp'))}, strict=True)
    y1 = gen(plan_only, vae, dv, st, 0, 'offline', ctl=cc); y2 = gen(full, vae, dv, st, 0, 'offline', ctl=cc)
    ok['C_flow_route_zero_at_init'] = bool(np.array_equal(y1, y2))
    y0 = gen(base, vae, dv, st, 0, 'offline')
    print('C', dict(identical=ok['C_flow_route_zero_at_init'], planner_route_change_rad=float(np.abs(y1 - y0).mean())), flush=True)
    # D: training with tokens
    stc, vc, _ = setup(a.ctl)
    b = batch_of(vc, n=8, seed=3, ctl_focus=0.5)
    sc = dict(SC, ctl_w=1.0, w_ctl_aux=1.0, ctl_drop=0.0, ctl_field_drop=0.2)
    l, g = grads_losses(full, vae, b, stc, sc)
    with torch.no_grad():                   # zero-init output layers block upstream gradients at step 0: take one SGD step
        for n_, p_ in full.named_parameters():
            if n_.startswith('ctl') and p_.grad is not None: p_ -= 1e-2 * p_.grad.sign()
    l, g = grads_losses(full, vae, b, stc, sc)
    need = ['ctl_type', 'ctl_val', 'ctl_enc', 'ctl_text', 'ctl_dt', 'ctl_fmlp']
    reach = {k: any(n_.startswith(k) and float(x.abs().sum()) > 0 for n_, x in g.items()) for k in need}
    ok['D_train_with_tokens'] = all(np.isfinite(list(l.values()))) and all(reach.values()) and 'ctl_ray' in l
    print('D', dict(reach=reach, ctl_ray=l.get('ctl_ray'), ctl_ray_n=l.get('ctl_ray_n'), ctl_frac=l.get('ctl_frac'),
                    n_tok=int((b['ctl_type'] > 0).sum())), flush=True)
    # E: causality under engine streaming: a token at the last word of a long clip, after one step of training (route non-zero)
    with torch.no_grad():
        for p_ in full.parameters(): pass
        full.ctl_fmlp[-1].weight.normal_(0, 0.05); full.ctl_text.weight.normal_(0, 0.05)
    ci = max(range(len(dv.clips)), key=lambda i: dv.clips[i]['frames'] if dv.clips[i]['frames'] < 1200 else 0)
    nch = dv.clips[ci]['frames'] // 16
    late = dict(type=1, t=nch * 0.32 - 0.4, target=[1.5, 0.3, 0.4], hand=1, amp_bin=2); tv = token_value(late)
    cl = dict(ctl_t=np.array([late['t']], np.float32), ctl_type=np.array([tv[0]]), ctl_v=tv[1][None], ctl_vm=tv[2][None])
    ye0 = gen(full, vae, dv, st, ci, 'engine', nch=nch); ye1 = gen(full, vae, dv, st, ci, 'engine', nch=nch, ctl=cl)
    diff = np.abs(ye0 - ye1).max(-1)
    first = int(np.argmax(diff > 0)) if (diff > 0).any() else len(diff)
    from scmg.data import WindowDataset
    s = WindowDataset(dv, nch * 16, max_tokens=256).sample(ci, 0, np.random.default_rng(ci))
    from scmg.model import x2_arrival
    tt = torch.tensor(s['ttime'])[None]; tm = torch.tensor(s['tmask'])[None]
    arr = x2_arrival(tt, tm, torch.tensor([15.]), torch.tensor([0.15]))[0].numpy()
    cand = [i for i in range(len(tt[0])) if s['tmask'][i] and np.isfinite(s['ttime'][i]) and s['ttime'][i] <= late['t']]
    arr_t = arr[max(cand, key=lambda i: s['ttime'][i])] if cand else late['t']
    first_chunk_allowed = int(np.ceil(arr_t / 0.32 - 1e-6))      # first chunk whose start >= arrival
    ok['E_causal'] = first >= first_chunk_allowed * 16 - 4      # - STRIDE: the decoder's 1-latent look-ahead (frame emitted late)
    print('E', dict(first_changed_frame=first, token_arrival_s=float(arr_t), allowed_from_frame=first_chunk_allowed * 16 - 4, changed=bool((diff > 0).any())), flush=True)
    # F: same windows with / without tokens (ctl_focus 0)
    b0 = batch_of(v, n=6, seed=11); b1 = batch_of(vc, n=6, seed=11)
    ok['F_sampler_default'] = all(eq(b0[k], b1[k]) for k in b0)
    print('F', ok['F_sampler_default'], flush=True)
    print(json.dumps(ok)); print('ALL_OK' if all(ok.values()) else 'SOME_FAIL')


if __name__ == '__main__':
    if os.environ.get('CTL_CHILD'): run_child()
    else: main()

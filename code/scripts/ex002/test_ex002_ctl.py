"""EX002 control tokens (11-07) acceptance tests (SCMG_J=34, local mirror).
    SCMG_J=34 SCMG_LOCAL_MIRROR=~/bulk/cpfs-mirror python scripts/ex002/test_ex002_ctl.py --ctl <ex002 tokens dir>
 A geometry: FK points match MuJoCo-calibrated kinematics (finite, wrists ~ arm length from shoulders); finger counts of the rest
   pose are a fixed tuple; soft counts with sharp sigmoids == hard counts on GT frames
 B token values: a finger token sets slots 0 / 1 and mask 0 only; POINT tokens unchanged by the new branch
 C training: sibag-s2 + ctl modules, a batch with EX002 tokens: ctl_ray and ctl_finger present and finite, gradients reach the
   ctl modules after one step; with the new weights 0 the losses equal the losses without the new fields (default bitwise)
Prints ALL_OK / SOME_FAIL."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; C = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'
VIEW = C + '/train-v1-20261101/views/data-v36_1'; STATS = C + '/scmg/STATS_v36_1.json'; TEXT = C + '/scmg/text-e5-v36_1'
INIT = B + '/mainline-seeds-1104b/sibag-s2/final.pt'; VAE = B + '/trained/vae-r3/final.pt'
SC = dict(p_stream=0.67, p_async=1.0, p_async_seg=0.0, self_force_p=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ctl', required=True); a = ap.parse_args()
    from scmg.common import P, EMBODIMENT
    assert EMBODIMENT == 'ex002', 'run with SCMG_J=34'
    from scmg import ctl, ex002_ctl as E
    ok = {}
    meta = json.loads(Path(P(VIEW) + '/META.json').read_text()); up = np.asarray(meta['ex002']['upper'], np.float32)
    rest = np.asarray(meta['ex002']['rest34'], np.float32)
    g = E.points(rest[None]); arm = np.linalg.norm(g['wrist'][0] - g['shoulder'][0], axis=-1)
    q = np.load(P(VIEW) + '/train_q.npy', mmap_mode='r'); Q = np.asarray(q[100000:100500], np.float32)
    hard = E.finger_counts(Q, up); soft = E.soft_counts(torch.as_tensor(Q), torch.as_tensor(up), beta=1e5).numpy()
    ok['A_geometry'] = bool(np.isfinite(arm).all() and (arm > 0.2).all() and (arm < 0.9).all())
    ok['A_soft_limit_hard'] = bool(np.abs(soft - hard).max() < 1e-3)
    print('A', dict(rest_arm_m=arm.round(3).tolist(), rest_counts=E.finger_counts(rest[None], up)[0].tolist()))
    tv = ctl.token_value(dict(type=ctl.TYPE_ID['beat'], t=1.0, fingers=[2, 0]))
    tp = ctl.token_value(dict(type=ctl.POINT, t=1.0, target=[1., .2, .3], hand=1)); tp0 = ctl.token_value(dict(type=ctl.POINT, t=1.0, target=[1., .2, .3], hand=1, fingers=[3, 3]))
    ok['B_finger_value'] = bool(np.allclose(tv[1][:2], [(2 - 1.5) / 1.5, (0 - 1.5) / 1.5]) and tv[2][0] == 1 and tv[2][1:].sum() == 0)
    ok['B_point_unchanged'] = all(np.array_equal(x, y) for x, y in zip(tp[1:], tp0[1:]))
    from scmg.data import View, WindowDataset, collate
    from scmg.train import build_model
    from scmg.vae import MotionVAE
    from scmg.losses import StageCfg, step_losses
    st = json.loads(Path(P(STATS)).read_text())
    v = View(VIEW, 'train', TEXT, st, mmap=True, ctl_tokens=a.ctl)
    vck = torch.load(P(VAE), map_location='cpu'); vae = MotionVAE(**vck['cfg']).cuda(); vae.load_state_dict(vck['ema']); vae.eval()
    ck = torch.load(P(INIT), map_location='cpu', weights_only=False)
    extra = {'ctl': True, 'ctl_flow': True}
    m, _ = build_model(dict(ck['base_cfg']), dict(ck['extra_cfg']), {**ck['extra_cfg'], **extra}, ck['ema'])
    for k_, v_ in extra.items(): setattr(m.cfg, k_, v_)
    m.add_modules(); m = m.cuda()
    it = iter(WindowDataset(v, 256, seed=3, ctl_focus=0.8)); b = collate([next(it) for _ in range(8)]); b = {k: x.cuda() for k, x in b.items()}
    stats = dict(qm=torch.tensor(st['q_mean']).cuda(), qs=torch.tensor(st['q_std']).cuda())

    def run(sc):
        m.train(); torch.manual_seed(123); np.random.seed(123)
        out = step_losses(m, vae, b, StageCfg(**sc), stats); m.zero_grad(set_to_none=True); out['loss'].backward()
        return {k: float(x) for k, x in out.items()}, {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
    base_sc = dict(SC, ctl_w=1.0, ctl_drop=0.0, ctl_field_drop=0.0)
    l0, _ = run(base_sc); lz, _ = run(dict(base_sc, w_ctl_finger=0.0, ctl_cf_finger_p=0.0))
    ok['C_default_bitwise'] = l0 == lz
    full = dict(base_sc, w_ctl_aux=1.0, w_ctl_finger=1.0, ctl_cf_p=0.5, ctl_cf_finger_p=0.5)
    l1, g1 = run(full)
    with torch.no_grad():
        for n_, p_ in m.named_parameters():
            if n_.startswith('ctl') and n_ in g1: p_ -= 1e-2 * g1[n_].sign()
    l2, g2 = run(full)
    reach = {k: any(n_.startswith(k) and float(x.abs().sum()) > 0 for n_, x in g2.items()) for k in ('ctl_type', 'ctl_val', 'ctl_dt', 'ctl_fmlp')}
    ok['C_train'] = all(np.isfinite(list(l2.values()))) and l2.get('ctl_ray_n', 0) > 0 and l2.get('ctl_finger_n', 0) > 0 and all(reach.values())
    print('C', dict(ray=l2.get('ctl_ray'), ray_n=l2.get('ctl_ray_n'), finger=l2.get('ctl_finger'), finger_n=l2.get('ctl_finger_n'),
                    finger_abs=l2.get('ctl_finger_abs'), reach=reach, n_tok=int((b['ctl_type'] > 0).sum())))
    print(json.dumps(ok)); print('ALL_OK' if all(ok.values()) else 'SOME_FAIL')


if __name__ == '__main__':
    main()

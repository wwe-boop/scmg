"""Leg-ground-truth FGD (design v4 §4.1, 11-05): upper-body (waist + arms) gesture quality under the ECHO-G official FGD.

For every stem the prediction is cropped to the common length with its ground truth (official crop_to_common_length), then
the ROOT block and the 12 LEG joints of the prediction are overwritten by the ground-truth frames, and the result goes
through the unmodified official FGD class (echo_g.evaluation.g1_motion_cls.FGD: same encoder, %32 tail drop, Frechet kernel).
The official FGD of the unmodified prediction is computed in the same pass (--also-official, on by default) so both numbers
come from one encoder load and one stem set.

robot_repr[T, 39] layout, taken from the official code (not assumed):
  0:6   base_orient_6d   } root      echo_g/evaluation/robot_repr.py BASE_ORIENT_6D_DIM / BASE_YAW_DELTA_DIM / BASE_VEL_LOCAL_DIM,
  6     base_yaw_delta   } (0:10)    g1_motion_cls._split_repr (rr[:, :6], rr[:, 6], rr[:, 7:10], rr[:, 10:]),
  7:10  base_vel_local   }           robot_fgd/featurize.py BASE_DIM = 10
  10:39 29 joint angles in G1_JOINT_NAMES order; G1_JOINT_NAMES[0:12] = left/right hip pitch/roll/yaw, knee, ankle pitch/roll
        -> legs = dims 10:22, waist 22:25, left arm 25:32, right arm 32:39.
The released encoder (weights/evaluation/g1_fgd_encoder.pt) has input_mode='pure2' (sin/cos of the 29 joints only), so the
root block does not enter FGD at all; it is still replaced so that the definition holds for root-covering encoders too.

    PYTHONPATH=$OFF/src python scripts/echog_g1/fgd_leggt.py --pred-dir P --ref-dir R --stems S --fgd-ckpt F [--out J]

Leg-neutral FGD (design v6 C3, 11-05; --leg-neutral): the 12 leg joints are set to the neutral pose (0 rad, the URDF
zero = straight standing legs) in BOTH the prediction and the ground truth, the prediction's root block is replaced by the
GT root (it does not enter the pure2 encoder anyway), then the same official FGD class. Unlike leg-GT FGD it does not copy
GT leg motion into the prediction, so a method's leg changes cannot leak GT information and the number measures the
upper body (waist + arms) only.
"""
import argparse, json, os, sys
import numpy as np
import torch

from echo_g.evaluation import g1_motion_cls as GM
from echo_g.evaluation.robot_repr import (G1_JOINT_NAMES, BASE_ORIENT_6D_DIM, BASE_YAW_DELTA_DIM, BASE_VEL_LOCAL_DIM,
                                          TOTAL_ROBOT_MOTION_DIM)

ROOT_DIMS = BASE_ORIENT_6D_DIM + BASE_YAW_DELTA_DIM + BASE_VEL_LOCAL_DIM            # 10
LEG_JOINTS = [i for i, n in enumerate(G1_JOINT_NAMES) if any(k in n for k in ('hip', 'knee', 'ankle'))]
LEG_DIMS = [ROOT_DIMS + i for i in LEG_JOINTS]                                      # 10..21
REPLACE_DIMS = list(range(ROOT_DIMS)) + LEG_DIMS                                    # 0..21
assert TOTAL_ROBOT_MOTION_DIM == 39 and ROOT_DIMS == 10 and LEG_DIMS == list(range(10, 22)), (ROOT_DIMS, LEG_DIMS)
assert all(G1_JOINT_NAMES[i].startswith(('left_', 'right_')) for i in LEG_JOINTS) and len(LEG_JOINTS) == 12


def leg_gt_replace(pr, gr):
    """(pred, gt) robot_repr [T, 39] tensors -> pred cropped to the common length with root + legs from gt."""
    pr, gr = GM.crop_to_common_length(pr, gr)
    out = pr.clone()
    out[:, REPLACE_DIMS] = gr[:, REPLACE_DIMS]
    return out, gr


LEG_NEUTRAL = 0.0


def leg_neutral(pr, gr):
    """(pred, gt) -> both cropped to the common length, legs = neutral pose in both, pred root = gt root."""
    pr, gr = GM.crop_to_common_length(pr, gr)
    po, go = pr.clone(), gr.clone()
    po[:, :ROOT_DIMS] = gr[:, :ROOT_DIMS]
    po[:, LEG_DIMS] = LEG_NEUTRAL; go[:, LEG_DIMS] = LEG_NEUTRAL
    return po, go


MODES = {'leggt': leg_gt_replace, 'legneutral': leg_neutral, 'official': GM.crop_to_common_length}


def read_stems(path):
    return list(dict.fromkeys(s.strip() for s in open(path) if s.strip()))


def load_pairs(pred_dir, ref_dir, stems, fps=30.0, strict=True):
    """[(stem, pred, gt)] as the official evaluator loads them (load_repr, >= 4 frames)."""
    out = []
    for s in stems:
        pp, rp = f'{pred_dir}/{s}.pt', f'{ref_dir}/{s}.pt'
        if not (os.path.isfile(pp) and os.path.isfile(rp)):
            if strict: raise SystemExit(f'missing {pp if not os.path.isfile(pp) else rp}')
            continue
        pr, gr = GM.load_repr(pp, fps), GM.load_repr(rp, fps)
        if pr is None or gr is None or pr.shape[0] < 4 or gr.shape[0] < 4:
            if strict: raise SystemExit(f'invalid clip {s}')
            continue
        out.append((s, pr, gr))
    return out


def _mode(leggt):
    return ('leggt' if leggt else 'official') if isinstance(leggt, bool) else leggt


def fgd_pairs(net, cfg, device, pairs, leggt):
    """leggt: True / False (leg-GT / official, as before) or a MODES key ('leggt', 'legneutral', 'official')."""
    fn = MODES[_mode(leggt)]
    f = GM.FGD(net, cfg, device)
    for _, pr, gr in pairs:
        pr, gr = fn(pr, gr)
        f.update(pr, gr)
    v, n_p, n_g = f.compute()
    return dict(FGD=v, latent_samples=[int(n_p), int(n_g)], clips=len(pairs))


def latents_pairs(net, cfg, device, pairs, mode):
    """per-clip official-encoder latents {stem: (pred [n, 192], gt [n, 192])} (for recording-level bootstrap)."""
    fn = MODES[_mode(mode)]; out = {}
    for s, pr, gr in pairs:
        pr, gr = fn(pr, gr)
        lp = GM.map2latent_feature(net, cfg, pr, device); lg = GM.map2latent_feature(net, cfg, gr, device)
        if lp is not None and lg is not None: out[s] = (lp, lg)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pred-dir', required=True); ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--stems', required=True, help='stem list file (e.g. val_split / dev_stems.txt)')
    ap.add_argument('--fgd-ckpt', required=True, help='ECHO-G official weights/evaluation/g1_fgd_encoder.pt')
    ap.add_argument('--fps', type=float, default=30.0); ap.add_argument('--tag', default='')
    ap.add_argument('--no-official', action='store_true', help='skip the unmodified official FGD')
    ap.add_argument('--leg-neutral', action='store_true', help='also the leg-neutral FGD (legs = 0 rad in pred and GT)')
    ap.add_argument('--allow-missing', action='store_true', help='skip stems without pred/ref (default: fail like --require-all-stems)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net, cfg = GM.load_g1_ae(a.fgd_ckpt, dev, a.fps)
    pairs = load_pairs(a.pred_dir, a.ref_dir, read_stems(a.stems), a.fps, strict=not a.allow_missing)
    res = dict(tag=a.tag or a.pred_dir, pred_dir=a.pred_dir, ref_dir=a.ref_dir, stems=a.stems, fgd_ckpt=a.fgd_ckpt,
               encoder_input_mode=cfg['input_mode'], replaced_dims=REPLACE_DIMS,
               replaced_joints=[G1_JOINT_NAMES[i] for i in LEG_JOINTS] + ['root(base_orient_6d, base_yaw_delta, base_vel_local)'],
               FGD_leggt=fgd_pairs(net, cfg, dev, pairs, True))
    if not a.no_official: res['FGD_official'] = fgd_pairs(net, cfg, dev, pairs, False)
    if a.leg_neutral: res['FGD_legneutral'] = fgd_pairs(net, cfg, dev, pairs, 'legneutral')
    line = f"[{res['tag']}] N={len(pairs)} FGD_leggt={res['FGD_leggt']['FGD']:.4f}"
    if 'FGD_official' in res: line += f" FGD_official={res['FGD_official']['FGD']:.4f}"
    if 'FGD_legneutral' in res: line += f" FGD_legneutral={res['FGD_legneutral']['FGD']:.4f}"
    print(line, flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, 'w'), indent=1)
    return res


if __name__ == '__main__':
    main()

"""G1 executable-contract loss (11-05, SCMG safety module; user: safety goes into the main table, optimise it on G1).

Contract measured by scripts/echog_g1/executability.py: URDF joint position limits, URDF velocity limits, no MuJoCo
self-collision. Velocity is never binding for gestures (20-37 rad/s), so the loss covers:
  safe_lim   : sum over joints of the excess beyond the Unitree URDF [lower, upper] (rad), frame-weighted mean
  safe_lr    : (relu(D_LR - d_lr) / D_LR)^2, d_lr = min distance between {elbow, wrist roll/pitch/yaw} body origins of
               the left and the right arm (official FK). Calibrated against MuJoCo mesh contacts
               (scripts/echog_g1/collision_proxy_calib.py, 80 clips): every left-right contact frame has d_lr <= 0.08 m,
               GT never collides and its 0.5 % quantile is 0.129 m -> D_LR = 0.10 m penalises collisions, not human motion.
  safe_torso : (relu(D_T - d_t) / D_T)^2, d_t = min distance of those 8 arm points to the torso segment (torso_link
               origin + 0.35 m along its z axis). Weaker proxy (contact frames 0.12-0.20 m overlap non-contact ones);
               D_T = 0.14 m stays below the GT 0.5 % quantile 0.158 m.
Distances are invariant to the base pose, so the FK runs with identity base orientation and zero translation.
Inputs: PHYSICAL robot_repr frames [B, T, 39] (joints = channels 10:39, canonical order echo_g G1_JOINT_NAMES).
"""
import torch
from .g1_contact import _mods

# Unitree g1_custom_collision_29dof.urdf <limit lower upper> (rad), copied verbatim (checked by name at first use)
URDF_LIMITS = {
    'left_hip_pitch': (-2.5307, 2.8798), 'left_hip_roll': (-0.5236, 2.9671), 'left_hip_yaw': (-2.7576, 2.7576),
    'left_knee': (-0.087267, 2.8798), 'left_ankle_pitch': (-0.87267, 0.5236), 'left_ankle_roll': (-0.2618, 0.2618),
    'right_hip_pitch': (-2.5307, 2.8798), 'right_hip_roll': (-2.9671, 0.5236), 'right_hip_yaw': (-2.7576, 2.7576),
    'right_knee': (-0.087267, 2.8798), 'right_ankle_pitch': (-0.87267, 0.5236), 'right_ankle_roll': (-0.2618, 0.2618),
    'waist_yaw': (-2.618, 2.618), 'waist_roll': (-0.52, 0.52), 'waist_pitch': (-0.52, 0.52),
    'left_shoulder_pitch': (-3.0892, 2.6704), 'left_shoulder_roll': (-1.5882, 2.2515), 'left_shoulder_yaw': (-2.618, 2.618),
    'left_elbow': (-1.0472, 2.0944), 'left_wrist_roll': (-1.972222054, 1.972222054),
    'left_wrist_pitch': (-1.614429558, 1.614429558), 'left_wrist_yaw': (-1.614429558, 1.614429558),
    'right_shoulder_pitch': (-3.0892, 2.6704), 'right_shoulder_roll': (-2.2515, 1.5882), 'right_shoulder_yaw': (-2.618, 2.618),
    'right_elbow': (-1.0472, 2.0944), 'right_wrist_roll': (-1.972222054, 1.972222054),
    'right_wrist_pitch': (-1.614429558, 1.614429558), 'right_wrist_yaw': (-1.614429558, 1.614429558),
}
D_LR, D_T, TORSO_LEN = 0.10, 0.14, 0.35
_C = {}


def _consts(device, dtype):
    key = (str(device), dtype)
    if key not in _C:
        M = _mods(); G1K = M['G1K']
        try:
            from echo_g.evaluation.robot_repr import G1_JOINT_NAMES
        except ImportError:             # stubbed echo_g package (no PyYAML in the DLC image): robot_repr.py needs nothing else
            import importlib.util, os, sys
            from .g1_contact import OFFICIAL_SRC
            spec = importlib.util.spec_from_file_location('echo_g.evaluation.robot_repr',
                                                          os.path.join(OFFICIAL_SRC, 'echo_g', 'evaluation', 'robot_repr.py'))
            mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
            G1_JOINT_NAMES = mod.G1_JOINT_NAMES
        names = [n[:-6] if n.endswith('_joint') else n for n in G1_JOINT_NAMES]
        missing = [n for n in names if n not in URDF_LIMITS]
        if missing or len(names) != 29: raise ValueError(f'G1 joint names vs URDF limits: missing {missing}')
        lo = torch.tensor([URDF_LIMITS[n][0] for n in names], device=device, dtype=dtype)
        hi = torch.tensor([URDF_LIMITS[n][1] for n in names], device=device, dtype=dtype)
        BN = list(G1K.G1_BFS_BODY_NAMES)
        arm = lambda s: [BN.index(f'{s}_elbow_link'), BN.index(f'{s}_wrist_roll_link'), BN.index(f'{s}_wrist_pitch_link'), BN.index(f'{s}_wrist_yaw_link')]
        _C[key] = dict(lo=lo, hi=hi, L=arm('left'), R=arm('right'), T=BN.index('torso_link'))
    return _C[key]


def safety_terms(rr):
    """rr [B, T, 39] physical -> per-frame terms [B, T]: lim (rad), lr, torso (normalised squared hinges), d_lr, d_t (m)."""
    M = _mods(); G1K = M['G1K']; c = _consts(rr.device, rr.dtype)
    q = rr[..., 10:]
    lim = (torch.relu(c['lo'] - q) + torch.relu(q - c['hi'])).sum(-1)
    B, T = q.shape[:2]
    eye = torch.eye(3, device=rr.device, dtype=rr.dtype).expand(B, T, 3, 3)
    pos, rot = G1K.g1_forward_kinematics_with_rotations(q, eye, torch.zeros(B, T, 3, device=rr.device, dtype=rr.dtype))
    PL, PR = pos[:, :, c['L']], pos[:, :, c['R']]                                       # [B, T, 4, 3]
    d_lr = (PL[:, :, :, None] - PR[:, :, None]).norm(dim=-1).flatten(2).min(-1).values  # [B, T]
    a0 = pos[:, :, c['T']]; ab = rot[:, :, c['T'], :, 2] * TORSO_LEN                    # torso segment a0 -> a0 + ab
    P = torch.cat([PL, PR], 2)
    s = (((P - a0[:, :, None]) * ab[:, :, None]).sum(-1) / (ab * ab).sum(-1)[..., None]).clamp(0, 1)
    d_t = (P - (a0[:, :, None] + s[..., None] * ab[:, :, None])).norm(dim=-1).min(-1).values
    lr = (torch.relu(D_LR - d_lr) / D_LR) ** 2
    torso = (torch.relu(D_T - d_t) / D_T) ** 2
    return dict(lim=lim, lr=lr, torso=torso, d_lr=d_lr, d_t=d_t)


def contract_loss(pred, w):
    """pred [B, T, 39] physical, w [B, T] frame weights -> dict(safe_lim, safe_lr, safe_torso, safe_viol_frac)."""
    st = safety_terms(pred.float())
    den = w.sum().clamp(min=1.)
    out = {f'safe_{k}': (st[k] * w).sum() / den for k in ('lim', 'lr', 'torso')}
    with torch.no_grad():
        viol = ((st['lim'] > 0) | (st['d_lr'] < 0.08) | (st['d_t'] < 0.12)).float()
        out['safe_viol_frac'] = (viol * w).sum() / den
    return out

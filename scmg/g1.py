"""ECHO-G Unitree G1 motion (robot_repr [T, 39], schema projecthermes-g1-39d-standard-v2, 30 FPS) <-> SCMG 50 Hz frames.

Channel layout (official docs/DATA_FORMAT.md + echo_g/evaluation/robot_repr.py, verified against the files):
   0:6   base orientation, continuous 6D = first two ROWS of the 3x3 rotation matrix (Gram-Schmidt back, as the official
         rotation_6d_to_matrix); frame-0 yaw removed (unitless)
   6     base yaw change since the previous frame (rad / frame; frame 0 = 0)
   7:10  yaw-local root velocity x, y, z (m/s; frame 0 = 0 in the release)
  10:39  29 G1 joint angles (rad) in canonical order: left leg 6, right leg 6, waist 3, left arm 7, right arm 7
The evaluator integrates the root path as yaw[t] = yaw0 + sum_{1..t} yaw_delta, trans[t] = sum_{1..t} R_yaw(t) v(t) / fps.

Resampling (30 <-> 50 FPS), every channel group handled in its own space so that the physical meaning is kept:
  joints, root velocity (m/s)     cubic spline in time (C2), per channel; velocity frame 0 (a 0 placeholder) is replaced by
                                  frame 1 before fitting and written back as 0
  yaw change (rad/frame)          cumulative heading psi = cumsum(yaw_delta[1:]) is splined, then differenced on the new grid
                                  -> the integrated heading is preserved (no x fps/fps' rescaling error)
  base orientation (6D)           6D -> rotation matrix -> scipy RotationSpline (C2 in rotation space) -> 6D
Times: frame i at i / fps. 30 -> 50 uses n50 = round(T30 * 5 / 3) frames (same duration); the last <= 1 new frame past the
last source time is clamped to it (hold, no extrapolation). 50 -> 30 never extrapolates (n50 frames cover T30 / 30 s).
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

SRC_FPS = 30.0
JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
    'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint',
    'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
CHANNELS = ([f'base_rot6d_r{r}c{c}' for r in range(2) for c in range(3)] + ['base_yaw_delta', 'base_vel_x', 'base_vel_y',
            'base_vel_z'] + JOINT_NAMES)
ROOT = list(range(0, 10)); ROT6D = list(range(0, 6)); YAW = 6; VEL = [7, 8, 9]
LEGS = list(range(10, 22)); WAIST = list(range(22, 25)); ARMS = list(range(25, 39)); JOINTS = list(range(10, 39))
UNITS = ['1'] * 6 + ['rad/frame'] + ['m/s'] * 3 + ['rad'] * 29
IDENTITY6D = np.array([1., 0., 0., 0., 1., 0.])
assert len(CHANNELS) == 39 and len(JOINT_NAMES) == 29


def rot6d_to_matrix(d6):
    """Official convention (echo_g.evaluation.rotation.rotation_6d_to_matrix): rows b1, b2, b1 x b2."""
    d6 = np.asarray(d6, np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12)
    return np.stack([b1, b2, np.cross(b1, b2)], -2)


def matrix_to_rot6d(R):
    return np.asarray(R)[..., :2, :].reshape(*np.shape(R)[:-2], 6)


def n_frames_out(n_in, fps_in, fps_out):
    return int(round(n_in * fps_out / fps_in))


def resample(rr, fps_in, fps_out, n_out=None, kind='cubic'):
    """rr [T, 39] physical robot_repr at fps_in -> [n_out, 39] at fps_out (float64). kind 'cubic' (default) | 'linear'
    (linear: joints/velocity/heading linear, orientation Slerp; only for the resampling comparison)."""
    rr = np.asarray(rr, np.float64); T = len(rr)
    if rr.ndim != 2 or rr.shape[1] != 39 or T < 2: raise ValueError(f'robot_repr shape {rr.shape}')
    n_out = n_frames_out(T, fps_in, fps_out) if n_out is None else int(n_out)
    t_in = np.arange(T) / fps_in
    t_out = np.minimum(np.arange(n_out) / fps_out, t_in[-1])
    out = np.empty((n_out, 39))
    lin = lambda y: np.stack([np.interp(t_out, t_in, y[:, j]) for j in range(y.shape[1])], 1)
    fit = (lambda y: CubicSpline(t_in, y, axis=0)(t_out)) if kind == 'cubic' else lin
    out[:, 10:] = fit(rr[:, 10:])
    v = rr[:, 7:10].copy(); v[0] = v[1]
    out[:, 7:10] = fit(v); out[0, 7:10] = 0.
    psi = np.concatenate([[0.], np.cumsum(rr[1:, YAW])])
    p = fit(psi[:, None])[:, 0]
    out[0, YAW] = 0.; out[1:, YAW] = np.diff(p)
    rot = Rotation.from_matrix(rot6d_to_matrix(rr[:, :6]))
    if kind == 'cubic':
        r_out = RotationSpline(t_in, rot)(t_out)
    else:
        from scipy.spatial.transform import Slerp
        r_out = Slerp(t_in, rot)(t_out)
    out[:, :6] = matrix_to_rot6d(r_out.as_matrix())
    return out


def to50(rr30, kind='cubic'):
    return resample(rr30, SRC_FPS, 50.0, kind=kind)


def to30(rr50, n30, kind='cubic'):
    """Generated 50 Hz frames -> exactly n30 frames at 30 FPS (requires (len-1)/50 >= (n30-1)/30)."""
    if (len(rr50) - 1) / 50. + 1e-9 < (n30 - 1) / 30.: raise ValueError(f'{len(rr50)} frames @50 do not cover {n30} @30')
    return resample(rr50, 50.0, SRC_FPS, n_out=n30, kind=kind)


def mask_channels(rr, mode):
    """Report-only 'leg masking' (11-04 user; the official benchmark always uses all 39 channels): returns a copy with
    mode 'legs'      : the 12 leg joints set to 0 rad
    mode 'legs_root' : legs 0 + root (identity orientation, yaw change 0, root velocity 0)."""
    x = np.array(rr, np.float64, copy=True)
    if mode not in ('legs', 'legs_root'): raise ValueError(mode)
    x[:, LEGS] = 0.
    if mode == 'legs_root':
        x[:, ROT6D] = IDENTITY6D; x[:, YAW] = 0.; x[:, VEL] = 0.
    return x

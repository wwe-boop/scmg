"""Geometry shared by every module.

50 Hz joint frames -> motion latents at 12.5 Hz (STRIDE = 4 frames = 80 ms) -> motion chunks of CHUNK = 4 latents
(16 frames = 320 ms). The audio time axis is the master clock: audio feature frame f (20 ms hop) belongs to latent
f // STRIDE and chunk f // (STRIDE*CHUNK).
"""
import os
FPS = 50
CPFS = '/mnt/cpfs/zbl-cpfs-new'


def P(path):
    """Local mirror of CPFS inputs (SCMG_LOCAL_MIRROR=<dir> holding the same relative paths, e.g. ~/bulk/cpfs-mirror on
    the 5090 host where CPFS is only an sshfs mount): returns the mirrored path when it exists, else the original."""
    m = os.environ.get('SCMG_LOCAL_MIRROR'); p = str(path)
    if m and p.startswith(CPFS):
        q = m.rstrip('/') + p[len(CPFS):]
        if os.path.exists(q): return q
    return p
STRIDE = int(os.environ.get('SCMG_STRIDE', '4'))   # frames per latent: 4 -> 12.5 Hz latents (default), 2 -> 25 Hz
if STRIDE not in (2, 4): raise ValueError('SCMG_STRIDE must be 2 or 4')
CHUNK_FRAMES = 16          # a motion chunk is always 320 ms
CHUNK = CHUNK_FRAMES // STRIDE   # latents per chunk (4 at 12.5 Hz, 8 at 25 Hz)
U80 = 4 // STRIDE          # latents per 80 ms: audio horizons / CLI horizons are given in 80 ms units
J = 34                     # EX002 joints in the training view (waist 2, head 2, arms 2x7, three-finger hands 2x8)
FEAT = 89                  # 80 mel + 8 prosody (onset, prominence, energy, logf0, voiced, beat, beat_left, beat_right) + valid
SPK = 192                  # CAM++ speaker embedding
TEXT_DIM = 768             # multilingual-e5-base token features (frozen semantic encoder)
N_SOURCE = 8               # null + 6 original sources + bagger_reversed (id 7)
PROSODY_COLS = {'onset': 80, 'prominence': 81, 'energy': 82, 'logf0': 83, 'voiced': 84, 'beat': 85, 'beat_left': 86,
                'beat_right': 87, 'valid': 88}

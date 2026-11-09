"""Coarse semantic-gesture classes for the Semantic Gesture Head (user 11-03, CoordSpeaker-style g_k):
0 none, 1 point_left, 2 point_right, 3 expand, 4 contract, 5 emphasis, 6 other.
Mapped from the view VOCAB keys (semsynth executed events; SeG lexicon names in seg-lexicon-20260922/CATALOG.private.json)."""
import numpy as np

NAMES = ['none', 'point_left', 'point_right', 'expand', 'contract', 'emphasis', 'other']
N_SEM = len(NAMES)
COARSE = {
    'point:left': 1, 'point:right': 2,
    # expand: PALMS EXPAND, ARMS ENCOMPASS, HANDS REVEAL, ARMS REACH, ARMS SURROUND, HAND OPEN
    'seg.184': 3, 'seg.008': 3, 'seg.140': 3, 'seg.014': 3, 'seg.018': 3, 'seg.112': 3,
    # contract: HANDS UNITE, FINGERS SHUT, HANDS DRAW-BACK, FIST CLENCH
    'seg.151': 4, 'seg.050': 4, 'seg.131': 4, 'seg.059': 4,
    # emphasis: HANDS EMPHASIS, FIST BEAT, FOREFINGER BEAT, PALM LOWER, HAND CHOP, FOREARM CUT, PALM THRUST, ELBOW FALL, tap
    'seg.133': 5, 'seg.057': 5, 'seg.069': 5, 'seg.172': 5, 'seg.101': 5, 'seg.066': 5, 'seg.176': 5, 'seg.036': 5,
    'ext.tap': 5,
}


def coarse_lut(vocab):
    """VOCAB id -> coarse id (int16); id 0 (none) -> 0, id 1 (unknown) -> -1, every other class -> its coarse id (default other)."""
    lut = np.full(max(vocab.values()) + 1, 6, np.int16)
    lut[0] = 0; lut[1] = -1
    for k, i in vocab.items():
        if i >= 2: lut[i] = COARSE.get(k, 6)
    return lut

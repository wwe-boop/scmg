"""Duration guards shared by EX002 mapping and native G1 evaluation."""
import math
import numpy as np


def check_duration(n, source_fps, ref_n, ref_fps):
    # The floor builder uses ceil(T50 / 50 * 30). Only that sampling roundoff is allowed.
    if n < 2 or ref_n < 2 or math.ceil(n * ref_fps / source_fps) != ref_n:
        raise ValueError(f'duration mismatch: {n} frames @ {source_fps} Hz vs {ref_n} @ {ref_fps} Hz')


def replace_upper(ref, pred, columns):
    check_duration(len(pred), 30, len(ref), 30)
    out = ref.copy()
    out[:, columns] = pred[:, columns]
    return out

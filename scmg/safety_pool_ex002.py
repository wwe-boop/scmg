"""11-08 EX002 3c-6: the exact collision / distance work of one projection round spread over W worker processes.
The per-frame calls (lin_rows = exact test + bisection + distance rows; collides) are independent across frames and deterministic,
so the results are bit-identical to the serial calls; only where they run changes. One persistent worker pool per layer; each worker
holds its own EX002 robot / MuJoCo data (EX002RMLimits().E, EX002Collision).
    pool = CollisionPool(4); L = ProjectionLayer(..., pool=pool)"""
from __future__ import annotations
import multiprocessing as mp
import numpy as np

_C = {}


def _init():
    import os
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from .safety_ex002_3f import EX002RMLimits
    from .safety_proj_ex002 import EX002Collision
    lim = EX002RMLimits(); _C['coll'] = EX002Collision(lim.E)


def _lin(args):
    from .safety_proj import lin_rows
    q, q_safe, act, nk = args
    ids = [] if nk.pop('want_ids', False) else None
    if ids is not None: nk['ids'] = ids
    rows, ql = lin_rows(_C['coll'], q, q_safe, act, **nk)
    return rows, ql, ids


def _collides(q):
    return _C['coll'].collides(q)


class CollisionPool:
    def __init__(self, workers=4):
        self.workers = workers
        self.pool = mp.get_context('fork').Pool(workers, initializer=_init)
        self.pool.map(_collides, [np.zeros(34)] * (4 * workers), chunksize=1)   # warm-up: every worker has built its robot model

    def lin_rows(self, jobs):
        """jobs: [(q, q_safe, act, near_kwargs incl. optional want_ids)] -> [(rows, linearisation point, ids or None)]"""
        return self.pool.map(_lin, jobs, chunksize=max(1, len(jobs) // (4 * self.workers)))

    def collides(self, qs):
        return self.pool.map(_collides, [np.asarray(q, float) for q in qs], chunksize=max(1, len(qs) // (4 * self.workers)))

    def close(self):
        self.pool.close(); self.pool.join()

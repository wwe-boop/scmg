"""11-08 EX002 module 3c-7 (design D1, exp/duplex docs/criteria/EX002_CBF_1108.md): block-separable chunk projection.
3c-4b (ProjectionLayer 'l2', recursive, tail_smooth 1e-3, cap 8) with the one 34-joint chunk QP split into three QPs on the same
horizon: finger blocks F_L / F_R (8 joints each) carry the collision rows whose gradient support is inside that hand's finger joints;
the arm block A (18 joints) carries every other row with the finger joints fixed at the finger solution (their columns moved into the
row offset). Position / derivative / terminal rows and the objective are separable per joint, so they split exactly. The certificate
is unchanged: the exact WBC test on the combined pose of every frame and midpoint; cuts are split by the same support rule.
Default path untouched: this is a separate class; ProjectionLayer is not modified."""
from __future__ import annotations
import numpy as np
from . import safety_proj as S

FL, FR, ARM = np.arange(18, 26), np.arange(26, 34), np.arange(0, 18)


class SubLim:
    """the Limits interface ChunkQP needs, restricted to joint indices idx."""
    def __init__(self, lim, idx):
        self.idx = idx; self.n = len(idx); self.lo, self.hi = lim.lo[idx], lim.hi[idx]; self._lim = lim

    def step_bound(self, k, shrink=1.0):
        return self._lim.step_bound(k, shrink)[self.idx]


def block_of(g, rtol=1e-9):
    """support rule: components below rtol x max |g| are round-off of the analytic Jacobian (same-hand finger pairs carry ~1e-16 arm
    entries); the exact test certifies the result either way."""
    a = np.abs(g); nz = np.nonzero(a > rtol * max(a.max(), 1e-300))[0]
    if len(nz) and nz.min() >= FL[0] and nz.max() <= FL[-1]: return 'L'
    if len(nz) and nz.min() >= FR[0] and nz.max() <= FR[-1]: return 'R'
    return 'A'


class BlockProjectionLayer(S.ProjectionLayer):
    def __init__(self, *a, order='fingers_first', **k):
        assert order in ('fingers_first', 'arm_first')
        super().__init__(*a, **k); self.order = order; self.name = f'proj_l2_blocks_{order}'

    def reset(self):
        super().reset(); self.stats.update(block_solves=0, rounds_per_chunk=[], mixed_rows=0, finger_rows=0)

    def _project(self, H, target_k, lin_k, ref, qp=None):
        assert qp is None and self.obj == 'l2' and self.pool is None and not (self.pred_d > 0 or self.carry)
        lim, J = self.lim, self.n
        k = len(target_k); N = k + self.B
        blk = {b: (idx, S.ChunkQP(SubLim(lim, idx), H[:, idx] if len(H) else H.reshape(0, len(idx)), N, terminal=True,
                                  pos_margin=self.pm, shrink=self.shrink)) for b, idx in (('L', FL), ('R', FR), ('A', ARM))}
        if self.recursive and self.plan is not None:
            for b, (idx, q_) in blk.items():
                if b != 'A': q_.xref = ref[:, idx]
        mixed = []                                   # (t, g [J], dist, qlin [J], mid)
        lin = np.concatenate([lin_k, np.repeat(lin_k[-1:], self.B, 0)], 0)
        memo = {}

        def add(t, rows, ql, mid):
            for g, dist in rows:
                b = block_of(g)
                if b == 'A': mixed.append((t, g, dist, ql, mid)); self.stats['mixed_rows'] += 1; continue
                idx, q_ = blk[b]; self.stats['finger_rows'] += 1
                if mid: q_.add_collision_mid(t, [(g[idx], dist)], ql[idx], self.ds)
                else: q_.add_collision(t, [(g[idx], dist)], ql[idx], self.ds)
            return {block_of(g) for g, _ in rows}

        for t in range(N):
            key = (lin[t].tobytes(), ref[t].tobytes())
            if key not in memo: memo[key] = S.lin_rows(self.coll, lin[t], ref[t], self.act)
            rows, ql = memo[key]
            if rows: add(t, rows, ql, False)
        tgt = np.full((N, J), np.nan); tgt[:k] = target_k
        W = np.tile(self.w, (N, 1))
        xF = {'L': None, 'R': None}; dirty = {'L': True, 'R': True}
        x = np.zeros((N, J)); qA_base = blk['A'][1]; nA0 = len(qA_base.crow)
        guess = np.concatenate([np.clip(target_k, lim.lo, lim.hi), np.repeat(np.clip(target_k[-1:], lim.lo, lim.hi), self.B, 0)], 0)
        rounds = 0
        for it in range(self.it):
            rounds += 1
            for b in ('L', 'R'):
                if dirty[b]:
                    idx, q_ = blk[b]; xb, st = q_.solve(tgt[:, idx], W[:, idx], tail_smooth=self.tail_smooth)
                    self.stats['qp_solves'] += 1; self.stats['block_solves'] += 1
                    if xb is None: self.stats['qp_fail'] += 1; self.stats['rounds_per_chunk'].append(rounds); return None, None
                    xF[b] = xb; dirty[b] = False
            fing = x.copy()
            if self.order == 'arm_first' and it == 0: fing[:, FL], fing[:, FR] = guess[:, FL], guess[:, FR]
            else: fing[:, FL], fing[:, FR] = xF['L'], xF['R']
            qA = qA_base; qA.crow, qA.cl, qA.cids = qA.crow[:nA0], qA.cl[:nA0], qA.cids[:nA0]
            Hl = H[-1] if len(H) else None
            for t, g, dist, ql, mid in mixed:            # finger columns into the offset at the current finger trajectory
                gA, gF = g[ARM], np.concatenate([g[FL], g[FR]]); FI = np.concatenate([FL, FR])
                if mid:
                    fp = 0.5 * (fing[t, FI] + (fing[t - 1, FI] if t >= 1 else Hl[FI]))
                    qA.add_collision_mid(t, [(gA, dist - gF @ ql[FI] + gF @ fp)], ql[ARM], self.ds)
                else:
                    qA.add_collision(t, [(gA, dist - gF @ ql[FI] + gF @ fing[t, FI])], ql[ARM], self.ds)
            if self.recursive and self.plan is not None: qA.xref = ref[:, ARM]
            xa, st = qA.solve(tgt[:, ARM], W[:, ARM], tail_smooth=self.tail_smooth)
            self.stats['qp_solves'] += 1; self.stats['block_solves'] += 1
            if xa is None: self.stats['qp_fail'] += 1; self.stats['rounds_per_chunk'].append(rounds); return None, None
            x = np.zeros((N, J)); x[:, ARM] = xa; x[:, FL], x[:, FR] = xF['L'], xF['R']
            Xh = np.concatenate([x, np.repeat(x[-1:], 3, 0)], 0)
            if not (S.pos_ok(x, lim) and S.diff_ok(H, Xh, lim)[0]):
                self.stats['verify_fail'] += 1; self.stats['rounds_per_chunk'].append(rounds); return None, None
            bad = [t for t in range(N) if self.coll.collides(x[t])]
            badm = []
            if self.coll.midpoints and not bad:
                prev = np.concatenate([H[-1:], x[:-1]], 0) if len(H) else x[:-1]
                cur = x if len(H) else x[1:]; off = 0 if len(H) else 1
                badm = [t + off for t in range(len(cur)) if self.coll.collides(0.5 * (cur[t] + prev[t]))]
            if not bad and not badm:
                self.stats['rounds_per_chunk'].append(rounds); return x, None
            self.stats['coll_iters'] += 1
            for t in bad:
                rows, ql = S.lin_rows(self.coll, x[t], ref[t], self.act)
                if rows:
                    for b in add(t, rows, ql, False) - {'A'}: dirty[b] = True
            for t in badm:
                pm = 0.5 * (x[t] + (x[t - 1] if t >= 1 else H[-1])); rm = 0.5 * (ref[t] + (ref[t - 1] if t >= 1 else H[-1]))
                rows, ql = S.lin_rows(self.coll, pm, rm, self.act)
                if rows:
                    for b in add(t, rows, ql, True) - {'A'}: dirty[b] = True
            if self.order == 'arm_first' and it == 0: pass
        self.stats['rounds_per_chunk'].append(rounds)
        return None, None

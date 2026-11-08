"""11-08 EX002 duplex ladder L2 (docs/criteria/EX002_DUPLEX_LADDER_1107.md REVISION): rest-keyframe insertion at the interruption.
From f_i the waist (q34 0-1) and both arms (4-17) follow a quintic (minimum-jerk when the end state is at rest) from the stream's state
at f_i to the speaker's rest pose (human_ref R_sil, 34 joints) with zero end velocity / acceleration, and then HOLD the rest pose to
the end of the window (no release back to the generator until the next turn); head (2-3) and fingers (18-33) keep the continuation.
Ramp code reused from B's module 9b (exp/ctl-port ec51a7b scripts/ex002/ex002_keyframe.py: state, quintic, the 9b duration rule
T = clip(sqrt(5.77 max_j |dq_j| / a95_j), 0.3, 0.8) over the 14 arm joints, a95 = tokens/ARM_ACC_P95.json); the start state is taken
from the 5-frame moving average of the executed stream around f_i (B: the r3 decoder has 4-frame acceleration spikes).
The executed prefix (frames < f_i) is the method's own uninterrupted output U, as in retract_post_ex002.py.
    python scripts/ex002/rest_ramp_ex002.py --in-dir Y --prefix-from U --out-dir O --interrupts I.json --human H [--keyframe-code K]"""
import argparse, json, sys
from pathlib import Path
import numpy as np

KF = '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/scmg/releases/ctlport-ec51a7b/scripts/ex002'
UP = [0, 1] + list(range(4, 18))
A95 = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002/tokens/ARM_ACC_P95.json'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--in-dir', required=True); ap.add_argument('--prefix-from', required=True)
    ap.add_argument('--out-dir', required=True); ap.add_argument('--interrupts', required=True); ap.add_argument('--human', required=True)
    ap.add_argument('--keyframe-code', default=KF)
    a = ap.parse_args()
    sys.path.insert(0, a.keyframe_code)
    from ex002_keyframe import state, quintic, FPS          # B's 9b ramp primitives (not forked)
    a95 = np.asarray(json.loads(Path(A95).read_text())['a95'], np.float64)               # arm joints 4..17
    H = json.loads(Path(a.human if a.human.endswith('.json') else a.human + '.json').read_text())
    rest = np.asarray(H['rest']['R_sil']['joints'], np.float64); assert len(rest) == 34, 'R_sil must be the 34-joint EX002 rest pose'
    ints = json.loads(Path(a.interrupts).read_text())['interrupts']; out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    n = 0; Ts = []
    for r in ints:
        name = f"{r['ci']}@{r['item']}@{r['frac']:.1f}.npy"; fp = Path(a.in_dir) / name
        if not fp.exists(): continue
        f = r['f_i']; y = np.load(fp).astype(np.float64); U = np.load(Path(a.prefix_from) / f"{r['ci']}.npy").astype(np.float64)
        z = np.concatenate([U[:f], y[f:]]); T = len(z)
        sm = z.copy()                       # 5-frame moving average at f-1, f, f+1 (the frames state() differentiates)
        for t_ in (f - 1, f, f + 1):
            if 2 <= t_ < T - 2: sm[t_] = z[t_ - 2:t_ + 3].mean(0)
        _, v0, a0 = state(sm[:, UP], f); p0 = z[f, UP]       # exact position at f (no step), smoothed velocity / acceleration
        dq = np.abs(rest[4:18] - z[f, 4:18]); Tr = float(np.clip(np.sqrt(5.77 * np.max(dq / a95)), 0.3, 0.8)); nr = int(round(Tr * FPS))
        o = z.copy(); zz = np.zeros(len(UP)); e = min(T, f + nr)
        o[f:e, UP] = quintic(p0, v0, a0, rest[UP], zz, zz, Tr, nr)[:e - f]
        o[e:, UP] = rest[UP][None]
        np.save(out / name, o.astype(np.float32)); n += 1; Ts.append(Tr)
    print('REST_RAMP_DONE', n, 'T median %.2f s' % float(np.median(Ts)) if Ts else '')


if __name__ == '__main__':
    main()

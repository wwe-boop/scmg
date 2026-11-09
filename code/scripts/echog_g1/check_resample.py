"""30 -> 50 -> 30 FPS round trip of ECHO-G GT motion (scmg/g1.py): per-channel-group error and the official world-space
jerk / pose-FK position error (official evaluator functions, read-only copy of the release code on PYTHONPATH).
    check_resample.py <motion_39d_30fps dir> <stem list> <n clips> [out json]"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import g1
from echo_g.evaluation import g1_motion_cls as E

mdir, lst, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
stems = [s for s in Path(lst).read_text().split() if s][:n]
T32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
res = {}
for kind in ('cubic', 'linear'):
    err = {k: [] for k in ('rot6d', 'yaw', 'vel', 'joints', 'legs', 'arms')}
    jo, jr, j50, wts, pos_err, drift = [], [], [], [], [], []
    for s in stems:
        d = torch.load(mdir / f'{s}.pt', map_location='cpu', weights_only=True)
        r = d['robot_repr'].double().numpy()
        r50 = g1.to50(r, kind); rb = g1.to30(r50, len(r), kind)
        e = np.abs(rb - r)
        err['rot6d'].append(e[:, :6].max()); err['yaw'].append(e[:, 6].max()); err['vel'].append(e[:, 7:10].max())
        err['joints'].append(e[:, 10:].max()); err['legs'].append(e[:, 10:22].max()); err['arms'].append(e[:, 25:].max())
        pw = E.fk_positions_world(T32(r), 30.); pb = E.fk_positions_world(T32(rb), 30.); p50 = E.fk_positions_world(T32(r50), 50.)
        jo.append(E.Jerk._jerk(pw, 30.)); jr.append(E.Jerk._jerk(pb, 30.)); j50.append(E.Jerk._jerk(p50, 50.))
        wts.append(len(r) - 3)
        pp = E.fk_positions(T32(r)); pq = E.fk_positions(T32(rb))
        pos_err.append(float(np.linalg.norm(pp - pq, axis=-1).mean()))
        drift.append(float(np.linalg.norm(pw[-1, 0] - pb[-1, 0])))      # end-of-clip pelvis drift (m) of the integrated root
    w = np.asarray(wts, float)
    res[kind] = dict(clips=len(stems), max_abs_err={k: float(np.max(v)) for k, v in err.items()},
                     median_clip_max_err={k: float(np.median(v)) for k, v in err.items()},
                     jerk_world_len_weighted=dict(original=float(np.average(jo, weights=w)), roundtrip=float(np.average(jr, weights=w)),
                                                  at_50fps=float(np.average(j50, weights=w))),
                     pose_fk_mean_pos_err_m=float(np.mean(pos_err)), end_root_drift_m=dict(mean=float(np.mean(drift)), max=float(np.max(drift))))
print(json.dumps(res, indent=1))
if len(sys.argv) > 4: Path(sys.argv[4]).write_text(json.dumps(res, indent=1))

"""(c)1 robot-side re-gate of the existing Seamless release07 RESULTs (criteria docs/criteria/VIDEO_REGATE_1107.md). Read-only on
the retarget outputs; no re-retargeting. Per item:
  R0  strict_repair_pass (current: repair > 5 cm on > 20 % of frames -> reject)
  R1  executable after the CBF-QP (scmg/safety.py limits34 margin 0.03 rad on data-v36_1 META bounds): no pos / vel / acc violation
  R2  no self-collision: the retargeter's own exact check on every stored float32 frame (WBC GJK on the MuJoCo EX002 model,
      strict.collision_failed_frames + finger gap), and strict.physical_pass
  R3  no position violation of the controlled / model joints (strict.position_violation_frames, model_position_violation_frames);
      the wrist near-limit frame rate (any wrist joint outside [lo + 0.03, hi - 0.03]) is reported for G2
  R4  repair only near the body: per frame and hand, near = the retargeter's own near weight wnear > 0.5 (core._near_remap:
      smoothstep of the gaussian-filtered human hand-body distance between near_on / near_off); far repair fraction = frames with
      repair > 5 cm on a hand that is NOT near; R4 passes if it is <= 0.20 (the R0 threshold, applied to far frames only)
Runs with the retargeter's python (.venv-sim) from the release07 directory. Writes one JSON line per item.
    python regate_seamless.py <snapshot.jsonl> <out.jsonl> <worker i> <workers n>
"""
import json, os, sys, time
sys.path.insert(0, '/mnt/cpfs/zbl-cpfs-new/USERS/zehan/retarget-release07-headroom-20261026')
sys.path.insert(0, os.environ['SCMG_CODE'])
import numpy as np
from scipy.ndimage import gaussian_filter1d
from retarget_adapter import load
from validate_motion import C
from smplx2ex002_3f.source_policy import options_for
from scmg.safety import limits34, CBFFilter, violation_rate

META = json.load(open('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_1/META.json'))['ex002']
LIM = limits34(META['lower'], META['upper'])
WRIST = [8, 9, 10, 15, 16, 17]


def one(row):
    d = os.path.dirname(row['result']); res = json.load(open(row['result'])); st = res['strict']
    q = np.load(d + '/q_ctrl.npy').astype(np.float64); rc = np.load(d + '/repair_cm.npy')
    fr = np.load(d + '/frames.npz'); summ = json.load(open(d + '/diag.json')).get('summary', {})
    seq, fps = load(row['source']); o = dict(C.DEFAULTS); o.update(options_for(row['source'], {}))
    hs = C.human_side(seq, fps, o); T = min(len(q), hs['T'], len(rc))
    near = np.zeros((T, 2), bool)
    for j, s in enumerate(('left', 'right')):
        dd = gaussian_filter1d(hs['prox'][s]['dist'], 2, mode='nearest')
        near[:, j] = (C._smoothstep((o['near_off'] - dd) / (o['near_off'] - o['near_on'])) > .5)[:T]
    rep = rc[:T] > 5
    far_rep = float((rep & ~near).any(1).mean()); near_rep = float((rep & near).any(1).mean())
    cb = CBFFilter(LIM, q[0]); qc = np.stack([cb.step(x) for x in q]); v = violation_rate(qc, LIM)
    wr = q[:, WRIST]; wr_out = float(((wr < LIM['lo'][WRIST]) | (wr > LIM['hi'][WRIST])).any(1).mean())
    wc = np.stack([fr['left_wrist_cm'], fr['right_wrist_cm']], 1)[:T]
    return dict(id=row['id'], seconds=row['seconds'], subset=row['source'].get('subset'), speaker=row['source'].get('source_id'),
                R0=bool(st['strict_repair_pass']), repair_frac=float(st['repair_gt5cm_fraction']),
                R1=bool(v['pos'] == 0 and v['vel'] == 0 and v['acc'] == 0), post_jerk=v['jerk'], raw_any=violation_rate(q, LIM)['any'],
                R2=bool(st['physical_pass'] and not st['collision_failed_frames'] and not st['finger_gap_failed_frames']),
                R3=bool(st['position_violation_frames'] == 0 and st['model_position_violation_frames'] == 0), wrist_out_frac=wr_out,
                far_repair_frac=far_rep, near_repair_frac=near_rep, near_frac=float(near.any(1).mean()), R4=far_rep <= .2,
                far_wrist_cm=float(wc[~near].mean()) if (~near).any() else None, near_wrist_cm=float(wc[near].mean()) if near.any() else None,
                backoff=summ.get('safety_backoff_scale'), self_contact_frac=res['summary'].get('self_contact_frac'))


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]; out = sys.argv[2]; i, n = int(sys.argv[3]), int(sys.argv[4])
    done = set()
    if os.path.exists(out): done = {json.loads(l)['id'] for l in open(out) if l.strip()}
    t0 = time.time(); k = 0
    with open(out, 'a') as f:
        for r in rows[i::n]:
            if r['id'] in done: continue
            try: x = one(r)
            except Exception as e: x = dict(id=r['id'], error=repr(e)[:300])
            f.write(json.dumps(x) + '\n'); f.flush(); k += 1
            if k % 100 == 0: print(k, f'{time.time() - t0:.0f}s', flush=True)
    print('REGATE_DONE', i, k)


if __name__ == '__main__':
    main()
